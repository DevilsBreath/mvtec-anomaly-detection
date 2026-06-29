# src/models/patchcore.py
"""
Custom PatchCore Implementation
Based on: "Towards Total Recall in Industrial Anomaly Detection" (Roth et al., CVPR 2022)

This is the custom implementation used for experiments beyond the anomalib baseline.
Understand every component here — you'll be asked about it in interviews.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torchvision import models
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm
import faiss


class FeatureExtractor(nn.Module):
    """
    Extracts intermediate feature maps from a pretrained CNN backbone.
    Backbone is frozen — we only use it as a fixed feature extractor.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = None,
        weights_dir: Optional[str] = None,
    ):
        super().__init__()
        if layers is None:
            layers = ["layer2", "layer3"]

        if weights_dir is None:
            weights_dir = Path(__file__).resolve().parents[2] / "Model"
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Load pretrained backbone from a project-local folder.
        if backbone_name == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
            backbone = models.wide_resnet50_2(weights=None)
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            backbone = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        state_dict = torch.hub.load_state_dict_from_url(
            weights.url,
            model_dir=str(self.weights_dir),
            progress=True,
            check_hash=True,
        )
        backbone.load_state_dict(state_dict)

        # Freeze all parameters — we never update these
        for param in backbone.parameters():
            param.requires_grad = False

        # Register hooks to capture intermediate feature maps
        self.features = {}
        self.hooks = []
        self.layers = layers

        for layer_name in layers:
            layer = dict(backbone.named_children())[layer_name]
            hook = layer.register_forward_hook(self._make_hook(layer_name))
            self.hooks.append(hook)

        self.backbone = backbone
        self.backbone.eval()

    def _make_hook(self, layer_name: str):
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.features = {}
        with torch.no_grad():
            _ = self.backbone(x)
        return dict(self.features)

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


class LocallyAwarePatchFeatures:
    """
    Aggregates patch features from neighbourhood context.
    This is the "locally aware" part of PatchCore — each patch feature
    incorporates information from its spatial neighbours.
    """

    @staticmethod
    def aggregate(
        features: Dict[str, torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None,
        kernel_size: int = 3,
    ) -> torch.Tensor:
        """
        Combines multi-scale features and applies neighbourhood aggregation.

        Args:
            features: Dict of {layer_name: feature_map (B, C, H, W)}
            target_size: Resize all maps to this spatial size before combining
            kernel_size: Neighbourhood aggregation kernel size

        Returns:
            Aggregated patch features: (B, C_total, H, W)
        """
        feature_maps = list(features.values())

        # Determine target spatial size from largest feature map
        if target_size is None:
            max_h = max(f.shape[2] for f in feature_maps)
            max_w = max(f.shape[3] for f in feature_maps)
            target_size = (max_h, max_w)

        # Resize all feature maps to same spatial size
        resized = []
        for fmap in feature_maps:
            if fmap.shape[2:] != target_size:
                fmap = F.interpolate(fmap, size=target_size, mode="bilinear", align_corners=False)
            resized.append(fmap)

        # Concatenate along channel dimension
        combined = torch.cat(resized, dim=1)  # (B, C1+C2, H, W)

        # Neighbourhood aggregation via average pooling
        # This makes each patch feature context-aware
        if kernel_size > 1:
            padding = kernel_size // 2
            combined = F.avg_pool2d(combined, kernel_size=kernel_size, stride=1, padding=padding)

        return combined


class CoresetSampler:
    """
    Greedy coreset subsampling — selects a representative subset of patches
    to store in the memory bank. Without this, memory bank grows too large.
    """

    @staticmethod
    def sample(features: np.ndarray, ratio: float = 0.1) -> np.ndarray:
        """
        Greedy coreset approximation.
        Selects the subset that maximises coverage of the full feature space.

        Args:
            features: (N, D) array of patch features
            ratio: Fraction of patches to keep

        Returns:
            Selected patch features: (M, D) where M = ratio * N
        """
        n_samples = max(1, int(len(features) * ratio))
        selected_indices = []

        # Start with a random point
        current_idx = np.random.randint(len(features))
        selected_indices.append(current_idx)

        # Distances from each point to nearest selected point
        min_distances = np.full(len(features), np.inf)

        for _ in tqdm(range(n_samples - 1), desc="Coreset sampling", leave=False):
            # Update min distances with latest selected point
            new_distances = np.linalg.norm(
                features - features[current_idx], axis=1
            )
            min_distances = np.minimum(min_distances, new_distances)

            # Select point furthest from all selected points
            current_idx = np.argmax(min_distances)
            selected_indices.append(current_idx)

        return features[selected_indices]


class PatchCore:
    """
    Full PatchCore pipeline.

    Usage:
        model = PatchCore(backbone="wide_resnet50_2", device="cuda")
        model.fit(train_dataloader)          # build memory bank
        scores, maps = model.predict(test_dataloader)
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] = None,
        coreset_ratio: float = 0.1,
        num_neighbors: int = 9,
        device: str = "cuda",
        weights_dir: Optional[str] = None,
    ):
        if layers is None:
            layers = ["layer2", "layer3"]

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors

        self.feature_extractor = FeatureExtractor(
            backbone,
            layers,
            weights_dir=weights_dir,
        ).to(self.device)
        self.patch_aggregator = LocallyAwarePatchFeatures()
        self.memory_bank = None  # Set during fit()
        self.index = None        # FAISS index for fast nearest-neighbour search

    def _extract_patch_features(self, dataloader) -> np.ndarray:
        """Extracts patch features from all training images."""
        all_features = []

        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch["image"].to(self.device)
            raw_features = self.feature_extractor(images)
            patch_features = self.patch_aggregator.aggregate(raw_features)

            # Reshape: (B, C, H, W) -> (B*H*W, C)
            B, C, H, W = patch_features.shape
            patch_features = patch_features.permute(0, 2, 3, 1).reshape(-1, C)
            all_features.append(patch_features.cpu().numpy())

        return np.concatenate(all_features, axis=0).astype(np.float32)

    def fit(self, train_dataloader) -> None:
        """Builds the memory bank from normal training images."""
        print("Building memory bank...")
        patch_features = self._extract_patch_features(train_dataloader)

        print(f"Total patches: {len(patch_features):,} | Applying coreset subsampling...")
        self.memory_bank = CoresetSampler.sample(patch_features, self.coreset_ratio)
        print(f"Memory bank size: {len(self.memory_bank):,} patches")

        # Build FAISS index for fast nearest-neighbour search
        d = self.memory_bank.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.memory_bank)
        print("Memory bank ready.")

    def predict(self, test_dataloader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs inference on test data.

        Returns:
            image_scores: (N,) anomaly score per image
            anomaly_maps: (N, H, W) pixel-level anomaly maps
            labels: (N,) ground truth labels
        """
        assert self.index is not None, "Call fit() before predict()"

        all_scores = []
        all_maps = []
        all_labels = []

        for batch in tqdm(test_dataloader, desc="Running inference"):
            images = batch["image"].to(self.device)
            labels = batch["label"].numpy()

            raw_features = self.feature_extractor(images)
            patch_features = self.patch_aggregator.aggregate(raw_features)

            B, C, H, W = patch_features.shape
            patches_flat = patch_features.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy().astype(np.float32)

            # K-NN search in memory bank
            distances, _ = self.index.search(patches_flat, self.num_neighbors)

            # Anomaly score = distance to nearest normal patch
            # Using max of k-NN distances for robustness
            patch_scores = distances[:, 0]  # distance to 1-NN

            # Reshape back to spatial maps
            patch_scores = patch_scores.reshape(B, H, W)

            # Image-level score = max patch score
            image_scores = patch_scores.max(axis=(1, 2))

            all_scores.append(image_scores)
            all_maps.append(patch_scores)
            all_labels.append(labels)

        return (
            np.concatenate(all_scores),
            np.concatenate(all_maps),
            np.concatenate(all_labels),
        )

    def save(self, path: str) -> None:
        """Save memory bank to disk."""
        np.save(path, self.memory_bank)
        print(f"Memory bank saved to {path}")

    def load(self, path: str) -> None:
        """Load memory bank from disk."""
        self.memory_bank = np.load(path)
        d = self.memory_bank.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.memory_bank)
        print(f"Memory bank loaded: {len(self.memory_bank):,} patches")
