# src/evaluation/metrics.py
"""
Evaluation Metrics for Anomaly Detection
Implements: Image AUROC, Pixel AUROC, PRO Score, and per-category reporting.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from scipy.ndimage import label as scipy_label
from typing import Dict, List, Tuple, Optional
import warnings
import cv2


def compute_image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Image-level AUROC: Can the model distinguish normal vs. anomalous images?

    Args:
        labels: (N,) binary ground truth (0=normal, 1=anomalous)
        scores: (N,) anomaly scores (higher = more anomalous)

    Returns:
        AUROC score in [0, 1]
    """
    if len(np.unique(labels)) < 2:
        warnings.warn("Only one class present in labels. AUROC is undefined.")
        return float("nan")
    return roc_auc_score(labels, scores)


def compute_pixel_auroc(
    gt_masks: np.ndarray,
    anomaly_maps: np.ndarray,
) -> float:
    """
    Pixel-level AUROC: Can the model localize defects at pixel level?

    Args:
        gt_masks: (N, H, W) binary ground truth masks
        anomaly_maps: (N, H, W) pixel-level anomaly scores

    Returns:
        Pixel AUROC score
    """
    if gt_masks.shape[1:] != anomaly_maps.shape[1:]:
        anomaly_maps = _resize_anomaly_maps(gt_masks, anomaly_maps)

    gt_flat = gt_masks.flatten()
    pred_flat = anomaly_maps.flatten()

    if len(np.unique(gt_flat)) < 2:
        warnings.warn("No anomalous pixels in ground truth. Pixel AUROC undefined.")
        return float("nan")

    return roc_auc_score(gt_flat, pred_flat)


def compute_pro_score(
    gt_masks: np.ndarray,
    anomaly_maps: np.ndarray,
    num_thresholds: int = 100,
    max_fpr: float = 0.3,
) -> float:
    """
    Per-Region Overlap (PRO) Score.
    Unlike pixel AUROC, this weights each connected defect region equally
    regardless of size — fairer for datasets with mixed defect sizes.

    Args:
        gt_masks: (N, H, W) binary ground truth masks
        anomaly_maps: (N, H, W) anomaly scores
        num_thresholds: number of threshold points
        max_fpr: integration limit (standard = 0.3)

    Returns:
        PRO score (area under PRO curve, normalized)
    """
    if gt_masks.shape[1:] != anomaly_maps.shape[1:]:
        anomaly_maps = _resize_anomaly_maps(gt_masks, anomaly_maps)

    thresholds = np.linspace(anomaly_maps.min(), anomaly_maps.max(), num_thresholds)
    fprs = []
    pros = []

    for thresh in thresholds:
        binary_pred = (anomaly_maps >= thresh).astype(np.uint8)

        region_overlaps = []
        total_normal_pixels = 0
        false_positive_pixels = 0

        for gt_mask, pred_mask in zip(gt_masks, binary_pred):
            # Count false positives on normal pixels
            normal_pixels = (gt_mask == 0)
            total_normal_pixels += normal_pixels.sum()
            false_positive_pixels += (pred_mask[normal_pixels] == 1).sum()

            # Per-region overlap for anomalous regions
            labeled_gt, n_regions = scipy_label(gt_mask)
            for region_idx in range(1, n_regions + 1):
                region_mask = (labeled_gt == region_idx)
                overlap = (pred_mask[region_mask] == 1).sum() / region_mask.sum()
                region_overlaps.append(overlap)

        fpr = false_positive_pixels / (total_normal_pixels + 1e-8)
        pro = np.mean(region_overlaps) if region_overlaps else 0.0

        fprs.append(fpr)
        pros.append(pro)

    # Normalize and integrate up to max_fpr
    fprs = np.array(fprs)
    pros = np.array(pros)

    mask = fprs <= max_fpr
    if mask.sum() < 2:
        return float("nan")

    # Trapezoidal integration requires FPR to be sorted in ascending order.
    sorted_indices = np.argsort(fprs[mask])
    sorted_fprs = fprs[mask][sorted_indices]
    sorted_pros = pros[mask][sorted_indices]

    pro_score = np.trapz(sorted_pros, sorted_fprs) / max_fpr
    return float(pro_score)


def _resize_anomaly_maps(gt_masks: np.ndarray, anomaly_maps: np.ndarray) -> np.ndarray:
    """Resize anomaly maps to match the spatial resolution of the masks."""
    target_h, target_w = gt_masks.shape[1:]
    resized_maps = np.empty((anomaly_maps.shape[0], target_h, target_w), dtype=np.float32)

    for idx, anomaly_map in enumerate(anomaly_maps):
        resized_maps[idx] = cv2.resize(
            anomaly_map.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )

    return resized_maps


def compute_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    gt_masks: Optional[np.ndarray] = None,
    anomaly_maps: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes all relevant metrics for a single category.

    Returns:
        Dict with image_auroc, pixel_auroc (if masks provided), pro_score (if masks provided)
    """
    results = {}

    results["image_auroc"] = compute_image_auroc(labels, scores)

    if gt_masks is not None and anomaly_maps is not None:
        if gt_masks.shape[0] != anomaly_maps.shape[0]:
            raise ValueError(
                "gt_masks and anomaly_maps must contain the same number of samples"
            )
        results["pixel_auroc"] = compute_pixel_auroc(gt_masks, anomaly_maps)
        results["pro_score"] = compute_pro_score(gt_masks, anomaly_maps)

    return results


def summarize_results(per_category_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Computes mean metrics across all categories.
    This is the headline number you report on your resume.
    """
    all_image_auroc = [v["image_auroc"] for v in per_category_results.values()
                       if not np.isnan(v.get("image_auroc", float("nan")))]
    all_pixel_auroc = [v["pixel_auroc"] for v in per_category_results.values()
                       if not np.isnan(v.get("pixel_auroc", float("nan")))]
    all_pro = [v["pro_score"] for v in per_category_results.values()
               if not np.isnan(v.get("pro_score", float("nan")))]

    summary = {
        "mean_image_auroc": np.mean(all_image_auroc) if all_image_auroc else float("nan"),
        "mean_pixel_auroc": np.mean(all_pixel_auroc) if all_pixel_auroc else float("nan"),
        "mean_pro_score": np.mean(all_pro) if all_pro else float("nan"),
        "n_categories": len(per_category_results),
    }
    return summary


def print_results_table(
    per_category_results: Dict[str, Dict[str, float]],
    summary: Dict[str, float],
) -> None:
    """Prints a clean results table to terminal."""
    print("\n" + "=" * 65)
    print(f"{'Category':<20} {'Image AUROC':>12} {'Pixel AUROC':>12} {'PRO Score':>10}")
    print("-" * 65)
    for cat, metrics in sorted(per_category_results.items()):
        img_auc = f"{metrics.get('image_auroc', float('nan')):.4f}"
        pix_auc = f"{metrics.get('pixel_auroc', float('nan')):.4f}"
        pro = f"{metrics.get('pro_score', float('nan')):.4f}"
        print(f"{cat:<20} {img_auc:>12} {pix_auc:>12} {pro:>10}")
    print("=" * 65)
    print(f"{'MEAN':<20} {summary['mean_image_auroc']:>12.4f} "
          f"{summary['mean_pixel_auroc']:>12.4f} "
          f"{summary['mean_pro_score']:>10.4f}")
    print("=" * 65 + "\n")
