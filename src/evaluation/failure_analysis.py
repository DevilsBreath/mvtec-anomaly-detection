# src/evaluation/failure_analysis.py
"""
Failure Case Analysis
Systematic analysis of false positives and false negatives.
This is the part that separates serious projects from tutorial clones.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd


def find_failure_cases(
    image_paths: List[str],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, List[int]]:
    """
    Identifies false positives and false negatives given a threshold.

    Args:
        image_paths: list of image file paths
        labels: (N,) ground truth labels (0=normal, 1=anomalous)
        scores: (N,) anomaly scores
        threshold: decision boundary

    Returns:
        Dict with 'false_positives' and 'false_negatives' indices
    """
    predictions = (scores >= threshold).astype(int)

    false_positives = np.where((predictions == 1) & (labels == 0))[0].tolist()
    false_negatives = np.where((predictions == 0) & (labels == 1))[0].tolist()
    true_positives = np.where((predictions == 1) & (labels == 1))[0].tolist()
    true_negatives = np.where((predictions == 0) & (labels == 0))[0].tolist()

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
    }


def optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Finds threshold that maximises F1 score.
    Use this for failure analysis (not for AUROC computation).
    """
    from sklearn.metrics import f1_score
    thresholds = np.percentile(scores, np.linspace(0, 100, 200))
    best_thresh, best_f1 = 0.0, 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh


def visualize_failure_cases(
    failure_indices: List[int],
    image_paths: List[str],
    anomaly_maps: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    failure_type: str,
    output_path: str,
    max_samples: int = 8,
) -> None:
    """
    Creates a grid visualization of failure cases with anomaly maps overlaid.

    Args:
        failure_indices: indices of failure cases to visualize
        image_paths: all image paths
        anomaly_maps: (N, H, W) anomaly score maps
        labels: ground truth labels
        scores: anomaly scores
        failure_type: 'false_positive' or 'false_negative'
        output_path: where to save the figure
        max_samples: cap visualization at this many samples
    """
    indices = failure_indices[:max_samples]
    if not indices:
        print(f"No {failure_type} cases found.")
        return

    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    title_color = "red" if failure_type == "false_positive" else "orange"

    for row, idx in enumerate(indices):
        img = cv2.imread(image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (anomaly_maps.shape[2], anomaly_maps.shape[1]))

        amap = anomaly_maps[idx]
        amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

        axes[row, 0].imshow(img_resized)
        axes[row, 0].set_title(f"Original\nLabel: {'Anomalous' if labels[idx] else 'Normal'}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(amap, cmap="hot")
        axes[row, 1].set_title(f"Anomaly Map\nScore: {scores[idx]:.4f}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"Overlay\n({failure_type.replace('_', ' ').title()})")
        axes[row, 2].axis("off")

    fig.suptitle(
        f"{failure_type.replace('_', ' ').title()} Cases ({n} shown)",
        fontsize=14, color=title_color, fontweight="bold"
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_failure_report(
    category: str,
    image_paths: List[str],
    labels: np.ndarray,
    scores: np.ndarray,
    anomaly_maps: np.ndarray,
    output_dir: str,
) -> pd.DataFrame:
    """
    Full failure analysis pipeline for a single category.
    Saves visualizations and returns a summary DataFrame.
    """
    threshold = optimal_threshold(labels, scores)
    cases = find_failure_cases(image_paths, labels, scores, threshold)

    print(f"\n[{category}] Threshold: {threshold:.4f}")
    print(f"  TP: {len(cases['true_positives'])} | "
          f"TN: {len(cases['true_negatives'])} | "
          f"FP: {len(cases['false_positives'])} | "
          f"FN: {len(cases['false_negatives'])}")

    out = Path(output_dir) / category
    out.mkdir(parents=True, exist_ok=True)

    # Visualize false positives
    visualize_failure_cases(
        cases["false_positives"], image_paths, anomaly_maps, labels, scores,
        failure_type="false_positive",
        output_path=str(out / "false_positives.png"),
    )

    # Visualize false negatives
    visualize_failure_cases(
        cases["false_negatives"], image_paths, anomaly_maps, labels, scores,
        failure_type="false_negative",
        output_path=str(out / "false_negatives.png"),
    )

    # Summary DataFrame
    records = []
    for case_type, indices in cases.items():
        for idx in indices:
            records.append({
                "category": category,
                "image_path": image_paths[idx],
                "true_label": labels[idx],
                "anomaly_score": scores[idx],
                "case_type": case_type,
            })

    return pd.DataFrame(records)
