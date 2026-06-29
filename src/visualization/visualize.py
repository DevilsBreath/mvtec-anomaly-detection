# src/visualization/visualize.py
"""
Visualization utilities for anomaly detection results.
Clean outputs for README, reports, and presentations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve


def plot_anomaly_maps(
    images: np.ndarray,
    anomaly_maps: np.ndarray,
    gt_masks: Optional[np.ndarray],
    labels: np.ndarray,
    scores: np.ndarray,
    output_path: str,
    n_samples: int = 5,
) -> None:
    """
    Plots a grid: original | gt mask | anomaly heatmap | overlay.
    """
    n_cols = 4 if gt_masks is not None else 3
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))

    # Pick balanced samples (normal + anomalous)
    normal_idx = np.where(labels == 0)[0]
    anomal_idx = np.where(labels == 1)[0]
    n_normal = min(n_samples // 2, len(normal_idx))
    n_anomal = min(n_samples - n_normal, len(anomal_idx))
    selected = list(np.random.choice(normal_idx, n_normal, replace=False)) + \
               list(np.random.choice(anomal_idx, n_anomal, replace=False))

    for row, idx in enumerate(selected[:n_samples]):
        img = images[idx] if images[idx].max() <= 1.0 else images[idx] / 255.0
        amap = anomaly_maps[idx]
        amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

        # Heatmap
        heatmap = cv2.applyColorMap(
            (amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # Overlay
        img_uint8 = (img * 255).astype(np.uint8)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        if img_uint8.shape[:2] != heatmap_uint8.shape[:2]:
            img_uint8 = cv2.resize(img_uint8, (heatmap_uint8.shape[1], heatmap_uint8.shape[0]))
        overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_uint8, 0.4, 0) / 255.0

        col = 0
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{'Anomalous' if labels[idx] else 'Normal'}\nScore: {scores[idx]:.3f}")
        axes[row, col].axis("off")
        col += 1

        if gt_masks is not None:
            axes[row, col].imshow(gt_masks[idx], cmap="gray")
            axes[row, col].set_title("GT Mask")
            axes[row, col].axis("off")
            col += 1

        axes[row, col].imshow(amap, cmap="hot")
        axes[row, col].set_title("Anomaly Map")
        axes[row, col].axis("off")
        col += 1

        axes[row, col].imshow(overlay)
        axes[row, col].set_title("Overlay")
        axes[row, col].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_roc_curves(
    per_category_results: Dict[str, Dict],
    output_path: str,
) -> None:
    """
    Plots ROC curves for all categories on one figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab20", len(per_category_results))

    for i, (cat, data) in enumerate(sorted(per_category_results.items())):
        fpr, tpr, _ = roc_curve(data["labels"], data["scores"])
        auroc = data["image_auroc"]
        ax.plot(fpr, tpr, color=cmap(i), lw=1.5, label=f"{cat} ({auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Categories (Image-Level)", fontsize=14)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_results_bar_chart(
    per_category_results: Dict[str, Dict[str, float]],
    metric: str = "image_auroc",
    output_path: str = "results/bar_chart.png",
) -> None:
    """
    Bar chart of per-category performance — great for README.
    """
    categories = sorted(per_category_results.keys())
    values = [per_category_results[c].get(metric, 0) for c in categories]
    mean_val = np.mean(values)

    colors = ["#e74c3c" if v < 0.95 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(mean_val, color="navy", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_val:.4f}")
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=1, alpha=0.7,
               label="0.95 reference")

    ax.set_ylim(0.8, 1.01)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"PatchCore — {metric.replace('_', ' ').title()} per Category", fontsize=13)
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison(
    patchcore_results: Dict[str, float],
    efficientad_results: Dict[str, float],
    metric: str = "image_auroc",
    output_path: str = "results/comparisons/model_comparison.png",
) -> None:
    """
    Side-by-side bar chart comparing PatchCore vs EfficientAD.
    """
    categories = sorted(set(patchcore_results) & set(efficientad_results))
    x = np.arange(len(categories))
    width = 0.35

    pc_vals = [patchcore_results[c] for c in categories]
    ead_vals = [efficientad_results[c] for c in categories]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, pc_vals, width, label=f"PatchCore (mean={np.mean(pc_vals):.3f})",
           color="#3498db", edgecolor="white")
    ax.bar(x + width / 2, ead_vals, width, label=f"EfficientAD (mean={np.mean(ead_vals):.3f})",
           color="#e67e22", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"PatchCore vs EfficientAD — {metric.replace('_', ' ').title()}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
