"""
Main script: PatchCore baseline on all MVTec categories.
Uses anomalib for the baseline run (fast, reliable).
Switch USE_CUSTOM_IMPL=True to use the custom implementation.

Usage:
    python scripts/run_patchcore.py --config configs/patchcore.yaml
    python scripts/run_patchcore.py --config configs/patchcore.yaml --categories bottle carpet
    python scripts/run_patchcore.py --config configs/patchcore.yaml --custom
    python scripts/run_patchcore.py --config configs/patchcore.yaml --resume
"""

import argparse
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mvtec_dataset import get_dataloader, validate_mvtec_structure, MVTEC_CATEGORIES
from src.evaluation.metrics import compute_all_metrics, summarize_results, print_results_table
from src.evaluation.failure_analysis import generate_failure_report
from src.visualization.visualize import plot_results_bar_chart
from src.utils.mlflow_utils import setup_mlflow, log_full_experiment, log_category_run

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Resume logic — checks which categories already have valid results on disk
# ---------------------------------------------------------------------------

def load_existing_results(results_dir: str) -> dict:
    """
    Loads previously completed results from results.json if it exists.
    Returns a dict of {category: metrics} for categories already done.
    """
    results_path = Path(results_dir) / "results.json"
    if not results_path.exists():
        return {}
    try:
        with open(results_path) as f:
            data = json.load(f)
        existing = data.get("per_category", {})
        # Only count categories that have a valid image_auroc (not nan, not missing)
        valid = {
            cat: metrics for cat, metrics in existing.items()
            if isinstance(metrics.get("image_auroc"), float)
            and not np.isnan(metrics["image_auroc"])
        }
        if valid:
            print(f"\n[RESUME] Found {len(valid)} completed categories: {list(valid.keys())}")
        return valid
    except Exception as e:
        print(f"[WARN] Could not load existing results: {e}")
        return {}


def filter_pending_categories(categories: list, existing: dict) -> list:
    """Returns only categories that haven't been successfully run yet."""
    pending = [c for c in categories if c not in existing]
    skipped = [c for c in categories if c in existing]
    if skipped:
        print(f"[SKIP] Already completed: {skipped}")
    if pending:
        print(f"[RUN]  Pending: {pending}")
    return pending


# ---------------------------------------------------------------------------
# anomalib PatchCore run
# ---------------------------------------------------------------------------

def run_anomalib_patchcore(cfg, categories: list) -> dict:
    """
    Runs PatchCore via anomalib library.
    Rich removed intentionally — anomalib owns the Rich live display.
    """
    from anomalib.models import Patchcore
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    per_category_results = {}
    total = len(categories)

    try:
        for i, category in enumerate(categories):
            print(f"\n[{i+1}/{total}] Category: {category}")
            print("-" * 40)

            datamodule = MVTecAD(
                root=cfg.dataset.root,
                category=category,
                train_batch_size=cfg.dataset.train_batch_size,
                eval_batch_size=cfg.dataset.test_batch_size,
                num_workers=cfg.dataset.num_workers,
            )

            model = Patchcore(
                backbone=cfg.model.backbone,
                layers=list(cfg.model.layers_to_extract),
                coreset_sampling_ratio=cfg.model.coreset_sampling_ratio,
                num_neighbors=cfg.model.num_neighbors,
            )

            engine = Engine(
                default_root_dir=str(Path(cfg.output.results_dir) / category),
            )

            start = time.time()
            engine.fit(model=model, datamodule=datamodule)
            test_results = engine.test(model=model, datamodule=datamodule)
            elapsed = time.time() - start

            metrics = {}
            if test_results:
                result_dict = test_results[0] if isinstance(test_results, list) else test_results
                metrics["image_auroc"] = result_dict.get("image_AUROC", float("nan"))
                metrics["pixel_auroc"] = result_dict.get("pixel_AUROC", float("nan"))
            metrics["inference_time_s"] = elapsed

            per_category_results[category] = metrics

            img_auc = metrics.get("image_auroc", float("nan"))
            pix_auc = metrics.get("pixel_auroc", float("nan"))
            print(f"  Image AUROC : {img_auc:.4f}")
            print(f"  Pixel AUROC : {pix_auc:.4f}")
            print(f"  Time        : {elapsed:.1f}s")

            # Log to MLflow after each category — safe even if later ones crash
            log_category_run(
                category=category,
                model_name="patchcore",
                metrics={k: v for k, v in metrics.items() if isinstance(v, float)},
                params=OmegaConf.to_container(cfg.model, resolve=True),
            )

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Run stopped by user. Completed categories saved.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise

    return per_category_results


# ---------------------------------------------------------------------------
# Custom PatchCore run
# ---------------------------------------------------------------------------

def run_custom_patchcore(cfg, categories: list) -> dict:
    """
    Runs the custom PatchCore implementation from src/models/patchcore.py.
    """
    from src.models.patchcore import PatchCore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    per_category_results = {}
    total = len(categories)

    for i, category in enumerate(categories):
        print(f"\n[{i+1}/{total}] Category: {category}")
        print("-" * 40)

        train_loader = get_dataloader(
            root=cfg.dataset.root,
            category=category,
            split="train",
            image_size=tuple(cfg.dataset.image_size),
            batch_size=cfg.dataset.train_batch_size,
            num_workers=cfg.dataset.num_workers,
            return_masks=False,
        )

        test_loader = get_dataloader(
            root=cfg.dataset.root,
            category=category,
            split="test",
            image_size=tuple(cfg.dataset.image_size),
            batch_size=cfg.dataset.test_batch_size,
            num_workers=cfg.dataset.num_workers,
            return_masks=True,
        )

        model = PatchCore(
            backbone=cfg.model.backbone,
            layers=list(cfg.model.layers_to_extract),
            coreset_ratio=cfg.model.coreset_sampling_ratio,
            num_neighbors=cfg.model.num_neighbors,
            device=device,
        )

        start = time.time()
        model.fit(train_loader)
        scores, anomaly_maps, labels = model.predict(test_loader)
        elapsed = time.time() - start

        gt_masks = []
        image_paths = []
        for batch in test_loader:
            gt_masks.append(batch["mask"].squeeze(1).numpy())
            image_paths.extend(batch["image_path"])
        gt_masks = np.concatenate(gt_masks)

        metrics = compute_all_metrics(labels, scores, gt_masks, anomaly_maps)
        metrics["inference_time_s"] = elapsed
        metrics["labels"] = labels.tolist()
        metrics["scores"] = scores.tolist()

        per_category_results[category] = metrics

        print(f"  Image AUROC : {metrics['image_auroc']:.4f}")
        print(f"  Pixel AUROC : {metrics.get('pixel_auroc', float('nan')):.4f}")
        print(f"  PRO Score   : {metrics.get('pro_score', float('nan')):.4f}")
        print(f"  Time        : {elapsed:.1f}s")

        clean_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
        log_category_run(
            category=category,
            model_name="patchcore",
            metrics=clean_metrics,
            params=OmegaConf.to_container(cfg.model, resolve=True),
        )

        model_save_dir = Path(cfg.output.results_dir) / category
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_save_dir / "memory_bank.npy"))

        if cfg.evaluation.save_visualizations:
            generate_failure_report(
                category=category,
                image_paths=image_paths,
                labels=labels,
                scores=scores,
                anomaly_maps=anomaly_maps,
                output_dir=str(Path(cfg.output.results_dir) / "failure_analysis"),
            )

    return per_category_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PatchCore on MVTec AD")
    parser.add_argument("--config", default="configs/patchcore.yaml")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Subset of categories to run. Default: all 15.")
    parser.add_argument("--custom", action="store_true",
                        help="Use custom PatchCore implementation instead of anomalib.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip categories that already have valid results in results.json.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    categories = args.categories or list(cfg.categories)

    print("\nPatchCore on MVTec AD")
    print(f"Implementation : {'Custom' if args.custom else 'anomalib'}")
    print(f"Resume mode    : {args.resume}")
    print(f"Categories     : {categories}\n")

    if not validate_mvtec_structure(cfg.dataset.root):
        print("[ERROR] Dataset structure invalid. Fix before running.")
        sys.exit(1)

    setup_mlflow(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)

    # Resume: load already-completed results and skip those categories
    existing_results = {}
    if args.resume:
        existing_results = load_existing_results(cfg.output.results_dir)
        categories = filter_pending_categories(categories, existing_results)

    if not categories:
        print("\nAll categories already completed. Nothing to run.")
        print("Remove --resume or delete results/patchcore/results.json to re-run.")
        sys.exit(0)

    # Run pending categories
    if args.custom:
        new_results = run_custom_patchcore(cfg, categories)
    else:
        new_results = run_anomalib_patchcore(cfg, categories)

    # Merge new results with previously completed ones
    all_results = {**existing_results, **new_results}

    # Strip non-numeric values before summarizing
    clean_results = {
        cat: {k: v for k, v in metrics.items() if isinstance(v, float)}
        for cat, metrics in all_results.items()
    }
    summary = summarize_results(clean_results)
    print_results_table(clean_results, summary)

    # Save merged results to disk (overwrites with full picture)
    Path(cfg.output.results_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(cfg.output.results_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump({"per_category": clean_results, "summary": summary}, f, indent=2)
    print(f"\nResults saved: {results_path}")

    plot_results_bar_chart(
        clean_results,
        metric="image_auroc",
        output_path=str(Path(cfg.output.results_dir) / "image_auroc_bar.png"),
    )

    params = OmegaConf.to_container(cfg.model, resolve=True)
    log_full_experiment(
        model_name="patchcore",
        per_category_results=clean_results,
        summary=summary,
        params=params,
        results_dir=cfg.output.results_dir,
    )

    print("\nDone.")
    print(f"Mean Image AUROC : {summary['mean_image_auroc']:.4f}")
    print(f"Mean Pixel AUROC : {summary['mean_pixel_auroc']:.4f}")


if __name__ == "__main__":
    main()