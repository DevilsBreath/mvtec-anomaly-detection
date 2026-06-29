# scripts/run_efficientad.py
"""
EfficientAD comparison on subset of MVTec categories.
Compares inference speed and accuracy against PatchCore.

Usage:
    python scripts/run_efficientad.py --config configs/efficientad.yaml
"""

import argparse
import sys
import json
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import compute_all_metrics, summarize_results, print_results_table
from src.visualization.visualize import plot_model_comparison
from src.utils.mlflow_utils import setup_mlflow, log_full_experiment, log_category_run

console = Console()


def benchmark_inference(model, dataloader, n_runs: int = 3) -> dict:
    """
    Benchmarks inference speed: FPS and latency.
    Runs multiple times and averages to reduce variance.
    """
    import torch
    times = []
    n_images = 0

    for _ in range(n_runs):
        start = time.perf_counter()
        for batch in dataloader:
            n_images += len(batch["image"])
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    fps = (n_images / n_runs) / avg_time
    latency_ms = (avg_time / (n_images / n_runs)) * 1000

    return {"fps": fps, "latency_ms": latency_ms}


def run_efficientad_anomalib(cfg, categories: list) -> dict:
    """Runs EfficientAD via anomalib."""
    from anomalib.models import EfficientAd
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    per_category_results = {}

    for category in categories:
        console.print(f"\n[bold cyan]EfficientAD — Category: {category}[/bold cyan]")

        datamodule = MVTecAD(
            root=cfg.dataset.root,
            category=category,
            train_batch_size=cfg.dataset.train_batch_size,
            eval_batch_size=cfg.dataset.test_batch_size,
            num_workers=cfg.dataset.num_workers,
        )

        model = EfficientAd()

        engine = Engine(
            default_root_dir=str(Path(cfg.output.results_dir) / category),
            max_steps=10000,
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
        metrics["total_time_s"] = elapsed

        per_category_results[category] = metrics
        console.print(f"  Image AUROC: {metrics.get('image_auroc', 'N/A'):.4f}")

        # Log to MLflow after each category
        log_category_run(
            category=category,
            model_name="efficientad",
            metrics={k: v for k, v in metrics.items() if isinstance(v, float)},
            params={},
        )

    return per_category_results


def compare_models(
    patchcore_results_path: str,
    efficientad_results: dict,
    output_dir: str,
) -> None:
    """
    Loads existing PatchCore results and generates comparison plots.
    """
    with open(patchcore_results_path) as f:
        pc_data = json.load(f)

    pc_results = {
        cat: metrics["image_auroc"]
        for cat, metrics in pc_data["per_category"].items()
        if "image_auroc" in metrics
    }

    ead_results = {
        cat: metrics["image_auroc"]
        for cat, metrics in efficientad_results.items()
        if "image_auroc" in metrics
    }

    # Only compare categories present in both
    common_cats = set(pc_results) & set(ead_results)
    pc_filtered = {c: pc_results[c] for c in common_cats}
    ead_filtered = {c: ead_results[c] for c in common_cats}

    plot_model_comparison(
        patchcore_results=pc_filtered,
        efficientad_results=ead_filtered,
        metric="image_auroc",
        output_path=str(Path(output_dir) / "patchcore_vs_efficientad.png"),
    )

    # Print comparison table
    console.print("\n[bold]PatchCore vs EfficientAD Comparison[/bold]")
    console.print(f"{'Category':<20} {'PatchCore':>12} {'EfficientAD':>12} {'Delta':>10}")
    console.print("-" * 55)
    for cat in sorted(common_cats):
        delta = ead_filtered[cat] - pc_filtered[cat]
        sign = "+" if delta >= 0 else ""
        console.print(
            f"{cat:<20} {pc_filtered[cat]:>12.4f} {ead_filtered[cat]:>12.4f} "
            f"[{'green' if delta >= 0 else 'red'}]{sign}{delta:>9.4f}[/]"
        )

    pc_mean = np.mean(list(pc_filtered.values()))
    ead_mean = np.mean(list(ead_filtered.values()))
    console.print("-" * 55)
    console.print(f"{'MEAN':<20} {pc_mean:>12.4f} {ead_mean:>12.4f}")


def main():
    parser = argparse.ArgumentParser(description="EfficientAD on MVTec AD")
    parser.add_argument("--config", default="configs/efficientad.yaml")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument(
        "--patchcore-results",
        default="results/patchcore/results.json",
        help="Path to PatchCore results for comparison",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    categories = args.categories or list(cfg.categories)

    console.print(f"\n[bold green]EfficientAD on MVTec AD[/bold green]")
    console.print(f"Categories: {categories}\n")

    setup_mlflow(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)

    per_category_results = run_efficientad_anomalib(cfg, categories)

    clean_results = {
        cat: {k: v for k, v in m.items() if isinstance(v, float)}
        for cat, m in per_category_results.items()
    }
    summary = summarize_results(clean_results)
    print_results_table(clean_results, summary)

    # Save results
    Path(cfg.output.results_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(cfg.output.results_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump({"per_category": clean_results, "summary": summary}, f, indent=2)

    # Compare against PatchCore if results exist
    if Path(args.patchcore_results).exists():
        compare_models(
            patchcore_results_path=args.patchcore_results,
            efficientad_results=clean_results,
            output_dir="results/comparisons",
        )
    else:
        console.print(
            f"\n[yellow]PatchCore results not found at {args.patchcore_results}. "
            "Run run_patchcore.py first for comparison.[/yellow]"
        )

    # Log to MLflow
    log_full_experiment(
        model_name="efficientad",
        per_category_results=clean_results,
        summary=summary,
        params={},
        results_dir=cfg.output.results_dir,
    )


if __name__ == "__main__":
    main()
