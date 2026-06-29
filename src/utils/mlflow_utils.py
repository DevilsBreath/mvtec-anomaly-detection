# src/utils/mlflow_utils.py
"""
MLflow experiment tracking utilities.
Wraps MLflow API for clean, consistent logging across experiments.
"""

try:
    import mlflow
    import mlflow.pytorch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "MLflow is not installed. Install project dependencies with "
        "`pip install -r requirements.txt` before running tracking scripts."
    ) from exc

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


def setup_mlflow(tracking_uri: str = "sqlite:///mlflow.db", experiment_name: str = "anomaly_detection"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"View UI: run `mlflow ui --backend-store-uri {tracking_uri}` then open http://localhost:5000")


def log_category_run(
    category: str,
    model_name: str,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    artifacts: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Logs a single category experiment run to MLflow.

    Args:
        category: MVTec category name (e.g. 'bottle')
        model_name: 'patchcore' or 'efficientad'
        metrics: dict of metric_name -> value
        params: model hyperparameters
        artifacts: dict of artifact_name -> file_path (e.g. plots)
        tags: additional tags

    Returns:
        run_id
    """
    run_tags = {
        "model": model_name,
        "category": category,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    if tags:
        run_tags.update(tags)

    with mlflow.start_run(run_name=f"{model_name}_{category}") as run:
        # Log params
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log metrics
        for k, v in metrics.items():
            if v is not None and not np.isnan(v):
                mlflow.log_metric(k, v)

        # Log tags
        mlflow.set_tags(run_tags)

        # Log artifact files (plots, etc.)
        if artifacts:
            for name, path in artifacts.items():
                if Path(path).exists():
                    mlflow.log_artifact(path, artifact_path=name)

        return run.info.run_id


def log_full_experiment(
    model_name: str,
    per_category_results: Dict[str, Dict[str, float]],
    summary: Dict[str, float],
    params: Dict[str, Any],
    results_dir: str,
) -> None:
    """
    Logs the complete multi-category experiment as a parent run with child runs.
    Creates a clean hierarchical view in the MLflow UI.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"{model_name}_full_mvtec") as parent_run:
        # Log summary metrics on parent run
        for k, v in summary.items():
            if isinstance(v, float) and not np.isnan(v):
                mlflow.log_metric(f"summary_{k}", v)

        mlflow.log_params(params)
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("dataset", "MVTec AD")
        mlflow.set_tag("n_categories", str(len(per_category_results)))

        # Log summary JSON as artifact
        summary_path = Path(results_dir) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "model": model_name,
                "params": params,
                "per_category": per_category_results,
                "summary": summary,
            }, f, indent=2)
        mlflow.log_artifact(str(summary_path))

        # Child runs per category
        for category, metrics in per_category_results.items():
            with mlflow.start_run(
                run_name=f"{model_name}_{category}",
                nested=True
            ):
                for k, v in metrics.items():
                    if isinstance(v, float) and not np.isnan(v):
                        mlflow.log_metric(k, v)
                mlflow.log_param("category", category)
                mlflow.set_tag("model", model_name)

    print(f"Experiment logged. Parent run ID: {parent_run.info.run_id}")
