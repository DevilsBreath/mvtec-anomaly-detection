# test_mlflow.py — run this from project root
import sys
sys.path.insert(0, '.')
from src.utils.mlflow_utils import setup_mlflow, log_full_experiment

setup_mlflow("sqlite:///mlflow.db", "patchcore_baseline")

# Dummy results to verify logging works
dummy_results = {
    "bottle": {"image_auroc": 0.998, "pixel_auroc": 0.981, "pro_score": 0.912},
    "carpet": {"image_auroc": 0.984, "pixel_auroc": 0.975, "pro_score": 0.883},
}
dummy_summary = {
    "mean_image_auroc": 0.991,
    "mean_pixel_auroc": 0.978,
    "mean_pro_score": 0.897,
    "n_categories": 2.0,
}
dummy_params = {"backbone": "wide_resnet50_2", "coreset_ratio": 0.1}

import os
os.makedirs("results/patchcore", exist_ok=True)

log_full_experiment(
    model_name="patchcore",
    per_category_results=dummy_results,
    summary=dummy_summary,
    params=dummy_params,
    results_dir="results/patchcore",
)
print("Done — refresh MLflow UI")