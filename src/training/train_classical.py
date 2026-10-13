"""Training and Patient-Independent Evaluation Script for Classical Models.

Trains Logistic Regression, Random Forest, and XGBoost on patient chb01,
validates on patient chb02, and tests on held-out patient chb03.
Saves model checkpoints to artifacts/models/ and benchmark logs to artifacts/metrics/.
"""

import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_project_root
from src.utils.seed import set_seed
from src.models.classical import get_classical_model
from src.evaluation.metrics import calculate_metrics, save_metrics_json


def load_dataset_bundle(npz_path: Path) -> Dict[str, Any]:
    """Load preprocessed features, labels, and metadata from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "features": data["features"],
        "labels": data["labels"],
        "patient_id": str(data["patient_id"]),
        "recording_name": str(data["recording_name"]),
        "feature_names": [str(x) for x in data["feature_names"]],
    }


def train_and_evaluate_all():
    set_seed(42)
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    models_dir = root / "artifacts" / "models"
    metrics_dir = root / "artifacts" / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" NeuroGuard: Patient-Independent Classical Model Training & Benchmark")
    print("=" * 75)

    # 1. Load Data Partitions
    train_file = processed_dir / "chb01_chb01_03.npz"
    val_file = processed_dir / "chb02_chb02_16.npz"
    test_file = processed_dir / "chb03_chb03_01.npz"

    for f in [train_file, val_file, test_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing processed data partition: {f}")

    d_train = load_dataset_bundle(train_file)
    d_val = load_dataset_bundle(val_file)
    d_test = load_dataset_bundle(test_file)

    X_train, y_train = d_train["features"], d_train["labels"]
    X_val, y_val = d_val["features"], d_val["labels"]
    X_test, y_test = d_test["features"], d_test["labels"]

    print(f"[*] Training Partition:   Patient {d_train['patient_id']} | Shape: {X_train.shape} | Seizures: {np.sum(y_train==1)}/{len(y_train)}")
    print(f"[*] Validation Partition: Patient {d_val['patient_id']} | Shape: {X_val.shape} | Seizures: {np.sum(y_val==1)}/{len(y_val)}")
    print(f"[*] Held-Out Test Part.:  Patient {d_test['patient_id']} | Shape: {X_test.shape} | Seizures: {np.sum(y_test==1)}/{len(y_test)}")
    print("-" * 75)

    models_to_train = [
        ("logistic_regression", {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}),
        ("random_forest", {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"}),
        ("xgboost", {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05}),
    ]

    benchmark_results: Dict[str, Any] = {
        "experiment": "patient_independent_classical_baselines",
        "partitions": {
            "train_patient": d_train["patient_id"],
            "val_patient": d_val["patient_id"],
            "test_patient": d_test["patient_id"],
        },
        "models": {},
    }

    print(f"{'Model Name':<20} | {'Set':<5} | {'Acc':<6} | {'Recall':<6} | {'Spec':<6} | {'Prec':<6} | {'F1':<6} | {'ROC-AUC':<7} | {'PR-AUC':<6}")
    print("-" * 75)

    for model_name, kwargs in models_to_train:
        clf = get_classical_model(model_name, **kwargs)

        # Fit model on training set
        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # Evaluate on Validation
        val_probs = clf.predict_proba(X_val)
        val_preds = clf.predict(X_val)
        val_metrics = calculate_metrics(y_val, val_preds, val_probs)

        # Evaluate on Held-Out Test
        test_probs = clf.predict_proba(X_test)
        test_preds = clf.predict(X_test)
        test_metrics = calculate_metrics(y_test, test_preds, test_probs)

        # Save model checkpoint
        ckpt_path = models_dir / f"{model_name}.joblib"
        clf.save(ckpt_path)

        # Log results
        benchmark_results["models"][model_name] = {
            "checkpoint_path": str(ckpt_path),
            "validation_metrics": val_metrics.to_dict(),
            "test_metrics": test_metrics.to_dict(),
        }

        print(
            f"{clf.name:<20} | {'VAL':<5} | {val_metrics.accuracy:<6.4f} | {val_metrics.recall:<6.4f} | "
            f"{val_metrics.specificity:<6.4f} | {val_metrics.precision:<6.4f} | {val_metrics.f1:<6.4f} | "
            f"{val_metrics.roc_auc:<7.4f} | {val_metrics.pr_auc:<6.4f}"
        )
        print(
            f"{clf.name:<20} | {'TEST':<5} | {test_metrics.accuracy:<6.4f} | {test_metrics.recall:<6.4f} | "
            f"{test_metrics.specificity:<6.4f} | {test_metrics.precision:<6.4f} | {test_metrics.f1:<6.4f} | "
            f"{test_metrics.roc_auc:<7.4f} | {test_metrics.pr_auc:<6.4f}"
        )
        print("-" * 75)

    # Save benchmark JSON
    benchmark_json = metrics_dir / "classical_models_benchmark.json"
    save_metrics_json(benchmark_results, benchmark_json)
    print(f"\n[OK] Benchmark results saved to: {benchmark_json}")
    print(f"[OK] Checkpoints saved to: {models_dir}")
    print("=" * 75)


if __name__ == "__main__":
    train_and_evaluate_all()
