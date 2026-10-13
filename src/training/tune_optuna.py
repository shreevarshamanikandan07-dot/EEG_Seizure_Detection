"""Hyperparameter Optimization with Optuna and MLflow Tracking.

Automates Bayesian hyperparameter tuning across learning rate, tree depth,
scale_pos_weight, and estimator count to maximize patient-independent validation F1.
Logs every trial to MLflow and evaluates the best model on held-out test patient chb03.
"""

import sys
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import optuna

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_project_root
from src.utils.seed import set_seed
from src.training.train_classical import load_dataset_bundle
from src.models.classical import XGBoostBaseline
from src.evaluation.metrics import calculate_metrics, save_metrics_json
from src.training.mlflow_logger import MLflowTracker

# Suppress verbose Optuna trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_study(n_trials: int = 8) -> Dict[str, Any]:
    """Run Optuna study on XGBoost with patient-independent evaluation."""
    set_seed(42)
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    models_dir = root / "artifacts" / "models"
    metrics_dir = root / "artifacts" / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    tracker = MLflowTracker(experiment_name="Optuna_Hyperparameter_Optimization")

    # Load Partitions
    d_train = load_dataset_bundle(processed_dir / "chb01_chb01_03.npz")
    d_val = load_dataset_bundle(processed_dir / "chb02_chb02_16.npz")
    d_test = load_dataset_bundle(processed_dir / "chb03_chb03_01.npz")

    X_train, y_train = d_train["features"], d_train["labels"]
    X_val, y_val = d_val["features"], d_val["labels"]
    X_test, y_test = d_test["features"], d_test["labels"]

    print("=" * 75)
    print(" NeuroGuard: Optuna Hyperparameter Optimization")
    print(f" Trials: {n_trials} | Objective: Maximize Validation F1-Score")
    print("=" * 75)

    trial_history = []

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        n_est = trial.suggest_int("n_estimators", 40, 140, step=20)
        max_d = trial.suggest_int("max_depth", 3, 7)
        lr = trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
        spw = trial.suggest_float("scale_pos_weight", 3.0, 10.0, step=1.0)

        clf = XGBoostBaseline(
            n_estimators=n_est,
            max_depth=max_d,
            learning_rate=lr,
            scale_pos_weight=spw,
            random_state=42,
        )

        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        val_probs = clf.predict_proba(X_val)
        val_preds = clf.predict(X_val)
        val_m = calculate_metrics(y_val, val_preds, val_probs)

        # Log to MLflow
        run_name = f"optuna_trial_{trial.number}"
        tracker.start_run(run_name=run_name, tags={"model": "XGBoost", "trial": str(trial.number)})
        tracker.log_params({
            "n_estimators": n_est,
            "max_depth": max_d,
            "learning_rate": lr,
            "scale_pos_weight": spw,
        })
        tracker.log_metrics({
            "val_f1": val_m.f1,
            "val_roc_auc": val_m.roc_auc,
            "val_precision": val_m.precision,
            "val_recall": val_m.recall,
            "val_accuracy": val_m.accuracy,
        })
        tracker.end_run()

        trial_history.append({
            "trial": trial.number,
            "params": trial.params,
            "val_f1": val_m.f1,
            "val_roc_auc": val_m.roc_auc,
        })

        print(
            f"  Trial {trial.number+1}/{n_trials} | F1: {val_m.f1:.4f} | ROC-AUC: {val_m.roc_auc:.4f} | "
            f"params: max_depth={max_d}, lr={lr:.3f}, scale_pos_weight={spw:.1f}"
        )
        return val_m.f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_p = study.best_params
    print("-" * 75)
    print(f"[*] Optimal Trial: #{study.best_trial.number} with Best Validation F1: {study.best_value:.4f}")
    print(f"[*] Best Parameters: {best_p}")

    # Retrain best model and evaluate on held-out test patient chb03
    best_clf = XGBoostBaseline(
        n_estimators=best_p["n_estimators"],
        max_depth=best_p["max_depth"],
        learning_rate=best_p["learning_rate"],
        scale_pos_weight=best_p["scale_pos_weight"],
        random_state=42,
    )
    best_clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    test_probs = best_clf.predict_proba(X_test)
    test_preds = best_clf.predict(X_test)
    test_m = calculate_metrics(y_test, test_preds, test_probs)

    best_ckpt = models_dir / "xgboost_optuna_best.joblib"
    best_clf.save(best_ckpt)

    print("\n" + "=" * 75)
    print(" Best Tuned Model Performance on Held-Out Test Set (Patient chb03):")
    print(test_m.summary_table())
    print("=" * 75)

    study_summary = {
        "best_trial_number": study.best_trial.number,
        "best_val_f1": study.best_value,
        "best_parameters": best_p,
        "test_metrics": test_m.to_dict(),
        "trial_history": trial_history,
        "best_checkpoint": str(best_ckpt),
    }

    study_json = metrics_dir / "optuna_study.json"
    save_metrics_json(study_summary, study_json)
    print(f"[OK] Optuna study summary saved to: {study_json}")
    return study_summary


if __name__ == "__main__":
    run_optuna_study(n_trials=8)
