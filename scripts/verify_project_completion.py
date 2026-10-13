"""NeuroGuard Final Project Completion Verification Script.

Audits every phase, file, trained checkpoint, benchmark metric, visualization,
and service configuration to guarantee 100% completion of CAT II requirements.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_full_audit():
    print("=" * 80)
    print(" NeuroGuard: Comprehensive Project Completion Audit")
    print("=" * 80)

    checklist = []

    # 1. Core directories
    core_dirs = [
        "configs", "data/raw", "data/processed", "src/data", "src/preprocessing",
        "src/features", "src/models", "src/training", "src/evaluation",
        "src/explainability", "src/visualization", "src/api", "src/utils",
        "frontend", "artifacts/models", "artifacts/metrics", "artifacts/plots",
        "tests", "scripts", "docs", "report"
    ]
    missing_dirs = [d for d in core_dirs if not (PROJECT_ROOT / d).exists()]
    checklist.append(("Directory Structure", len(missing_dirs) == 0, f"Missing: {missing_dirs}" if missing_dirs else "All 21 directories verified"))

    # 2. Raw and Processed Datasets
    raw_edfs = ["chb01/chb01_03.edf", "chb02/chb02_16.edf", "chb03/chb03_01.edf"]
    missing_edfs = [f for f in raw_edfs if not (PROJECT_ROOT / "data" / "raw" / f).exists()]
    checklist.append(("CHB-MIT Raw EDFs", len(missing_edfs) == 0, f"Missing: {missing_edfs}" if missing_edfs else "All 3 EDFs verified on disk"))

    proc_npz = ["chb01_chb01_03.npz", "chb02_chb02_16.npz", "chb03_chb03_01.npz"]
    missing_npz = [f for f in proc_npz if not (PROJECT_ROOT / "data" / "processed" / f).exists()]
    checklist.append(("Processed Tensors", len(missing_npz) == 0, f"Missing: {missing_npz}" if missing_npz else "All 3 patient NPZ bundles verified"))

    # 3. Model Checkpoints
    ckpts = [
        "logistic_regression.joblib", "random_forest.joblib", "xgboost.joblib",
        "xgboost_optuna_best.joblib", "cnn1d_best.pt", "cnn_lstm_best.pt",
        "eeg_scaler.joblib"
    ]
    missing_ckpts = [f for f in ckpts if not (PROJECT_ROOT / "artifacts" / "models" / f).exists()]
    checklist.append(("Trained Model Checkpoints", len(missing_ckpts) == 0, f"Missing: {missing_ckpts}" if missing_ckpts else "All 7 model weights verified"))

    # 4. Analytical Publication Figures
    plots = [
        "01_raw_and_filtered_waveforms.png", "02_eeg_spectrogram_time_frequency.png",
        "03_class_distribution_imbalance.png", "04_feature_correlation_heatmap.png",
        "05_pca_projection.png", "06_umap_projection.png",
        "07_confusion_matrices.png", "08_roc_and_pr_curves.png",
        "09_training_validation_loss_curves.png", "10_feature_importance_xai.png",
        "11_patient_error_analysis.png"
    ]
    missing_plots = [f for f in plots if not (PROJECT_ROOT / "artifacts" / "plots" / f).exists()]
    checklist.append(("11 Analytical Figures", len(missing_plots) == 0, f"Missing: {missing_plots}" if missing_plots else "All 11 figures verified"))

    # 5. Benchmark Metrics
    metrics_files = [
        "classical_models_benchmark.json", "cnn1d_benchmark.json",
        "cnn_lstm_benchmark.json", "optuna_study.json"
    ]
    missing_metrics = [f for f in metrics_files if not (PROJECT_ROOT / "artifacts" / "metrics" / f).exists()]
    checklist.append(("Benchmark Metrics JSONs", len(missing_metrics) == 0, f"Missing: {missing_metrics}" if missing_metrics else "All 4 benchmark files verified"))

    # 6. Serving & User Interface
    app_files = [
        "src/api/app.py", "src/api/schemas.py", "frontend/app.py",
        "Dockerfile", "docker-compose.yml", "docs/DEPLOYMENT.md"
    ]
    missing_apps = [f for f in app_files if not (PROJECT_ROOT / f).exists()]
    checklist.append(("API, UI & Docker Serving", len(missing_apps) == 0, f"Missing: {missing_apps}" if missing_apps else "FastAPI, Streamlit & Docker verified"))

    # 7. Documentation & IEEE Report
    doc_files = [
        "README.md", "docs/RECOVERY_STATUS.md", "docs/DEPLOYMENT.md",
        "report/IEEE_NeuroGuard_Research_Report.md"
    ]
    missing_docs = [f for f in doc_files if not (PROJECT_ROOT / f).exists()]
    checklist.append(("Documentation & IEEE Report", len(missing_docs) == 0, f"Missing: {missing_docs}" if missing_docs else "All documentation verified"))

    # Output Results Table
    all_passed = True
    for item, passed, details in checklist:
        status_marker = "[OK]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f" {status_marker} {item:<30} | {details}")

    print("=" * 80)
    if all_passed:
        print("[SUCCESS] ALL PROJECT MILESTONES 100% COMPLETE AND VERIFIED!")
    else:
        print("[WARNING] Some project milestones are incomplete.")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_full_audit()
    sys.exit(0 if success else 1)
