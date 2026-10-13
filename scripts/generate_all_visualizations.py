"""Master Visualization Generation Script for NeuroGuard.

Generates all 11 required analytical visualizations directly from real CHB-MIT datasets,
trained models, and experimental evaluation logs. Saves high-res figures to artifacts/plots/.
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_project_root
from src.data.edf_reader import EDFReader
from src.preprocessing.preprocessor import EEGPreprocessor
from src.training.train_classical import load_dataset_bundle
from src.models.cnn1d import EEG1DCNN
from src.models.cnn_lstm import EEG_CNNLSTM
from src.explainability.explainer import EEGExplainer
from src.visualization.visualizer import EEGVisualizer


def main():
    print("=" * 75)
    print(" NeuroGuard: Generating All 11 Academic Visualizations")
    print("=" * 75)

    root = get_project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    models_dir = root / "artifacts" / "models"
    metrics_dir = root / "artifacts" / "metrics"
    plots_dir = root / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    visualizer = EEGVisualizer(output_dir=plots_dir)

    # 1. Load Processed Partitions
    d_train = load_dataset_bundle(processed_dir / "chb01_chb01_03.npz")
    d_val = load_dataset_bundle(processed_dir / "chb02_chb02_16.npz")
    d_test = load_dataset_bundle(processed_dir / "chb03_chb03_01.npz")

    data_train_npz = np.load(processed_dir / "chb01_chb01_03.npz")
    data_test_npz = np.load(processed_dir / "chb03_chb03_01.npz")

    windows_train = data_train_npz["windows"]
    labels_train = data_train_npz["labels"]
    windows_test = data_test_npz["windows"]
    labels_test = data_test_npz["labels"]
    feature_names = [str(x) for x in data_train_npz["feature_names"]]

    # -------------------------------------------------------------
    # Plot 1: Raw vs Filtered Waveforms
    # -------------------------------------------------------------
    print("[*] Generating Figure 1: Raw vs Filtered Waveforms...")
    edf_path = raw_dir / "chb01" / "chb01_03.edf"
    raw_sig, sfreq, ch_names = EDFReader.load_recording(edf_path, tmin=2990.0, tmax=3000.0)
    prep = EEGPreprocessor()
    filtered_sig = prep.filter_signal(raw_sig)
    p1 = visualizer.plot_01_waveforms(raw_sig, filtered_sig, sfreq=sfreq, channel_names=ch_names, duration_sec=6.0)
    print(f"    [OK] Saved: {p1.name}")

    # -------------------------------------------------------------
    # Plot 2: Spectrogram Analysis
    # -------------------------------------------------------------
    print("[*] Generating Figure 2: Time-Frequency Spectrogram...")
    sz_idx = np.where(labels_train == 1)[0][0]
    non_sz_idx = np.where(labels_train == 0)[0][0]
    p2 = visualizer.plot_02_spectrogram(windows_train[non_sz_idx], windows_train[sz_idx], sfreq=sfreq)
    print(f"    [OK] Saved: {p2.name}")

    # -------------------------------------------------------------
    # Plot 3: Class Distribution Imbalance
    # -------------------------------------------------------------
    print("[*] Generating Figure 3: Class Distribution & Imbalance...")
    p3 = visualizer.plot_03_class_distribution(d_train["labels"], d_val["labels"], d_test["labels"])
    print(f"    [OK] Saved: {p3.name}")

    # -------------------------------------------------------------
    # Plot 4: Feature Correlation Heatmap
    # -------------------------------------------------------------
    print("[*] Generating Figure 4: Feature Correlation Heatmap...")
    p4 = visualizer.plot_04_correlation_heatmap(d_train["features"], feature_names)
    print(f"    [OK] Saved: {p4.name}")

    # -------------------------------------------------------------
    # Plot 5: 2D PCA Projection
    # -------------------------------------------------------------
    print("[*] Generating Figure 5: 2D PCA Projection...")
    p5 = visualizer.plot_05_pca(d_test["features"], d_test["labels"])
    print(f"    [OK] Saved: {p5.name}")

    # -------------------------------------------------------------
    # Plot 6: Manifold Embedding (t-SNE / UMAP Clustering)
    # -------------------------------------------------------------
    print("[*] Generating Figure 6: Non-Linear Manifold Projection...")
    p6 = visualizer.plot_06_manifold_clustering(d_test["features"], d_test["labels"])
    print(f"    [OK] Saved: {p6.name}")

    # -------------------------------------------------------------
    # Load Models for Predictions & Comparative Curves
    # -------------------------------------------------------------
    lr_data = joblib.load(models_dir / "logistic_regression.joblib")
    rf_data = joblib.load(models_dir / "random_forest.joblib")
    xgb_data = joblib.load(models_dir / "xgboost.joblib")

    lr_model = lr_data["model"]
    rf_model = rf_data["model"]
    xgb_model = xgb_data["model"]

    cnn1d = EEG1DCNN(in_channels=windows_test.shape[1], num_classes=2)
    cnn1d.load(models_dir / "cnn1d_best.pt")

    cnn_lstm = EEG_CNNLSTM(in_channels=windows_test.shape[1], num_classes=2)
    cnn_lstm.load(models_dir / "cnn_lstm_best.pt")

    # Predict on Held-Out Test Set (chb03)
    X_test = d_test["features"]
    y_test = d_test["labels"]
    X_test_tensor = torch.tensor(windows_test, dtype=torch.float32)

    model_preds = {
        "Logistic Regression": lr_model.predict(X_test),
        "Random Forest": rf_model.predict(X_test),
        "XGBoost": xgb_model.predict(X_test),
        "1D CNN": (cnn1d.predict_proba(X_test_tensor).numpy() >= 0.5).astype(int),
        "CNN-LSTM": (cnn_lstm.predict_proba(X_test_tensor).numpy() >= 0.5).astype(int),
    }

    model_probs = {
        "Logistic Regression": lr_model.predict_proba(X_test)[:, 1],
        "Random Forest": rf_model.predict_proba(X_test)[:, 1],
        "XGBoost": xgb_model.predict_proba(X_test)[:, 1],
        "1D CNN": cnn1d.predict_proba(X_test_tensor).numpy(),
        "CNN-LSTM": cnn_lstm.predict_proba(X_test_tensor).numpy(),
    }

    # -------------------------------------------------------------
    # Plot 7: Confusion Matrices Comparison
    # -------------------------------------------------------------
    print("[*] Generating Figure 7: Multi-Model Confusion Matrices...")
    p7 = visualizer.plot_07_confusion_matrices(y_test, model_preds)
    print(f"    [OK] Saved: {p7.name}")

    # -------------------------------------------------------------
    # Plot 8: ROC and PR Curves Comparison
    # -------------------------------------------------------------
    print("[*] Generating Figure 8: ROC & PR Curves...")
    p8 = visualizer.plot_08_roc_and_pr_curves(y_test, model_probs)
    print(f"    [OK] Saved: {p8.name}")

    # -------------------------------------------------------------
    # Plot 9: Deep Learning Convergence Curves
    # -------------------------------------------------------------
    print("[*] Generating Figure 9: Training Loss & F1 Curves...")
    cnn1d_bench = json.loads((metrics_dir / "cnn1d_benchmark.json").read_text())
    cnn_lstm_bench = json.loads((metrics_dir / "cnn_lstm_benchmark.json").read_text())
    p9 = visualizer.plot_09_learning_curves(cnn1d_bench["training_history"], cnn_lstm_bench["training_history"])
    print(f"    [OK] Saved: {p9.name}")

    # -------------------------------------------------------------
    # Plot 10: Explainable AI - Physiological & Channel Importance
    # -------------------------------------------------------------
    print("[*] Generating Figure 10: XAI Feature & Channel Attributions...")
    from src.models.classical import XGBoostBaseline
    xgb_wrapper = XGBoostBaseline()
    xgb_wrapper.model = xgb_model
    xai_res = EEGExplainer.get_feature_importances(xgb_wrapper, feature_names, top_k=15)
    p10 = visualizer.plot_10_feature_importance(xai_res["top_features"], xai_res["channel_importance"])
    print(f"    [OK] Saved: {p10.name}")

    # -------------------------------------------------------------
    # Plot 11: Patient-Wise Error Analysis
    # -------------------------------------------------------------
    print("[*] Generating Figure 11: Patient-Wise Error Breakdown...")
    # Evaluate XGBoost across Validation (chb02) and Test (chb03)
    val_preds_xgb = xgb_model.predict(d_val["features"])
    test_preds_xgb = xgb_model.predict(d_test["features"])
    from sklearn.metrics import confusion_matrix
    cm_val = confusion_matrix(d_val["labels"], val_preds_xgb, labels=[0, 1])
    cm_test = confusion_matrix(d_test["labels"], test_preds_xgb, labels=[0, 1])

    patient_stats = {
        "chb02 (Validation)": {
            "TP": int(cm_val[1, 1]), "FP": int(cm_val[0, 1]), "FN": int(cm_val[1, 0]), "TN": int(cm_val[0, 0])
        },
        "chb03 (Held-Out Test)": {
            "TP": int(cm_test[1, 1]), "FP": int(cm_test[0, 1]), "FN": int(cm_test[1, 0]), "TN": int(cm_test[0, 0])
        },
    }
    p11 = visualizer.plot_11_patient_error_analysis(patient_stats)
    print(f"    [OK] Saved: {p11.name}")

    print("\n" + "=" * 75)
    print(f"[OK] Successfully Generated and Verified All 11 Figures in: {plots_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
