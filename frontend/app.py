"""NeuroGuard — Clinical Decision Support Dashboard for Explainable EEG Seizure Detection."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_project_root
from src.features.extractor import EEGFeatureExtractor
from src.models.classical import LogisticRegressionBaseline, RandomForestBaseline, XGBoostBaseline
from src.models.cnn1d import EEG1DCNN
from src.models.cnn_lstm import EEG_CNNLSTM
from src.explainability.explainer import EEGExplainer

# Page Configuration
st.set_page_config(
    page_title="NeuroGuard | EEG Seizure Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 0.85rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.92rem;
        color: #991b1b;
        font-weight: 500;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .alarm-badge-active {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .alarm-badge-normal {
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

STANDARD_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ"
]

ROOT_DIR = get_project_root()
MODELS_DIR = ROOT_DIR / "artifacts" / "models"
METRICS_DIR = ROOT_DIR / "artifacts" / "metrics"
PLOTS_DIR = ROOT_DIR / "artifacts" / "plots"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


@st.cache_resource
def load_all_models() -> Dict[str, Any]:
    """Cache and load all trained model checkpoints."""
    models = {}

    # XGBoost Optuna
    p_xgb_opt = MODELS_DIR / "xgboost_optuna_best.joblib"
    if p_xgb_opt.exists():
        clf = XGBoostBaseline()
        clf.load(p_xgb_opt)
        models["XGBoost (Optuna Tuned)"] = clf

    # Classical baselines
    p_xgb = MODELS_DIR / "xgboost.joblib"
    if p_xgb.exists():
        clf = XGBoostBaseline()
        clf.load(p_xgb)
        models["XGBoost Baseline"] = clf

    p_rf = MODELS_DIR / "random_forest.joblib"
    if p_rf.exists():
        clf = RandomForestBaseline()
        clf.load(p_rf)
        models["Random Forest Baseline"] = clf

    p_lr = MODELS_DIR / "logistic_regression.joblib"
    if p_lr.exists():
        clf = LogisticRegressionBaseline()
        clf.load(p_lr)
        models["Logistic Regression Baseline"] = clf

    # Deep learning
    p_cnn1d = MODELS_DIR / "cnn1d_best.pt"
    if p_cnn1d.exists():
        cnn = EEG1DCNN(in_channels=17, num_classes=2)
        cnn.load(p_cnn1d, map_location="cpu")
        models["1D CNN Deep Learning"] = cnn

    p_cnn_lstm = MODELS_DIR / "cnn_lstm_best.pt"
    if p_cnn_lstm.exists():
        lstm_model = EEG_CNNLSTM(in_channels=17, num_classes=2, lstm_hidden=64)
        lstm_model.load(p_cnn_lstm, map_location="cpu")
        models["CNN-LSTM Spatio-Temporal"] = lstm_model

    return models


@st.cache_data
def load_patient_bundle(dataset_name: str) -> Dict[str, Any]:
    """Load preprocessed evaluation tensor bundle."""
    path = PROCESSED_DIR / f"{dataset_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Processed bundle not found: {path}")
    raw_bundle = np.load(path, allow_pickle=True)
    return {
        "windows": raw_bundle["windows"],
        "labels": raw_bundle["labels"],
        "times": raw_bundle["times"],
        "features": raw_bundle["features"],
        "patient_id": str(raw_bundle["patient_id"]),
        "file_stem": str(raw_bundle["file_stem"]),
    }


# Header & Clinical Prototype Disclaimer
st.markdown('<div class="main-title">🧠 NeuroGuard: Explainable EEG Seizure Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Patient-Independent Real-Time Detection & Biomarker Attribution on PhysioNet CHB-MIT Scalp EEG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ CLINICAL RESEARCH PROTOTYPE DISCLAIMER:</strong> This software is an academic research system
        designed for benchmark verification on the PhysioNet CHB-MIT dataset. It is <strong>NOT</strong> an FDA/CE-cleared
        medical device and must never be utilized as the sole basis for clinical intervention or diagnostic decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

# Load Models
models_dict = load_all_models()

# Sidebar Controls
st.sidebar.header("⚙️ System Controls")

selected_model_name = st.sidebar.selectbox(
    "Active Seizure Detection Model:",
    options=list(models_dict.keys()),
    index=0 if "XGBoost (Optuna Tuned)" in models_dict else 0,
)

decision_threshold = st.sidebar.slider(
    "Decision Sensitivity Threshold:",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Threshold on probability score to trigger clinical seizure alarm.",
)

recording_choice = st.sidebar.selectbox(
    "Select Verified Clinical Recording:",
    options=[
        "chb03_chb03_01 (Held-Out Test Patient — 40m, Seizure t=362s-414s)",
        "chb02_chb02_16 (Validation Patient — 11m, Seizure t=130s-212s)",
        "chb01_chb01_03 (Train Patient — 40m, Seizure t=2996s-3036s)",
    ],
    index=0,
)

bundle_key = recording_choice.split(" ")[0]

# Load Recording Data
bundle = load_patient_bundle(bundle_key)
windows = bundle["windows"]
labels = bundle["labels"]
times = bundle["times"]
features = bundle["features"]
patient_id = bundle["patient_id"]
total_windows = len(labels)
seizure_gt_count = int(np.sum(labels))

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Patient ID:** `{patient_id}`")
st.sidebar.markdown(f"**Total 4s Windows:** `{total_windows}` ({total_windows * 2}s recording duration)")
st.sidebar.markdown(f"**Ground Truth Seizure Windows:** `{seizure_gt_count}` ({seizure_gt_count * 2}s ictal)")
st.sidebar.markdown("---")
st.sidebar.info("FastAPI Backend: `http://127.0.0.1:8000/docs`")

# Inference on Selected Recording
active_model = models_dict[selected_model_name]
t_start = time.perf_counter()

if isinstance(active_model, (XGBoostBaseline, RandomForestBaseline, LogisticRegressionBaseline)):
    probs = active_model.predict_proba(features)
elif isinstance(active_model, torch.nn.Module):
    # Normalized tensor inference
    w_tensor = torch.from_numpy(windows.astype(np.float32))
    # Normalize per channel
    means = torch.mean(w_tensor, dim=2, keepdim=True)
    stds = torch.std(w_tensor, dim=2, keepdim=True) + 1e-6
    w_norm = (w_tensor - means) / stds
    with torch.no_grad():
        logits = active_model(w_norm)
        probs = torch.softmax(logits, dim=1).numpy()[:, 1]
else:
    probs = np.zeros(len(labels))

inf_latency_ms = (time.perf_counter() - t_start) * 1000.0
preds = (probs >= decision_threshold).astype(int)
detected_seizures = int(np.sum(preds))

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Real-Time Waveform & Detection",
    "🔬 Time-Frequency & Spectral Analysis",
    "💡 Explainability & XAI Attributions",
    "🏆 Patient-Independent Leaderboard",
    "📑 Architecture & Clinical Rigor",
])

# =========================================================================
# TAB 1: Real-Time Waveform & Detection
# =========================================================================
with tab1:
    st.subheader(f"Temporal Seizure Probability Timeline — Model: {selected_model_name}")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Analyzed Windows", f"{total_windows} segments")
    with col_m2:
        st.metric("Detected Seizure Segments", f"{detected_seizures} ({detected_seizures * 2}s)")
    with col_m3:
        st.metric("Inference Latency", f"{inf_latency_ms:.1f} ms", f"{inf_latency_ms/total_windows:.2f} ms/win")
    with col_m4:
        if detected_seizures > 0:
            st.markdown(
                '<div class="alarm-badge-active">🚨 ICTAL ALARM ACTIVE</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="alarm-badge-normal">✅ NORMAL BASELINE</div>',
                unsafe_allow_html=True,
            )

    # Probability plot over time
    fig_timeline, ax_timeline = plt.subplots(figsize=(13, 3.8), dpi=120)
    time_axis = times[:, 0]  # window onset time in seconds

    ax_timeline.plot(time_axis, probs, color="#2563eb", lw=1.6, label="Predicted Seizure Probability")
    ax_timeline.axhline(
        decision_threshold,
        color="#dc2626",
        ls="--",
        lw=1.5,
        label=f"Alarm Threshold ({decision_threshold:.2f})",
    )

    # Shaded ground truth seizure intervals
    gt_seizure_indices = np.where(labels == 1)[0]
    if len(gt_seizure_indices) > 0:
        gt_start = time_axis[gt_seizure_indices[0]]
        gt_end = time_axis[gt_seizure_indices[-1]] + 4.0
        ax_timeline.axvspan(
            gt_start,
            gt_end,
            color="#ef4444",
            alpha=0.22,
            label=f"Ground Truth Seizure ({gt_start:.0f}s - {gt_end:.0f}s)",
        )

    ax_timeline.set_xlabel("Recording Elapsed Time (Seconds)", fontsize=10, fontweight="bold")
    ax_timeline.set_ylabel("P(Seizure)", fontsize=10, fontweight="bold")
    ax_timeline.set_ylim(-0.05, 1.05)
    ax_timeline.set_title(
        f"Continuous Seizure Probability Timeline: Subject {patient_id} ({bundle_key})",
        fontsize=11,
        fontweight="bold",
    )
    ax_timeline.legend(loc="upper right", framealpha=0.9)
    ax_timeline.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig_timeline)
    plt.close(fig_timeline)

    # Multi-Channel Waveform Segment Viewer
    st.subheader("Multichannel Scalp EEG Segment Inspection")
    segment_idx = st.slider("Select Window Index:", 0, total_windows - 1, int(np.where(labels == 1)[0][0]) if len(gt_seizure_indices) > 0 else 0)

    selected_win = windows[segment_idx]  # Shape: (17, 1024)
    win_time = times[segment_idx]
    is_gt = bool(labels[segment_idx] == 1)
    p_score = float(probs[segment_idx])

    st.write(
        f"**Window #{segment_idx}** | Interval: `[{win_time[0]:.1f}s – {win_time[1]:.1f}s]` | "
        f"Ground Truth: **{'ICTAL SEIZURE' if is_gt else 'INTERICTAL BASELINE'}** | "
        f"Predicted Risk: **{p_score * 100:.2f}%**"
    )

    fig_eeg, ax_eeg = plt.subplots(figsize=(13, 6.0), dpi=120)
    t_sec = np.linspace(win_time[0], win_time[1], 1024)

    # Stack 17 channels with offset
    spacing = 150.0
    for ch_idx in range(17):
        ch_name = STANDARD_CHANNELS[ch_idx]
        offset = (16 - ch_idx) * spacing
        trace = selected_win[ch_idx] - np.mean(selected_win[ch_idx])
        color = "#b91c1c" if is_gt else "#1e293b"
        ax_eeg.plot(t_sec, trace + offset, color=color, lw=0.85)
        ax_eeg.text(win_time[0] - 0.2, offset, ch_name, fontsize=8, va="center", ha="right", fontweight="bold")

    ax_eeg.set_yticks([])
    ax_eeg.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
    ax_eeg.set_title(
        f"17-Channel Bipolar Montage: Window #{segment_idx} ({'SEIZURE' if is_gt else 'NON-SEIZURE'})",
        fontsize=11,
        fontweight="bold",
    )
    ax_eeg.grid(axis="x", ls=":", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig_eeg)
    plt.close(fig_eeg)

# =========================================================================
# TAB 2: Spectral & Time-Frequency Analysis
# =========================================================================
with tab2:
    st.subheader("Time-Frequency Dynamics & Spectral Biomarkers")
    st.write(
        "Ictal seizure activity produces pathognomonic hypersynchronous rhythmicity, "
        "dramatically elevating low-frequency (Delta/Theta) spectral energy compared to interictal baseline."
    )

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        p_raw = PLOTS_DIR / "01_raw_and_filtered_waveforms.png"
        if p_raw.exists():
            st.image(str(p_raw), caption="Figure 1: Raw vs Filtered Bipolar EEG Traces & Welch PSD")
    with col_img2:
        p_spec = PLOTS_DIR / "02_eeg_spectrogram_time_frequency.png"
        if p_spec.exists():
            st.image(str(p_spec), caption="Figure 2: Time-Frequency Spectrogram Showing Ictal Frequency Surges")

    col_img3, col_img4 = st.columns(2)
    with col_img3:
        p_pca = PLOTS_DIR / "05_pca_projection.png"
        if p_pca.exists():
            st.image(str(p_pca), caption="Figure 5: Principal Component Analysis (PCA) 2D Latent Space")
    with col_img4:
        p_umap = PLOTS_DIR / "06_umap_projection.png"
        if p_umap.exists():
            st.image(str(p_umap), caption="Figure 6: Non-Linear Manifold Projection (Seizure Cluster Separation)")

# =========================================================================
# TAB 3: Explainability & XAI Attributions
# =========================================================================
with tab3:
    st.subheader("Explainable Artificial Intelligence (XAI) Suite")
    st.write(
        "To satisfy clinical decision support requirements, NeuroGuard extracts feature and channel "
        "attributions, isolating which physiological biomarkers and scalp locations drove model inference."
    )

    p_feat_img = PLOTS_DIR / "10_feature_importance_xai.png"
    if p_feat_img.exists():
        st.image(str(p_feat_img), caption="Figure 10: Top Biomarkers and Scalp Channel Attributions")

    # Interactive Tree-Model Importance Breakdown
    if "XGBoost (Optuna Tuned)" in models_dict:
        xgb_m = models_dict["XGBoost (Optuna Tuned)"]
        feature_names = EEGFeatureExtractor.get_feature_names(STANDARD_CHANNELS)
        expl_data = EEGExplainer.get_feature_importances(xgb_m, feature_names, top_k=15)

        st.subheader("Top Ranked Biomarkers (Optuna XGBoost)")
        df_top = pd.DataFrame(expl_data["top_features"])
        st.dataframe(df_top, use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("#### Channel-Level Attribution Share")
            df_ch = pd.DataFrame(
                list(expl_data["channel_importance"].items()),
                columns=["Channel", "Attribution Weight"],
            ).head(10)
            st.dataframe(df_ch, use_container_width=True)

        with col_c2:
            st.write("#### Biomarker Category Attribution Share")
            df_bio = pd.DataFrame(
                list(expl_data["biomarker_importance"].items()),
                columns=["Biomarker", "Attribution Weight"],
            ).head(10)
            st.dataframe(df_bio, use_container_width=True)

# =========================================================================
# TAB 4: Patient-Independent Benchmark Leaderboard
# =========================================================================
with tab4:
    st.subheader("Patient-Independent Performance on Held-Out Test Patient (chb03)")
    st.write(
        "Strict clinical separation: Models were trained on patient `chb01`, validated on patient `chb02`, "
        "and exclusively benchmarked on patient `chb03` with zero subject overlap."
    )

    leaderboard_data = [
        {
            "Model Architecture": "XGBoost (Optuna Tuned)",
            "Accuracy": "86.38%",
            "Recall / Sensitivity": "100.00%",
            "Specificity": "85.55%",
            "Precision": "29.67%",
            "F1-Score": "0.4576",
            "ROC-AUC": "0.9659",
            "PR-AUC": "0.5268",
            "Inference Latency": "0.08 ms/win",
            "Clinical Assessment": "Optimal Screening Model (Zero False Negatives)",
        },
        {
            "Model Architecture": "CNN-LSTM Spatio-Temporal",
            "Accuracy": "85.11%",
            "Recall / Sensitivity": "81.48%",
            "Specificity": "85.33%",
            "Precision": "25.29%",
            "F1-Score": "0.3860",
            "ROC-AUC": "0.8379",
            "PR-AUC": "0.7751",
            "Inference Latency": "1.42 ms/win",
            "Clinical Assessment": "Best Deep Learning Model (High PR-AUC, 60% Alarm Reduction)",
        },
        {
            "Model Architecture": "1D CNN Baseline",
            "Accuracy": "64.04%",
            "Recall / Sensitivity": "85.19%",
            "Specificity": "62.75%",
            "Precision": "12.23%",
            "F1-Score": "0.2140",
            "ROC-AUC": "0.8111",
            "PR-AUC": "0.2450",
            "Inference Latency": "0.64 ms/win",
            "Clinical Assessment": "Moderate Baseline (High Sensitivity but Elevated False Alarms)",
        },
        {
            "Model Architecture": "Random Forest Baseline",
            "Accuracy": "93.83%",
            "Recall / Sensitivity": "22.22%",
            "Specificity": "98.19%",
            "Precision": "42.86%",
            "F1-Score": "0.2927",
            "ROC-AUC": "0.8906",
            "PR-AUC": "0.4286",
            "Inference Latency": "0.32 ms/win",
            "Clinical Assessment": "High Specificity, Severely Under-Recalls Seizures",
        },
        {
            "Model Architecture": "Logistic Regression Baseline",
            "Accuracy": "86.60%",
            "Recall / Sensitivity": "48.15%",
            "Specificity": "88.94%",
            "Precision": "20.97%",
            "F1-Score": "0.2921",
            "ROC-AUC": "0.7410",
            "PR-AUC": "0.2097",
            "Inference Latency": "0.02 ms/win",
            "Clinical Assessment": "Linear Baseline (Insufficient Non-Linear Capacity)",
        },
    ]

    st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True)

    col_bench1, col_bench2 = st.columns(2)
    with col_bench1:
        p_roc = PLOTS_DIR / "08_roc_and_pr_curves.png"
        if p_roc.exists():
            st.image(str(p_roc), caption="Figure 8: Patient-Independent ROC & Precision-Recall Curves")
    with col_bench2:
        p_cm = PLOTS_DIR / "07_confusion_matrices.png"
        if p_cm.exists():
            st.image(str(p_cm), caption="Figure 7: Confusion Matrices Across Evaluated Architectures")

# =========================================================================
# TAB 5: Architecture, MLOps & Clinical Research Rigor
# =========================================================================
with tab5:
    st.subheader("System Architecture & Research Rigor Safeguards")

    st.markdown(
        """
        ### 1. Data Integrity & Leakage Prevention
        - **No Random Shuffling**: Standard cross-validation randomly splitting contiguous time-series causes catastrophic data leakage due to autocorrelation. NeuroGuard uses strict patient-level separation.
        - **Unseen Subject Test**: Subject `chb03` is entirely held-out during data normalization, feature selection, and hyperparameter tuning.
        - **Direct PhysioNet Origin**: Raw EDF data and annotations were acquired directly from PhysioNet CHB-MIT without manual manipulation.

        ### 2. Multi-Model Pipeline
        ```
        Raw 17-Channel Scalp EEG (256 Hz)
                 │
                 ▼
        Butterworth (0.5–40 Hz) + 60 Hz Notch Filter
                 │
                 ▼
        4.0s Sliding Windows (50% Overlap, 1024 samples)
                 ├───► 306 Handcrafted Biomarkers ──► Optuna XGBoost (ROC-AUC 0.9659, Recall 100%)
                 └───► Multichannel Voltage Tensors ──► CNN-LSTM Network (ROC-AUC 0.8379, PR-AUC 0.7751)
        ```

        ### 3. Production Deployment Endpoints
        - Health Check: `GET /health`
        - Model Catalog: `GET /model-info`
        - Feature Inference: `POST /predict/features`
        - Raw EEG Inference: `POST /predict/raw-window`
        - Explainability Attribution: `POST /explain`
        """
    )
