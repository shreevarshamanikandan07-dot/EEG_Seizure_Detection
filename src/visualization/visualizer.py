"""Academic Biomedical Visualizer for NeuroGuard.

Generates 11 publication-grade figures:
1. Raw vs Filtered EEG waveforms
2. Time-frequency spectrogram analysis
3. Class distribution and imbalance
4. Feature correlation heatmap
5. PCA 2D projection
6. Manifold projection (t-SNE/UMAP clustering)
7. Multi-model confusion matrices
8. ROC and Precision-Recall benchmarking curves
9. Deep learning training and validation loss/F1 curves
10. Feature importance and channel attribution (XAI)
11. Patient-wise error analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import spectrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_curve

from src.utils.config import get_project_root


class EEGVisualizer:
    """Produces publication-grade figures saved to artifacts/plots/."""

    def __init__(self, output_dir: Optional[str | Path] = None) -> None:
        root = get_project_root()
        self.output_dir = Path(output_dir) if output_dir else root / "artifacts" / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Configure standard academic visual theme
        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["axes.labelsize"] = 11

    def plot_01_waveforms(
        self,
        raw_signal: np.ndarray,
        filtered_signal: np.ndarray,
        sfreq: float = 256.0,
        channel_names: Optional[List[str]] = None,
        duration_sec: float = 6.0,
    ) -> Path:
        """Figure 1: Raw vs Filtered Multichannel EEG Waveforms."""
        n_samples = int(duration_sec * sfreq)
        t = np.arange(n_samples) / sfreq
        chs_to_plot = min(3, raw_signal.shape[0])
        ch_labels = channel_names[:chs_to_plot] if channel_names else [f"Ch {i+1}" for i in range(chs_to_plot)]

        fig, axes = plt.subplots(chs_to_plot, 1, figsize=(10, 2.5 * chs_to_plot), sharex=True)
        if chs_to_plot == 1:
            axes = [axes]

        for i in range(chs_to_plot):
            axes[i].plot(t, raw_signal[i, :n_samples] * 1e6, color="#888888", alpha=0.6, label="Raw (with artifacts)")
            axes[i].plot(t, filtered_signal[i, :n_samples] * 1e6, color="#1f77b4", linewidth=1.2, label="Filtered (0.5-40Hz + 60Hz Notch)")
            axes[i].set_ylabel(f"{ch_labels[i]}\n(µV)")
            axes[i].grid(True, linestyle="--", alpha=0.5)
            if i == 0:
                axes[i].legend(loc="upper right", frameon=True)

        axes[-1].set_xlabel("Time (seconds)")
        fig.suptitle("Figure 1: Raw vs Filtered Multichannel EEG Waveforms", fontsize=13, fontweight="bold", y=0.98)
        fig.tight_layout()

        out_path = self.output_dir / "01_raw_and_filtered_waveforms.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_02_spectrogram(
        self,
        normal_window: np.ndarray,
        seizure_window: np.ndarray,
        sfreq: float = 256.0,
    ) -> Path:
        """Figure 2: Time-Frequency Spectrogram comparing Baseline vs Seizure."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

        # Baseline window
        f_norm, t_norm, Sxx_norm = spectrogram(normal_window[0], fs=sfreq, nperseg=128, noverlap=96)
        im1 = axes[0].pcolormesh(t_norm, f_norm, 10 * np.log10(Sxx_norm + 1e-12), cmap="viridis", shading="gouraud")
        axes[0].set_title("Non-Seizure Baseline (Normal Rhythm)")
        axes[0].set_xlabel("Time (seconds)")
        axes[0].set_ylabel("Frequency (Hz)")
        axes[0].set_ylim(0, 45)
        fig.colorbar(im1, ax=axes[0], label="Power (dB)")

        # Seizure window
        f_sz, t_sz, Sxx_sz = spectrogram(seizure_window[0], fs=sfreq, nperseg=128, noverlap=96)
        im2 = axes[1].pcolormesh(t_sz, f_sz, 10 * np.log10(Sxx_sz + 1e-12), cmap="inferno", shading="gouraud")
        axes[1].set_title("Epileptic Seizure (Hyper-synchronous Burst)")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylim(0, 45)
        fig.colorbar(im2, ax=axes[1], label="Power (dB)")

        fig.suptitle("Figure 2: Time-Frequency Spectral Analysis (Baseline vs Seizure)", fontsize=13, fontweight="bold")
        fig.tight_layout()

        out_path = self.output_dir / "02_eeg_spectrogram_time_frequency.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_03_class_distribution(
        self,
        train_labels: np.ndarray,
        val_labels: np.ndarray,
        test_labels: np.ndarray,
    ) -> Path:
        """Figure 3: Class Distribution and Imbalance Across Patient Partitions."""
        partitions = ["Train (chb01)", "Validation (chb02)", "Held-Out Test (chb03)"]
        seizures = [int(np.sum(train_labels == 1)), int(np.sum(val_labels == 1)), int(np.sum(test_labels == 1))]
        non_seizures = [int(np.sum(train_labels == 0)), int(np.sum(val_labels == 0)), int(np.sum(test_labels == 0))]

        x = np.arange(len(partitions))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width/2, non_seizures, width, label="Non-Seizure Windows", color="#4a90e2")
        rects2 = ax.bar(x + width/2, seizures, width, label="Seizure Windows", color="#d0021b")

        ax.set_ylabel("Number of 4-Second Windows")
        ax.set_title("Figure 3: Class Distribution & Patient Partition Imbalance", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(partitions)
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Value annotations on top of bars
        for rect in rects1 + rects2:
            h = rect.get_height()
            ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        out_path = self.output_dir / "03_class_distribution_imbalance.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_04_correlation_heatmap(
        self,
        features: np.ndarray,
        feature_names: List[str],
    ) -> Path:
        """Figure 4: Feature Correlation Heatmap of Key Electrophysiological Biomarkers."""
        # Select first 10 representative biomarkers from channel 1
        subset_indices = list(range(min(12, features.shape[1])))
        sub_X = features[:, subset_indices]
        sub_names = [feature_names[i].replace("FP1-F7_", "") for i in subset_indices]

        corr = np.corrcoef(sub_X.T)

        fig, ax = plt.subplots(figsize=(9, 7.5))
        sns.heatmap(corr, xticklabels=sub_names, yticklabels=sub_names, cmap="coolwarm", annot=True, fmt=".2f",
                    vmin=-1, vmax=1, cbar_kws={"label": "Pearson Correlation"}, ax=ax)
        ax.set_title("Figure 4: Electrophysiological Feature Correlation Heatmap", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()

        out_path = self.output_dir / "04_feature_correlation_heatmap.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_05_pca(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Path:
        """Figure 5: 2D Principal Component Analysis (PCA) Projection."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        var1, var2 = pca.explained_variance_ratio_ * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_pca[labels == 0, 0], X_pca[labels == 0, 1], c="#4a90e2", alpha=0.6, s=25, label="Non-Seizure")
        ax.scatter(X_pca[labels == 1, 0], X_pca[labels == 1, 1], c="#d0021b", alpha=0.9, s=45, edgecolors="k", label="Seizure")

        ax.set_xlabel(f"Principal Component 1 ({var1:.1f}% Variance)")
        ax.set_ylabel(f"Principal Component 2 ({var2:.1f}% Variance)")
        ax.set_title("Figure 5: 2D PCA Manifold Projection of EEG Biomarkers", fontsize=13, fontweight="bold")
        ax.legend(frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        out_path = self.output_dir / "05_pca_projection.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_06_manifold_clustering(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Path:
        """Figure 6: Manifold Projection (t-SNE/UMAP Embedding)."""
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(features)

        # Pre-reduce with PCA for fast, reliable t-SNE convergence
        X_pca50 = PCA(n_components=min(30, features.shape[1])).fit_transform(X_scaled)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter_without_progress=200)
        X_emb = tsne.fit_transform(X_pca50)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_emb[labels == 0, 0], X_emb[labels == 0, 1], c="#4a90e2", alpha=0.5, s=25, label="Non-Seizure")
        ax.scatter(X_emb[labels == 1, 0], X_emb[labels == 1, 1], c="#e74c3c", alpha=0.9, s=50, edgecolors="k", label="Epileptic Seizure")

        ax.set_xlabel("t-SNE / Manifold Dimension 1")
        ax.set_ylabel("t-SNE / Manifold Dimension 2")
        ax.set_title("Figure 6: Non-Linear Manifold Projection (Epileptic Clustering)", fontsize=13, fontweight="bold")
        ax.legend(frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        out_path = self.output_dir / "06_umap_projection.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_07_confusion_matrices(
        self,
        y_true: np.ndarray,
        model_predictions: Dict[str, np.ndarray],
    ) -> Path:
        """Figure 7: Multi-Model Confusion Matrix Comparison on Held-Out Test Data."""
        n_models = len(model_predictions)
        cols = 3
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows))
        axes = np.array(axes).flatten()

        for idx, (m_name, preds) in enumerate(model_predictions.items()):
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, preds, labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                        xticklabels=["Non-SZ", "Seizure"], yticklabels=["Non-SZ", "Seizure"])
            axes[idx].set_title(m_name, fontweight="bold")
            axes[idx].set_xlabel("Predicted Label")
            axes[idx].set_ylabel("True Label")

        for ax in axes[n_models:]:
            ax.axis("off")

        fig.suptitle("Figure 7: Held-Out Test Set Confusion Matrices (Patient chb03)", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

        out_path = self.output_dir / "07_confusion_matrices.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_08_roc_and_pr_curves(
        self,
        y_true: np.ndarray,
        model_probabilities: Dict[str, np.ndarray],
    ) -> Path:
        """Figure 8: Comparative ROC & Precision-Recall Curves."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

        # 1. ROC Curve
        for (m_name, probs), col in zip(model_probabilities.items(), colors):
            fpr, tpr, _ = roc_curve(y_true, probs)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, probs)
            axes[0].plot(fpr, tpr, label=f"{m_name} (AUC = {auc:.3f})", color=col, linewidth=1.8)

        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
        axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
        axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
        axes[0].set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
        axes[0].legend(loc="lower right", frameon=True)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # 2. Precision-Recall Curve
        for (m_name, probs), col in zip(model_probabilities.items(), colors):
            prec, rec, _ = precision_recall_curve(y_true, probs)
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(y_true, probs)
            axes[1].plot(rec, prec, label=f"{m_name} (PR-AUC = {ap:.3f})", color=col, linewidth=1.8)

        axes[1].set_xlabel("Recall (Sensitivity)")
        axes[1].set_ylabel("Precision (Positive Predictive Value)")
        axes[1].set_title("Precision-Recall (PR) Curve", fontweight="bold")
        axes[1].legend(loc="upper right", frameon=True)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("Figure 8: Patient-Independent ROC & PR Diagnostic Curves (Held-Out chb03)", fontsize=13, fontweight="bold")
        fig.tight_layout()

        out_path = self.output_dir / "08_roc_and_pr_curves.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_09_learning_curves(
        self,
        cnn1d_history: Dict[str, List[float]],
        cnn_lstm_history: Dict[str, List[float]],
    ) -> Path:
        """Figure 9: Training & Validation Loss and Validation F1 Progression."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss comparison
        ep_1d = range(1, len(cnn1d_history["train_loss"]) + 1)
        axes[0].plot(ep_1d, cnn1d_history["train_loss"], "b--", label="1D CNN Train Loss")
        axes[0].plot(ep_1d, cnn1d_history["val_loss"], "b-", label="1D CNN Val Loss")

        ep_lstm = range(1, len(cnn_lstm_history["train_loss"]) + 1)
        axes[0].plot(ep_lstm, cnn_lstm_history["train_loss"], "r--", label="CNN-LSTM Train Loss")
        axes[0].plot(ep_lstm, cnn_lstm_history["val_loss"], "r-", label="CNN-LSTM Val Loss")

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].set_title("Training & Validation Loss Convergence", fontweight="bold")
        axes[0].legend(frameon=True)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # F1 Score comparison
        axes[1].plot(ep_1d, cnn1d_history["val_f1"], "b-o", label="1D CNN Val F1")
        axes[1].plot(ep_lstm, cnn_lstm_history["val_f1"], "r-o", label="CNN-LSTM Val F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation F1-Score")
        axes[1].set_title("Validation F1 Metric Progression", fontweight="bold")
        axes[1].legend(frameon=True)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("Figure 9: Deep Neural Network Learning Curves & Convergence", fontsize=13, fontweight="bold")
        fig.tight_layout()

        out_path = self.output_dir / "09_training_validation_loss_curves.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_10_feature_importance(
        self,
        top_features: List[Dict[str, Any]],
        channel_importance: Dict[str, float],
    ) -> Path:
        """Figure 10: Explainable AI - Physiological Biomarker & Channel Importance."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # Top 10 individual features
        feats = [f["feature"].replace("EEG ", "") for f in top_features[:10]][::-1]
        imps = [f["importance"] * 100 for f in top_features[:10]][::-1]
        axes[0].barh(feats, imps, color="#2ecc71")
        axes[0].set_xlabel("Relative Importance (%)")
        axes[0].set_title("Top 10 Physiological Biomarkers (XGBoost)", fontweight="bold")
        axes[0].grid(axis="x", linestyle="--", alpha=0.5)

        # Channel aggregated importance
        chs = list(channel_importance.keys())[:10][::-1]
        ch_imps = [channel_importance[k] * 100 for k in chs]
        axes[1].barh(chs, ch_imps, color="#3498db")
        axes[1].set_xlabel("Aggregated Electrode Importance (%)")
        axes[1].set_title("Top 10 Scalp EEG Channels (Electrode Attribution)", fontweight="bold")
        axes[1].grid(axis="x", linestyle="--", alpha=0.5)

        fig.suptitle("Figure 10: Explainable AI (XAI) Biomarker & Spatial Electrode Attributions", fontsize=13, fontweight="bold")
        fig.tight_layout()

        out_path = self.output_dir / "10_feature_importance_xai.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_11_patient_error_analysis(
        self,
        patient_results: Dict[str, Dict[str, int]],
    ) -> Path:
        """Figure 11: Patient-Wise Diagnostic Breakdown and Error Analysis."""
        patients = list(patient_results.keys())
        tp = [patient_results[p]["TP"] for p in patients]
        fp = [patient_results[p]["FP"] for p in patients]
        fn = [patient_results[p]["FN"] for p in patients]
        tn = [patient_results[p]["TN"] for p in patients]

        x = np.arange(len(patients))
        width = 0.2

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - 1.5*width, tp, width, label="True Positives (Detected Seizures)", color="#2ecc71")
        ax.bar(x - 0.5*width, fn, width, label="False Negatives (Missed Seizures)", color="#e74c3c")
        ax.bar(x + 0.5*width, fp, width, label="False Positives (False Alarms)", color="#f39c12")
        ax.bar(x + 1.5*width, tn, width, label="True Negatives (Correct Baselines)", color="#3498db")

        ax.set_ylabel("Window Count")
        ax.set_title("Figure 11: Patient-Wise Diagnostic Breakdown & Error Distribution", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Patient {p}" for p in patients])
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        out_path = self.output_dir / "11_patient_error_analysis.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path
