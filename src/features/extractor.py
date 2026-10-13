"""EEG Biomarker Feature Extraction Module.

Extracts comprehensive time-domain and frequency-domain physiological biomarkers
from multichannel EEG windows for classical machine-learning models (Logistic
Regression, Random Forest, XGBoost).
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
from scipy import stats
from scipy.signal import welch


class EEGFeatureExtractor:
    """Extracts electrophysiological features across time and frequency domains."""

    FREQ_BANDS = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 40.0),
    }

    FEATURE_NAMES_PER_CHANNEL = [
        "mean",
        "variance",
        "std",
        "rms",
        "kurtosis",
        "skewness",
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",
        "zero_crossings",
        "peak_to_peak",
        "delta_power",
        "theta_power",
        "alpha_power",
        "beta_power",
        "gamma_power",
        "spectral_entropy",
        "dominant_freq",
    ]

    def __init__(self, sampling_rate: float = 256.0) -> None:
        self.sampling_rate = float(sampling_rate)

    @classmethod
    def get_feature_names(cls, channel_names: List[str]) -> List[str]:
        """Generate human-readable feature column names."""
        names: List[str] = []
        for ch in channel_names:
            for feat in cls.FEATURE_NAMES_PER_CHANNEL:
                names.append(f"{ch}_{feat}")
        return names

    def extract_single_channel_features(self, sig: np.ndarray) -> np.ndarray:
        """Extract all 18 biomarker features from a 1D EEG signal segment.

        Args:
            sig: 1D array of shape (n_samples,).

        Returns:
            np.ndarray: 1D vector of 18 extracted feature values.
        """
        # Time-domain metrics
        mean_val = float(np.mean(sig))
        var_val = float(np.var(sig))
        std_val = float(np.std(sig))
        rms_val = float(np.sqrt(np.mean(sig**2)))
        kurt_val = float(stats.kurtosis(sig, bias=False)) if len(sig) > 4 else 0.0
        skew_val = float(stats.skew(sig, bias=False)) if len(sig) > 2 else 0.0

        # Hjorth parameters
        diff1 = np.diff(sig)
        diff2 = np.diff(diff1)
        var_diff1 = np.var(diff1) if len(diff1) > 0 else 0.0
        var_diff2 = np.var(diff2) if len(diff2) > 0 else 0.0

        eps = 1e-12
        hjorth_act = var_val
        hjorth_mob = np.sqrt(var_diff1 / (var_val + eps))
        mob_diff = np.sqrt(var_diff2 / (var_diff1 + eps))
        hjorth_comp = mob_diff / (hjorth_mob + eps)

        # Zero crossing rate
        zero_crossings = float(np.sum(np.diff(np.signbit(sig)) != 0) / len(sig))
        ptp_val = float(np.ptp(sig))

        # Frequency-domain metrics using Welch's PSD
        # nperseg calibrated to window (e.g. 256 samples = 1 sec resolution)
        nperseg = min(len(sig), int(self.sampling_rate))
        freqs, psd = welch(sig, fs=self.sampling_rate, nperseg=nperseg)

        # Sub-band power integrals
        band_powers = []
        total_power = np.sum(psd) + eps
        for _, (low_f, high_f) in self.FREQ_BANDS.items():
            mask = (freqs >= low_f) & (freqs < high_f)
            band_pow = float(np.sum(psd[mask])) if np.any(mask) else 0.0
            band_powers.append(band_pow)

        # Normalized Spectral Entropy
        psd_norm = psd / total_power
        psd_norm = psd_norm[psd_norm > 0]
        spectral_ent = -float(np.sum(psd_norm * np.log2(psd_norm))) / (np.log2(len(freqs)) + eps)

        # Dominant frequency
        dom_idx = np.argmax(psd)
        dom_freq = float(freqs[dom_idx])

        features = [
            mean_val,
            var_val,
            std_val,
            rms_val,
            kurt_val,
            skew_val,
            hjorth_act,
            hjorth_mob,
            hjorth_comp,
            zero_crossings,
            ptp_val,
            *band_powers,  # 5 band powers
            spectral_ent,
            dom_freq,
        ]
        return np.array(features, dtype=np.float32)

    def extract_window_features(
        self,
        window: np.ndarray,
        channel_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Extract features across all channels for a single EEG window.

        Args:
            window: 2D array of shape (n_channels, n_samples).
            channel_names: Optional channel labels.

        Returns:
            np.ndarray: 1D feature vector of shape (n_channels * 18,).
        """
        n_channels = window.shape[0]
        feats = []
        for ch_idx in range(n_channels):
            ch_feats = self.extract_single_channel_features(window[ch_idx])
            feats.append(ch_feats)
        return np.concatenate(feats)

    def extract_batch_features(
        self,
        windows: np.ndarray,
        channel_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features for a batch of EEG windows.

        Args:
            windows: 3D array of shape (n_windows, n_channels, n_samples).
            channel_names: Optional list of channel names.

        Returns:
            Tuple[np.ndarray, List[str]]:
                - Feature matrix of shape (n_windows, n_features)
                - List of feature column names
        """
        n_windows, n_channels, _ = windows.shape
        feature_list = []
        for i in range(n_windows):
            feat_vec = self.extract_window_features(windows[i], channel_names)
            feature_list.append(feat_vec)

        feats_matrix = np.array(feature_list, dtype=np.float32)

        if channel_names is None:
            channel_names = [f"Ch{i}" for i in range(n_channels)]
        feat_names = self.get_feature_names(channel_names)

        return feats_matrix, feat_names
