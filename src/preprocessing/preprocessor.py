"""Biomedical EEG Preprocessing Module.

Implements zero-phase digital filtering (Butterworth bandpass, notch filter),
signal artifact handling, and training-only channel normalization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import joblib
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt


class EEGPreprocessor:
    """Biomedical signal filter and normalizer for continuous multichannel EEG."""

    def __init__(
        self,
        sampling_rate: float = 256.0,
        bandpass_low: float = 0.5,
        bandpass_high: float = 40.0,
        bandpass_order: int = 4,
        notch_freq: float = 60.0,
        notch_q: float = 30.0,
    ) -> None:
        self.sampling_rate = float(sampling_rate)
        self.bandpass_low = float(bandpass_low)
        self.bandpass_high = float(bandpass_high)
        self.bandpass_order = int(bandpass_order)
        self.notch_freq = float(notch_freq)
        self.notch_q = float(notch_q)

        # Precompute Butterworth filter in second-order sections (SOS) for numerical stability
        nyq = 0.5 * self.sampling_rate
        low_norm = self.bandpass_low / nyq
        high_norm = self.bandpass_high / nyq
        self.sos = butter(
            self.bandpass_order,
            [low_norm, high_norm],
            btype="bandpass",
            output="sos",
        )

        # Precompute Notch filter
        self.notch_b, self.notch_a = iirnotch(
            self.notch_freq,
            self.notch_q,
            fs=self.sampling_rate,
        )

        # Normalization statistics (fitted strictly on training data)
        self.channel_means: Optional[np.ndarray] = None
        self.channel_stds: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def filter_signal(self, data: np.ndarray) -> np.ndarray:
        """Apply zero-phase bandpass and notch filtering to EEG data.

        Args:
            data: NumPy array of shape (n_channels, n_samples) or (n_samples,).

        Returns:
            np.ndarray: Filtered EEG signal of identical shape.
        """
        # Ensure 2D (channels, samples)
        orig_1d = False
        if data.ndim == 1:
            data_2d = data[np.newaxis, :]
            orig_1d = True
        else:
            data_2d = data.copy()

        # Handle invalid values (NaN, Inf) if present
        if not np.all(np.isfinite(data_2d)):
            data_2d = np.nan_to_num(data_2d, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Zero-phase Butterworth Bandpass Filter (0.5 - 40 Hz) using SOS
        filtered = sosfiltfilt(self.sos, data_2d, axis=-1)

        # 2. Zero-phase 60 Hz Notch Filter (power-line noise removal)
        from scipy.signal import filtfilt
        filtered = filtfilt(self.notch_b, self.notch_a, filtered, axis=-1)

        return filtered[0] if orig_1d else filtered

    def fit_normalizer(self, train_data: np.ndarray) -> EEGPreprocessor:
        """Compute per-channel mean and standard deviation strictly on training data.

        Args:
            train_data: Array of shape (n_channels, n_samples) from training patient(s).

        Returns:
            self
        """
        if train_data.ndim == 1:
            train_data = train_data[np.newaxis, :]

        # Compute per-channel statistics
        self.channel_means = np.mean(train_data, axis=-1, keepdims=True)
        self.channel_stds = np.std(train_data, axis=-1, keepdims=True)

        # Prevent division by zero
        eps = 1e-8
        self.channel_stds = np.where(self.channel_stds < eps, 1.0, self.channel_stds)
        self.is_fitted = True
        return self

    def normalize_signal(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalize EEG channels using pre-fitted training statistics.

        Args:
            data: Signal array of shape (n_channels, n_samples).

        Returns:
            np.ndarray: Normalized signal.
        """
        if not self.is_fitted or self.channel_means is None or self.channel_stds is None:
            # If not yet fitted, fallback to per-recording robust zero-centering
            med = np.median(data, axis=-1, keepdims=True)
            iqr = np.percentile(data, 75, axis=-1, keepdims=True) - np.percentile(data, 25, axis=-1, keepdims=True)
            iqr = np.where(iqr < 1e-8, 1.0, iqr)
            return (data - med) / iqr

        if data.shape[0] != self.channel_means.shape[0]:
            # Channel dimension mismatch: fallback to per-channel local z-score
            local_mean = np.mean(data, axis=-1, keepdims=True)
            local_std = np.std(data, axis=-1, keepdims=True)
            local_std = np.where(local_std < 1e-8, 1.0, local_std)
            return (data - local_mean) / local_std

        return (data - self.channel_means) / self.channel_stds

    def process(self, data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Filter and optionally normalize EEG data.

        Args:
            data: Raw EEG array of shape (n_channels, n_samples).
            normalize: Whether to apply normalization.

        Returns:
            np.ndarray: Cleaned, filtered, and normalized EEG array.
        """
        filtered = self.filter_signal(data)
        if normalize:
            return self.normalize_signal(filtered)
        return filtered

    def save_scaler(self, path: str | Path) -> None:
        """Persist fitted normalization statistics to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "channel_means": self.channel_means,
            "channel_stds": self.channel_stds,
            "is_fitted": self.is_fitted,
            "sampling_rate": self.sampling_rate,
            "bandpass_low": self.bandpass_low,
            "bandpass_high": self.bandpass_high,
            "notch_freq": self.notch_freq,
        }, target)

    def load_scaler(self, path: str | Path) -> None:
        """Load normalization statistics from disk."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Scaler file not found at: {target}")
        data = joblib.load(target)
        self.channel_means = data["channel_means"]
        self.channel_stds = data["channel_stds"]
        self.is_fitted = data["is_fitted"]
