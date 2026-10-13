"""Unit tests for EEG preprocessing and window segmentation."""

import pytest
import numpy as np
from pathlib import Path
from src.preprocessing.preprocessor import EEGPreprocessor
from src.preprocessing.segmenter import EEGSegmenter
from src.data.annotation_parser import SeizureEvent


def test_bandpass_and_notch_filter():
    """Verify that bandpass and notch filters attenuate noise and retain in-band signals."""
    prep = EEGPreprocessor(
        sampling_rate=256.0,
        bandpass_low=0.5,
        bandpass_high=40.0,
        notch_freq=60.0,
    )

    t = np.linspace(0, 10, int(10 * 256.0), endpoint=False)
    # 10 Hz (in-band alpha rhythm) + 60 Hz (power-line noise) + 0.1 Hz (DC drift)
    sig_10hz = np.sin(2 * np.pi * 10 * t)
    sig_60hz = 2.0 * np.sin(2 * np.pi * 60 * t)
    sig_dc = 5.0 * np.sin(2 * np.pi * 0.1 * t)

    combined = sig_10hz + sig_60hz + sig_dc
    filtered = prep.filter_signal(combined)

    assert filtered.shape == combined.shape
    # Steady state (excluding boundary filter transient) should isolate ~1.0 amplitude 10 Hz tone
    steady_state = filtered[int(256):int(-256)]
    assert np.max(np.abs(steady_state)) < 1.5
    # Variance of filtered signal should be massively reduced compared to raw combined signal (8.0V peak)
    assert np.var(filtered) < 0.5 * np.var(combined)


def test_normalizer_fit_and_transform(tmp_path: Path):
    """Verify training normalizer fits means/stds and saves/loads cleanly."""
    prep = EEGPreprocessor()

    # Generate synthetic 4-channel data
    data_train = np.random.randn(4, 1000) * 10.0 + 5.0  # mean=5, std=10
    prep.fit_normalizer(data_train)

    assert prep.is_fitted
    assert prep.channel_means is not None
    assert prep.channel_means.shape == (4, 1)

    normalized = prep.normalize_signal(data_train)
    # Channel means should be ~0 and stds should be ~1
    np.testing.assert_allclose(np.mean(normalized, axis=-1), np.zeros(4), atol=1e-2)
    np.testing.assert_allclose(np.std(normalized, axis=-1), np.ones(4), atol=1e-2)

    # Test save and load scaler
    scaler_file = tmp_path / "scaler.joblib"
    prep.save_scaler(scaler_file)
    assert scaler_file.exists()

    prep2 = EEGPreprocessor()
    prep2.load_scaler(scaler_file)
    assert prep2.is_fitted
    np.testing.assert_allclose(prep2.channel_means, prep.channel_means)


def test_eeg_segmenter_shapes_and_labels():
    """Verify segmenter produces correct window shapes and ground truth labels."""
    sfreq = 256.0
    duration_sec = 20.0
    n_channels = 3
    n_samples = int(duration_sec * sfreq)

    data = np.random.randn(n_channels, n_samples)
    # Place a seizure between t=5.0s and t=11.0s (6 seconds duration)
    seizures = [SeizureEvent(index=1, start_sec=5.0, end_sec=11.0)]

    segmenter = EEGSegmenter(
        window_duration_sec=4.0,  # 1024 samples
        step_duration_sec=2.0,    # 512 samples
        seizure_overlap_threshold=0.5,
    )

    batch = segmenter.segment_recording(
        data=data,
        sampling_rate=sfreq,
        seizures=seizures,
        patient_id="test_pat",
        recording_name="test_rec.edf",
    )

    # 20 seconds with 4s windows and 2s steps:
    # Windows start at 0, 2, 4, 6, 8, 10, 12, 14, 16 -> 9 total windows
    assert batch.windows.shape[1] == n_channels
    assert batch.windows.shape[2] == int(4.0 * sfreq)
    assert len(batch.labels) == len(batch.windows)

    # Check that seizure labels exist
    assert batch.n_seizure_windows > 0
    assert batch.n_non_seizure_windows > 0
    # Window [6.0, 10.0] is completely inside [5.0, 11.0] -> must be labeled 1
    # Find window starting at 6.0
    win_idx = [i for i, (st, en) in enumerate(batch.window_times) if np.isclose(st, 6.0)][0]
    assert batch.labels[win_idx] == 1
