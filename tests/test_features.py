"""Unit tests for EEG biomarker feature extraction."""

import pytest
import numpy as np
from src.features.extractor import EEGFeatureExtractor


def test_feature_extractor_dimensions():
    """Verify that feature extractor returns correct number of features."""
    sfreq = 256.0
    extractor = EEGFeatureExtractor(sampling_rate=sfreq)

    # Synthetic window: 4 channels, 1024 samples (4 seconds at 256 Hz)
    n_channels = 4
    n_samples = 1024
    window = np.random.randn(n_channels, n_samples).astype(np.float32)

    # Single channel feature extraction
    single_feats = extractor.extract_single_channel_features(window[0])
    assert len(single_feats) == 18

    # Multi-channel single window extraction
    win_feats = extractor.extract_window_features(window)
    assert len(win_feats) == 4 * 18  # 72 features

    # Batch extraction (3 windows, 4 channels, 1024 samples)
    batch_windows = np.random.randn(3, n_channels, n_samples).astype(np.float32)
    feats_matrix, feat_names = extractor.extract_batch_features(
        batch_windows,
        channel_names=["FP1-F7", "F7-T7", "T7-P7", "P7-O1"]
    )

    assert feats_matrix.shape == (3, 72)
    assert len(feat_names) == 72
    assert "FP1-F7_hjorth_mobility" in feat_names
    assert "P7-O1_alpha_power" in feat_names
    assert "F7-T7_spectral_entropy" in feat_names


def test_feature_extractor_frequency_sensitivity():
    """Verify that frequency band power correctly detects dominant rhythms."""
    sfreq = 256.0
    extractor = EEGFeatureExtractor(sampling_rate=sfreq)

    t = np.linspace(0, 4, int(4 * sfreq), endpoint=False)
    # Pure 10 Hz alpha wave (8-13 Hz)
    pure_alpha = np.sin(2 * np.pi * 10 * t)
    feats_alpha = extractor.extract_single_channel_features(pure_alpha)

    # Feature index mapping:
    # 0:mean, 1:var, 2:std, 3:rms, 4:kurt, 5:skew, 6:hjorth_act, 7:hjorth_mob, 8:hjorth_comp,
    # 9:zcr, 10:ptp, 11:delta, 12:theta, 13:alpha, 14:beta, 15:gamma, 16:entropy, 17:dom_freq
    alpha_power = feats_alpha[13]
    delta_power = feats_alpha[11]
    beta_power = feats_alpha[14]
    dom_freq = feats_alpha[17]

    assert alpha_power > delta_power
    assert alpha_power > beta_power
    assert np.isclose(dom_freq, 10.0, atol=1.0)
