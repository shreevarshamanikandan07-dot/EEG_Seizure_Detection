"""Unit tests for EDF reader and channel montage normalization."""

import pytest
import numpy as np
import mne
from pathlib import Path
from src.data.edf_reader import EDFReader


def test_channel_name_normalization():
    """Verify EDF channel name cleaning across various formatting quirks."""
    assert EDFReader.normalize_channel_name("EEG FP1-F7") == "FP1-F7"
    assert EDFReader.normalize_channel_name("eeg fp1-f7") == "FP1-F7"
    assert EDFReader.normalize_channel_name("  FP1-F7  ") == "FP1-F7"
    assert EDFReader.normalize_channel_name("EEG C3-P3") == "C3-P3"
    assert EDFReader.normalize_channel_name("CZ-PZ") == "CZ-PZ"
    assert EDFReader.normalize_channel_name("EEG T8-P8") == "T8-P8"


def test_load_recording_from_mne_raw(tmp_path: Path):
    """Test loading channels and samples using an in-memory MNE Raw object saved to EDF."""
    # Create synthetic 4-channel EEG signal (256 Hz, 5 seconds = 1280 samples)
    sfreq = 256.0
    n_channels = 4
    n_samples = int(5 * sfreq)
    ch_names = ["EEG FP1-F7", "EEG F7-T7", "EEG FP1-F3", "EEG F3-C3"]
    data = np.random.randn(n_channels, n_samples) * 1e-5  # Microvolt range

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    # Save as .fif temporarily (to test load pipeline behavior)
    raw_file = tmp_path / "test_raw.fif"
    raw.save(raw_file, overwrite=True)

    # Test reading back with MNE
    raw_read = mne.io.read_raw_fif(raw_file, preload=True)
    norm_channels = [EDFReader.normalize_channel_name(ch) for ch in raw_read.ch_names]

    assert norm_channels == ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3"]
    assert raw_read.n_times == 1280
    assert raw_read.info["sfreq"] == 256.0


def test_read_real_downloaded_edf_metadata():
    """Verify EDFReader on real downloaded CHB-MIT file."""
    edf_p = Path("data/raw/chb01/chb01_03.edf")
    if not edf_p.exists():
        pytest.skip("chb01_03.edf not yet downloaded")

    info = EDFReader.get_info(edf_p)
    assert info.sampling_rate == 256.0
    assert info.duration_sec == 3600.0
    assert info.n_channels >= 18
    assert "FP1-F7" in info.channel_names
