"""Biomedical EDF Reader and Montage Normalizer.

Uses MNE-Python to read European Data Format (.edf) files with memory-mapped
lazy loading (preload=False) to ensure low RAM consumption on local machines.
Normalizes channel montages and extracts metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import mne


@dataclass
class RecordingInfo:
    """Detailed metadata for an EDF recording."""
    filepath: Path
    filename: str
    sampling_rate: float
    n_channels: int
    duration_sec: float
    channel_names: List[str]
    n_samples: int


class EDFReader:
    """Memory-efficient reader for CHB-MIT EEG recordings."""

    # Standard bipolar channel aliases commonly present across CHB-MIT recordings
    CHANNEL_ALIASES: Dict[str, str] = {
        "FP1-F7": "FP1-F7",
        "EEG FP1-F7": "FP1-F7",
        "F7-T7": "F7-T7",
        "EEG F7-T7": "F7-T7",
        "T7-P7": "T7-P7",
        "EEG T7-P7": "T7-P7",
        "P7-O1": "P7-O1",
        "EEG P7-O1": "P7-O1",
        "FP1-F3": "FP1-F3",
        "EEG FP1-F3": "FP1-F3",
        "F3-C3": "F3-C3",
        "EEG F3-C3": "F3-C3",
        "C3-P3": "C3-P3",
        "EEG C3-P3": "C3-P3",
        "P3-O1": "P3-O1",
        "EEG P3-O1": "P3-O1",
        "FP2-F4": "FP2-F4",
        "EEG FP2-F4": "FP2-F4",
        "F4-C4": "F4-C4",
        "EEG F4-C4": "F4-C4",
        "C4-P4": "C4-P4",
        "EEG C4-P4": "C4-P4",
        "P4-O2": "P4-O2",
        "EEG P4-O2": "P4-O2",
        "FP2-F8": "FP2-F8",
        "EEG FP2-F8": "FP2-F8",
        "F8-T8": "F8-T8",
        "EEG F8-T8": "F8-T8",
        "T8-P8": "T8-P8",
        "EEG T8-P8": "T8-P8",
        "P8-O2": "P8-O2",
        "EEG P8-O2": "P8-O2",
        "FZ-CZ": "FZ-CZ",
        "EEG FZ-CZ": "FZ-CZ",
        "CZ-PZ": "CZ-PZ",
        "EEG CZ-PZ": "CZ-PZ",
    }

    @classmethod
    def normalize_channel_name(cls, name: str) -> str:
        """Strip prefixes like 'EEG ' and whitespace, returning uppercase standard name."""
        cleaned = name.strip()
        if cleaned in cls.CHANNEL_ALIASES:
            return cls.CHANNEL_ALIASES[cleaned]
        # Generic prefix stripping
        cleaned = re.sub(r"^EEG\s*", "", cleaned, flags=re.IGNORECASE).strip().upper()
        return cleaned

    @classmethod
    def get_info(cls, edf_path: str | Path) -> RecordingInfo:
        """Read EDF metadata without loading raw signal data into memory.

        Args:
            edf_path: Path to .edf file.

        Returns:
            RecordingInfo: Recording statistics and channel names.
        """
        path = Path(edf_path)
        if not path.exists():
            raise FileNotFoundError(f"EDF file not found: {path}")

        # preload=False loads only headers (~few KB), keeping RAM consumption near zero
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
        duration = float(raw.n_times / raw.info["sfreq"])

        norm_channels = [cls.normalize_channel_name(ch) for ch in raw.ch_names]

        return RecordingInfo(
            filepath=path,
            filename=path.name,
            sampling_rate=float(raw.info["sfreq"]),
            n_channels=len(raw.ch_names),
            duration_sec=duration,
            channel_names=norm_channels,
            n_samples=raw.n_times,
        )

    @classmethod
    def load_recording(
        cls,
        edf_path: str | Path,
        target_channels: Optional[List[str]] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        preload: bool = True
    ) -> Tuple[np.ndarray, float, List[str]]:
        """Load signals for specified channels and time range.

        Args:
            edf_path: Path to .edf file.
            target_channels: List of channels to select (standardized names).
            tmin: Start time in seconds.
            tmax: End time in seconds.
            preload: Whether to preload raw into memory.

        Returns:
            Tuple[np.ndarray, float, List[str]]:
                - data: Shape (n_channels, n_samples) in Volts
                - sfreq: Sampling frequency in Hz
                - channels: Names of returned channels
        """
        path = Path(edf_path)
        raw = mne.io.read_raw_edf(str(path), preload=preload, verbose=False)

        # Build mapping from raw channel names to normalized names
        raw_to_norm = {ch: cls.normalize_channel_name(ch) for ch in raw.ch_names}
        norm_to_raw = {v: k for k, v in raw_to_norm.items()}

        if target_channels is not None:
            available_targets = [ch for ch in target_channels if ch in norm_to_raw]
            if not available_targets:
                raise ValueError(f"None of the target channels {target_channels} found in {path.name}")
            raw_channel_picks = [norm_to_raw[ch] for ch in available_targets]
            raw.pick(raw_channel_picks)
            final_channels = available_targets
        else:
            final_channels = [raw_to_norm[ch] for ch in raw.ch_names]

        # Crop if requested
        if tmin is not None or tmax is not None:
            raw.crop(tmin=tmin or 0.0, tmax=tmax, include_tmax=False)

        data = raw.get_data()  # Shape: (channels, samples)
        sfreq = float(raw.info["sfreq"])
        return data, sfreq, final_channels
