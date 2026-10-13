"""EEG Window Segmentation and Ground-Truth Labeling Module.

Slices multichannel continuous EEG recordings into fixed-duration sliding windows,
assigns labels based on strict overlap with clinical seizure annotations,
and manages non-seizure subsampling to balance training distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from src.data.annotation_parser import SeizureEvent


@dataclass
class WindowBatch:
    """Batch of segmented EEG windows and ground-truth metadata."""
    windows: np.ndarray        # Shape: (n_windows, n_channels, window_samples)
    labels: np.ndarray         # Shape: (n_windows,) with 0 = non-seizure, 1 = seizure
    window_times: np.ndarray   # Shape: (n_windows, 2) [start_sec, end_sec]
    patient_id: str
    recording_name: str

    @property
    def n_windows(self) -> int:
        return len(self.labels)

    @property
    def n_seizure_windows(self) -> int:
        return int(np.sum(self.labels == 1))

    @property
    def n_non_seizure_windows(self) -> int:
        return int(np.sum(self.labels == 0))


class EEGSegmenter:
    """Sliding-window generator with annotation-aligned labeling."""

    def __init__(
        self,
        window_duration_sec: float = 4.0,
        step_duration_sec: float = 2.0,
        seizure_overlap_threshold: float = 0.5,
        exclude_ambiguous_boundaries: bool = True,
    ) -> None:
        self.window_duration_sec = float(window_duration_sec)
        self.step_duration_sec = float(step_duration_sec)
        self.seizure_overlap_threshold = float(seizure_overlap_threshold)
        self.exclude_ambiguous_boundaries = exclude_ambiguous_boundaries

    def segment_recording(
        self,
        data: np.ndarray,
        sampling_rate: float,
        seizures: Optional[List[SeizureEvent]] = None,
        patient_id: str = "unknown",
        recording_name: str = "unknown",
        non_seizure_subsample_ratio: Optional[int] = None,
        random_seed: int = 42,
    ) -> WindowBatch:
        """Segment continuous EEG array into windowed tensor and derive labels.

        Args:
            data: Continuous signal of shape (n_channels, n_samples).
            sampling_rate: Recording sampling rate in Hz (e.g. 256.0).
            seizures: List of SeizureEvent objects for this recording.
            patient_id: Patient ID string (e.g. 'chb01').
            recording_name: Recording filename (e.g. 'chb01_03.edf').
            non_seizure_subsample_ratio: If set (e.g. 4), keeps 1 in every N
                non-seizure windows to balance training sets.
            random_seed: Random seed for deterministic subsampling.

        Returns:
            WindowBatch: Structured window tensors, binary labels, and time bounds.
        """
        n_channels, n_samples = data.shape
        window_samples = int(round(self.window_duration_sec * sampling_rate))
        step_samples = int(round(self.step_duration_sec * sampling_rate))

        if n_samples < window_samples:
            raise ValueError(
                f"Recording length ({n_samples} samples) is shorter than "
                f"window size ({window_samples} samples)."
            )

        seizure_list = seizures or []

        windows_list = []
        labels_list = []
        times_list = []

        start_idx = 0
        while start_idx + window_samples <= n_samples:
            end_idx = start_idx + window_samples
            win_start_sec = start_idx / sampling_rate
            win_end_sec = end_idx / sampling_rate

            # Compute total overlap with any annotated seizure
            total_overlap_sec = 0.0
            for sz in seizure_list:
                total_overlap_sec += sz.overlaps_window(win_start_sec, win_end_sec)

            overlap_ratio = total_overlap_sec / self.window_duration_sec

            # Assign label according to strict boundary policy
            if overlap_ratio >= self.seizure_overlap_threshold:
                # Confirmed seizure window
                label = 1
                windows_list.append(data[:, start_idx:end_idx])
                labels_list.append(label)
                times_list.append((win_start_sec, win_end_sec))
            elif overlap_ratio == 0.0:
                # Confirmed pure non-seizure window
                label = 0
                windows_list.append(data[:, start_idx:end_idx])
                labels_list.append(label)
                times_list.append((win_start_sec, win_end_sec))
            else:
                # Ambiguous boundary transition window (0 < overlap < threshold)
                if not self.exclude_ambiguous_boundaries:
                    label = 0
                    windows_list.append(data[:, start_idx:end_idx])
                    labels_list.append(label)
                    times_list.append((win_start_sec, win_end_sec))
                # Otherwise skipped from training to avoid label contamination

            start_idx += step_samples

        if not windows_list:
            raise RuntimeError(f"No valid windows generated for recording {recording_name}")

        windows_arr = np.array(windows_list, dtype=np.float32)
        labels_arr = np.array(labels_list, dtype=np.int64)
        times_arr = np.array(times_list, dtype=np.float32)

        # Apply non-seizure subsampling if requested (only applied to label=0 windows)
        if non_seizure_subsample_ratio is not None and non_seizure_subsample_ratio > 1:
            seizure_indices = np.where(labels_arr == 1)[0]
            non_seizure_indices = np.where(labels_arr == 0)[0]

            # Subsample non-seizure windows uniformly or with seed
            subsampled_non_seizures = non_seizure_indices[::non_seizure_subsample_ratio]
            keep_indices = np.sort(np.concatenate([seizure_indices, subsampled_non_seizures]))

            windows_arr = windows_arr[keep_indices]
            labels_arr = labels_arr[keep_indices]
            times_arr = times_arr[keep_indices]

        return WindowBatch(
            windows=windows_arr,
            labels=labels_arr,
            window_times=times_arr,
            patient_id=patient_id,
            recording_name=recording_name,
        )
