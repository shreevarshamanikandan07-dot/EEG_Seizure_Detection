"""End-to-End EEG Preprocessing and Feature Pipeline.

Orchestrates EDF loading, filtering, segmentation, feature extraction,
and data persistence for both deep-learning tensors and classical feature matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.utils.config import Config, get_project_root
from src.data.edf_reader import EDFReader
from src.data.annotation_parser import AnnotationParser, SeizureEvent
from src.preprocessing.preprocessor import EEGPreprocessor
from src.preprocessing.segmenter import EEGSegmenter, WindowBatch
from src.features.extractor import EEGFeatureExtractor


class ProcessingPipeline:
    """Unified pipeline for processing continuous EEG into model-ready datasets."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()
        prep_cfg = self.cfg.preprocessing
        bp_cfg = prep_cfg.get("bandpass", {})
        notch_cfg = prep_cfg.get("notch", {})
        seg_cfg = prep_cfg.get("segmentation", {})
        ds_cfg = self.cfg.dataset

        self.sampling_rate = float(ds_cfg.get("sampling_rate", 256.0))
        self.standard_channels = ds_cfg.get("standard_channels", [])

        self.preprocessor = EEGPreprocessor(
            sampling_rate=self.sampling_rate,
            bandpass_low=float(bp_cfg.get("low_freq", 0.5)),
            bandpass_high=float(bp_cfg.get("high_freq", 40.0)),
            bandpass_order=int(bp_cfg.get("order", 4)),
            notch_freq=float(notch_cfg.get("freq", 60.0)),
            notch_q=float(notch_cfg.get("quality_factor", 30.0)),
        )

        self.segmenter = EEGSegmenter(
            window_duration_sec=float(seg_cfg.get("window_duration_sec", 4.0)),
            step_duration_sec=float(seg_cfg.get("step_duration_sec", 2.0)),
            seizure_overlap_threshold=float(seg_cfg.get("seizure_overlap_threshold", 0.5)),
            exclude_ambiguous_boundaries=True,
        )

        self.feature_extractor = EEGFeatureExtractor(sampling_rate=self.sampling_rate)

    def process_file(
        self,
        edf_path: str | Path,
        summary_path: Optional[str | Path] = None,
        patient_id: Optional[str] = None,
        subsample_ratio: Optional[int] = None,
        fit_normalizer: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[str], str]]:
        """Process a single EDF file from raw signal to windowed tensors and features.

        Args:
            edf_path: Path to raw .edf file.
            summary_path: Optional path to patient's summary .txt file.
            patient_id: Optional patient identifier string.
            subsample_ratio: Non-seizure subsampling ratio for class balancing.
            fit_normalizer: If True, fits normalizer on this recording (use for training patient).

        Returns:
            Dict containing:
                - 'windows': ndarray of shape (N, C, samples)
                - 'features': ndarray of shape (N, D)
                - 'labels': ndarray of shape (N,)
                - 'window_times': ndarray of shape (N, 2)
                - 'channels': list of channel names
                - 'feature_names': list of feature column names
                - 'patient_id': patient string
                - 'recording_name': EDF filename string
        """
        edf_file = Path(edf_path)
        pat_id = patient_id or edf_file.parent.name
        rec_name = edf_file.name

        # 1. Parse Seizures from Summary if available
        seizures: List[SeizureEvent] = []
        if summary_path is not None:
            summary = AnnotationParser.parse_summary_file(summary_path)
            rec_meta = summary.recordings.get(rec_name)
            if rec_meta is not None:
                seizures = rec_meta.seizures

        # 2. Load recording with standard channels
        raw_data, sfreq, actual_channels = EDFReader.load_recording(
            edf_path=edf_file,
            target_channels=self.standard_channels if self.standard_channels else None,
            preload=True,
        )

        # 3. Fit normalizer on training data if requested
        if fit_normalizer:
            self.preprocessor.fit_normalizer(raw_data)

        # 4. Filter and normalize continuous signal
        clean_data = self.preprocessor.process(raw_data, normalize=True)

        # 5. Segment into sliding windows
        window_batch = self.segmenter.segment_recording(
            data=clean_data,
            sampling_rate=sfreq,
            seizures=seizures,
            patient_id=pat_id,
            recording_name=rec_name,
            non_seizure_subsample_ratio=subsample_ratio,
        )

        # 6. Extract tabular physiological biomarkers
        features_matrix, feat_names = self.feature_extractor.extract_batch_features(
            windows=window_batch.windows,
            channel_names=actual_channels,
        )

        return {
            "windows": window_batch.windows,
            "features": features_matrix,
            "labels": window_batch.labels,
            "window_times": window_batch.window_times,
            "channels": actual_channels,
            "feature_names": feat_names,
            "patient_id": pat_id,
            "recording_name": rec_name,
        }

    def save_processed(
        self,
        result: Dict[str, Union[np.ndarray, List[str], str]],
        output_dir: Optional[str | Path] = None,
    ) -> Path:
        """Persist processed dataset bundle to compressed .npz archive."""
        root = get_project_root()
        out_dir = Path(output_dir) if output_dir else root / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{result['patient_id']}_{Path(str(result['recording_name'])).stem}.npz"
        out_path = out_dir / filename

        np.savez_compressed(
            out_path,
            windows=result["windows"],
            features=result["features"],
            labels=result["labels"],
            window_times=result["window_times"],
            channels=np.array(result["channels"]),
            feature_names=np.array(result["feature_names"]),
            patient_id=str(result["patient_id"]),
            recording_name=str(result["recording_name"]),
        )
        return out_path
