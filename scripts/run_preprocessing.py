"""Run full preprocessing and feature extraction on downloaded CHB-MIT subset.

Processes:
1. Training partition (chb01): Fits normalizer, filters, segments, extracts features.
2. Validation partition (chb02): Applies training normalizer, filters, segments, extracts features.
3. Held-out test partition (chb03): Applies training normalizer, filters, segments, extracts features.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import ProcessingPipeline
from src.utils.config import get_project_root


def main():
    print("=" * 70)
    print(" NeuroGuard: EEG Preprocessing & Feature Extraction Pipeline")
    print("=" * 70)

    root = get_project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    models_dir = root / "artifacts" / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ProcessingPipeline()

    tasks = [
        {
            "partition": "TRAIN",
            "patient_id": "chb01",
            "edf_file": "chb01_03.edf",
            "fit_normalizer": True,
            "subsample_ratio": 4,
        },
        {
            "partition": "VALIDATION",
            "patient_id": "chb02",
            "edf_file": "chb02_16.edf",
            "fit_normalizer": False,
            "subsample_ratio": 4,
        },
        {
            "partition": "TEST (HELD-OUT)",
            "patient_id": "chb03",
            "edf_file": "chb03_01.edf",
            "fit_normalizer": False,
            "subsample_ratio": 4,
        },
    ]

    for t in tasks:
        pat = t["patient_id"]
        edf_name = t["edf_file"]
        edf_path = raw_dir / pat / edf_name
        summary_path = raw_dir / pat / f"{pat}-summary.txt"

        print(f"\n[*] Processing [{t['partition']}] - Patient: {pat}, File: {edf_name} ...")
        if not edf_path.exists():
            print(f"    [!] Error: EDF file not found at {edf_path}")
            continue

        res = pipeline.process_file(
            edf_path=edf_path,
            summary_path=summary_path,
            patient_id=pat,
            subsample_ratio=t["subsample_ratio"],
            fit_normalizer=t["fit_normalizer"],
        )

        out_path = pipeline.save_processed(res, output_dir=processed_dir)

        windows = res["windows"]
        features = res["features"]
        labels = res["labels"]
        n_seizures = int(np.sum(labels == 1))
        n_non_seizures = int(np.sum(labels == 0))

        print(f"    [OK] Saved: {out_path.name}")
        print(f"         Windows Shape (for Deep Learning): {windows.shape} (N, Channels, Samples)")
        print(f"         Features Shape (for Classical ML): {features.shape} (N, Biomarkers)")
        print(f"         Class Distribution: Seizure={n_seizures}, Non-Seizure={n_non_seizures} (Total={len(labels)})")
        print(f"         Channels: {len(res['channels'])} | Feature Dimension: {len(res['feature_names'])}")

    # Persist the training normalizer scaler
    scaler_path = models_dir / "eeg_scaler.joblib"
    pipeline.preprocessor.save_scaler(scaler_path)
    print(f"\n[OK] Training Normalizer saved to: {scaler_path}")
    print("\n" + "=" * 70)
    print(" [OK] Phase 4 Preprocessing & Feature Extraction Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
