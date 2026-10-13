"""Script to download and verify the validation subset of CHB-MIT Scalp EEG.

Downloads patient-independent subset:
- Train patient: chb01 (summary + chb01_03.edf [seizure at 2996s-3036s])
- Validation patient: chb02 (summary + chb02_16.edf [seizure at 130s-212s])
- Test patient: chb03 (summary + chb03_01.edf [seizure at 362s-414s])
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader import CHBMITDownloader
from src.data.annotation_parser import AnnotationParser
from src.data.edf_reader import EDFReader


def main():
    print("=" * 70)
    print(" NeuroGuard: Downloading CHB-MIT Biomedical EEG Subset")
    print("=" * 70)

    downloader = CHBMITDownloader()
    free_gb = downloader.check_disk_space()
    print(f"[*] Verified Free Disk Space: {free_gb:.2f} GB")

    target_subset = {
        "chb01": ["chb01_03.edf"],
        "chb02": ["chb02_16.edf"],
        "chb03": ["chb03_01.edf"],
    }

    print("\n[*] Target Subset Specification:")
    for pat, files in target_subset.items():
        print(f"    - Patient {pat}: {files}")

    try:
        downloaded = downloader.download_subset(target_subset)
    except Exception as e:
        print(f"\n[!] Download encounter error: {e}")
        print("\n[!] Please verify your network connection.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" Verifying Downloaded Recordings and Parsing Annotations")
    print("=" * 70)

    for pat, paths in downloaded.items():
        summary_path = [p for p in paths if p.name.endswith(".txt")][0]
        edf_paths = [p for p in paths if p.name.endswith(".edf")]

        print(f"\n[+] Patient: {pat}")
        patient_summary = AnnotationParser.parse_summary_file(summary_path)
        print(f"    Sampling Rate: {patient_summary.sampling_rate} Hz")
        print(f"    Total Annotated Channels: {len(patient_summary.channels)}")
        print(f"    Total Recordings Listed in Summary: {patient_summary.total_recordings}")

        for edf_p in edf_paths:
            info = EDFReader.get_info(edf_p)
            print(f"    -> EDF File: {info.filename}")
            print(f"       Duration: {info.duration_sec:.1f} sec ({info.duration_sec/60:.1f} minutes)")
            print(f"       Channels in file: {info.n_channels}")
            print(f"       Sampling frequency: {info.sampling_rate} Hz")

            # Check seizures in this file
            rec_meta = patient_summary.recordings.get(info.filename)
            if rec_meta and rec_meta.has_seizures:
                for sz in rec_meta.seizures:
                    print(f"       [!] Seizure #{sz.index}: Start = {sz.start_sec}s, End = {sz.end_sec}s (Duration = {sz.duration_sec}s)")
            else:
                print("       No seizures annotated for this file.")

    print("\n[OK] Phase 3 Dataset Verification Complete!")


if __name__ == "__main__":
    main()
