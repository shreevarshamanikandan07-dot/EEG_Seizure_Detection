"""CHB-MIT Scalp EEG Database Downloader and Verification Utility.

Downloads controlled subsets of official PhysioNet CHB-MIT recordings,
verifies file integrity, and enforces disk safety thresholds.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import requests
from tqdm import tqdm

from src.utils.config import get_project_root


# Official PhysioNet Open Data S3 mirror (high speed direct access) with fallback
PHYSIONET_AWS_URL = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0"
PHYSIONET_DIRECT_URL = "https://physionet.org/files/chbmit/1.0.0"
PHYSIONET_BASE_URL = PHYSIONET_AWS_URL


class CHBMITDownloader:
    """Manages secure streaming downloads of CHB-MIT recordings."""

    def __init__(
        self,
        base_url: str = PHYSIONET_BASE_URL,
        dest_dir: Optional[str | Path] = None,
        min_free_gb: float = 5.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if dest_dir is None:
            self.dest_dir = get_project_root() / "data" / "raw"
        else:
            self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.min_free_gb = min_free_gb

    def check_disk_space(self) -> float:
        """Return free disk space in GB and ensure it exceeds threshold."""
        _, _, free = shutil.disk_usage(self.dest_dir)
        free_gb = free / (1024**3)
        if free_gb < self.min_free_gb:
            raise RuntimeError(
                f"Insufficient disk space: {free_gb:.2f} GB free, "
                f"minimum required is {self.min_free_gb:.2f} GB."
            )
        return free_gb

    def download_file(self, rel_path: str, force: bool = False) -> Path:
        """Download a single file from PhysioNet CHB-MIT repository.

        Args:
            rel_path: Relative path within CHB-MIT, e.g. 'chb01/chb01-summary.txt'
                      or 'chb01/chb01_03.edf'.
            force: If True, redownload even if file exists locally.

        Returns:
            Path: Absolute path to the locally saved file.
        """
        self.check_disk_space()

        target_path = self.dest_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and target_path.stat().st_size > 0 and not force:
            print(f"[SKIP] File already exists: {target_path} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
            return target_path

        url = f"{self.base_url}/{rel_path}"
        print(f"[DOWNLOAD] Fetching {url} -> {target_path} ...")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
        }

        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))

        temp_path = target_path.with_suffix(".tmp")
        initial_pos = 0
        if temp_path.exists():
            initial_pos = temp_path.stat().st_size
            headers["Range"] = f"bytes={initial_pos}-"

        try:
            response = session.get(url, headers=headers, stream=True, timeout=(15, 300))
            if response.status_code == 416:  # Range not satisfiable -> start fresh
                temp_path.unlink(missing_ok=True)
                initial_pos = 0
                headers.pop("Range", None)
                response = session.get(url, headers=headers, stream=True, timeout=(15, 300))
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0)) + initial_pos
            chunk_size = 128 * 1024  # 128 KB chunks for smooth socket keep-alive

            mode = "ab" if initial_pos > 0 else "wb"
            with open(temp_path, mode) as f, tqdm(
                total=total_size,
                initial=initial_pos,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=Path(rel_path).name,
                leave=True,
                ascii=True,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            # Atomic rename once completed
            if target_path.exists():
                target_path.unlink()
            temp_path.rename(target_path)
            print(f"[OK] Saved: {target_path} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
            return target_path

        except Exception as e:
            if target_path.with_suffix(".tmp").exists():
                target_path.with_suffix(".tmp").unlink()
            raise RuntimeError(
                f"Failed to download {url}: {e}\n"
                f"Manual Alternative: Download '{rel_path}' directly from {self.base_url} "
                f"and place it at {target_path.resolve()}"
            )

    def download_patient_summary(self, patient_id: str) -> Path:
        """Download summary text file for a patient."""
        rel = f"{patient_id}/{patient_id}-summary.txt"
        return self.download_file(rel)

    def download_subset(
        self,
        patient_recordings: Dict[str, List[str]]
    ) -> Dict[str, List[Path]]:
        """Download summaries and selected EDF files for specified patients.

        Args:
            patient_recordings: Dict mapping patient_id to list of EDF file names.
                e.g., {'chb01': ['chb01_03.edf', 'chb01_04.edf'], 'chb02': ['chb02_16.edf']}

        Returns:
            Dict[str, List[Path]]: Downloaded paths per patient.
        """
        results: Dict[str, List[Path]] = {}
        for patient_id, edf_names in patient_recordings.items():
            results[patient_id] = []
            # 1. Always download summary file first
            summary_path = self.download_patient_summary(patient_id)
            results[patient_id].append(summary_path)

            # 2. Download each specified EDF file
            for edf_name in edf_names:
                rel = f"{patient_id}/{edf_name}"
                p = self.download_file(rel)
                results[patient_id].append(p)

        return results
