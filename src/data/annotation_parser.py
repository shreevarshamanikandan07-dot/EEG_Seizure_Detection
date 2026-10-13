"""CHB-MIT Annotation and Summary Parser.

Parses official PhysioNet CHB-MIT Scalp EEG Database summary files (*-summary.txt)
to extract exact recording metadata, sampling rates, channel montages,
and precise seizure onset and offset timestamps (in seconds).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SeizureEvent:
    """Represents a single clinical seizure occurrence."""
    index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def overlaps_window(self, win_start: float, win_end: float) -> float:
        """Calculate overlap duration between this seizure and an arbitrary time window."""
        overlap_start = max(self.start_sec, win_start)
        overlap_end = min(self.end_sec, win_end)
        return max(0.0, overlap_end - overlap_start)


@dataclass
class EDFMetadata:
    """Metadata and seizure labels for an individual EDF recording."""
    filename: str
    start_time_str: Optional[str] = None
    end_time_str: Optional[str] = None
    seizure_count: int = 0
    seizures: List[SeizureEvent] = field(default_factory=list)

    @property
    def has_seizures(self) -> bool:
        return len(self.seizures) > 0


@dataclass
class PatientSummary:
    """Parsed summary information for a patient directory."""
    patient_id: str
    sampling_rate: int
    channels: List[str]
    recordings: Dict[str, EDFMetadata] = field(default_factory=dict)

    @property
    def total_recordings(self) -> int:
        return len(self.recordings)

    @property
    def total_seizures(self) -> int:
        return sum(rec.seizure_count for rec in self.recordings.values())


class AnnotationParser:
    """Robust parser for CHB-MIT summary text files."""

    @staticmethod
    def parse_summary_file(summary_path: str | Path) -> PatientSummary:
        """Parse a *-summary.txt file from CHB-MIT.

        Args:
            summary_path: Path to the summary text file.

        Returns:
            PatientSummary: Structured metadata and seizure intervals.
        """
        path = Path(summary_path)
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")

        text = path.read_text(encoding="utf-8", errors="ignore")
        patient_id = path.stem.replace("-summary", "").replace("_summary", "")

        # 1. Parse Data Sampling Rate
        sr_match = re.search(r"Data Sampling Rate:\s*(\d+)\s*Hz", text, re.IGNORECASE)
        sampling_rate = int(sr_match.group(1)) if sr_match else 256

        # 2. Parse Channel Montage
        channels: List[str] = []
        channel_pattern = re.compile(r"Channel\s+\d+:\s*([A-Za-z0-9\-]+)")
        for line in text.splitlines():
            m = channel_pattern.search(line)
            if m:
                ch_name = m.group(1).strip()
                if ch_name not in channels:
                    channels.append(ch_name)

        # 3. Parse Individual File Blocks
        recordings: Dict[str, EDFMetadata] = {}
        # Split by "File Name:"
        file_blocks = re.split(r"File Name:\s*", text)

        for block in file_blocks[1:]:  # First chunk is header before first File Name
            lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
            if not lines:
                continue

            filename = lines[0].strip()
            # Normalize to basename
            filename = Path(filename).name

            start_time = None
            end_time = None
            seizure_count = 0
            seizures: List[SeizureEvent] = []

            # Match start & end times
            m_start = re.search(r"File Start Time:\s*([0-9:]+)", block)
            if m_start:
                start_time = m_start.group(1)

            m_end = re.search(r"File End Time:\s*([0-9:]+)", block)
            if m_end:
                end_time = m_end.group(1)

            m_count = re.search(r"Number of Seizures in File:\s*(\d+)", block)
            if m_count:
                seizure_count = int(m_count.group(1))

            # Match Seizure Start & End Times
            # Patterns in CHB-MIT:
            # "Seizure Start Time: 2996 seconds"
            # "Seizure 1 Start Time: 2996 seconds"
            # "Seizure [1] Start Time: 2996 seconds"
            starts = re.findall(r"Seizure(?:\s+\d+|\s+\[\d+\])?\s+Start Time:\s*(\d+)\s*seconds", block, re.IGNORECASE)
            ends = re.findall(r"Seizure(?:\s+\d+|\s+\[\d+\])?\s+End Time:\s*(\d+)\s*seconds", block, re.IGNORECASE)

            for i, (st, en) in enumerate(zip(starts, ends)):
                st_sec = float(st)
                en_sec = float(en)
                if en_sec > st_sec:
                    seizures.append(SeizureEvent(index=i + 1, start_sec=st_sec, end_sec=en_sec))

            recordings[filename] = EDFMetadata(
                filename=filename,
                start_time_str=start_time,
                end_time_str=end_time,
                seizure_count=max(seizure_count, len(seizures)),
                seizures=seizures
            )

        return PatientSummary(
            patient_id=patient_id,
            sampling_rate=sampling_rate,
            channels=channels,
            recordings=recordings
        )
