"""Unit tests for CHB-MIT annotation and summary parser."""

import pytest
from pathlib import Path
from src.data.annotation_parser import AnnotationParser, SeizureEvent


SAMPLE_SUMMARY_TEXT = """Data Sampling Rate: 256 Hz

Channels in EDF Files:
Channel 1: FP1-F7
Channel 2: F7-T7
Channel 3: T7-P7
Channel 4: P7-O1
Channel 5: FP1-F3
Channel 6: F3-C3

File Name: chb01_01.edf
File Start Time: 11:42:54
File End Time: 12:42:54
Number of Seizures in File: 0

File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb01_04.edf
File Start Time: 14:53:44
File End Time: 15:53:44
Number of Seizures in File: 2
Seizure 1 Start Time: 1467 seconds
Seizure 1 End Time: 1494 seconds
Seizure 2 Start Time: 2000 seconds
Seizure 2 End Time: 2050 seconds
"""


def test_parse_summary_text(tmp_path: Path):
    """Test parsing standard CHB-MIT summary file."""
    summary_file = tmp_path / "chb01-summary.txt"
    summary_file.write_text(SAMPLE_SUMMARY_TEXT, encoding="utf-8")

    parsed = AnnotationParser.parse_summary_file(summary_file)
    assert parsed.patient_id == "chb01"
    assert parsed.sampling_rate == 256
    assert len(parsed.channels) == 6
    assert "FP1-F7" in parsed.channels
    assert "F3-C3" in parsed.channels

    assert parsed.total_recordings == 3
    assert parsed.total_seizures == 3

    # File 1: 0 seizures
    rec1 = parsed.recordings["chb01_01.edf"]
    assert rec1.seizure_count == 0
    assert not rec1.has_seizures

    # File 3: 1 seizure
    rec3 = parsed.recordings["chb01_03.edf"]
    assert rec3.seizure_count == 1
    assert rec3.has_seizures
    assert rec3.seizures[0].start_sec == 2996
    assert rec3.seizures[0].end_sec == 3036
    assert rec3.seizures[0].duration_sec == 40.0

    # File 4: 2 seizures
    rec4 = parsed.recordings["chb01_04.edf"]
    assert rec4.seizure_count == 2
    assert len(rec4.seizures) == 2
    assert rec4.seizures[0].start_sec == 1467
    assert rec4.seizures[1].start_sec == 2000


def test_seizure_overlap_calculation():
    """Test overlap calculation between seizure event and arbitrary time window."""
    sz = SeizureEvent(index=1, start_sec=100.0, end_sec=150.0)

    # Window completely before seizure
    assert sz.overlaps_window(50.0, 90.0) == 0.0

    # Window completely after seizure
    assert sz.overlaps_window(160.0, 200.0) == 0.0

    # Window partially overlapping onset (90 to 110: overlap is 10s)
    assert sz.overlaps_window(90.0, 110.0) == 10.0

    # Window completely inside seizure (110 to 120: overlap is 10s)
    assert sz.overlaps_window(110.0, 120.0) == 10.0

    # Window completely enclosing seizure (90 to 160: overlap is 50s)
    assert sz.overlaps_window(90.0, 160.0) == 50.0
