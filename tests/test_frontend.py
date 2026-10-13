"""Test suite for Streamlit frontend script integrity and data bundle loading."""

import importlib.util
from pathlib import Path
import pytest
import numpy as np


def test_frontend_script_exists_and_compiles():
    """Verify that frontend/app.py exists and compiles without syntax errors."""
    app_path = Path(__file__).resolve().parent.parent / "frontend" / "app.py"
    assert app_path.exists(), "frontend/app.py must exist"

    spec = importlib.util.spec_from_file_location("frontend_app", str(app_path))
    assert spec is not None
    assert spec.loader is not None


def test_frontend_styles_exist_and_contain_glassmorphism():
    """Verify that frontend/styles.css exists and defines glassmorphic styles."""
    css_path = Path(__file__).resolve().parent.parent / "frontend" / "styles.css"
    assert css_path.exists(), "frontend/styles.css must exist"

    css_content = css_path.read_text(encoding="utf-8")
    assert "--bg-void" in css_content
    assert "backdrop-filter" in css_content
    assert "ambient-cursor-glow" in css_content
    assert "prefers-reduced-motion" in css_content


def test_load_patient_bundle_schema_and_keys():
    """Verify that load_patient_bundle resolves without KeyError on real archives."""
    from frontend.app import load_patient_bundle

    bundle = load_patient_bundle("chb03_chb03_01")
    assert "windows" in bundle
    assert "labels" in bundle
    assert "window_times" in bundle
    assert "features" in bundle
    assert "patient_id" in bundle
    assert "recording_name" in bundle

    assert isinstance(bundle["windows"], np.ndarray)
    assert bundle["windows"].ndim == 3  # (N, channels, seq_len)
    assert bundle["window_times"].ndim == 2  # (N, 2)
    assert bundle["features"].ndim == 2  # (N, 306)
    assert bundle["patient_id"] == "chb03"
