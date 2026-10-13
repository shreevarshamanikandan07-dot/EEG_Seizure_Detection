"""Test suite for Streamlit frontend script integrity."""

import importlib.util
from pathlib import Path


def test_frontend_script_exists_and_compiles():
    """Verify that frontend/app.py exists and compiles without syntax errors."""
    app_path = Path(__file__).resolve().parent.parent / "frontend" / "app.py"
    assert app_path.exists(), "frontend/app.py must exist"

    spec = importlib.util.spec_from_file_location("frontend_app", str(app_path))
    assert spec is not None
    assert spec.loader is not None
