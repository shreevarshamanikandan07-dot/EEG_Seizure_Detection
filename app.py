"""NeuroGuard — Clinical Decision Support Dashboard for Explainable EEG Seizure Detection.

Hugging Face Spaces and Root Entrypoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root directory to sys.path so modules in src/ and frontend/ can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Execute the main Streamlit application
import runpy

frontend_app = PROJECT_ROOT / "frontend" / "app.py"
runpy.run_path(str(frontend_app), run_name="__main__")
