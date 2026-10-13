"""Unit tests for environment validation and system readiness."""

import sys
import platform
from pathlib import Path
import pytest
import numpy as np
import torch

from src.utils.config import get_project_root
from src.utils.seed import set_seed


def test_python_version():
    """Verify that Python version is at least 3.10."""
    assert sys.version_info >= (3, 10), f"Python version {platform.python_version()} is less than 3.10"


def test_project_root_exists():
    """Verify project root path resolves correctly and exists."""
    root = get_project_root()
    assert root.exists()
    assert root.is_dir()
    assert (root / "configs" / "config.yaml").exists()


def test_core_directories_exist():
    """Verify that all essential subdirectories exist."""
    root = get_project_root()
    required = [
        "configs",
        "data/raw",
        "data/processed",
        "data/interim",
        "src/data",
        "src/preprocessing",
        "src/features",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/explainability",
        "src/visualization",
        "src/api",
        "src/utils",
        "tests",
        "scripts",
        "artifacts/models",
        "artifacts/metrics",
        "artifacts/plots",
        "artifacts/reports",
    ]
    for rel_path in required:
        p = root / rel_path
        assert p.exists(), f"Required path {p} does not exist"
        assert p.is_dir(), f"Path {p} is not a directory"


def test_seed_reproducibility():
    """Verify that set_seed ensures reproducible random numbers."""
    set_seed(1234)
    np_rand_1 = np.random.rand(5)
    torch_rand_1 = torch.rand(5)

    set_seed(1234)
    np_rand_2 = np.random.rand(5)
    torch_rand_2 = torch.rand(5)

    np.testing.assert_allclose(np_rand_1, np_rand_2)
    assert torch.equal(torch_rand_1, torch_rand_2)
