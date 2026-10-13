"""Unit tests for configuration loading and validation."""

from pathlib import Path
import pytest
from src.utils.config import load_yaml_config, resolve_paths, Config, get_project_root


def test_load_yaml_config_defaults():
    """Verify loading default config.yaml returns expected dictionary structure."""
    cfg = load_yaml_config()
    assert isinstance(cfg, dict)
    assert "project" in cfg
    assert "paths" in cfg
    assert "dataset" in cfg
    assert "preprocessing" in cfg
    assert "features" in cfg
    assert "models" in cfg


def test_config_paths_resolution():
    """Verify that all configured paths are resolved to valid Path objects."""
    cfg = load_yaml_config()
    resolved = resolve_paths(cfg)
    assert "project_root" in resolved
    assert resolved["project_root"].exists()

    for key, path_obj in resolved.items():
        assert isinstance(path_obj, Path)
        assert path_obj.exists(), f"Resolved path for '{key}' does not exist: {path_obj}"


def test_config_object_properties():
    """Verify Config singleton helper methods and properties."""
    conf = Config()
    assert conf.project.get("name") == "NeuroGuard_EEG_Seizure_Detection"
    assert conf.dataset.get("sampling_rate") == 256
    assert "bandpass" in conf.preprocessing
    assert conf.preprocessing["bandpass"]["low_freq"] == 0.5
    assert conf.preprocessing["bandpass"]["high_freq"] == 40.0
    assert "cnn_1d" in conf.models
    assert conf.models["cnn_1d"]["in_channels"] == 18


def test_patient_split_specified():
    """Ensure patient-independent split is explicitly configured."""
    conf = Config()
    split = conf.dataset.get("patient_split", {})
    assert "train_patients" in split
    assert "val_patients" in split
    assert "test_patients" in split

    # Critical requirement: train, val, test must have disjoint patient sets!
    train_set = set(split["train_patients"])
    val_set = set(split["val_patients"])
    test_set = set(split["test_patients"])

    assert train_set.isdisjoint(val_set), "Train and validation patient sets overlap!"
    assert train_set.isdisjoint(test_set), "Train and test patient sets overlap!"
    assert val_set.isdisjoint(test_set), "Validation and test patient sets overlap!"
