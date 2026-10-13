"""Configuration manager for NeuroGuard.

Loads YAML configuration, validates schema, resolves relative paths,
and provides access throughout the project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the NeuroGuard project root directory."""
    # This file is at src/utils/config.py -> 2 levels up is project root
    return Path(__file__).resolve().parent.parent.parent


def load_yaml_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load and parse YAML configuration file.

    Args:
        config_path: Optional path to config.yaml. Defaults to configs/config.yaml.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    root = get_project_root()
    if config_path is None:
        target_path = root / "configs" / "config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Resolve and create all project directory paths from configuration.

    Args:
        cfg: Configuration dictionary containing a 'paths' section.

    Returns:
        Dict[str, Path]: Map of path keys to resolved Path objects.
    """
    root = get_project_root()
    resolved: Dict[str, Path] = {"project_root": root}

    paths_dict = cfg.get("paths", {})
    for key, rel_path in paths_dict.items():
        p = root / rel_path
        p.mkdir(parents=True, exist_ok=True)
        resolved[key] = p

    return resolved


class Config:
    """Singleton-style wrapper providing typed configuration attributes."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.raw: Dict[str, Any] = load_yaml_config(config_path)
        self.root: Path = get_project_root()
        self.paths: Dict[str, Path] = resolve_paths(self.raw)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def project(self) -> Dict[str, Any]:
        return self.raw.get("project", {})

    @property
    def dataset(self) -> Dict[str, Any]:
        return self.raw.get("dataset", {})

    @property
    def preprocessing(self) -> Dict[str, Any]:
        return self.raw.get("preprocessing", {})

    @property
    def features(self) -> Dict[str, Any]:
        return self.raw.get("features", {})

    @property
    def models(self) -> Dict[str, Any]:
        return self.raw.get("models", {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw.get("evaluation", {})

    @property
    def server(self) -> Dict[str, Any]:
        return self.raw.get("server", {})

    @property
    def frontend(self) -> Dict[str, Any]:
        return self.raw.get("frontend", {})


# Global instance for quick imports
config = Config()
