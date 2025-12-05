# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Utility functions and configuration loaders for the training pipeline."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# Path Configuration
# =============================================================================
PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
ARTIFACTS_DIR: Path = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "serving" / "models"))
CONFIG_PATH: Path = PROJECT_ROOT / "training" / "configs" / "params.yaml"

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: dict[str, Any], path: Path) -> None:
    """Save dictionary as JSON file.

    Args:
        obj: Dictionary to save.
        path: Target file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def next_model_version(model_base: Path) -> int:
    """Determine the next version number for a model.

    Scans the model directory for existing version folders (numeric names)
    and returns the next version number.

    Args:
        model_base: Base directory for model versions.

    Returns:
        Next version number (1 if no versions exist).
    """
    model_base.mkdir(parents=True, exist_ok=True)
    versions: list[int] = []
    for p in model_base.iterdir():
        if p.is_dir() and p.name.isdigit():
            versions.append(int(p.name))
    return (max(versions) + 1) if versions else 1


def timestamp_version() -> int:
    """Generate a version number based on current timestamp.

    Returns:
        Unix timestamp as integer.
    """
    return int(time.time())
