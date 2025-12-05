# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Manual rollback functionality for production models."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from pipelines.notify import send as notify
from training.src.utils import ARTIFACTS_DIR, MODELS_DIR


def rollback_to_previous() -> dict[str, Any]:
    """Rollback production model to the previous version.

    Uses registry_state.json to find the last known good version,
    or falls back to the second-to-last version in the directory.

    Returns:
        Dictionary with rollback details.

    Raises:
        RuntimeError: If there aren't enough versions to rollback.
    """
    prod_dir = MODELS_DIR / "model"
    versions = _get_sorted_versions(prod_dir)

    # Try to get target from saved state
    target_version = _get_target_from_state()

    if target_version is None:
        if len(versions) < 2:
            raise RuntimeError("Not enough production versions to rollback")
        target_version = versions[-2]

    # Create new version from target
    next_ver = (versions[-1] + 1) if versions else target_version
    src = prod_dir / str(target_version)
    dst = prod_dir / str(next_ver)

    if not src.exists():
        raise RuntimeError(f"Target rollback version not found: {src}")

    shutil.copytree(src, dst)

    info = {"rolled_back_to": target_version, "new_version": next_ver}
    print(info)
    notify("Rollback applied", "Primary rolled back to previous version", info)

    return info


def _get_sorted_versions(prod_dir: Path) -> list[int]:
    """Get sorted list of version numbers from production directory.

    Args:
        prod_dir: Path to production model directory.

    Returns:
        Sorted list of version integers.
    """
    if not prod_dir.exists():
        return []
    return sorted(
        int(p.name)
        for p in prod_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    )


def _get_target_from_state() -> int | None:
    """Get target rollback version from registry state file.

    Returns:
        Version number or None if not available.
    """
    state_path = ARTIFACTS_DIR / "registry_state.json"
    if not state_path.exists():
        return None

    try:
        state = json.loads(state_path.read_text())
        if state.get("last_good_prod_dir_version") is not None:
            return int(state["last_good_prod_dir_version"])
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    return None


if __name__ == "__main__":
    rollback_to_previous()
