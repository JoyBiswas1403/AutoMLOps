# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Canary promotion from staging to production."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from pipelines.notify import send as notify
from training.src.utils import ARTIFACTS_DIR, MODELS_DIR, load_config, save_json


def promote_canary_to_production() -> dict[str, Any]:
    """Promote the latest canary model to production.

    This function:
    1. Finds the latest canary version
    2. Copies it to the production directory
    3. Transitions the MLflow model to Production stage
    4. Saves registry state for potential rollback

    Returns:
        Dictionary with promotion details.

    Raises:
        RuntimeError: If no canary versions are found.
    """
    cfg = load_config()
    model_name = cfg["experiment"]["model_name"]

    canary_dir = MODELS_DIR / f"{model_name}_canary"
    prod_dir = MODELS_DIR / f"{model_name}"
    prod_dir.mkdir(parents=True, exist_ok=True)

    # Find latest canary version
    canary_versions = _get_versions(canary_dir)
    if not canary_versions:
        raise RuntimeError("No canary versions found")
    latest_canary = max(canary_versions)

    # Track current production version for rollback
    prod_versions = _get_versions(prod_dir)
    last_good_prod_version = max(prod_versions) if prod_versions else None

    # Determine destination version
    proposed = latest_canary
    dst = prod_dir / str(proposed)
    if dst.exists():
        proposed = (max(prod_versions) + 1) if prod_versions else 1
        dst = prod_dir / str(proposed)

    # Copy canary to production
    src = canary_dir / str(latest_canary)
    shutil.copytree(src, dst)

    # Transition MLflow model to Production
    mv_version = _transition_to_production(model_name)

    # Save state for rollback
    state = {
        "last_good_prod_dir_version": last_good_prod_version,
        "current_prod_dir_version": proposed,
        "last_prod_registry_version": mv_version,
    }
    save_json(state, ARTIFACTS_DIR / "registry_state.json")

    info = {
        "promoted_version": proposed,
        "source": str(src),
        "dest": str(dst),
        "registry": state,
    }
    print(info)
    notify("Promotion complete", "Canary promoted to production", info)

    return info


def _get_versions(directory: Path) -> list[int]:
    """Get list of version numbers from a directory.

    Args:
        directory: Path to model directory.

    Returns:
        List of version integers.
    """
    if not directory.exists():
        return []
    return [
        int(p.name)
        for p in directory.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]


def _transition_to_production(model_name: str) -> int | None:
    """Transition the latest MLflow model version to Production.

    Args:
        model_name: Name of the registered model.

    Returns:
        Version number that was transitioned, or None if failed.
    """
    try:
        client = MlflowClient()
        rm = client.get_registered_model(model_name)
        mv = max(rm.latest_versions, key=lambda v: int(v.version))
        mv_version = int(mv.version)
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        return mv_version
    except Exception as e:
        print(f"Warning: MLflow transition failed: {e}")
        return None


if __name__ == "__main__":
    promote_canary_to_production()
