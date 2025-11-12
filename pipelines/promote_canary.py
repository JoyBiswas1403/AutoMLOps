import shutil
import json
from pathlib import Path
from mlflow.tracking import MlflowClient
from training.src.utils import MODELS_DIR, ARTIFACTS_DIR, load_config, save_json
from pipelines.notify import send as notify


def promote_canary_to_production():
    cfg = load_config()
    model_name = cfg["experiment"]["model_name"]

    canary_dir = MODELS_DIR / f"{model_name}_canary"
    prod_dir = MODELS_DIR / f"{model_name}"
    prod_dir.mkdir(parents=True, exist_ok=True)

    # find latest canary version
    versions = [int(p.name) for p in canary_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not versions:
        raise RuntimeError("No canary versions found")
    latest_canary = max(versions)

    # determine current production latest (for last known good)
    prod_versions = [int(p.name) for p in prod_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    last_good_prod_version = max(prod_versions) if prod_versions else None

    # copy to production as next version (reuse canary version number if free; else append +1)
    proposed = latest_canary
    dst = prod_dir / str(proposed)
    if dst.exists():
        proposed = (max(prod_versions) + 1) if prod_versions else 1
        dst = prod_dir / str(proposed)
    src = canary_dir / str(latest_canary)
    shutil.copytree(src, dst)

    # Transition MLflow model stage from Staging to Production for latest
    mv_version = None
    try:
        client = MlflowClient()
        rm = client.get_registered_model(model_name)
        mv = max(rm.latest_versions, key=lambda v: int(v.version))
        mv_version = int(mv.version)
        client.transition_model_version_stage(
            name=model_name, version=mv.version, stage="Production", archive_existing_versions=True
        )
    except Exception:
        pass

    # Persist registry state
    state_path = ARTIFACTS_DIR / "registry_state.json"
    state = {
        "last_good_prod_dir_version": last_good_prod_version,
        "current_prod_dir_version": proposed,
        "last_prod_registry_version": mv_version,
    }
    save_json(state, state_path)

    info = {"promoted_version": proposed, "source": str(src), "dest": str(dst), "registry": state}
    print(info)
    notify("Promotion complete", "Canary promoted to production", info)


if __name__ == "__main__":
    promote_canary_to_production()
