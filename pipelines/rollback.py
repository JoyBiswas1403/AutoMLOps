import shutil
import json
from pathlib import Path
from training.src.utils import MODELS_DIR, ARTIFACTS_DIR
from pipelines.notify import send as notify


def rollback_to_previous():
    prod_dir = MODELS_DIR / "model"
    versions = sorted([int(p.name) for p in prod_dir.iterdir() if p.is_dir() and p.name.isdigit()])

    # Try registry_state.json first
    state_path = ARTIFACTS_DIR / "registry_state.json"
    target_version = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            if state.get("last_good_prod_dir_version") is not None:
                target_version = int(state["last_good_prod_dir_version"])
        except Exception:
            target_version = None

    if target_version is None:
        if len(versions) < 2:
            raise RuntimeError("Not enough production versions to rollback")
        target_version = versions[-2]

    next_ver = (versions[-1] + 1) if versions else target_version
    src = prod_dir / str(target_version)
    dst = prod_dir / str(next_ver)
    if not src.exists():
        raise RuntimeError(f"Target rollback version not found: {src}")
    shutil.copytree(src, dst)

    info = {"rolled_back_to": target_version, "new_version": next_ver}
    print(info)
    notify("Rollback applied", "Primary rolled back to previous version", info)


if __name__ == "__main__":
    rollback_to_previous()
