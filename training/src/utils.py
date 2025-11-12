import os
import time
import yaml
import json
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "serving" / "models"))
CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "params.yaml"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def next_model_version(model_base: Path) -> int:
    model_base.mkdir(parents=True, exist_ok=True)
    versions = []
    for p in model_base.iterdir():
        if p.is_dir() and p.name.isdigit():
            versions.append(int(p.name))
    return (max(versions) + 1) if versions else 1


def timestamp_version() -> int:
    return int(time.time())
