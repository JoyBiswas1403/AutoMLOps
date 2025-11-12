from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pathlib import Path
from .utils import load_config, ARTIFACTS_DIR

RAW_DIR = ARTIFACTS_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def generate_and_split():
    cfg = load_config()
    fcfg = cfg["features"]
    ecfg = cfg["experiment"]

    X, y = make_classification(
        n_samples=10000,
        n_features=fcfg["n_features"],
        n_informative=fcfg["n_informative"],
        n_redundant=fcfg["n_redundant"],
        n_classes=fcfg["n_classes"],
        random_state=ecfg["random_state"],
        class_sep=1.5,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=ecfg["test_size"], random_state=ecfg["random_state"]
    )
    val_rel = cfg["experiment"]["val_size"] / (1 - cfg["experiment"]["test_size"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_rel, random_state=ecfg["random_state"]
    )

    (RAW_DIR / "train.csv").write_text(pd.concat([X_train, y_train], axis=1).to_csv(index=False))
    (RAW_DIR / "val.csv").write_text(pd.concat([X_val, y_val], axis=1).to_csv(index=False))
    (RAW_DIR / "test.csv").write_text(pd.concat([X_test, y_test], axis=1).to_csv(index=False))

    return {
        "train": str(RAW_DIR / "train.csv"),
        "val": str(RAW_DIR / "val.csv"),
        "test": str(RAW_DIR / "test.csv"),
    }


if __name__ == "__main__":
    paths = generate_and_split()
    print(paths)
