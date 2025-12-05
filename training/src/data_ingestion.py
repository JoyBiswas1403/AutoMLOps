# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Data ingestion module for generating and splitting datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from .utils import ARTIFACTS_DIR, load_config

# =============================================================================
# Data Directories
# =============================================================================
RAW_DIR: Path = ARTIFACTS_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 20,
    n_informative: int = 10,
    n_redundant: int = 2,
    n_classes: int = 2,
    random_state: int = 42,
    class_sep: float = 1.5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic classification data.

    Args:
        n_samples: Number of samples to generate.
        n_features: Total number of features.
        n_informative: Number of informative features.
        n_redundant: Number of redundant features.
        n_classes: Number of classes.
        random_state: Random seed for reproducibility.
        class_sep: Factor controlling class separation.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state,
        class_sep=class_sep,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


def load_real_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load real Credit Card Fraud dataset.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    from .data_loader import load_real_dataset

    return load_real_dataset()


def generate_and_split(
    data_source: Literal["synthetic", "real"] | None = None,
) -> dict[str, str]:
    """Generate or load data and split into train/val/test sets.

    Uses configuration from params.yaml to determine feature counts,
    split ratios, and data source.

    Args:
        data_source: Override config to use "synthetic" or "real" data.

    Returns:
        Dictionary with paths to train, val, and test CSV files.
    """
    cfg = load_config()
    fcfg = cfg["features"]
    ecfg = cfg["experiment"]

    # Determine data source
    if data_source is None:
        data_source = cfg.get("data", {}).get("source", "synthetic")

    if data_source == "real":
        print("Loading real Credit Card Fraud dataset...")
        X, y = load_real_data()
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    else:
        print("Generating synthetic data...")
        X, y = generate_synthetic_data(
            n_samples=fcfg.get("n_samples", 10000),
            n_features=fcfg["n_features"],
            n_informative=fcfg["n_informative"],
            n_redundant=fcfg["n_redundant"],
            n_classes=fcfg["n_classes"],
            random_state=ecfg["random_state"],
            class_sep=1.5,
        )

    # Split: train -> (val, test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=ecfg["test_size"], random_state=ecfg["random_state"]
    )

    val_rel = ecfg["val_size"] / (1 - ecfg["test_size"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_rel, random_state=ecfg["random_state"]
    )

    # Combine features and target
    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    val_df = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    # Save to CSV
    train_path = RAW_DIR / "train.csv"
    val_path = RAW_DIR / "val.csv"
    test_path = RAW_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    print(f"Positive rate: {y_train.mean():.2%}")

    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
    }


if __name__ == "__main__":
    paths = generate_and_split()
    print(paths)
