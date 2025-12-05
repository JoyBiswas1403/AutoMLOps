# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Data loader module supporting both synthetic and real datasets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import ARTIFACTS_DIR, PROJECT_ROOT

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Credit Card Fraud Dataset info
CREDIT_CARD_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
CREDIT_CARD_LOCAL = DATA_DIR / "creditcard.csv"


def download_credit_card_dataset() -> Path:
    """Download the Credit Card Fraud dataset if not present.

    The dataset is from Kaggle but also available via TensorFlow's data mirror.
    Contains 284,807 transactions with 492 frauds (~0.17% positive class).

    Features:
    - Time: Seconds elapsed between this transaction and first transaction
    - V1-V28: PCA-transformed features (anonymized)
    - Amount: Transaction amount
    - Class: Target (1=fraud, 0=legitimate)

    Returns:
        Path to the downloaded CSV file.

    Raises:
        RuntimeError: If download fails.
    """
    if CREDIT_CARD_LOCAL.exists():
        print(f"Dataset already exists at {CREDIT_CARD_LOCAL}")
        return CREDIT_CARD_LOCAL

    print(f"Downloading Credit Card Fraud dataset from {CREDIT_CARD_URL}...")

    try:
        import urllib.request

        urllib.request.urlretrieve(CREDIT_CARD_URL, CREDIT_CARD_LOCAL)
        print(f"Downloaded to {CREDIT_CARD_LOCAL}")

        # Verify the file
        df = pd.read_csv(CREDIT_CARD_LOCAL)
        print(f"Dataset shape: {df.shape}")
        print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")

        return CREDIT_CARD_LOCAL
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}") from e


def load_real_dataset(
    path: Path | str | None = None,
    target_col: str = "Class",
    drop_cols: list[str] | None = None,
    undersample: bool = True,
    undersample_ratio: float = 5.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a real dataset from CSV.

    Args:
        path: Path to CSV file. If None, uses Credit Card dataset.
        target_col: Name of the target column.
        drop_cols: Columns to drop (e.g., Time).
        undersample: Whether to undersample majority class.
        undersample_ratio: Ratio of negative to positive samples.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    if path is None:
        path = download_credit_card_dataset()
    else:
        path = Path(path)

    df = pd.read_csv(path)

    # Default columns to drop for credit card dataset
    if drop_cols is None:
        drop_cols = ["Time"] if "Time" in df.columns else []

    # Separate features and target
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle class imbalance with undersampling
    if undersample and y.value_counts().min() < y.value_counts().max() * 0.1:
        X, y = _undersample(X, y, undersample_ratio, random_state)

    # Rename columns to standard format
    feature_cols = [f"f{i}" for i in range(X.shape[1])]
    X.columns = feature_cols

    return X, pd.Series(y.values, name="target")


def _undersample(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Undersample majority class to handle class imbalance.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        ratio: Desired ratio of negative to positive samples.
        random_state: Random seed.

    Returns:
        Undersampled (X, y) tuple.
    """
    np.random.seed(random_state)

    # Find minority and majority
    class_counts = y.value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    minority_count = class_counts[minority_class]
    desired_majority = int(minority_count * ratio)

    # Get indices
    minority_idx = y[y == minority_class].index
    majority_idx = y[y == majority_class].index

    # Sample majority class
    if len(majority_idx) > desired_majority:
        sampled_majority_idx = np.random.choice(
            majority_idx, size=desired_majority, replace=False
        )
    else:
        sampled_majority_idx = majority_idx

    # Combine
    final_idx = np.concatenate([minority_idx, sampled_majority_idx])
    np.random.shuffle(final_idx)

    return X.loc[final_idx].reset_index(drop=True), y.loc[final_idx].reset_index(drop=True)


def get_dataset_info() -> dict[str, Any]:
    """Get information about the available dataset.

    Returns:
        Dictionary with dataset statistics.
    """
    if not CREDIT_CARD_LOCAL.exists():
        return {"available": False, "path": str(CREDIT_CARD_LOCAL)}

    df = pd.read_csv(CREDIT_CARD_LOCAL)
    return {
        "available": True,
        "path": str(CREDIT_CARD_LOCAL),
        "shape": df.shape,
        "columns": list(df.columns),
        "target_distribution": df["Class"].value_counts().to_dict(),
        "fraud_percentage": df["Class"].mean() * 100,
    }


if __name__ == "__main__":
    # Download dataset
    path = download_credit_card_dataset()
    print(f"\nDataset info: {get_dataset_info()}")

    # Test loading
    X, y = load_real_dataset()
    print(f"\nLoaded data shape: X={X.shape}, y={y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
