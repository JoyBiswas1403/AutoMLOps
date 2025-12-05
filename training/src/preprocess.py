# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Preprocessing module for feature scaling and schema management."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import ARTIFACTS_DIR

# =============================================================================
# Directories
# =============================================================================
PROCESSED_DIR: Path = ARTIFACTS_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fit_and_transform(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
) -> dict[str, Any]:
    """Fit a scaler on training data and transform all datasets.

    This function:
    1. Fits a StandardScaler on the training features
    2. Transforms train, validation, and test sets
    3. Saves the scaler and feature schema for serving
    4. Exports training data to Parquet for Feast

    Args:
        train_csv: Path to training CSV file.
        val_csv: Path to validation CSV file.
        test_csv: Path to test CSV file.

    Returns:
        Dictionary containing paths to scaler, schema, and processed files.
    """
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)

    # Add row_id for Feast offline store
    for df in (train, val, test):
        if "row_id" not in df.columns:
            df.insert(0, "row_id", range(len(df)))

    feature_cols = [c for c in train.columns if c not in ("target", "row_id")]
    n_features = len(feature_cols)

    print(f"Processing {n_features} features: {feature_cols[:5]}...{feature_cols[-3:]}")

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_val = scaler.transform(val[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    # Save processed datasets
    _save_processed_dataset(X_train, feature_cols, train, PROCESSED_DIR / "train.csv")
    _save_processed_dataset(X_val, feature_cols, val, PROCESSED_DIR / "val.csv")
    _save_processed_dataset(X_test, feature_cols, test, PROCESSED_DIR / "test.csv")

    # Save scaler for serving
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    # Save feature schema
    schema_path = ARTIFACTS_DIR / "schema.json"
    schema = {
        "feature_order": feature_cols,
        "n_features": n_features,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    schema_path.write_text(json.dumps(schema, indent=2))
    print(f"Saved schema to {schema_path}")

    # Export to Feast offline store (Parquet)
    _export_to_feast(PROCESSED_DIR / "train.csv")

    return {
        "scaler_path": str(scaler_path),
        "schema_path": str(schema_path),
        "processed": {
            "train": str(PROCESSED_DIR / "train.csv"),
            "val": str(PROCESSED_DIR / "val.csv"),
            "test": str(PROCESSED_DIR / "test.csv"),
        },
        "n_features": n_features,
    }


def _save_processed_dataset(
    X_scaled: Any,
    feature_cols: list[str],
    original_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save scaled features with metadata to CSV.

    Args:
        X_scaled: Scaled feature array.
        feature_cols: List of feature column names.
        original_df: Original DataFrame with row_id and target.
        output_path: Path for output CSV.
    """
    df = pd.DataFrame(X_scaled, columns=feature_cols)
    df["row_id"] = original_df["row_id"].values
    df["target"] = original_df["target"].values
    df.to_csv(output_path, index=False)


def _export_to_feast(processed_train_path: Path) -> None:
    """Export processed training data to Feast offline store.

    Args:
        processed_train_path: Path to processed training CSV.
    """
    try:
        feast_offline = ARTIFACTS_DIR.parent / "feature_store" / "feast_feature_repo" / "data" / "offline"
        feast_offline.mkdir(parents=True, exist_ok=True)
        parquet_path = feast_offline / "train.parquet"
        pd.read_csv(processed_train_path).to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"Warning: Feast export failed: {e}")


if __name__ == "__main__":
    raise SystemExit("Run via training/src/train.py")
