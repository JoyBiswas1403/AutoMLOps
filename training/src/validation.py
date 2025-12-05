# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Data validation module using Pandera for schema enforcement."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    pa = None

from .utils import ARTIFACTS_DIR

# =============================================================================
# Validation Report Directory
# =============================================================================
VALIDATION_DIR: Path = ARTIFACTS_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


def create_feature_schema(
    n_features: int = 29,
    feature_prefix: str = "f",
    include_target: bool = True,
) -> "DataFrameSchema | None":
    """Create a Pandera schema for feature validation.

    Args:
        n_features: Number of features expected.
        feature_prefix: Prefix for feature column names.
        include_target: Whether to include target column.

    Returns:
        Pandera DataFrameSchema or None if Pandera not available.
    """
    if not PANDERA_AVAILABLE:
        print("Warning: Pandera not installed. Run: pip install pandera")
        return None

    columns = {}

    # Feature columns
    for i in range(n_features):
        col_name = f"{feature_prefix}{i}"
        columns[col_name] = Column(
            float,
            nullable=False,
            checks=[
                Check.not_equal_to(np.inf, error="Infinite value detected"),
                Check.not_equal_to(-np.inf, error="Negative infinite value detected"),
            ],
            coerce=True,
        )

    # Target column
    if include_target:
        columns["target"] = Column(
            int,
            nullable=False,
            checks=[
                Check.isin([0, 1], error="Target must be 0 or 1"),
            ],
            coerce=True,
        )

    return DataFrameSchema(
        columns=columns,
        strict=False,  # Allow extra columns
        coerce=True,
    )


def create_inference_schema(
    n_features: int = 29,
) -> "DataFrameSchema | None":
    """Create schema for inference input validation.

    Args:
        n_features: Number of features expected.

    Returns:
        Pandera DataFrameSchema for inference inputs.
    """
    if not PANDERA_AVAILABLE:
        return None

    columns = {}
    for i in range(n_features):
        columns[f"f{i}"] = Column(
            float,
            nullable=False,
            checks=[
                Check(lambda s: s.notna().all(), error="Missing values not allowed"),
                Check(lambda s: np.isfinite(s).all(), error="Non-finite values detected"),
            ],
            coerce=True,
        )

    return DataFrameSchema(columns=columns, strict=False, coerce=True)


def validate_dataframe(
    df: pd.DataFrame,
    schema: Any | None = None,
    n_features: int | None = None,
    include_target: bool = True,
) -> dict[str, Any]:
    """Validate a DataFrame against schema.

    Args:
        df: DataFrame to validate.
        schema: Optional pre-built schema.
        n_features: Number of features (auto-detected if None).
        include_target: Whether target column is expected.

    Returns:
        Validation result dictionary.
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Basic checks (always run, even without Pandera)
    result["stats"] = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Check for missing values
    missing_per_col = df.isna().sum()
    cols_with_missing = missing_per_col[missing_per_col > 0]
    if len(cols_with_missing) > 0:
        result["warnings"].append(f"Missing values in: {cols_with_missing.to_dict()}")

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not np.isfinite(df[col]).all():
            result["errors"].append(f"Infinite values in column: {col}")
            result["valid"] = False

    # Check target distribution if present
    if "target" in df.columns:
        target_dist = df["target"].value_counts(normalize=True).to_dict()
        result["stats"]["target_distribution"] = target_dist

        # Warn if highly imbalanced
        if min(target_dist.values()) < 0.05:
            result["warnings"].append("Target is highly imbalanced (<5% minority)")

    # Pandera schema validation
    if PANDERA_AVAILABLE:
        if schema is None:
            if n_features is None:
                n_features = len([c for c in df.columns if c not in ("target", "row_id")])
            schema = create_feature_schema(n_features, include_target=include_target)

        if schema is not None:
            try:
                schema.validate(df, lazy=True)
            except pa.errors.SchemaErrors as e:
                result["valid"] = False
                for err in e.failure_cases.to_dict("records"):
                    result["errors"].append(f"{err.get('column', 'unknown')}: {err.get('check', 'failed')}")

    return result


def validate_training_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Validate all training data splits.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.

    Returns:
        Combined validation results.
    """
    results = {
        "train": validate_dataframe(train_df, include_target=True),
        "val": validate_dataframe(val_df, include_target=True),
        "test": validate_dataframe(test_df, include_target=True),
    }

    # Overall validity
    results["all_valid"] = all(r["valid"] for r in results.values())

    # Save validation report
    report_path = save_validation_report(results)
    results["report_path"] = str(report_path)

    return results


def save_validation_report(results: dict[str, Any]) -> Path:
    """Save validation results to a markdown report.

    Args:
        results: Validation results dictionary.

    Returns:
        Path to the saved report.
    """
    report_path = VALIDATION_DIR / "validation_report.md"

    content = f"""# Data Validation Report

**Generated:** {datetime.utcnow().isoformat()}Z
**Status:** {"✅ PASSED" if results.get("all_valid", False) else "❌ FAILED"}

## Summary

| Split | Valid | Rows | Missing | Duplicates |
|-------|-------|------|---------|------------|
"""
    for split in ["train", "val", "test"]:
        if split in results:
            r = results[split]
            s = r.get("stats", {})
            status = "✅" if r["valid"] else "❌"
            content += f"| {split.title()} | {status} | {s.get('n_rows', 0):,} | {s.get('missing_values', 0)} | {s.get('duplicate_rows', 0)} |\n"

    # Errors section
    all_errors = []
    for split in ["train", "val", "test"]:
        if split in results:
            for err in results[split].get("errors", []):
                all_errors.append(f"- **{split}:** {err}")

    if all_errors:
        content += "\n## Errors\n\n" + "\n".join(all_errors)

    # Warnings section
    all_warnings = []
    for split in ["train", "val", "test"]:
        if split in results:
            for warn in results[split].get("warnings", []):
                all_warnings.append(f"- **{split}:** {warn}")

    if all_warnings:
        content += "\n\n## Warnings\n\n" + "\n".join(all_warnings)

    report_path.write_text(content, encoding="utf-8")
    return report_path


def validate_inference_input(
    instances: list[list[float]],
    n_features: int = 29,
) -> dict[str, Any]:
    """Validate inference API input.

    Args:
        instances: List of feature vectors.
        n_features: Expected number of features.

    Returns:
        Validation result with valid flag and errors.
    """
    result = {"valid": True, "errors": []}

    if not instances:
        result["valid"] = False
        result["errors"].append("Empty input: no instances provided")
        return result

    for i, instance in enumerate(instances):
        if len(instance) != n_features:
            result["valid"] = False
            result["errors"].append(
                f"Instance {i}: expected {n_features} features, got {len(instance)}"
            )

        if any(not np.isfinite(v) for v in instance):
            result["valid"] = False
            result["errors"].append(f"Instance {i}: contains non-finite values")

    return result


if __name__ == "__main__":
    # Test validation
    print(f"Pandera available: {PANDERA_AVAILABLE}")

    # Create sample data
    df = pd.DataFrame({
        "f0": [1.0, 2.0, 3.0],
        "f1": [4.0, 5.0, 6.0],
        "target": [0, 1, 0],
    })

    result = validate_dataframe(df, n_features=2)
    print(f"Validation result: {result}")
