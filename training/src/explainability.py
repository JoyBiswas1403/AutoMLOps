# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Model explainability module using SHAP."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ARTIFACTS_DIR

# =============================================================================
# Directories
# =============================================================================
EXPLAINABILITY_DIR: Path = ARTIFACTS_DIR / "explainability"
EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)


def compute_shap_values(
    model: Any,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    max_samples: int = 100,
) -> dict[str, Any]:
    """Compute SHAP values for model explanations.

    Uses DeepExplainer for neural networks (TensorFlow/Keras models).

    Args:
        model: Trained Keras/TensorFlow model.
        X_train: Training data for background (subset used).
        X_explain: Data to explain (typically test set).
        feature_names: List of feature names.
        max_samples: Maximum samples to use for explanation.

    Returns:
        Dictionary containing SHAP values and feature importance.
    """
    try:
        import shap
    except ImportError:
        print("Warning: SHAP not installed. Run: pip install shap")
        return {}

    print(f"Computing SHAP values for {min(len(X_explain), max_samples)} samples...")

    # Limit samples for efficiency
    background = X_train[:min(100, len(X_train))]
    X_sample = X_explain[:min(max_samples, len(X_explain))]

    try:
        # Use DeepExplainer for neural networks
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_sample)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Compute feature importance (mean absolute SHAP value)
        feature_importance = np.abs(shap_values).mean(axis=0)

        # Handle 2D case (model output dimension)
        if len(feature_importance.shape) > 1:
            feature_importance = feature_importance.flatten()

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importance,
        }).sort_values("importance", ascending=False)

        print(f"Top 5 features: {importance_df.head()['feature'].tolist()}")

        return {
            "shap_values": shap_values,
            "X_sample": X_sample,
            "feature_names": feature_names,
            "feature_importance": importance_df,
            "explainer": explainer,
        }

    except Exception as e:
        print(f"Warning: SHAP computation failed: {e}")
        # Fallback: use gradient-based importance
        return _compute_gradient_importance(model, X_sample, feature_names)


def _compute_gradient_importance(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Fallback: compute feature importance using gradients.

    Args:
        model: Keras model.
        X: Input samples.
        feature_names: Feature names.

    Returns:
        Dictionary with feature importance.
    """
    try:
        import tensorflow as tf

        X_tensor = tf.constant(X, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = model(X_tensor)

        gradients = tape.gradient(predictions, X_tensor)
        importance = np.abs(gradients.numpy()).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return {
            "feature_importance": importance_df,
            "method": "gradient",
        }
    except Exception as e:
        print(f"Warning: Gradient importance failed: {e}")
        return {}


def generate_shap_plots(
    shap_result: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate and save SHAP visualization plots.

    Args:
        shap_result: Result from compute_shap_values.
        output_dir: Directory to save plots.

    Returns:
        Dictionary mapping plot names to file paths.
    """
    if not shap_result:
        return {}

    try:
        import shap
    except ImportError:
        return {}

    output_dir = output_dir or EXPLAINABILITY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}

    # 1. Feature Importance Bar Plot
    try:
        importance_df = shap_result.get("feature_importance")
        if importance_df is not None:
            plt.figure(figsize=(10, 8))
            top_n = min(15, len(importance_df))
            top_features = importance_df.head(top_n)

            plt.barh(
                range(top_n),
                top_features["importance"].values[::-1],
                color="steelblue",
            )
            plt.yticks(range(top_n), top_features["feature"].values[::-1])
            plt.xlabel("Mean |SHAP Value|")
            plt.title("Feature Importance (SHAP)")
            plt.tight_layout()

            path = output_dir / "feature_importance.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            plots["feature_importance"] = path
            print(f"Saved: {path}")
    except Exception as e:
        print(f"Warning: Feature importance plot failed: {e}")

    # 2. SHAP Summary Plot (Beeswarm)
    if "shap_values" in shap_result:
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_result["shap_values"],
                shap_result["X_sample"],
                feature_names=shap_result["feature_names"],
                show=False,
                max_display=15,
            )
            plt.tight_layout()

            path = output_dir / "shap_summary.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            plots["shap_summary"] = path
            print(f"Saved: {path}")
        except Exception as e:
            print(f"Warning: Summary plot failed: {e}")

    # 3. Save feature importance as CSV
    try:
        importance_df = shap_result.get("feature_importance")
        if importance_df is not None:
            csv_path = output_dir / "feature_importance.csv"
            importance_df.to_csv(csv_path, index=False)
            plots["feature_importance_csv"] = csv_path
    except Exception as e:
        print(f"Warning: CSV export failed: {e}")

    return plots


def generate_explainability_report(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Generate complete explainability report.

    This is the main entry point for model explainability.

    Args:
        model: Trained model.
        X_train: Training data.
        X_test: Test data to explain.
        feature_names: Feature names.

    Returns:
        Dictionary with SHAP results and plot paths.
    """
    print("\n" + "="*60)
    print("Generating Model Explainability Report")
    print("="*60 + "\n")

    # Compute SHAP values
    shap_result = compute_shap_values(
        model=model,
        X_train=X_train,
        X_explain=X_test,
        feature_names=feature_names,
    )

    # Generate plots
    plots = generate_shap_plots(shap_result)

    # Get top features for model card
    top_features = []
    if "feature_importance" in shap_result:
        top_features = shap_result["feature_importance"].head(10)["feature"].tolist()

    return {
        "shap_result": shap_result,
        "plots": plots,
        "top_features": top_features,
        "explainability_dir": str(EXPLAINABILITY_DIR),
    }


if __name__ == "__main__":
    print("Explainability module loaded. Run via training pipeline.")
