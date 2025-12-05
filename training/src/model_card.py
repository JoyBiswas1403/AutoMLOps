# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Model card generation for documentation and governance."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import ARTIFACTS_DIR


def generate_model_card(
    metrics: dict[str, float],
    params: dict[str, Any],
    data_source: str = "synthetic",
    n_features: int = 20,
    train_samples: int = 0,
    top_features: list[str] | None = None,
    notes: str = "",
) -> Path:
    """Generate a model card documenting model details and performance.

    Args:
        metrics: Dictionary of evaluation metrics.
        params: Dictionary of training parameters.
        data_source: Data source used ("synthetic" or "real").
        n_features: Number of input features.
        train_samples: Number of training samples.
        top_features: List of top important features from SHAP.
        notes: Optional additional notes.

    Returns:
        Path to the generated model card file.
    """
    report_dir = ARTIFACTS_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "model_card.md"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Dataset information
    if data_source == "real":
        dataset_info = """
## Dataset: Credit Card Fraud Detection

- **Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description:** European cardholder transactions, September 2013
- **Features:** 28 PCA-transformed features (V1-V28) + Amount
- **Class Distribution:** Highly imbalanced (~0.17% fraud)
"""
    else:
        dataset_info = """
## Dataset: Synthetic Classification Data

- **Source:** sklearn.datasets.make_classification
- **Purpose:** Pipeline testing and demonstration
"""

    content = f"""# Model Card - AutoMLOps Fraud Detection

**Generated:** {timestamp}
**Version:** 1.0
**Framework:** TensorFlow/Keras
**Data Source:** {data_source.title()}

## Overview

Binary classifier for fraud detection, trained as part of the AutoMLOps pipeline.
Uses a feedforward neural network architecture optimized for identifying fraudulent transactions.

{dataset_info}

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Features | {n_features} |
| Training Samples | {train_samples:,} |
"""
    for key, value in params.items():
        content += f"| {key} | {value} |\n"

    content += """
## Evaluation Metrics (Test Set)

| Metric | Value |
|--------|-------|
"""
    for key, value in metrics.items():
        content += f"| {key.replace('_', ' ').title()} | {value:.4f} |\n"

    # Add feature importance section if available
    if top_features:
        content += """
## Feature Importance (SHAP)

Top contributing features based on SHAP analysis:

| Rank | Feature |
|------|---------|
"""
        for i, feat in enumerate(top_features[:10], 1):
            content += f"| {i} | {feat} |\n"

    content += f"""
## Model Architecture

```
Input Layer ({n_features} features)
    ↓
Dense (128, ReLU) + Dropout(0.2)
    ↓
Dense (64, ReLU) + Dropout(0.2)
    ↓
Dense (32, ReLU) + Dropout(0.2)
    ↓
Output (1, Sigmoid) → Fraud Probability
```

## Intended Use

- **Primary:** Demonstration of MLOps best practices
- **Secondary:** Fraud detection in financial transactions

## Limitations

- PCA-transformed features limit interpretability
- Performance may vary on different time periods/regions
- Undersampling may affect generalization

## Ethical Considerations

- No demographic information available to assess bias
- Decisions should be reviewed by human analysts
- May produce false positives/negatives

## Monitoring

- **Data Drift:** PSI and KS-test metrics
- **Performance:** AUC and accuracy tracking
- **Retraining:** Automatic when drift exceeds thresholds
"""

    if notes:
        content += f"\n## Notes\n\n{notes}\n"

    path.write_text(content, encoding="utf-8")
    return path