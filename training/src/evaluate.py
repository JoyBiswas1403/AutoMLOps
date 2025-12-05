# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Model evaluation utilities for computing classification metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Evaluate binary classification predictions.

    Computes AUC, accuracy, precision, recall, and F1 score.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dictionary containing evaluation metrics:
        - auc: Area Under ROC Curve
        - acc: Accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
    """
    y_pred = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return {
        "auc": auc,
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


if __name__ == "__main__":
    raise SystemExit("Run within training workflow")
