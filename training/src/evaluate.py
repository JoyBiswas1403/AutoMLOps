from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from pathlib import Path


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {"auc": auc, "acc": acc, "precision": float(p), "recall": float(r), "f1": float(f1)}


if __name__ == "__main__":
    raise SystemExit("Run within training workflow")
