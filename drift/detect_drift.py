# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Data drift detection using KS-test, PSI, Evidently, and whylogs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from scipy.stats import ks_2samp

try:
    import whylogs as why
except ImportError:
    why = None  # type: ignore

try:
    from evidently.ui.workspace import Workspace
except ImportError:
    Workspace = None  # type: ignore

from pipelines.notify import send as notify
from training.src import data_ingestion, preprocess
from training.src.train import train_and_deploy
from training.src.utils import ARTIFACTS_DIR, load_config

# =============================================================================
# Directories
# =============================================================================
PROCESSED_DIR: Path = ARTIFACTS_DIR / "processed"
RAW_DIR: Path = ARTIFACTS_DIR / "raw"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) for a single feature.

    PSI measures the shift in distribution between two datasets.
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Slight change
    - PSI >= 0.2: Significant change

    Args:
        expected: Reference distribution values.
        actual: Current distribution values.
        bins: Number of bins for histogram.

    Returns:
        PSI score (0 indicates identical distributions).
    """
    quantiles = np.linspace(0, 100, bins + 1)
    cuts = np.unique(np.percentile(expected, quantiles))

    # Avoid identical bins
    if len(cuts) < 3:
        return 0.0

    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)

    e_perc = e_counts / max(e_counts.sum(), 1)
    a_perc = a_counts / max(a_counts.sum(), 1)

    # Avoid division by zero
    e_perc = np.clip(e_perc, 1e-6, None)
    a_perc = np.clip(a_perc, 1e-6, None)

    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def simulate_new_batch(reference_csv: Path, shift: float = 0.5) -> pd.DataFrame:
    """Generate a simulated drifted batch for testing.

    Adds a constant shift to alternating features to simulate
    concept drift in production data.

    Args:
        reference_csv: Path to reference dataset.
        shift: Amount to shift feature values.

    Returns:
        DataFrame with simulated drift applied.
    """
    ref = pd.read_csv(reference_csv)
    feature_cols = [c for c in ref.columns if c != "target"]
    new = ref.copy()

    # Add shift to alternating features
    for i, col in enumerate(feature_cols):
        if i % 2 == 0:
            new[col] = new[col] + shift

    return new


def compute_drift_metrics(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Compute drift metrics for all features.

    Args:
        reference: Reference dataset.
        current: Current dataset to compare.
        feature_cols: List of feature column names.

    Returns:
        Dictionary with KS and PSI statistics per feature and means.
    """
    ks_stats: list[float] = []
    psi_scores: list[float] = []

    for col in feature_cols:
        ks_stat = ks_2samp(reference[col], current[col]).statistic
        ks_stats.append(float(ks_stat))
        psi_scores.append(psi(reference[col].values, current[col].values))

    return {
        "ks_mean": float(np.mean(ks_stats)),
        "psi_mean": float(np.mean(psi_scores)),
        "ks_per_feature": ks_stats,
        "psi_per_feature": psi_scores,
    }


def generate_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> Path | None:
    """Generate an Evidently drift report.

    Args:
        reference: Reference dataset.
        current: Current dataset.

    Returns:
        Path to HTML report or None if generation failed.
    """
    try:
        ref_df = reference.drop(columns=["target"], errors="ignore")
        cur_df = current.drop(columns=["target"], errors="ignore")

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)

        report_path = REPORTS_DIR / "drift_report.html"
        report.save_html(str(report_path))
        return report_path
    except Exception as e:
        print(f"Warning: Evidently report generation failed: {e}")
        return None


def generate_whylogs_profile(data: pd.DataFrame) -> Path | None:
    """Generate a whylogs profile for the dataset.

    Args:
        data: Dataset to profile.

    Returns:
        Path to binary profile or None if generation failed.
    """
    if why is None:
        return None

    try:
        profile = why.log(pandas=data.drop(columns=["target"], errors="ignore"))
        profile_path = REPORTS_DIR / "batch_profile.bin"
        profile.writer("binary").option("path", str(profile_path)).write()
        return profile_path
    except Exception as e:
        print(f"Warning: whylogs profile generation failed: {e}")
        return None


def push_metrics_to_prometheus(ks_mean: float, psi_mean: float) -> None:
    """Push drift metrics to Prometheus Pushgateway.

    Args:
        ks_mean: Mean KS statistic.
        psi_mean: Mean PSI score.
    """
    try:
        registry = CollectorRegistry()
        g_ks = Gauge("drift_ks_mean", "Mean KS drift score", registry=registry)
        g_psi = Gauge("drift_psi_mean", "Mean PSI drift score", registry=registry)
        g_ks.set(ks_mean)
        g_psi.set(psi_mean)
        push_to_gateway("pushgateway:9091", job="drift", registry=registry)
    except Exception as e:
        print(f"Warning: Failed to push to Prometheus: {e}")


def save_to_evidently_workspace(report_path: Path | None) -> None:
    """Save report to Evidently UI workspace.

    Args:
        report_path: Path to the HTML report.
    """
    if Workspace is None or report_path is None:
        return

    try:
        ws_path = os.getenv(
            "EVIDENTLY_WORKSPACE",
            str(ARTIFACTS_DIR.parent / "evidently_workspace"),
        )
        if os.path.exists(ws_path):
            ws = Workspace(ws_path)
        else:
            ws = Workspace.create(ws_path)

        project = ws.get_or_create_project("drift-monitoring")
        if os.path.exists(report_path):
            project.add_report(str(report_path))
        ws.save()
    except Exception as e:
        print(f"Warning: Failed to save to Evidently workspace: {e}")


def detect_and_optionally_retrain(simulate: bool = False) -> dict[str, Any]:
    """Main drift detection function with optional retraining.

    This function:
    1. Loads reference and current data
    2. Computes drift metrics (KS-test, PSI)
    3. Generates Evidently and whylogs reports
    4. Logs results to MLflow
    5. Triggers retraining if drift exceeds thresholds

    Args:
        simulate: If True, generate a simulated drifted batch.

    Returns:
        Dictionary with drift metrics and retraining status.
    """
    cfg = load_config()
    mon = cfg["monitoring"]

    # Ensure reference data exists
    if not (RAW_DIR / "train.csv").exists():
        data_ingestion.generate_and_split()
        preprocess.fit_and_transform(
            RAW_DIR / "train.csv",
            RAW_DIR / "val.csv",
            RAW_DIR / "test.csv",
        )

    ref = pd.read_csv(PROCESSED_DIR / "train.csv")
    feature_cols = [c for c in ref.columns if c not in ("target", "row_id")]

    # Get current batch
    if simulate:
        sim_raw = simulate_new_batch(RAW_DIR / "train.csv", shift=0.7)
        sim_raw.to_csv(RAW_DIR / "new_batch.csv", index=False)
        scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
        X_batch = scaler.transform(sim_raw[feature_cols])
        batch = pd.DataFrame(X_batch, columns=feature_cols)
        batch["target"] = sim_raw["target"].values
    else:
        batch_path = RAW_DIR / "new_batch.csv"
        if not batch_path.exists():
            raise FileNotFoundError(
                "No new batch found. Use --simulate or create artifacts/raw/new_batch.csv"
            )
        scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
        raw_batch = pd.read_csv(batch_path)
        X_batch = scaler.transform(raw_batch[feature_cols])
        batch = pd.DataFrame(X_batch, columns=feature_cols)
        batch["target"] = raw_batch["target"].values

    # Compute drift metrics
    result = compute_drift_metrics(ref, batch, feature_cols)

    # Generate reports
    report_path = generate_evidently_report(ref, batch)
    why_path = generate_whylogs_profile(batch)

    # Log to MLflow
    mlflow.set_experiment(cfg["experiment"]["name"])
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_metrics({
            "ks_mean": result["ks_mean"],
            "psi_mean": result["psi_mean"],
        })
        if report_path:
            mlflow.log_artifact(str(report_path), artifact_path="drift")
        if why_path:
            mlflow.log_artifact(str(why_path), artifact_path="drift")

        drift_detected = (
            result["ks_mean"] >= mon["drift_threshold_ks"]
            or result["psi_mean"] >= mon["drift_threshold_psi"]
        )
        mlflow.set_tag("drift_detected", str(drift_detected))

    # Push to Prometheus
    push_metrics_to_prometheus(result["ks_mean"], result["psi_mean"])

    # Save to Evidently workspace
    save_to_evidently_workspace(report_path)

    # Trigger retraining if drift detected
    if drift_detected:
        print("Drift detected. Triggering retraining...")
        try:
            notify(
                "Drift detected",
                "Retraining triggered due to drift",
                {"ks_mean": result["ks_mean"], "psi_mean": result["psi_mean"]},
            )
        except Exception as e:
            print(f"Warning: Notification failed: {e}")
        train_and_deploy()
        result["retrained"] = True
    else:
        result["retrained"] = False

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect data drift and optionally retrain")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate a drifted batch for testing",
    )
    args = parser.parse_args()
    output = detect_and_optionally_retrain(simulate=args.simulate)
    print(output)
