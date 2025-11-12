from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
import mlflow
from training.src.utils import load_config, ARTIFACTS_DIR
from training.src import data_ingestion, preprocess
from training.src.train import train_and_deploy
from pipelines.notify import send as notify

# Evidently/whylogs
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import whylogs as why

PROCESSED_DIR = ARTIFACTS_DIR / "processed"
RAW_DIR = ARTIFACTS_DIR / "raw"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    # Calculate Population Stability Index for a single feature
    quantiles = np.linspace(0, 100, bins + 1)
    cuts = np.unique(np.percentile(expected, quantiles))
    # Avoid identical bins
    if len(cuts) < 3:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e_perc = e_counts / max(e_counts.sum(), 1)
    a_perc = a_counts / max(a_counts.sum(), 1)
    e_perc = np.clip(e_perc, 1e-6, None)
    a_perc = np.clip(a_perc, 1e-6, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def simulate_new_batch(reference_csv: Path, shift: float = 0.5) -> pd.DataFrame:
    ref = pd.read_csv(reference_csv)
    feature_cols = [c for c in ref.columns if c != "target"]
    new = ref.copy()
    # Add shift to half of features to simulate drift
    for i, c in enumerate(feature_cols):
        if i % 2 == 0:
            new[c] = new[c] + shift
    return new


def detect_and_optionally_retrain(simulate: bool = False) -> dict:
    cfg = load_config()
    mon = cfg["monitoring"]

    # Ensure we have reference data
    if not (RAW_DIR / "train.csv").exists():
        data_ingestion.generate_and_split()
        preprocess.fit_and_transform(RAW_DIR / "train.csv", RAW_DIR / "val.csv", RAW_DIR / "test.csv")

    ref = pd.read_csv(PROCESSED_DIR / "train.csv")
    feature_cols = [c for c in ref.columns if c != "target"]

    if simulate:
        # Build a drifted batch and save
        sim_raw = simulate_new_batch(RAW_DIR / "train.csv", shift=0.7)
        sim_raw.to_csv(RAW_DIR / "new_batch.csv", index=False)
        # Process it using the existing scaler
        from sklearn.preprocessing import StandardScaler
        import joblib
        scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
        Xb = scaler.transform(sim_raw[feature_cols])
        batch = pd.DataFrame(Xb, columns=feature_cols).assign(target=sim_raw["target"])
    else:
        # If a batch is present
        batch_path = RAW_DIR / "new_batch.csv"
        if not batch_path.exists():
            raise FileNotFoundError("No new batch found. Provide --simulate or create artifacts/raw/new_batch.csv")
        # Process with existing scaler
        import joblib
        scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
        raw_batch = pd.read_csv(batch_path)
        Xb = scaler.transform(raw_batch[feature_cols])
        batch = pd.DataFrame(Xb, columns=feature_cols).assign(target=raw_batch["target"])

    # Compute drift metrics per feature
    ks_stats = []
    psis = []
    for c in feature_cols:
        ks = ks_2samp(ref[c], batch[c]).statistic
        ks_stats.append(ks)
        psis.append(psi(ref[c].values, batch[c].values))

    ks_mean = float(np.mean(ks_stats))
    psi_mean = float(np.mean(psis))

    result = {"ks_mean": ks_mean, "psi_mean": psi_mean, "ks_per_feature": ks_stats, "psi_per_feature": psis}

    # Generate Evidently drift report (HTML)
    try:
        ref_df = ref.drop(columns=["target"]) if "target" in ref.columns else ref
        cur_df = batch.drop(columns=["target"]) if "target" in batch.columns else batch
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
        report_path = REPORTS_DIR / "drift_report.html"
        report.save_html(str(report_path))
    except Exception as e:
        report_path = None

    # Generate whylogs profile
    try:
        profile = why.log(pandas=batch.drop(columns=["target"]))
        why_path = REPORTS_DIR / "batch_profile.bin"
        profile.writer("binary").option("path", str(why_path)).write()
    except Exception:
        why_path = None

    # Log to MLflow
    mlflow.set_experiment(cfg["experiment"]["name"])
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_metrics({"ks_mean": ks_mean, "psi_mean": psi_mean})
        if report_path:
            mlflow.log_artifact(str(report_path), artifact_path="drift")
        if why_path:
            mlflow.log_artifact(str(why_path), artifact_path="drift")
        drift_flag = (ks_mean >= mon["drift_threshold_ks"]) or (psi_mean >= mon["drift_threshold_psi"])
        mlflow.set_tag("drift_detected", str(drift_flag))

    if drift_flag:
        print("Drift detected. Triggering retraining...")
        notify("Drift detected", "Retraining triggered due to drift", {"ks_mean": ks_mean, "psi_mean": psi_mean})
        train_and_deploy()
        result["retrained"] = True
    else:
        result["retrained"] = False

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true", help="Simulate a drifted batch")
    args = parser.parse_args()
    out = detect_and_optionally_retrain(simulate=args.simulate)
    print(out)
