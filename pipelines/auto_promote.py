import os
import time
import math
import requests
from typing import Optional
from mlflow.tracking import MlflowClient

from pipelines.promote_canary import promote_canary_to_production
from training.src.utils import load_config

PROM_URL = os.getenv("PROM_URL", "http://prometheus:9090")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def q(expr: str) -> Optional[float]:
    r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": expr}, timeout=5)
    r.raise_for_status()
    data = r.json().get("data", {}).get("result", [])
    if not data:
        return None
    try:
        return float(data[0]["value"][1])
    except Exception:
        return None


def evaluate_canary(threshold_auc_delta: float = -0.005,
                    latency_ratio_max: float = 1.1,
                    max_error_rate: float = 0.01,
                    min_canary_rps: float = 0.02) -> bool:
    cfg = load_config()
    model_name = cfg["experiment"]["model_name"]
    client = MlflowClient(tracking_uri=MLFLOW_URI)

    try:
        mv_prod = client.get_latest_versions(model_name, stages=["Production"]) or []
        mv_stg = client.get_latest_versions(model_name, stages=["Staging"]) or []
    except Exception:
        mv_prod, mv_stg = [], []

    auc_prod = None
    if mv_prod:
        run_prod = client.get_run(mv_prod[0].run_id)
        auc_prod = run_prod.data.metrics.get("test_auc")
    auc_canary = None
    if mv_stg:
        run_stg = client.get_run(mv_stg[0].run_id)
        auc_canary = run_stg.data.metrics.get("test_auc")

    # AUC criterion
    auc_ok = True
    if auc_prod is not None and auc_canary is not None:
        auc_ok = (auc_canary - auc_prod) >= threshold_auc_delta

    # Prometheus latency and error criteria (5m window)
    p90_primary = q('histogram_quantile(0.9, sum(rate(fastapi_inference_latency_ms_bucket{route="primary"}[5m])) by (le, route))')
    p90_canary  = q('histogram_quantile(0.9, sum(rate(fastapi_inference_latency_ms_bucket{route="canary"}[5m])) by (le, route))')

    rps_primary = q('sum(rate(fastapi_requests_total{route="primary",status="200"}[5m]))') or 0.0
    rps_canary  = q('sum(rate(fastapi_requests_total{route="canary",status="200"}[5m]))') or 0.0

    errs_primary = q('sum(rate(fastapi_requests_total{route="primary",status!="200"}[5m]))') or 0.0
    errs_canary  = q('sum(rate(fastapi_requests_total{route="canary",status!="200"}[5m]))') or 0.0

    err_rate_primary = errs_primary / max(rps_primary + errs_primary, 1e-9)
    err_rate_canary  = errs_canary / max(rps_canary + errs_canary, 1e-9)

    traffic_ok = rps_canary >= min_canary_rps

    latency_ok = True if (p90_primary is None or p90_canary is None) else (p90_canary <= (p90_primary * latency_ratio_max))
    error_ok = (err_rate_canary <= max_error_rate) and (err_rate_canary <= err_rate_primary * 1.2)

    return bool(auc_ok and traffic_ok and latency_ok and error_ok)


def main(dry_run: bool = True) -> bool:
    decision = evaluate_canary()
    if decision and not dry_run:
        promote_canary_to_production()
    return decision


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    out = main(dry_run=(not args.apply))
    print({"promote": out, "applied": args.apply and out})
