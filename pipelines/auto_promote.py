# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Automatic canary promotion based on Prometheus metrics and MLflow model comparison."""
from __future__ import annotations

import argparse
import os
from typing import Any

import requests
from mlflow.tracking import MlflowClient

from pipelines.promote_canary import promote_canary_to_production
from training.src.utils import load_config

# =============================================================================
# Configuration
# =============================================================================
PROM_URL: str = os.getenv("PROM_URL", "http://prometheus:9090")
MLFLOW_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def query_prometheus(expr: str) -> float | None:
    """Query Prometheus for a metric value.

    Args:
        expr: PromQL expression.

    Returns:
        Metric value or None if query fails.
    """
    try:
        response = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": expr},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json().get("data", {}).get("result", [])
        if not data:
            return None
        return float(data[0]["value"][1])
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None


def evaluate_canary(
    threshold_auc_delta: float = -0.005,
    latency_ratio_max: float = 1.1,
    max_error_rate: float = 0.01,
    min_canary_rps: float = 0.02,
) -> bool:
    """Evaluate if canary model should be promoted.

    Checks:
    1. AUC improvement (canary vs production)
    2. Latency comparison (canary should not be significantly slower)
    3. Error rate within acceptable limits
    4. Sufficient canary traffic for statistical validity

    Args:
        threshold_auc_delta: Minimum AUC improvement required.
        latency_ratio_max: Maximum allowed latency ratio (canary/primary).
        max_error_rate: Maximum allowed error rate for canary.
        min_canary_rps: Minimum requests per second to canary.

    Returns:
        True if canary passes all criteria.
    """
    cfg = load_config()
    model_name = cfg["experiment"]["model_name"]
    client = MlflowClient(tracking_uri=MLFLOW_URI)

    # Get model versions
    auc_prod, auc_canary = _get_model_aucs(client, model_name)

    # AUC criterion
    auc_ok = True
    if auc_prod is not None and auc_canary is not None:
        auc_ok = (auc_canary - auc_prod) >= threshold_auc_delta

    # Latency and traffic metrics from Prometheus
    metrics = _get_traffic_metrics()

    # Traffic criterion
    traffic_ok = metrics["rps_canary"] >= min_canary_rps

    # Latency criterion
    latency_ok = True
    if metrics["p90_primary"] and metrics["p90_canary"]:
        latency_ok = metrics["p90_canary"] <= (metrics["p90_primary"] * latency_ratio_max)

    # Error rate criterion
    error_ok = (
        metrics["err_rate_canary"] <= max_error_rate
        and metrics["err_rate_canary"] <= metrics["err_rate_primary"] * 1.2
    )

    return bool(auc_ok and traffic_ok and latency_ok and error_ok)


def _get_model_aucs(client: MlflowClient, model_name: str) -> tuple[float | None, float | None]:
    """Get AUC metrics for production and staging models.

    Args:
        client: MLflow client.
        model_name: Name of the registered model.

    Returns:
        Tuple of (production AUC, staging AUC).
    """
    try:
        mv_prod = client.get_latest_versions(model_name, stages=["Production"]) or []
        mv_stg = client.get_latest_versions(model_name, stages=["Staging"]) or []
    except Exception:
        return None, None

    auc_prod = None
    if mv_prod:
        run_prod = client.get_run(mv_prod[0].run_id)
        auc_prod = run_prod.data.metrics.get("test_auc")

    auc_canary = None
    if mv_stg:
        run_stg = client.get_run(mv_stg[0].run_id)
        auc_canary = run_stg.data.metrics.get("test_auc")

    return auc_prod, auc_canary


def _get_traffic_metrics() -> dict[str, Any]:
    """Get traffic metrics from Prometheus.

    Returns:
        Dictionary with latency and error rate metrics.
    """
    p90_primary = query_prometheus(
        'histogram_quantile(0.9, sum(rate(fastapi_inference_latency_ms_bucket{route="primary"}[5m])) by (le, route))'
    )
    p90_canary = query_prometheus(
        'histogram_quantile(0.9, sum(rate(fastapi_inference_latency_ms_bucket{route="canary"}[5m])) by (le, route))'
    )

    rps_primary = query_prometheus(
        'sum(rate(fastapi_requests_total{route="primary",status="200"}[5m]))'
    ) or 0.0
    rps_canary = query_prometheus(
        'sum(rate(fastapi_requests_total{route="canary",status="200"}[5m]))'
    ) or 0.0

    errs_primary = query_prometheus(
        'sum(rate(fastapi_requests_total{route="primary",status!="200"}[5m]))'
    ) or 0.0
    errs_canary = query_prometheus(
        'sum(rate(fastapi_requests_total{route="canary",status!="200"}[5m]))'
    ) or 0.0

    err_rate_primary = errs_primary / max(rps_primary + errs_primary, 1e-9)
    err_rate_canary = errs_canary / max(rps_canary + errs_canary, 1e-9)

    return {
        "p90_primary": p90_primary,
        "p90_canary": p90_canary,
        "rps_primary": rps_primary,
        "rps_canary": rps_canary,
        "err_rate_primary": err_rate_primary,
        "err_rate_canary": err_rate_canary,
    }


def main(dry_run: bool = True) -> dict[str, Any]:
    """Main entry point for auto-promotion.

    Args:
        dry_run: If True, only evaluate without promoting.

    Returns:
        Dictionary with promotion decision and action taken.
    """
    decision = evaluate_canary()
    result = {"should_promote": decision, "applied": False}

    if decision and not dry_run:
        promote_canary_to_production()
        result["applied"] = True

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-promote canary based on metrics")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the promotion (default: dry-run)",
    )
    args = parser.parse_args()
    output = main(dry_run=not args.apply)
    print({"promote": output["should_promote"], "applied": output["applied"]})
