# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Automatic rollback based on Prometheus metrics."""
from __future__ import annotations

import argparse
import os
from typing import Any

import requests

from pipelines.rollback import rollback_to_previous

# =============================================================================
# Configuration
# =============================================================================
PROM_URL: str = os.getenv("PROM_URL", "http://prometheus:9090")


def query_prometheus(expr: str) -> float | None:
    """Query Prometheus for a metric value.

    Args:
        expr: PromQL expression.

    Returns:
        Metric value or None if query fails or returns no data.
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


def should_rollback(
    latency_ratio_max: float = 1.3,
    error_rate_max: float = 0.05,
) -> bool:
    """Determine if a rollback should be triggered.

    Checks:
    1. If primary latency is significantly higher than canary
    2. If primary error rate exceeds threshold

    Args:
        latency_ratio_max: Maximum allowed latency ratio (primary/canary).
        error_rate_max: Maximum allowed error rate.

    Returns:
        True if rollback should be triggered.
    """
    # Query latency metrics
    p90_primary = query_prometheus(
        'histogram_quantile(0.9, sum(rate(fastapi_inference_latency_ms_bucket{route="primary"}[5m])) by (le, route))'
    )
    p90_canary = query_prometheus(
        'histogram_quantile(0.9, sum(rate(fastapi_inference_latency_ms_bucket{route="canary"}[5m])) by (le, route))'
    )

    # Query traffic metrics
    rps_primary = query_prometheus(
        'sum(rate(fastapi_requests_total{route="primary",status="200"}[5m]))'
    ) or 0.0
    errs_primary = query_prometheus(
        'sum(rate(fastapi_requests_total{route="primary",status!="200"}[5m]))'
    ) or 0.0

    # Calculate error rate
    total_primary = rps_primary + errs_primary
    err_rate_primary = errs_primary / max(total_primary, 1e-9)

    # Check conditions
    latency_bad = False
    if p90_primary is not None and p90_canary is not None and p90_canary > 0:
        latency_bad = p90_primary > (p90_canary * latency_ratio_max)

    error_bad = err_rate_primary > error_rate_max

    return bool(latency_bad or error_bad)


def main(apply: bool = True) -> dict[str, Any]:
    """Main entry point for auto-rollback.

    Args:
        apply: If True, perform the rollback. If False, dry-run only.

    Returns:
        Dictionary with rollback decision and action taken.
    """
    should_roll = should_rollback()
    result = {"should_rollback": should_roll, "applied": False}

    if should_roll and apply:
        rollback_to_previous()
        result["applied"] = True

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-rollback based on metrics")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the rollback (default: dry-run)",
    )
    args = parser.parse_args()
    output = main(apply=args.apply)
    print(output)
