import os
from pipelines.rollback import rollback_to_previous
import requests

PROM_URL = os.getenv("PROM_URL", "http://prometheus:9090")


def q(expr: str):
    r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": expr}, timeout=5)
    r.raise_for_status()
    data = r.json().get("data", {}).get("result", [])
    if not data:
        return None
    return float(data[0]["value"][1])


def should_rollback(latency_ratio_max: float = 1.3, error_rate_max: float = 0.05) -> bool:
    p90_primary = q('histogram_quantile(0.9, sum(rate(fastapi_inference_latency_ms_bucket{route="primary"}[5m])) by (le, route))')
    p90_canary  = q('histogram_quantile(0.9, sum(rate(fastapi_inference_latency_ms_bucket{route="canary"}[5m])) by (le, route))')

    rps_primary = q('sum(rate(fastapi_requests_total{route="primary",status="200"}[5m]))') or 0.0
    errs_primary = q('sum(rate(fastapi_requests_total{route="primary",status!="200"}[5m]))') or 0.0
    err_rate_primary = errs_primary / max(rps_primary + errs_primary, 1e-9)

    latency_bad = False if (p90_primary is None or p90_canary is None) else (p90_primary > p90_canary * latency_ratio_max)
    error_bad = err_rate_primary > error_rate_max
    return bool(latency_bad or error_bad)


def main(apply: bool = True) -> bool:
    if should_rollback():
        if apply:
            rollback_to_previous()
        return True
    return False


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    out = main(apply=args.apply)
    print({"rolled_back": out, "applied": args.apply and out})
