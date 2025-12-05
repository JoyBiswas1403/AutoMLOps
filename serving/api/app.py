# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""FastAPI inference service with canary routing and Prometheus metrics."""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

# =============================================================================
# Configuration
# =============================================================================
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
TF_SERVING_URL_PRIMARY: str = os.getenv(
    "TF_SERVING_URL_PRIMARY", "http://localhost:8501/v1/models/model:predict"
)
TF_SERVING_URL_CANARY: str | None = os.getenv("TF_SERVING_URL_CANARY")
CANARY_ENABLED: bool = os.getenv("CANARY_ENABLED", "false").lower() == "true"
CANARY_PERCENT: int = int(os.getenv("CANARY_PERCENT", "0"))
SCALER_PATH: str = os.getenv("SCALER_PATH", "/app/artifacts/scaler.joblib")
SCHEMA_PATH: str = os.getenv("SCHEMA_PATH", "/app/artifacts/schema.json")
MODEL_NAME: str = os.getenv("MODEL_NAME", "model")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# =============================================================================
# Prometheus Metrics
# =============================================================================
REQ_COUNTER = Counter(
    "fastapi_requests_total",
    "Total requests",
    ["route", "status"],
)
LAT_HIST = Histogram(
    "fastapi_inference_latency_ms",
    "Inference latency (ms)",
    ["route"],
    buckets=(5, 10, 25, 50, 100, 200, 500, 1000),
)
G_MODEL_AUC = Gauge("model_test_auc", "Model test AUC by stage", ["stage"])
G_MODEL_ACC = Gauge("model_test_acc", "Model test ACC by stage", ["stage"])
G_MODEL_VERSION = Gauge("model_version", "Model version served by route", ["route"])
G_CANARY_PERCENT = Gauge("canary_traffic_percent", "Configured canary traffic percent")

# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="AutoMLOps Inference API",
    description="ML inference service with canary routing and observability",
    version="1.2",
)


class PredictRequest(BaseModel):
    """Request body for prediction endpoint."""

    instances: list[list[float]]


class PredictResponse(BaseModel):
    """Response body for prediction endpoint."""

    route: str
    predictions: list[Any]
    latency_ms: float


@app.post("/traffic")
async def set_canary_percent(percent: int) -> dict[str, int]:
    """Set the canary traffic percentage.

    Args:
        percent: Percentage of traffic to route to canary (0-100).

    Returns:
        Current canary percentage configuration.

    Raises:
        HTTPException: If percent is not between 0 and 100.
    """
    global CANARY_PERCENT

    if percent < 0 or percent > 100:
        raise HTTPException(status_code=400, detail="percent must be 0..100")

    CANARY_PERCENT = int(percent)
    try:
        G_CANARY_PERCENT.set(CANARY_PERCENT)
    except Exception:
        pass

    return {"canary_percent": CANARY_PERCENT}


def load_scaler_and_schema() -> tuple[Any, list[str]]:
    """Load the scaler and feature schema from artifacts.

    Returns:
        Tuple of (scaler, feature_order list).

    Raises:
        RuntimeError: If loading fails.
    """
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler at {SCALER_PATH}: {e}") from e

    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        feature_order = schema["feature_order"]
    except Exception as e:
        raise RuntimeError(f"Failed to load schema at {SCHEMA_PATH}: {e}") from e

    return scaler, feature_order


def _check_tf_serving_status(name: str, url: str | None) -> int | None:
    """Check TensorFlow Serving endpoint status.

    Args:
        name: Name of the endpoint (primary/canary).
        url: TF Serving URL.

    Returns:
        HTTP status code or None if not configured.
    """
    if not url:
        return None

    try:
        base = url.replace(":predict", "")
        response = requests.get(base, timeout=2)
        status_code = response.status_code

        # Try to get version info
        try:
            info = response.json()
            version = info.get("model_version_status", [{}])[0].get("version")
            if version:
                G_MODEL_VERSION.labels(route=name).set(float(version))
        except (ValueError, KeyError, IndexError):
            pass

        return status_code
    except requests.RequestException:
        return 0


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint with TensorFlow Serving status.

    Returns:
        Health status including TF Serving endpoint statuses.
    """
    statuses: dict[str, int | None] = {}

    endpoints = {
        "primary": TF_SERVING_URL_PRIMARY,
        "canary": TF_SERVING_URL_CANARY,
    }

    for name, url in endpoints.items():
        statuses[name] = _check_tf_serving_status(name, url)

    # Update canary percent gauge
    try:
        G_CANARY_PERCENT.set(CANARY_PERCENT)
    except Exception:
        pass

    return {"status": "ok", "tf_serving": statuses}


def _refresh_mlflow_metrics() -> None:
    """Refresh MLflow model metrics in Prometheus gauges."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        # Production metrics
        prod = client.get_latest_versions(MODEL_NAME, stages=["Production"]) or []
        if prod:
            run = client.get_run(prod[0].run_id)
            auc = run.data.metrics.get("test_auc")
            acc = run.data.metrics.get("test_acc")
            if auc is not None:
                G_MODEL_AUC.labels(stage="Production").set(auc)
            if acc is not None:
                G_MODEL_ACC.labels(stage="Production").set(acc)

        # Staging metrics
        stg = client.get_latest_versions(MODEL_NAME, stages=["Staging"]) or []
        if stg:
            run = client.get_run(stg[0].run_id)
            auc = run.data.metrics.get("test_auc")
            acc = run.data.metrics.get("test_acc")
            if auc is not None:
                G_MODEL_AUC.labels(stage="Staging").set(auc)
            if acc is not None:
                G_MODEL_ACC.labels(stage="Staging").set(acc)
    except Exception:
        pass


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint.

    Returns:
        Prometheus metrics in text format.
    """
    _refresh_mlflow_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _select_route() -> tuple[str, str]:
    """Select routing (primary or canary) based on traffic split.

    Returns:
        Tuple of (route_name, serving_url).
    """
    if CANARY_ENABLED and TF_SERVING_URL_CANARY:
        if random.randint(1, 100) <= CANARY_PERCENT:
            return "canary", TF_SERVING_URL_CANARY

    return "primary", TF_SERVING_URL_PRIMARY


def _log_inference_to_mlflow(latency_ms: float, batch_size: int, route: str) -> None:
    """Log inference metrics to MLflow.

    Args:
        latency_ms: Inference latency in milliseconds.
        batch_size: Number of instances in the batch.
        route: Selected route (primary/canary).
    """
    try:
        with mlflow.start_run(run_name="inference", nested=True):
            mlflow.log_metric("inference_latency_ms", latency_ms)
            mlflow.log_param("batch_size", batch_size)
            mlflow.set_tag("route", route)
    except Exception:
        pass


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    """Make predictions using the deployed model.

    Handles input validation, scaling, routing (primary/canary),
    and metrics collection.

    Args:
        req: Prediction request with instances.

    Returns:
        Predictions with routing and latency information.

    Raises:
        HTTPException: If input validation fails or TF Serving errors.
    """
    scaler, feature_order = load_scaler_and_schema()

    # Validate input shape
    X = np.array(req.instances, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(feature_order):
        raise HTTPException(
            status_code=400,
            detail=f"Each instance must have {len(feature_order)} features",
        )

    # Validate input values
    try:
        from training.src.validation import validate_inference_input

        validation = validate_inference_input(req.instances, n_features=len(feature_order))
        if not validation["valid"]:
            raise HTTPException(
                status_code=422,
                detail=f"Input validation failed: {validation['errors']}",
            )
    except ImportError:
        # Fallback validation
        if not np.isfinite(X).all():
            raise HTTPException(
                status_code=422,
                detail="Input contains non-finite values (NaN or Inf)",
            )

    # Scale features
    X_scaled = scaler.transform(pd.DataFrame(X, columns=feature_order))
    payload = {"instances": X_scaled.tolist()}

    # Select route
    route, url = _select_route()

    # Make prediction request
    with LAT_HIST.labels(route=route).time():
        t0 = time.perf_counter()
        response = requests.post(url, json=payload, timeout=5)
        latency_ms = (time.perf_counter() - t0) * 1000

    if not response.ok:
        REQ_COUNTER.labels(route=route, status="502").inc()
        raise HTTPException(
            status_code=502,
            detail=f"TF Serving error ({route}): {response.text}",
        )

    REQ_COUNTER.labels(route=route, status="200").inc()
    predictions = response.json().get("predictions")

    # Log to MLflow
    _log_inference_to_mlflow(latency_ms, len(req.instances), route)

    return PredictResponse(
        route=route,
        predictions=predictions,
        latency_ms=latency_ms,
    )
