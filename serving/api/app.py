import os
import time
import json
import random
from typing import List

import numpy as np
import pandas as pd
import requests
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
TF_SERVING_URL_PRIMARY = os.getenv("TF_SERVING_URL_PRIMARY", "http://localhost:8501/v1/models/model:predict")
TF_SERVING_URL_CANARY = os.getenv("TF_SERVING_URL_CANARY")
CANARY_ENABLED = os.getenv("CANARY_ENABLED", "false").lower() == "true"
CANARY_PERCENT = int(os.getenv("CANARY_PERCENT", "0"))
SCALER_PATH = os.getenv("SCALER_PATH", "/app/artifacts/scaler.joblib")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "/app/artifacts/schema.json")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Prometheus metrics
REQ_COUNTER = Counter("fastapi_requests_total", "Total requests", ["route", "status"])
LAT_HIST = Histogram(
    "fastapi_inference_latency_ms",
    "Inference latency (ms)",
    ["route"],
    buckets=(5,10,25,50,100,200,500,1000)
)
# Gauges for MLflow model metrics
G_MODEL_AUC = Gauge("model_test_auc", "Model test AUC by stage", ["stage"]) 
G_MODEL_ACC = Gauge("model_test_acc", "Model test ACC by stage", ["stage"]) 
G_MODEL_VERSION = Gauge("model_version", "Model version served by route", ["route"]) 
G_CANARY_PERCENT = Gauge("canary_traffic_percent", "Configured canary traffic percent")

app = FastAPI(title="MLOps E2E API", version="1.2")


class PredictRequest(BaseModel):
    instances: List[List[float]]


@app.post("/traffic")
async def set_canary_percent(percent: int):
    global CANARY_PERCENT
    if percent < 0 or percent > 100:
        raise HTTPException(status_code=400, detail="percent must be 0..100")
    CANARY_PERCENT = int(percent)
    try:
        G_CANARY_PERCENT.set(CANARY_PERCENT)
    except Exception:
        pass
    return {"canary_percent": CANARY_PERCENT}


def load_scaler_and_schema():
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler at {SCALER_PATH}: {e}")
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        feature_order = schema["feature_order"]
    except Exception as e:
        raise RuntimeError(f"Failed to load schema at {SCHEMA_PATH}: {e}")
    return scaler, feature_order


@app.get("/health")
async def health():
    # Ping primary and canary TF Serving endpoints and set version gauge
    statuses = {}
    for name, url in {"primary": TF_SERVING_URL_PRIMARY, "canary": TF_SERVING_URL_CANARY}.items():
        if not url:
            statuses[name] = None
            continue
        try:
            base = url.replace(":predict", "")
            r = requests.get(base, timeout=2)
            statuses[name] = r.status_code
            # try to fetch version
            try:
                info = requests.get(base, timeout=2).json()
                vers = info.get("model_version_status", [{}])[0].get("version")
                if vers:
                    G_MODEL_VERSION.labels(route=name).set(float(vers))
            except Exception:
                pass
        except Exception:
            statuses[name] = 0
    # reflect canary percent
    try:
        G_CANARY_PERCENT.set(CANARY_PERCENT)
    except Exception:
        pass
    return {"status": "ok", "tf_serving": statuses}


@app.get("/metrics")
async def metrics():
    # Refresh MLflow model metric gauges on each scrape
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        # Production
        prod = client.get_latest_versions(os.getenv("MODEL_NAME", "model"), stages=["Production"]) or []
        if prod:
            run = client.get_run(prod[0].run_id)
            auc = run.data.metrics.get("test_auc")
            acc = run.data.metrics.get("test_acc")
            if auc is not None:
                G_MODEL_AUC.labels(stage="Production").set(auc)
            if acc is not None:
                G_MODEL_ACC.labels(stage="Production").set(acc)
        # Staging
        stg = client.get_latest_versions(os.getenv("MODEL_NAME", "model"), stages=["Staging"]) or []
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
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(req: PredictRequest):
    scaler, feature_order = load_scaler_and_schema()
    X = np.array(req.instances, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(feature_order):
        raise HTTPException(status_code=400, detail=f"Each instance must have {len(feature_order)} features")

    X_scaled = scaler.transform(pd.DataFrame(X, columns=feature_order))
    payload = {"instances": X_scaled.tolist()}

    route = "primary"
    url = TF_SERVING_URL_PRIMARY
    if CANARY_ENABLED and TF_SERVING_URL_CANARY and random.randint(1,100) <= CANARY_PERCENT:
        route = "canary"
        url = TF_SERVING_URL_CANARY

    with LAT_HIST.labels(route=route).time():
        t0 = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=5)
        latency_ms = (time.perf_counter() - t0) * 1000

    if not resp.ok:
        REQ_COUNTER.labels(route=route, status="502").inc()
        raise HTTPException(status_code=502, detail=f"TF Serving error ({route}): {resp.text}")
    else:
        REQ_COUNTER.labels(route=route, status="200").inc()

    preds = resp.json().get("predictions")
    try:
        with mlflow.start_run(run_name="inference", nested=True):
            mlflow.log_metric("inference_latency_ms", latency_ms)
            mlflow.log_param("batch_size", len(req.instances))
            mlflow.set_tag("route", route)
    except Exception:
        pass

    return {"route": route, "predictions": preds, "latency_ms": latency_ms}
