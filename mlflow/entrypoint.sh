#!/bin/sh

: "${MLFLOW_BACKEND_URI:=postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow}"
: "${MLFLOW_ARTIFACT_ROOT:=s3://mlflow}"

exec mlflow server \
  --backend-store-uri "$MLFLOW_BACKEND_URI" \
  --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port 5000
