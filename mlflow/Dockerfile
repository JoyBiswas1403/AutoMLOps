FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow[extras]==2.17.0 gunicorn boto3 psycopg2-binary

EXPOSE 5000
ENTRYPOINT ["/bin/sh","-c","mlflow server --backend-store-uri \"$MLFLOW_BACKEND_URI\" --default-artifact-root \"$MLFLOW_ARTIFACT_ROOT\" --host 0.0.0.0 --port 5000"]
