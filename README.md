# AutoMLOps: End-to-End MLOps (MLflow + Drift + TF Serving + FastAPI)

[![CI](https://github.com/JoyBiswas1403/AutoMLOps/actions/workflows/ci_full.yml/badge.svg)](https://github.com/JoyBiswas1403/AutoMLOps/actions/workflows/ci_full.yml)
![Made with Docker](https://img.shields.io/badge/Made%20with-Docker-informational)
![MLOps](https://img.shields.io/badge/Topic-MLOps-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.17-brightgreen)
![TensorFlow%20Serving](https://img.shields.io/badge/TensorFlow-Serving-orange)

This project demonstrates a production-style ML system with:
- Data generation/ingestion and preprocessing
- Model training with TensorFlow/Keras
- Experiment tracking with MLflow
- Automated data drift detection and retraining
- Deployment via TensorFlow Serving
- FastAPI backend for predictions

Architecture

```mermaid
flowchart LR
  A[Data Generation/ Ingestion] --> B[Preprocess/ Feature Eng]
  B --> C[Training (Keras)]
  C -->|Autolog| D[MLflow Tracking/Registry]
  C -->|SavedModel| E[TF Serving - Canary]
  C -->|Model Card| D
  E -->|Promote| F[TF Serving - Production]
  F --> G[FastAPI Inference]
  G -->|Latency/Err/Vol| H[Prometheus]
  H --> I[Grafana]
  J[Drift Detector (Evidently/whylogs)] -->|KS/PSI| H
  J -->|Retrain Trigger| C
  K[Airflow] -.-> C
  K -.-> J
  K -.-> E
```

Quickstart
1) Start core services (MLflow, TF Serving, API):
   - Install Docker Desktop
   - Copy `.env.example` to `.env` (loads default Postgres/MinIO creds and ports)
   - Run: `docker compose -f "./docker-compose.yml" up -d --build`

2) Train and deploy the first model:
   - Run the trainer container: `docker compose run --rm trainer python -m training.src.train`
   - This logs to MLflow and exports a SavedModel to `serving/models/model_canary/<n>/`
   - Promote canary to prod: `docker compose run --rm -e PYTHONPATH=/app trainer python pipelines/promote_canary.py && docker compose restart tfserving`

3) Open UIs:
   - MLflow UI: http://localhost:5000
   - API docs: http://localhost:8000/docs

4) Make a prediction:
   - `POST http://localhost:8000/predict` with JSON body like
     `{ "instances": [[0.1, -1.23, ...]] }`
   - On Windows PowerShell you can run: `.\u005cscripts\curl_predict.ps1`

5) Simulate data drift and auto-retrain:
   - `docker compose run --rm monitor python drift/detect_drift.py --simulate` (creates a drifted batch)
   - If drift > threshold, it triggers a new training run and deploys a new canary version (e.g., `.../2/`)

Services
- mlflow: Tracking server with local SQLite backend and local artifact store (volume)
- tfserving: Serves the latest TensorFlow SavedModel under `serving/models/model/<version>`
- api: FastAPI app applying the saved scaler then forwarding to TF Serving; logs latency to MLflow
- trainer: Python environment to run training / preprocessing code
- monitor (optional): Python environment to run drift checks periodically

Tech choices
- Simulated tabular classification dataset (sklearn make_classification)
- StandardScaler persisted with joblib; FastAPI uses the same scaler at inference
- KS test + PSI for drift; threshold is configurable
- Prefect flow provided as an example orchestrator (optional)

Screenshots (add yours for recruiters)
- MLflow run: docs/screenshots/mlflow_run.png
- Grafana dashboard: docs/screenshots/grafana_dashboard.png
- Evidently UI: docs/screenshots/evidently_drift.png
- FastAPI docs: docs/screenshots/fastapi_docs.png
  A[Data Generation/ Ingestion] --> B[Preprocess/ Feature Eng]
  B --> C[Training (Keras)]
  C -->|Autolog| D[MLflow Tracking/Registry]
  C -->|SavedModel| E[TF Serving - Canary]
  C -->|Model Card| D
  E -->|Promote| F[TF Serving - Production]
  F --> G[FastAPI Inference]
  G -->|Latency/Err/Vol| H[Prometheus]
  H --> I[Grafana]
  J[Drift Detector (Evidently/whylogs)] -->|KS/PSI| H
  J -->|Retrain Trigger| C
  K[Airflow] -.-> C
  K -.-> J
  K -.-> E
```

Repository layout
- docker-compose.yml: Orchestrates services
- mlflow/: MLflow server Dockerfile and entrypoint
- training/src/: data_ingestion, preprocess, train, evaluate, utils
- drift/: drift detection and retraining trigger
- serving/api/: FastAPI app and Dockerfile
- pipelines/: Prefect flow for end-to-end orchestration (optional)
- .github/workflows/ci.yml: Lint/test and build

Results and metrics (example data)
- Latency p90: ~50–80 ms (local, batch size 1)
- Error rate: < 1% (HTTP 5xx and validation errors tracked)
- Drift (KS/PSI): detectable with simulated shift of 0.7 on half features; KS_mean ~0.2, PSI_mean ~0.3
- Before/After retrain: AUC 0.985 → 0.989; ACC 0.962 → 0.968 (synthetic)
- Live metrics: drift_ks_mean, drift_psi_mean, fastapi_requests_total{route,status}, fastapi_inference_latency_ms_bucket, model_version{route}

Notes
- By default, artifacts/runs are persisted to Docker volumes; delete volumes to reset.
- Model updates create a new numeric version directory. TF Serving automatically picks the latest.
- For production, consider moving to remote artifact stores and a managed DB for MLflow.

Demo video (optional but highly recommended)
- Record a 2–3 minute screencast covering: training, MLflow UI, Grafana dashboards, /predict calls, simulated drift → retrain → promotion.
- Place the video link here: [Demo video](https://your-demo-link)
- Use docs/demo_script.md as a step-by-step checklist.

Repository topics (set in GitHub UI)
- Recommended: mlops, mlflow, tensorflow-serving, docker, grafana, prometheus, airflow, feature-store, feast, evidently, whylogs, canary-deployments, model-registry

Model & data cards
- A model card is generated at artifacts/reports/model_card.md and logged to MLflow under artifacts.
- You can extend this with data cards (dataset provenance, splits, quality checks) and link in README.

Troubleshooting
- If MLflow doesn’t start on Windows: CRLF line endings can break shell entrypoints. This stack runs MLflow directly via the Dockerfile to avoid CRLF issues.
- If Python imports fail in containers, use `python -m package.module` so relative imports resolve.
- If MLflow artifact upload errors mention S3 credentials, ensure `.env` exists (copied from `.env.example`).
- If TF Serving returns variable not found errors, retrain after upgrading to Keras 3 (we export SavedModel via `model.export()`); then promote and restart TF Serving.

Lessons learned
- Windows CRLF can break container entrypoints; prefer ENTRYPOINT with sh -c or enforce LF via .gitattributes.
- Exporting Keras 3 SavedModel via model.export() avoids variable binding errors in TF Serving.
- Keep credentials in .env for local S3 (MinIO) and pass through docker-compose env.
- Progressive rollout is simplest with dual-Serving + routing in the API; pair it with Prometheus checks and Airflow gates.
