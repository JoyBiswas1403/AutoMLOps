# End-to-End MLOps: MLflow + Data Drift + TF Serving + FastAPI

This project demonstrates a production-style ML system with:
- Data generation/ingestion and preprocessing
- Model training with TensorFlow/Keras
- Experiment tracking with MLflow
- Automated data drift detection and retraining
- Deployment via TensorFlow Serving
- FastAPI backend for predictions

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

Repository layout
- docker-compose.yml: Orchestrates services
- mlflow/: MLflow server Dockerfile and entrypoint
- training/src/: data_ingestion, preprocess, train, evaluate, utils
- drift/: drift detection and retraining trigger
- serving/api/: FastAPI app and Dockerfile
- pipelines/: Prefect flow for end-to-end orchestration (optional)
- .github/workflows/ci.yml: Lint/test and build

Notes
- By default, artifacts/runs are persisted to Docker volumes; delete volumes to reset.
- Model updates create a new numeric version directory. TF Serving automatically picks the latest.
- For production, consider moving to remote artifact stores and a managed DB for MLflow.

Troubleshooting
- If MLflow doesn’t start on Windows: CRLF line endings can break shell entrypoints. This stack runs MLflow directly via the Dockerfile to avoid CRLF issues.
- If Python imports fail in containers, use `python -m package.module` so relative imports resolve.
- If MLflow artifact upload errors mention S3 credentials, ensure `.env` exists (copied from `.env.example`).
- If TF Serving returns variable not found errors, retrain after upgrading to Keras 3 (we export SavedModel via `model.export()`); then promote and restart TF Serving.
