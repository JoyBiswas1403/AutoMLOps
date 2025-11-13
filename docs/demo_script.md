# Demo script (2–3 min)

1) Training + MLflow
- docker compose up -d --build
- docker compose run --rm trainer python -m training.src.train
- Show MLflow: new run under experiment e2e_mlops; see metrics and model card in artifacts

2) Deploy + Inference
- docker compose run --rm -e PYTHONPATH=/app trainer python pipelines/promote_canary.py
- docker compose restart tfserving
- scripts/curl_predict.ps1
- Show API /metrics and Grafana dashboard (requests, latency)

3) Canary rollout
- POST /traffic with 10, 25, 50, 100
- Watch Grafana split by route (primary vs canary)

4) Drift + Auto-Retrain
- docker compose run --rm monitor python drift/detect_drift.py --simulate
- Show Prometheus drift_ks_mean/drift_psi_mean rising; Evidently UI updated; MLflow new run; Airflow DAG shows stages

5) Close with outcomes
- Mention before/after AUC and latency
- Point to README sections and Streamlit comparison UI
