# AutoMLOps Runbook

Operational guide for running, troubleshooting, and maintaining the AutoMLOps platform.

## 🚀 Starting the Platform

### Full Stack
```bash
docker compose up -d --build
```

### Individual Services
```bash
docker compose up -d mlflow      # MLflow only
docker compose up -d api         # FastAPI only
docker compose up -d tfserving   # TF Serving only
```

### Check Status
```bash
docker compose ps
```

## 🔧 Common Operations

### Train a New Model
```bash
# Using real Credit Card Fraud dataset
docker compose run --rm trainer python -m training.src.train

# Using synthetic data
# Edit training/configs/params.yaml: data.source: synthetic
docker compose run --rm trainer python -m training.src.train
```

### Promote Canary to Production
```bash
docker compose run --rm -e PYTHONPATH=/app trainer python pipelines/promote_canary.py
docker compose restart tfserving
```

### Rollback to Previous Version
```bash
docker compose run --rm -e PYTHONPATH=/app trainer python pipelines/rollback.py
docker compose restart tfserving
```

### Simulate Data Drift
```bash
docker compose run --rm monitor python drift/detect_drift.py --simulate
```

### Check Model Health
```bash
curl http://localhost:8000/health
```

### Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"instances": [[0.0, -1.3, 2.1, ...]]}'
```

## 🐛 Troubleshooting

### Model Not Loading

**Symptom:** TF Serving returns 404 or model not found

**Check:**
1. Model exists in models directory:
   ```bash
   ls models/model/
   ```
2. Model has correct structure:
   ```bash
   ls models/model/1/  # Should have saved_model.pb
   ```
3. TF Serving logs:
   ```bash
   docker compose logs tfserving
   ```

**Fix:**
```bash
docker compose run --rm trainer python -m training.src.train
docker compose restart tfserving
```

### MLflow Not Accessible

**Symptom:** Cannot connect to http://localhost:5000

**Check:**
```bash
docker compose logs mlflow
docker compose ps mlflow
```

**Fix:**
```bash
docker compose restart mlflow
```

### API Returns 500

**Symptom:** Inference errors

**Check:**
```bash
docker compose logs api
```

**Common causes:**
- Missing scaler.joblib or schema.json
- TF Serving not responding
- Feature count mismatch

**Fix:**
```bash
# Retrain to regenerate artifacts
docker compose run --rm trainer python -m training.src.train
docker compose restart api
```

### Out of Memory

**Symptom:** Container killed by OOM

**Check:**
```bash
docker stats
```

**Fix:**
- Reduce batch_size in params.yaml
- Limit undersampling ratio
- Add memory limits to docker-compose.yml

### Drift Detection Failing

**Symptom:** Drift script errors

**Check:**
```bash
docker compose run --rm monitor python drift/detect_drift.py --verbose
```

**Common causes:**
- Missing reference data
- Schema mismatch

## 📊 Monitoring

### Key Metrics to Watch

| Metric | Location | Threshold |
|--------|----------|-----------|
| Inference Latency | Grafana | p99 < 100ms |
| Error Rate | Grafana | < 1% |
| Memory Usage | `docker stats` | < 80% |
| Model AUC | MLflow | > 0.90 |
| Drift PSI | Logs | < 0.25 |

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# MLflow health
curl http://localhost:5000/version

# TF Serving health
curl http://localhost:8501/v1/models/model
```

### Prometheus Queries
```promql
# Request rate
sum(rate(fastapi_requests_total[5m]))

# Latency p99
histogram_quantile(0.99, rate(fastapi_inference_latency_ms_bucket[5m]))

# Error rate
sum(rate(fastapi_requests_total{status!="200"}[5m])) /
sum(rate(fastapi_requests_total[5m]))
```

## 🔄 Maintenance

### Clean Up Docker
```bash
docker compose down -v  # Remove volumes
docker system prune -a  # Clean all
```

### Reset Artifacts
```bash
rm -rf artifacts/
docker compose run --rm trainer python -m training.src.train
```

### Update Dependencies
```bash
pip install -U -r training/requirements.txt
docker compose build --no-cache
```

### Backup MLflow
```bash
cp -r mlflow/mlruns backup/mlruns_$(date +%Y%m%d)
```

## 📞 Support

- GitHub Issues: https://github.com/JoyBiswas1403/AutoMLOps/issues
- Documentation: /docs/ARCHITECTURE.md
