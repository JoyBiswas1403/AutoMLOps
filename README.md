# 🚀 AutoMLOps — End-to-End MLOps Pipeline (MLflow + TF-Serving + Drift Detection + FastAPI)

[![Docker Compose](https://img.shields.io/badge/Docker--Compose-Ready-brightgreen?style=flat-square)](#)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-orange?style=flat-square)](#)
[![TensorFlow Serving](https://img.shields.io/badge/TF--Serving-Production%20Models-blue?style=flat-square)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-009688?style=flat-square)](#)
[![MLOps](https://img.shields.io/badge/MLOps-End--to--End-purple?style=flat-square)](#)

> **A fully modular, end-to-end Machine Learning Operations (MLOps) system featuring MLflow tracking, TensorFlow Serving deployment, FastAPI inference API, data-drift detection, automated retraining, and production-style orchestration — all runnable with a single `docker compose up`.**

This project demonstrates **real MLOps engineering** — the same workflow used at companies like Google, Uber, and Netflix to train, deploy, monitor, and retrain ML models at scale.

---

# 🎥 Demo (GIF / Video Placeholder)
> *(Replace with your GIF once recorded)*  
![demo-placeholder](docs/demo.gif)

---

# 🧠 Architecture Overview

```mermaid
flowchart TB
    subgraph DATA[Data Layer]
        data_gen[Data Generator] --> preprocess[Preprocessing]
    end

    subgraph TRAIN[Training Pipeline]
        preprocess --> trainer[Trainer]
        trainer --> mlflow[MLflow Tracking]
        trainer --> export_model[Model Export (SavedModel)]
    end

    subgraph DEPLOY[Deployment]
        export_model --> tfserving[TensorFlow Serving]
        tfserving --> api[FastAPI Inference API]
    end

    subgraph MONITOR[Monitoring & Retraining]
        api --> prometheus[Prometheus Metrics]
        prometheus --> grafana[Grafana Dashboard]
        preprocess --> drift[Data Drift Detector]
        drift -->|Drift Found| trainer
    end

    api --> users[Users / Applications]
    mlflow --> mlflow_ui[MLflow UI]
```


---

# ✨ Key Features

### ✅ 1. MLflow Tracking & Model Registry
- Automatic experiment logging  
- Versioned models stored in MLflow  
- Supports model promotion (Canary → Production)

### ✅ 2. TensorFlow Serving Deployment
- Saves model as TF SavedModel  
- High-throughput serving  
- Standardized inference interface

### ✅ 3. FastAPI Inference Service
- Clean `/predict` endpoint  
- Input validation  
- Consistent preprocessing with persisted scalers

### ✅ 4. Data Drift Detection
- KS-test & PSI implementation  
- `--simulate` mode for testing  
- Automatic retraining trigger

### ✅ 5. Automated Retraining Pipeline
- Detect drift → retrain → register new model → promote → restart TF-Serving  
- Full MLOps lifecycle

### ✅ 6. Observability Stack
- Prometheus → latency, throughput, error rates  
- Grafana → dashboards & drift visualization

### ✅ 7. Fully Containerized
- Docker + Docker Compose  
- Zero manual environment setup  
- Reproducible pipeline

---

# ⚡ Quickstart (2 Minutes)

### 1️⃣ Clone repo & setup env  
```bash
git clone https://github.com/JoyBiswas1403/AutoMLOps.git
cd AutoMLOps
cp .env.example .env
```

### 2️⃣ Spin up the entire MLOps stack  
```bash
docker compose up -d --build
```

Services launched:
- MLflow → http://localhost:5000  
- FastAPI → http://localhost:8000/docs  
- TensorFlow Serving → http://localhost:8501  
- Prometheus → http://localhost:9090  
- Grafana → http://localhost:3000  

### 3️⃣ Train a new model  
```bash
docker compose run --rm trainer python -m training.src.train
```

### 4️⃣ Promote canary → production  
```bash
docker compose run --rm -e PYTHONPATH=/app trainer python pipelines/promote_canary.py
docker compose restart tfserving
```

### 5️⃣ Make an inference request  
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"instances":[[0.1, 0.2, -1.3, ...]]}'
```

### 6️⃣ Simulate data drift  
```bash
docker compose run --rm monitor python drift/detect_drift.py --simulate
```

---

# 📊 Results (Example — Replace With Real Values)

| Model Version | Test AUC | Drift Detected | Notes |
|---------------|----------|----------------|--------|
| v1 (baseline) | 0.78     | —              | Initial training |
| v2 (retrained) | 0.84     | Yes            | Auto-retrain triggered |

**Latency:** ~35 ms  
**Throughput:** ~300 req/s  

---

# 🧾 Model Card (Auto-Generated Template)

```
# Model Card — AutoMLOps

**Model Name:** TabularClassifier  
**Version:** vX  
**Created On:** YYYY-MM-DD  
**Framework:** TensorFlow (SavedModel)

## Overview
Binary classifier trained on synthetic/generated dataset.

## Metrics
AUC:  
Accuracy:  
Precision / Recall:

## Intended Use
Demo for MLOps lifecycle, CI/CD, retraining, drift detection.

## Limitations
Synthetic data; not intended for real-world clinical/financial use.

## Ethical Considerations
Validate with real data + domain experts.
```

---

# 🖼️ Screenshots (Add these once ready)

### MLflow Tracking UI  
*(Insert screenshot here)*

### Grafana Dashboard  
*(Insert screenshot here)*

### FastAPI Docs  
*(Insert screenshot here)*

---

# 🧩 Project Structure

```
AutoMLOps/
│── drift/
│── grafana/
│── mlflow/
│── pipelines/
│── prometheus/
│── serving/
│── training/
│── .github/workflows/ci.yml
│── docker-compose.yml
│── README.md
│── requirements.txt
```

---

# 🚀 Roadmap
- [ ] Add demo GIF & screenshots  
- [ ] Add Evidently AI dashboards  
- [ ] Add canary traffic splitting (Nginx / router)  
- [ ] Add more unit tests + integration tests in CI  
- [ ] Add Data Versioning (DVC / LakeFS)  
- [ ] Deploy API to cloud (Render/AWS/GCP)

---

# 🤝 Contributing
PRs, issues, and suggestions are welcome — this project is designed to evolve into a complete MLOps reference system.

---

# 📄 License
MIT License  
