# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================

.PHONY: help install lint format test train serve up down clean

# Default target
help:
	@echo "AutoMLOps - Available Commands"
	@echo "=============================="
	@echo "  make install     - Install dependencies (dev mode)"
	@echo "  make lint        - Run linting (ruff + mypy)"
	@echo "  make format      - Format code with ruff"
	@echo "  make test        - Run tests with coverage"
	@echo "  make train       - Train a new model"
	@echo "  make promote     - Promote canary to production"
	@echo "  make up          - Start all services"
	@echo "  make down        - Stop all services"
	@echo "  make logs        - View logs"
	@echo "  make clean       - Clean artifacts"

# Install dependencies
install:
	pip install -e ".[dev]"
	pre-commit install

# Linting
lint:
	ruff check .
	mypy training serving pipelines drift --config-file pyproject.toml

# Format code
format:
	ruff format .
	ruff check --fix .

# Run tests
test:
	pytest --cov=training --cov=serving --cov=pipelines --cov=drift --cov-report=term-missing -v

# Train a model (requires services to be running)
train:
	docker compose run --rm trainer python -m training.src.train

# Promote canary to production
promote:
	docker compose run --rm -e PYTHONPATH=/app trainer python pipelines/promote_canary.py
	docker compose restart tfserving

# Start services
up:
	docker compose up -d --build

# Stop services
down:
	docker compose down

# View logs
logs:
	docker compose logs -f

# Clean artifacts (be careful!)
clean:
	rm -rf artifacts/raw artifacts/processed artifacts/reports
	rm -rf mlruns
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf **/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Simulate drift detection
drift:
	docker compose run --rm monitor python drift/detect_drift.py --simulate

# Check service health
health:
	@echo "MLflow: http://localhost:5000"
	@curl -s http://localhost:5000/version || echo "MLflow not running"
	@echo ""
	@echo "FastAPI: http://localhost:8000"
	@curl -s http://localhost:8000/health || echo "FastAPI not running"
	@echo ""
	@echo "TF Serving: http://localhost:8501"
	@curl -s http://localhost:8501/v1/models/model || echo "TF Serving not running"
