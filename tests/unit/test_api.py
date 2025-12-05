# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Unit tests for FastAPI application."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(mock_scaler, mock_schema, monkeypatch):
    """Create test client with mocked dependencies."""
    monkeypatch.setenv("SCALER_PATH", str(mock_scaler))
    monkeypatch.setenv("SCHEMA_PATH", str(mock_schema))
    monkeypatch.setenv("TF_SERVING_URL_PRIMARY", "http://localhost:8501/v1/models/model:predict")
    monkeypatch.setenv("CANARY_ENABLED", "false")
    
    # Need to reload the module to pick up new env vars
    from serving.api import app as api_module
    import importlib
    importlib.reload(api_module)
    
    return TestClient(api_module.app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_ok(self, api_client):
        """Test that health endpoint returns OK status."""
        with patch("serving.api.app.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {}
            
            response = api_client.get("/health")
            
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    def test_health_includes_tf_serving_status(self, api_client):
        """Test that health includes TF Serving status."""
        with patch("serving.api.app.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {}
            
            response = api_client.get("/health")
            
            assert "tf_serving" in response.json()


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_validates_input_shape(self, api_client):
        """Test that predict validates input feature count."""
        # Send wrong number of features (should be 20)
        payload = {"instances": [[0.1, 0.2, 0.3]]}  # Only 3 features
        
        with patch("serving.api.app.requests.post"):
            response = api_client.post("/predict", json=payload)
            
            assert response.status_code == 400
            assert "features" in response.json()["detail"].lower()

    def test_predict_returns_predictions(self, api_client):
        """Test that predict returns predictions for valid input."""
        payload = {"instances": [[0.0] * 20]}  # Correct number of features
        
        with patch("serving.api.app.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"predictions": [[0.75]]}
            mock_post.return_value = mock_response
            
            response = api_client.post("/predict", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "route" in data
            assert "latency_ms" in data

    def test_predict_handles_batch(self, api_client):
        """Test that predict handles batch predictions."""
        payload = {"instances": [[0.0] * 20, [1.0] * 20, [0.5] * 20]}
        
        with patch("serving.api.app.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"predictions": [[0.1], [0.9], [0.5]]}
            mock_post.return_value = mock_response
            
            response = api_client.post("/predict", json=payload)
            
            assert response.status_code == 200
            assert len(response.json()["predictions"]) == 3


class TestTrafficEndpoint:
    """Tests for traffic management endpoint."""

    def test_set_canary_percent_valid(self, api_client):
        """Test setting valid canary percentage."""
        response = api_client.post("/traffic?percent=25")
        
        assert response.status_code == 200
        assert response.json()["canary_percent"] == 25

    def test_set_canary_percent_invalid(self, api_client):
        """Test rejection of invalid canary percentage."""
        response = api_client.post("/traffic?percent=150")
        
        assert response.status_code == 400

    def test_set_canary_percent_zero(self, api_client):
        """Test setting canary percentage to zero."""
        response = api_client.post("/traffic?percent=0")
        
        assert response.status_code == 200
        assert response.json()["canary_percent"] == 0


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics_returns_prometheus_format(self, api_client):
        """Test that metrics endpoint returns Prometheus format."""
        with patch("serving.api.app._refresh_mlflow_metrics"):
            response = api_client.get("/metrics")
            
            assert response.status_code == 200
            assert "text/plain" in response.headers.get("content-type", "")
