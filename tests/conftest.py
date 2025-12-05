# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Generate sample feature data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    data = np.random.randn(n_samples, n_features)
    columns = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def sample_target() -> pd.Series:
    """Generate sample target data for testing."""
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100), name="target")


@pytest.fixture
def sample_dataset(sample_features: pd.DataFrame, sample_target: pd.Series) -> pd.DataFrame:
    """Generate complete sample dataset with features and target."""
    return pd.concat([sample_features, sample_target], axis=1)


@pytest.fixture
def sample_probabilities() -> np.ndarray:
    """Generate sample prediction probabilities."""
    np.random.seed(42)
    return np.random.rand(100)


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Provide mock configuration for tests."""
    return {
        "experiment": {
            "name": "test_experiment",
            "model_name": "test_model",
            "random_state": 42,
            "test_size": 0.2,
            "val_size": 0.2,
        },
        "training": {
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.001,
            "hidden_units": [32, 16],
            "dropout": 0.1,
        },
        "features": {
            "n_features": 20,
            "n_informative": 10,
            "n_redundant": 2,
            "n_classes": 2,
        },
        "monitoring": {
            "drift_threshold_ks": 0.15,
            "drift_threshold_psi": 0.25,
        },
    }


@pytest.fixture
def mock_artifacts_dir(temp_dir: Path) -> Path:
    """Create mock artifacts directory structure."""
    artifacts = temp_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    
    (artifacts / "raw").mkdir(exist_ok=True)
    (artifacts / "processed").mkdir(exist_ok=True)
    (artifacts / "reports").mkdir(exist_ok=True)
    
    return artifacts


@pytest.fixture
def mock_scaler(mock_artifacts_dir: Path) -> Path:
    """Create mock scaler file."""
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    scaler = StandardScaler()
    scaler.fit(np.random.randn(100, 20))
    
    scaler_path = mock_artifacts_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    return scaler_path


@pytest.fixture
def mock_schema(mock_artifacts_dir: Path) -> Path:
    """Create mock schema file."""
    schema = {"feature_order": [f"f{i}" for i in range(20)]}
    schema_path = mock_artifacts_dir / "schema.json"
    schema_path.write_text(json.dumps(schema))
    return schema_path


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for tests that don't need actual tracking."""
    with patch("mlflow.set_experiment"), \
         patch("mlflow.start_run"), \
         patch("mlflow.log_metrics"), \
         patch("mlflow.log_params"), \
         patch("mlflow.log_artifact"), \
         patch("mlflow.set_tag"), \
         patch("mlflow.tensorflow.autolog"), \
         patch("mlflow.tensorflow.log_model"):
        yield


@pytest.fixture
def mock_requests():
    """Mock requests for API tests."""
    with patch("requests.post") as mock_post, \
         patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"predictions": [[0.5]]}
        mock_post.return_value = mock_response
        mock_get.return_value = mock_response
        yield {"post": mock_post, "get": mock_get}


@pytest.fixture(autouse=True)
def reset_env(temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Reset environment variables for each test."""
    monkeypatch.setenv("ARTIFACTS_DIR", str(temp_dir / "artifacts"))
    monkeypatch.setenv("MODELS_DIR", str(temp_dir / "models"))
    monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))


# Test markers
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
