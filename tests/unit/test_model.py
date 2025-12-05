# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Unit tests for model building and training."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestBuildModel:
    """Tests for model building function."""

    def test_model_has_correct_input_shape(self):
        """Test that model accepts correct input shape."""
        from training.src.train import build_model
        
        model = build_model(
            input_dim=20,
            hidden_units=[64, 32],
            dropout=0.1,
            lr=0.001,
        )
        
        # Check input shape
        assert model.input_shape == (None, 20)

    def test_model_has_correct_output_shape(self):
        """Test that model outputs single value (binary classification)."""
        from training.src.train import build_model
        
        model = build_model(
            input_dim=20,
            hidden_units=[64, 32],
            dropout=0.1,
            lr=0.001,
        )
        
        # Check output shape
        assert model.output_shape == (None, 1)

    def test_model_has_hidden_layers(self):
        """Test that model has correct number of layers."""
        from training.src.train import build_model
        
        model = build_model(
            input_dim=20,
            hidden_units=[64, 32, 16],
            dropout=0.0,  # No dropout for simpler layer count
            lr=0.001,
        )
        
        # Input + 3 dense + output = we should have Dense layers
        dense_layers = [l for l in model.layers if "dense" in l.name.lower()]
        assert len(dense_layers) == 4  # 3 hidden + 1 output

    def test_model_with_dropout(self):
        """Test that model includes dropout layers when specified."""
        from training.src.train import build_model
        
        model = build_model(
            input_dim=20,
            hidden_units=[64, 32],
            dropout=0.5,
            lr=0.001,
        )
        
        dropout_layers = [l for l in model.layers if "dropout" in l.name.lower()]
        assert len(dropout_layers) == 2  # One per hidden layer

    def test_model_without_dropout(self):
        """Test that model has no dropout when dropout=0."""
        from training.src.train import build_model
        
        model = build_model(
            input_dim=20,
            hidden_units=[64, 32],
            dropout=0.0,
            lr=0.001,
        )
        
        dropout_layers = [l for l in model.layers if "dropout" in l.name.lower()]
        assert len(dropout_layers) == 0

    def test_model_compiles_successfully(self):
        """Test that model compiles without errors."""
        from training.src.train import build_model
        
        model = build_model(
            input_dim=20,
            hidden_units=[64, 32],
            dropout=0.1,
            lr=0.001,
        )
        
        # Check optimizer is set
        assert model.optimizer is not None
        assert model.loss is not None

    def test_model_can_predict(self):
        """Test that model can make predictions."""
        from training.src.train import build_model
        
        model = build_model(
            input_dim=20,
            hidden_units=[32, 16],
            dropout=0.0,
            lr=0.001,
        )
        
        # Create dummy input
        X = np.random.randn(10, 20).astype(np.float32)
        
        # Should not raise
        predictions = model.predict(X, verbose=0)
        
        assert predictions.shape == (10, 1)
        assert all(0 <= p <= 1 for p in predictions.ravel())
