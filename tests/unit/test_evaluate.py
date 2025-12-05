# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Unit tests for evaluation module."""
from __future__ import annotations

import numpy as np
import pytest

from training.src.evaluate import evaluate_predictions


class TestEvaluatePredictions:
    """Tests for prediction evaluation function."""

    def test_returns_required_metrics(self):
        """Test that all required metrics are returned."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        
        result = evaluate_predictions(y_true, y_prob)
        
        assert "auc" in result
        assert "acc" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result

    def test_perfect_predictions(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        
        result = evaluate_predictions(y_true, y_prob)
        
        assert result["auc"] == 1.0
        assert result["acc"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_random_predictions(self):
        """Test that random predictions give ~0.5 AUC."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.rand(1000)
        
        result = evaluate_predictions(y_true, y_prob)
        
        # Random should be around 0.5
        assert 0.4 < result["auc"] < 0.6

    def test_metrics_are_floats(self):
        """Test that all metrics are float type."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        
        result = evaluate_predictions(y_true, y_prob)
        
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not float"

    def test_metrics_in_valid_range(self):
        """Test that all metrics are in [0, 1] range."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        
        result = evaluate_predictions(y_true, y_prob)
        
        for key, value in result.items():
            assert 0 <= value <= 1, f"{key}={value} out of range"

    def test_threshold_at_half(self):
        """Test that threshold is at 0.5."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.4, 0.6, 0.4, 0.6])  # Two correct, two wrong
        
        result = evaluate_predictions(y_true, y_prob)
        
        # With threshold at 0.5: pred = [0, 1, 0, 1]
        # Matches: [0,0], [1,1] -> 2/4 = 0.5
        assert result["acc"] == 0.5
