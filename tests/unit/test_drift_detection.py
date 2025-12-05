# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Unit tests for drift detection module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift.detect_drift import psi, simulate_new_batch, compute_drift_metrics


class TestPSI:
    """Tests for Population Stability Index calculation."""

    def test_psi_identical_distributions(self):
        """Test PSI is ~0 for identical distributions."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        
        result = psi(data, data)
        
        assert result < 0.01  # Should be very close to 0

    def test_psi_increases_with_shift(self):
        """Test PSI increases when distributions shift."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual_small_shift = np.random.normal(0.3, 1, 1000)
        actual_large_shift = np.random.normal(1.0, 1, 1000)
        
        psi_small = psi(expected, actual_small_shift)
        psi_large = psi(expected, actual_large_shift)
        
        assert psi_large > psi_small

    def test_psi_non_negative(self):
        """Test PSI is always non-negative."""
        np.random.seed(42)
        for _ in range(10):
            expected = np.random.normal(0, 1, 500)
            actual = np.random.normal(np.random.uniform(-1, 1), 1, 500)
            
            result = psi(expected, actual)
            
            assert result >= 0

    def test_psi_with_scale_change(self):
        """Test PSI detects scale changes."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 2, 1000)  # Same mean, different std
        
        result = psi(expected, actual)
        
        assert result > 0.05  # Should detect distribution change


class TestSimulateNewBatch:
    """Tests for batch simulation function."""

    def test_simulate_creates_shifted_data(self, temp_dir):
        """Test that simulation creates shifted data."""
        # Create reference CSV
        np.random.seed(42)
        ref_data = pd.DataFrame({
            "f0": np.random.randn(100),
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        })
        ref_path = temp_dir / "ref.csv"
        ref_data.to_csv(ref_path, index=False)
        
        result = simulate_new_batch(ref_path, shift=0.5)
        
        # Check f0 is shifted (even index)
        assert result["f0"].mean() > ref_data["f0"].mean() + 0.3
        # Check f1 is NOT shifted (odd index)
        assert abs(result["f1"].mean() - ref_data["f1"].mean()) < 0.2

    def test_simulate_preserves_shape(self, temp_dir):
        """Test that simulation preserves data shape."""
        ref_data = pd.DataFrame({
            "f0": np.random.randn(50),
            "f1": np.random.randn(50),
            "target": np.random.randint(0, 2, 50),
        })
        ref_path = temp_dir / "ref.csv"
        ref_data.to_csv(ref_path, index=False)
        
        result = simulate_new_batch(ref_path, shift=0.5)
        
        assert result.shape == ref_data.shape

    def test_simulate_preserves_target(self, temp_dir):
        """Test that simulation preserves target values."""
        ref_data = pd.DataFrame({
            "f0": np.random.randn(50),
            "target": [0, 1] * 25,
        })
        ref_path = temp_dir / "ref.csv"
        ref_data.to_csv(ref_path, index=False)
        
        result = simulate_new_batch(ref_path, shift=0.5)
        
        assert list(result["target"]) == list(ref_data["target"])


class TestComputeDriftMetrics:
    """Tests for drift metrics computation."""

    def test_returns_required_fields(self):
        """Test that all required fields are returned."""
        np.random.seed(42)
        ref = pd.DataFrame({"f0": np.random.randn(100), "f1": np.random.randn(100)})
        curr = pd.DataFrame({"f0": np.random.randn(100), "f1": np.random.randn(100)})
        
        result = compute_drift_metrics(ref, curr, ["f0", "f1"])
        
        assert "ks_mean" in result
        assert "psi_mean" in result
        assert "ks_per_feature" in result
        assert "psi_per_feature" in result

    def test_detects_drift(self):
        """Test that metrics detect significant drift."""
        np.random.seed(42)
        ref = pd.DataFrame({"f0": np.random.normal(0, 1, 500)})
        curr = pd.DataFrame({"f0": np.random.normal(2, 1, 500)})  # Shifted by 2
        
        result = compute_drift_metrics(ref, curr, ["f0"])
        
        assert result["ks_mean"] > 0.3  # Should detect drift
        assert result["psi_mean"] > 0.1

    def test_no_drift_on_same_data(self):
        """Test low metrics on identical data."""
        np.random.seed(42)
        data = pd.DataFrame({"f0": np.random.randn(500), "f1": np.random.randn(500)})
        
        result = compute_drift_metrics(data, data.copy(), ["f0", "f1"])
        
        assert result["ks_mean"] < 0.1
        assert result["psi_mean"] < 0.1
