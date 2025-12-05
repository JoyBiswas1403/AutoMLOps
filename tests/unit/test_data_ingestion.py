# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Unit tests for data ingestion module."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


class TestGenerateSyntheticData:
    """Tests for synthetic data generation."""

    def test_generate_correct_shape(self):
        """Test that generated data has correct shape."""
        # Import here to avoid module-level issues with env vars
        from training.src.data_ingestion import generate_synthetic_data
        
        X, y = generate_synthetic_data(
            n_samples=500,
            n_features=15,
            random_state=42,
        )
        
        assert X.shape == (500, 15)
        assert len(y) == 500

    def test_generate_correct_columns(self):
        """Test that generated data has correct column names."""
        from training.src.data_ingestion import generate_synthetic_data
        
        X, y = generate_synthetic_data(n_samples=100, n_features=10)
        
        expected_cols = [f"f{i}" for i in range(10)]
        assert list(X.columns) == expected_cols
        assert y.name == "target"

    def test_generate_binary_target(self):
        """Test that target is binary (0 or 1)."""
        from training.src.data_ingestion import generate_synthetic_data
        
        _, y = generate_synthetic_data(n_samples=1000, n_classes=2)
        
        assert set(y.unique()).issubset({0, 1})

    def test_generate_reproducible(self):
        """Test that generation is reproducible with same seed."""
        from training.src.data_ingestion import generate_synthetic_data
        
        X1, y1 = generate_synthetic_data(n_samples=100, random_state=42)
        X2, y2 = generate_synthetic_data(n_samples=100, random_state=42)
        
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_generate_different_seeds(self):
        """Test that different seeds produce different data."""
        from training.src.data_ingestion import generate_synthetic_data
        
        X1, _ = generate_synthetic_data(n_samples=100, random_state=42)
        X2, _ = generate_synthetic_data(n_samples=100, random_state=123)
        
        assert not X1.equals(X2)


class TestGenerateAndSplit:
    """Tests for data generation and splitting."""

    def test_split_creates_files(self, temp_dir: Path, mock_config: dict):
        """Test that split creates train/val/test CSV files."""
        # Set up environment
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        with patch("training.src.data_ingestion.load_config", return_value=mock_config), \
             patch("training.src.data_ingestion.ARTIFACTS_DIR", artifacts_dir), \
             patch("training.src.data_ingestion.RAW_DIR", artifacts_dir / "raw"):
            
            (artifacts_dir / "raw").mkdir(exist_ok=True)
            
            from training.src.data_ingestion import generate_and_split
            
            paths = generate_and_split()
            
            assert Path(paths["train"]).exists()
            assert Path(paths["val"]).exists()
            assert Path(paths["test"]).exists()

    def test_split_ratios_approximate(self, temp_dir: Path, mock_config: dict):
        """Test that split ratios are approximately correct."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        with patch("training.src.data_ingestion.load_config", return_value=mock_config), \
             patch("training.src.data_ingestion.ARTIFACTS_DIR", artifacts_dir), \
             patch("training.src.data_ingestion.RAW_DIR", artifacts_dir / "raw"):
            
            (artifacts_dir / "raw").mkdir(exist_ok=True)
            
            from training.src.data_ingestion import generate_and_split
            
            paths = generate_and_split()
            
            train = pd.read_csv(paths["train"])
            val = pd.read_csv(paths["val"])
            test = pd.read_csv(paths["test"])
            
            total = len(train) + len(val) + len(test)
            
            # Approximate ratios (allow for rounding)
            assert 0.55 < len(train) / total < 0.70  # ~64%
            assert 0.10 < len(val) / total < 0.25    # ~16%
            assert 0.15 < len(test) / total < 0.25   # ~20%

    def test_split_has_target_column(self, temp_dir: Path, mock_config: dict):
        """Test that all splits contain target column."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        with patch("training.src.data_ingestion.load_config", return_value=mock_config), \
             patch("training.src.data_ingestion.ARTIFACTS_DIR", artifacts_dir), \
             patch("training.src.data_ingestion.RAW_DIR", artifacts_dir / "raw"):
            
            (artifacts_dir / "raw").mkdir(exist_ok=True)
            
            from training.src.data_ingestion import generate_and_split
            
            paths = generate_and_split()
            
            for split_name, path in paths.items():
                df = pd.read_csv(path)
                assert "target" in df.columns, f"Missing target in {split_name}"
