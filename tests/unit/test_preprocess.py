# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Unit tests for preprocessing module."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


class TestFitAndTransform:
    """Tests for fit_and_transform function."""

    def test_creates_scaler_file(self, temp_dir: Path, sample_dataset: pd.DataFrame):
        """Test that scaler file is created."""
        # Setup
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "processed").mkdir(exist_ok=True)
        
        # Create train/val/test CSVs
        train_path = temp_dir / "train.csv"
        val_path = temp_dir / "val.csv"
        test_path = temp_dir / "test.csv"
        
        sample_dataset.to_csv(train_path, index=False)
        sample_dataset.to_csv(val_path, index=False)
        sample_dataset.to_csv(test_path, index=False)
        
        with patch("training.src.preprocess.ARTIFACTS_DIR", artifacts_dir), \
             patch("training.src.preprocess.PROCESSED_DIR", artifacts_dir / "processed"):
            
            from training.src.preprocess import fit_and_transform
            
            result = fit_and_transform(train_path, val_path, test_path)
            
            assert Path(result["scaler_path"]).exists()

    def test_creates_schema_file(self, temp_dir: Path, sample_dataset: pd.DataFrame):
        """Test that schema file is created."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "processed").mkdir(exist_ok=True)
        
        train_path = temp_dir / "train.csv"
        val_path = temp_dir / "val.csv"
        test_path = temp_dir / "test.csv"
        
        sample_dataset.to_csv(train_path, index=False)
        sample_dataset.to_csv(val_path, index=False)
        sample_dataset.to_csv(test_path, index=False)
        
        with patch("training.src.preprocess.ARTIFACTS_DIR", artifacts_dir), \
             patch("training.src.preprocess.PROCESSED_DIR", artifacts_dir / "processed"):
            
            from training.src.preprocess import fit_and_transform
            
            result = fit_and_transform(train_path, val_path, test_path)
            
            assert Path(result["schema_path"]).exists()
            
            with open(result["schema_path"]) as f:
                schema = json.load(f)
            assert "feature_order" in schema

    def test_creates_processed_files(self, temp_dir: Path, sample_dataset: pd.DataFrame):
        """Test that processed CSV files are created."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "processed").mkdir(exist_ok=True)
        
        train_path = temp_dir / "train.csv"
        val_path = temp_dir / "val.csv"
        test_path = temp_dir / "test.csv"
        
        sample_dataset.to_csv(train_path, index=False)
        sample_dataset.to_csv(val_path, index=False)
        sample_dataset.to_csv(test_path, index=False)
        
        with patch("training.src.preprocess.ARTIFACTS_DIR", artifacts_dir), \
             patch("training.src.preprocess.PROCESSED_DIR", artifacts_dir / "processed"):
            
            from training.src.preprocess import fit_and_transform
            
            result = fit_and_transform(train_path, val_path, test_path)
            
            assert Path(result["processed"]["train"]).exists()
            assert Path(result["processed"]["val"]).exists()
            assert Path(result["processed"]["test"]).exists()

    def test_scaler_is_fitted(self, temp_dir: Path, sample_dataset: pd.DataFrame):
        """Test that saved scaler is fitted."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "processed").mkdir(exist_ok=True)
        
        train_path = temp_dir / "train.csv"
        val_path = temp_dir / "val.csv"
        test_path = temp_dir / "test.csv"
        
        sample_dataset.to_csv(train_path, index=False)
        sample_dataset.to_csv(val_path, index=False)
        sample_dataset.to_csv(test_path, index=False)
        
        with patch("training.src.preprocess.ARTIFACTS_DIR", artifacts_dir), \
             patch("training.src.preprocess.PROCESSED_DIR", artifacts_dir / "processed"):
            
            from training.src.preprocess import fit_and_transform
            
            result = fit_and_transform(train_path, val_path, test_path)
            
            scaler = joblib.load(result["scaler_path"])
            assert hasattr(scaler, "mean_")
            assert hasattr(scaler, "scale_")

    def test_processed_data_is_scaled(self, temp_dir: Path, sample_dataset: pd.DataFrame):
        """Test that processed data is properly scaled."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "processed").mkdir(exist_ok=True)
        
        train_path = temp_dir / "train.csv"
        val_path = temp_dir / "val.csv"
        test_path = temp_dir / "test.csv"
        
        sample_dataset.to_csv(train_path, index=False)
        sample_dataset.to_csv(val_path, index=False)
        sample_dataset.to_csv(test_path, index=False)
        
        with patch("training.src.preprocess.ARTIFACTS_DIR", artifacts_dir), \
             patch("training.src.preprocess.PROCESSED_DIR", artifacts_dir / "processed"):
            
            from training.src.preprocess import fit_and_transform
            
            result = fit_and_transform(train_path, val_path, test_path)
            
            processed_train = pd.read_csv(result["processed"]["train"])
            feature_cols = [c for c in processed_train.columns if c not in ("target", "row_id")]
            
            # Scaled data should have mean ~0 and std ~1
            for col in feature_cols[:5]:  # Check first 5 features
                assert abs(processed_train[col].mean()) < 0.5
                assert 0.5 < processed_train[col].std() < 1.5

    def test_adds_row_id(self, temp_dir: Path, sample_dataset: pd.DataFrame):
        """Test that row_id is added to processed data."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "processed").mkdir(exist_ok=True)
        
        train_path = temp_dir / "train.csv"
        val_path = temp_dir / "val.csv"
        test_path = temp_dir / "test.csv"
        
        sample_dataset.to_csv(train_path, index=False)
        sample_dataset.to_csv(val_path, index=False)
        sample_dataset.to_csv(test_path, index=False)
        
        with patch("training.src.preprocess.ARTIFACTS_DIR", artifacts_dir), \
             patch("training.src.preprocess.PROCESSED_DIR", artifacts_dir / "processed"):
            
            from training.src.preprocess import fit_and_transform
            
            result = fit_and_transform(train_path, val_path, test_path)
            
            processed = pd.read_csv(result["processed"]["train"])
            assert "row_id" in processed.columns
