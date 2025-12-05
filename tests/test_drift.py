# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Unit tests for drift detection PSI calculation."""
from __future__ import annotations

import numpy as np
import pytest

from drift.detect_drift import psi


class TestPSI:
    """Tests for Population Stability Index calculation."""

    def test_psi_increases_with_shift(self):
        """Test PSI increases when distribution shifts."""
        np.random.seed(42)
        a = np.random.normal(0, 1, size=1000)
        b = np.random.normal(0.5, 1, size=1000)
        
        s0 = psi(a, a)
        s1 = psi(a, b)
        
        assert s0 <= 1e-6
        assert s1 > s0
