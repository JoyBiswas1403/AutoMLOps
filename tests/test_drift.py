import importlib, os
import numpy as np
from drift.detect_drift import psi


def test_psi_increases_with_shift():
    a = np.random.normal(0, 1, size=1000)
    b = np.random.normal(0.5, 1, size=1000)
    s0 = psi(a, a)
    s1 = psi(a, b)
    assert s0 <= 1e-6
    assert s1 > s0
