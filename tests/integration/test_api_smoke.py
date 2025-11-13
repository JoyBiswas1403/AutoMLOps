import os
import pytest
import requests

RUN_SMOKE = os.getenv("RUN_SMOKE", "0") == "1"
API_URL = os.getenv("API_URL", "http://localhost:8000")

pytestmark = pytest.mark.skipif(not RUN_SMOKE, reason="Set RUN_SMOKE=1 to run integration smoke test")


def test_health():
    r = requests.get(f"{API_URL}/health", timeout=5)
    assert r.ok


def test_predict_zero_vector():
    body = {"instances": [[0.0] * 20]}
    r = requests.post(f"{API_URL}/predict", json=body, timeout=5)
    assert r.ok
    data = r.json()
    assert "predictions" in data and isinstance(data["predictions"], list)
