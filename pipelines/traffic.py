import os
import requests

API_URL = os.getenv("API_URL", "http://api:8000")


def set_canary(percent: int) -> bool:
    r = requests.post(f"{API_URL}/traffic", json=percent, timeout=5)
    r.raise_for_status()
    return r.json().get("canary_percent") == percent
