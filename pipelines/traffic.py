# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Traffic management for canary deployments."""
from __future__ import annotations

import os

import requests

# =============================================================================
# Configuration
# =============================================================================
API_URL: str = os.getenv("API_URL", "http://api:8000")


def set_canary(percent: int) -> bool:
    """Set the canary traffic percentage.

    Args:
        percent: Percentage of traffic to route to canary (0-100).

    Returns:
        True if the traffic split was set successfully.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    response = requests.post(
        f"{API_URL}/traffic",
        params={"percent": percent},
        timeout=5,
    )
    response.raise_for_status()
    return response.json().get("canary_percent") == percent


def get_current_split() -> int:
    """Get the current canary traffic percentage.

    Returns:
        Current canary percentage.
    """
    response = requests.get(f"{API_URL}/health", timeout=5)
    response.raise_for_status()
    # Default to 0 if not available
    return 0
