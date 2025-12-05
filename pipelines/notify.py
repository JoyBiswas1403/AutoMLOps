# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Notification module for sending alerts via webhooks (Slack/Discord)."""
from __future__ import annotations

import json
import os
from typing import Any

import requests

# =============================================================================
# Configuration
# =============================================================================
WEBHOOK_ENV_KEYS: list[str] = [
    "DISCORD_WEBHOOK_URL",
    "SLACK_WEBHOOK_URL",
    "WEBHOOK_URL",
]


def send(title: str, text: str, extra: dict[str, Any] | None = None) -> bool:
    """Send a notification via webhook.

    Supports both Slack and Discord webhook formats.

    Args:
        title: Notification title (bold).
        text: Main notification text.
        extra: Optional dictionary of additional data to include.

    Returns:
        True if notification was sent successfully, False otherwise.
    """
    url = _get_webhook_url()
    if not url:
        return False

    payload = _build_payload(title, text, extra, url)

    try:
        response = requests.post(url, json=payload, timeout=5)
        return response.ok
    except requests.RequestException as e:
        print(f"Warning: Notification failed: {e}")
        return False


def _get_webhook_url() -> str | None:
    """Get the first available webhook URL from environment.

    Returns:
        Webhook URL or None if not configured.
    """
    for key in WEBHOOK_ENV_KEYS:
        url = os.getenv(key)
        if url:
            return url
    return None


def _build_payload(
    title: str,
    text: str,
    extra: dict[str, Any] | None,
    url: str,
) -> dict[str, Any]:
    """Build webhook payload based on the service type.

    Args:
        title: Notification title.
        text: Notification text.
        extra: Additional data.
        url: Webhook URL (used to detect Discord vs Slack).

    Returns:
        Formatted payload dictionary.
    """
    extra_text = json.dumps(extra or {}, indent=2)

    # Discord webhooks
    if "discord" in url.lower():
        return {
            "content": f"**{title}**\n{text}",
            "embeds": [
                {
                    "title": "Details",
                    "description": f"```json\n{extra_text}\n```",
                    "color": 5814783,  # Blue color
                }
            ],
        }

    # Slack webhooks (default)
    return {
        "text": f"*{title}*\n{text}",
        "attachments": [{"text": extra_text}],
    }
