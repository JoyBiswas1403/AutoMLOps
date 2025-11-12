import os
import json
import requests

WEBHOOK_ENV_KEYS = [
    "SLACK_WEBHOOK_URL",
    "WEBHOOK_URL",
]


def send(title: str, text: str, extra: dict | None = None) -> bool:
    url = None
    for k in WEBHOOK_ENV_KEYS:
        if os.getenv(k):
            url = os.getenv(k)
            break
    if not url:
        return False
    payload = {
        "text": f"*{title}*\n{text}",
        "attachments": [
            {"text": json.dumps(extra or {}, indent=2)}
        ],
    }
    try:
        r = requests.post(url, json=payload, timeout=5)
        return r.ok
    except Exception:
        return False
