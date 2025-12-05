# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Test and demo script for notifications."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.notify import send


def test_notification() -> bool:
    """Send a test notification.

    Returns:
        True if notification was sent successfully.
    """
    title = "🧪 AutoMLOps Test Notification"
    message = "If you see this, notifications are working!"
    extra = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "test": True,
        "source": "test_notify.py",
    }

    print(f"Sending test notification...")
    print(f"  Title: {title}")
    print(f"  Message: {message}")
    print(f"  Extra: {extra}")

    result = send(title, message, extra)

    if result:
        print("✅ Notification sent successfully!")
    else:
        print("❌ Notification failed. Check your webhook URL in .env")
        print("\nTo set up Discord notifications:")
        print("1. Go to your Discord server")
        print("2. Server Settings > Integrations > Webhooks")
        print("3. Create a new webhook")
        print("4. Copy the webhook URL")
        print("5. Add to .env: DISCORD_WEBHOOK_URL=<your-url>")

    return result


def demo_training_complete() -> bool:
    """Demo notification for training completion."""
    title = "🎉 Training Complete"
    message = "Model training finished successfully"
    extra = {
        "model": "model",
        "version": 3,
        "metrics": {
            "test_auc": 0.9823,
            "test_acc": 0.9567,
        },
        "data_source": "real",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    return send(title, message, extra)


def demo_drift_detected() -> bool:
    """Demo notification for drift detection."""
    title = "⚠️ Data Drift Detected"
    message = "Significant drift found in production data"
    extra = {
        "drift_score_ks": 0.23,
        "drift_score_psi": 0.31,
        "affected_features": ["V14", "V17", "Amount"],
        "recommendation": "Trigger retraining",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    return send(title, message, extra)


def demo_promotion() -> bool:
    """Demo notification for model promotion."""
    title = "🚀 Model Promoted to Production"
    message = "Canary model promoted after successful evaluation"
    extra = {
        "model": "model",
        "from_version": 2,
        "to_version": 3,
        "canary_auc": 0.9823,
        "production_auc": 0.9756,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    return send(title, message, extra)


def demo_rollback() -> bool:
    """Demo notification for model rollback."""
    title = "🔙 Model Rolled Back"
    message = "Production model reverted due to performance issues"
    extra = {
        "model": "model",
        "rolled_back_to": 2,
        "reason": "Error rate exceeded threshold",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    return send(title, message, extra)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test notifications")
    parser.add_argument(
        "--demo",
        choices=["test", "training", "drift", "promotion", "rollback", "all"],
        default="test",
        help="Notification type to send",
    )
    args = parser.parse_args()

    demos = {
        "test": test_notification,
        "training": demo_training_complete,
        "drift": demo_drift_detected,
        "promotion": demo_promotion,
        "rollback": demo_rollback,
    }

    if args.demo == "all":
        results = []
        for name, func in demos.items():
            print(f"\n--- {name.upper()} ---")
            results.append(func())
        print(f"\n\nResults: {sum(results)}/{len(results)} successful")
    else:
        demos[args.demo]()


if __name__ == "__main__":
    main()
