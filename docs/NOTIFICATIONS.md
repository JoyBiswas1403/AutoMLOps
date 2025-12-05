# Discord Notifications Setup

AutoMLOps supports Discord (and Slack) webhooks for pipeline notifications.

## Quick Setup (Discord)

### 1. Create a Discord Webhook

1. Open your Discord server
2. Go to **Server Settings** → **Integrations** → **Webhooks**
3. Click **New Webhook**
4. Name it "AutoMLOps" (or any name)
5. Select the channel for notifications
6. Click **Copy Webhook URL**

### 2. Configure AutoMLOps

Add the webhook URL to your `.env` file:

```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_TOKEN
```

### 3. Test the Notification

```bash
# Test notification
python scripts/test_notify.py --demo test

# Demo all notification types
python scripts/test_notify.py --demo all
```

## Notification Types

| Event | Title | When |
|-------|-------|------|
| Training Complete | 🎉 Training Complete | After model training |
| Drift Detected | ⚠️ Data Drift Detected | During drift monitoring |
| Model Promoted | 🚀 Model Promoted | Canary → Production |
| Model Rollback | 🔙 Model Rolled Back | On performance issues |

## Example Discord Message

```
🎉 Training Complete
Model training finished successfully

{
  "model": "model",
  "version": 3,
  "metrics": {
    "test_auc": 0.9823,
    "test_acc": 0.9567
  }
}
```

## Alternative: Slack

For Slack, use:
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

## Troubleshooting

**Notification not sending?**
- Check webhook URL is correct
- Ensure `.env` is loaded
- Test with: `python scripts/test_notify.py`

**Wrong format?**
- Discord and Slack have different payload formats
- The notify module auto-detects based on URL
