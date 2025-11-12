$ErrorActionPreference = "Stop"

Write-Host "Simulating data drift and triggering retraining if thresholds exceeded..."
docker compose run --rm monitor python drift/detect_drift.py --simulate
