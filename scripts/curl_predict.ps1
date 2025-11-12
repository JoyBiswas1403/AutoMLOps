$ErrorActionPreference = "Stop"

$features = 20
$instance = @(for ($i=0; $i -lt $features; $i++) { 0.0 })
$bodyObj = @{ instances = ,$instance }
$body = $bodyObj | ConvertTo-Json -Depth 5

Write-Host "POST /predict with one zero-vector instance ($features features)"
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
