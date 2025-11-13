param(
  [int]$Percent = 10,
  [int]$Requests = 50
)

Write-Host "Setting canary traffic to $Percent%"
$null = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/traffic" -Body $Percent -ContentType 'application/json'

Write-Host "Sending $Requests requests to /predict"
for ($i=0; $i -lt $Requests; $i++) {
  $features = 20
  $instance = @(for ($j=0; $j -lt $features; $j++) { 0.0 })
  $bodyObj = @{ instances = ,$instance }
  $body = $bodyObj | ConvertTo-Json -Depth 5
  try {
    $resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" -ContentType "application/json" -Body $body
    $route = $resp.route
    Write-Host "[$i] route=$route"
  } catch {
    Write-Host "[$i] error"
  }
}
