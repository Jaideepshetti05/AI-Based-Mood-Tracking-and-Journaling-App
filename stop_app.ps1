param([int]$Port = 8000)
$connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $connections) { Write-Host "No Mood Journal process is listening on port $Port."; exit 0 }
foreach ($connection in $connections) {
    Stop-Process -Id $connection.OwningProcess -Force
    Write-Host "Stopped process $($connection.OwningProcess) on port $Port."
}
