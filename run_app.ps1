param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) { throw "Python was not found. Install Python 3.11+ and reopen PowerShell." }

if (-not (Test-Path (Join-Path $root "backend\requirements.txt"))) {
    throw "backend\requirements.txt was not found. Run this script from the repository root."
}

$env:MOOD_DATABASE_URL = if ($env:MOOD_DATABASE_URL) { $env:MOOD_DATABASE_URL } else { "sqlite:///./mood_journal.db" }
$env:MOOD_ENABLE_EMOTION_MODEL = if ($env:MOOD_ENABLE_EMOTION_MODEL) { $env:MOOD_ENABLE_EMOTION_MODEL } else { "false" }

Write-Host "Starting Mood Journal at http://${BindHost}:${Port}" -ForegroundColor Cyan
Write-Host "Database: local SQLite; transformer enabled: $env:MOOD_ENABLE_EMOTION_MODEL"
Write-Host "Press Ctrl+C to stop."

$backend = Join-Path $root "backend"
$logDirectory = Join-Path $root ".runtime"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stdoutLog = Join-Path $logDirectory "backend.out.log"
$stderrLog = Join-Path $logDirectory "backend.err.log"
$arguments = "-m uvicorn app.main:app --app-dir `"$backend`" --host $BindHost --port $Port"
$process = Start-Process -FilePath $python.Source -ArgumentList $arguments -WorkingDirectory $root `
    -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
try {
    $health = $null
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            $health = Invoke-RestMethod "http://${BindHost}:${Port}/api/health" -ErrorAction Stop
            break
        } catch {
            if ($attempt -eq 9) {
                $details = if (Test-Path $stderrLog) { Get-Content $stderrLog -Raw } else { $_.Exception.Message }
                throw "Backend did not start on port ${Port}: $details"
            }
        }
    }
    if ($health.status -ne "ok") { throw "Health endpoint returned an unexpected response." }
    if (-not $NoBrowser) { Start-Process "http://${BindHost}:${Port}/" }
    Wait-Process -Id $process.Id
}
finally {
    if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force }
}
