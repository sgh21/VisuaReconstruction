param(
    [string]$LogDir = "runs/clean_prior",
    [string]$CondaEnv = "VisionNet",
    [int]$Port = 6006
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host ">>> tensorboard --logdir $LogDir --port $Port" -ForegroundColor Cyan
& conda run -n $CondaEnv tensorboard --logdir $LogDir --port $Port
if ($LASTEXITCODE -ne 0) {
    throw "TensorBoard failed with exit code $LASTEXITCODE"
}
