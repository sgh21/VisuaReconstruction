param(
    [ValidateSet("train", "test", "all")]
    [string]$Mode = "all",

    [string[]]$Models = @("mae", "lraspp", "fcn"),

    [string]$DataRoot = "dataset",
    [string]$CondaEnv = "VisionNet",
    [int]$Epochs = 200,
    [int]$NumWorkers = 6,
    [double]$LearningRate = 1e-4,
    [int]$LimitSuction = 0,
    [double]$OverlayAlpha = 0.35,

    [switch]$EngineeringSize
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$ModelConfigs = @{
    "lraspp" = @{
        Model = "lraspp_mobilenet_v3_large"
        Run = "lraspp_mobilenet_v3_large_official"
        BatchSize = 48
        Degradation = "blur_noise_erase"
        EngineeringImageSize = @("512", "288")
    }
    "fcn" = @{
        Model = "fcn_resnet50"
        Run = "fcn_resnet50_official"
        BatchSize = 16
        Degradation = "blur_noise_erase"
        EngineeringImageSize = @("512", "288")
    }
    "mae" = @{
        Model = "mae_vit_b_16"
        Run = "mae_vit_b_16_official"
        BatchSize = 48
        Degradation = "blur_noise_patchmask"
        EngineeringImageSize = @("224", "224")
    }
}

function Invoke-Python {
    param([string[]]$Arguments)

    Write-Host ""
    Write-Host ">>> conda run -n $CondaEnv python $($Arguments -join ' ')" -ForegroundColor Cyan
    & conda run -n $CondaEnv python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Get-TrainArgs {
    param([hashtable]$Config)

    $RunDir = "runs/clean_prior/$($Config.Run)"
    $Args = @(
        "scripts/train_clean_prior.py",
        "--data-root", $DataRoot,
        "--model", $Config.Model,
        "--weights", "default",
        "--degradation", $Config.Degradation,
        "--epochs", "$Epochs",
        "--batch-size", "$($Config.BatchSize)",
        "--lr", "$LearningRate",
        "--num-workers", "$NumWorkers",
        "--run-dir", $RunDir
    )

    if ($EngineeringSize) {
        $Args += @("--image-size", $Config.EngineeringImageSize[0], $Config.EngineeringImageSize[1])
    } else {
        $Args += @("--official-image-size")
    }

    return $Args
}

function Get-TestArgs {
    param([hashtable]$Config)

    $RunDir = "runs/clean_prior/$($Config.Run)"
    $Checkpoint = "$RunDir/best.pt"
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Checkpoint not found: $Checkpoint. Run training first or use -Mode train."
    }

    $Args = @(
        "scripts/test_clean_prior.py",
        "--data-root", $DataRoot,
        "--checkpoint", $Checkpoint,
        "--output-dir", "$RunDir/test_outputs",
        "--overlay-alpha", "$OverlayAlpha",
        "--num-workers", "0"
    )

    if ($LimitSuction -gt 0) {
        $Args += @("--limit-suction", "$LimitSuction")
    }

    return $Args
}

foreach ($Alias in $Models) {
    $Key = $Alias.ToLower()
    if (-not $ModelConfigs.ContainsKey($Key)) {
        throw "Unknown model alias '$Alias'. Use one or more of: lraspp, fcn, mae."
    }

    $Config = $ModelConfigs[$Key]
    Write-Host ""
    Write-Host "===== $Key -> $($Config.Model) =====" -ForegroundColor Green

    if ($Mode -eq "train" -or $Mode -eq "all") {
        Invoke-Python -Arguments (Get-TrainArgs -Config $Config)
    }

    if ($Mode -eq "test" -or $Mode -eq "all") {
        Invoke-Python -Arguments (Get-TestArgs -Config $Config)
    }
}

Write-Host ""
Write-Host "Done. Results are under runs/clean_prior/." -ForegroundColor Green
