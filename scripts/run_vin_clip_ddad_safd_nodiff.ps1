param(
    [string]$Config = "cfgs/Vin_CLIP_DDAD_SAFD_NoDiff.yaml"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$steps = @(
    "a",
    "b",
    "clip",
    "cache_fusion",
    "fusion",
    "eval_all"
)

foreach ($mode in $steps) {
    Write-Host "==> Running mode $mode with $Config"
    & python main.py --config $Config --mode $mode
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$mode' failed with exit code $LASTEXITCODE."
    }
}

Write-Host "==> Completed CLIP + DDAD SAFD pipeline without diffusion."
