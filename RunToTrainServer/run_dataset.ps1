# ============================================================
# run_dataset.ps1  (SLURM / RPTU Elwetritsch)
# Phase 1 — Feature generation: gen_feature_bml_v2.py over each subfolder
# of Raw/Raw_BML, submitted as one SLURM array job (one task per folder).
#
# Feature export is CPU-only, so no GPU is requested by default.
#
# Usage (from the login node, run from anywhere):
#   pwsh ./RunToTrainServer/run_dataset.ps1
#   pwsh ./RunToTrainServer/run_dataset.ps1 -DryRun
#   pwsh ./RunToTrainServer/run_dataset.ps1 -Mem 64G -Time 04:00:00
# ============================================================
param(
    [string[]]$Folders = @("HUST", "LFP", "MATR"),
    [string]$Account   = "default",
    [string]$Qos       = "u-rptu",
    [string]$Gpu       = "",
    [int]$Cpus         = 4,
    [string]$Mem       = "32G",
    [string]$Time      = "04:00:00",
    [string]$CondaEnv  = "battery_ml",
    [int]$Throttle     = 3,
    [switch]$DryRun
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_SlurmCommon.ps1")

$slurm = @{
    Account = $Account; Qos = $Qos; Gpu = $Gpu; Cpus = $Cpus
    Mem = $Mem; Time = $Time; CondaEnv = $CondaEnv
}

$commands = foreach ($folder in $Folders) {
    "python -u gen_feature_bml_v2.py --data_dir `"./Raw/Raw_BML/$folder`" --out_dir `"./content_bml/$folder`""
}

$jobDir     = New-JobDir -RepoRoot $RepoRoot -Name "dataset"
$paramsFile = Join-Path $jobDir "params.txt"
Write-LfFile -Path $paramsFile -Lines $commands

Show-JobPlan -Title "PHASE 1: Feature generation (gen_feature_bml_v2.py)" `
             -Count $commands.Count -Slurm $slurm -Throttle $Throttle -JobDir $jobDir
Write-Host ("Folders     : {0}" -f ($Folders -join ", ")) -ForegroundColor White

$jobScript = New-ArrayJobScript -JobName "bml_dataset" -RepoRoot $RepoRoot `
                                -JobDir $jobDir -ParamsFile $paramsFile -Slurm $slurm

Submit-ArrayJob -ScriptPath $jobScript -Count $commands.Count -Throttle $Throttle -DryRun:$DryRun
