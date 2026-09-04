# ============================================================
# run_train_all_default.ps1  (SLURM / RPTU Elwetritsch)
# Phase 2 — Training with default hyperparameters: train_clf_bml_V2.py
# over each folder, submitted as one SLURM array job (one task per folder,
# one GPU each).
#
# Requires content_bml/<folder> to already exist (run run_dataset.ps1 first).
#
# Usage (from the login node, run from anywhere):
#   pwsh ./RunToTrainServer/run_train_all_default.ps1
#   pwsh ./RunToTrainServer/run_train_all_default.ps1 -DryRun
#   pwsh ./RunToTrainServer/run_train_all_default.ps1 -Gpu a100:1 -Time 20:00:00
# ============================================================
param(
    [string[]]$Folders = @("HUST", "LFP", "MATR"),
    [string]$Account   = "default",
    [string]$Qos       = "u-rptu",
    [string]$Gpu       = "v100:1",
    [int]$Cpus         = 4,
    [string]$Mem       = "32G",
    [string]$Time      = "10:00:00",
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
    "python -u train_clf_bml_V2.py --content_dir `"./content_bml/$folder`" --output_dir `"./checkpoints_clf_bml_$folder`""
}

$jobDir     = New-JobDir -RepoRoot $RepoRoot -Name "train_default"
$paramsFile = Join-Path $jobDir "params.txt"
Write-LfFile -Path $paramsFile -Lines $commands

Show-JobPlan -Title "PHASE 2: Training (train_clf_bml_V2.py)" `
             -Count $commands.Count -Slurm $slurm -Throttle $Throttle -JobDir $jobDir
Write-Host ("Folders     : {0}" -f ($Folders -join ", ")) -ForegroundColor White

$jobScript = New-ArrayJobScript -JobName "bml_default" -RepoRoot $RepoRoot `
                                -JobDir $jobDir -ParamsFile $paramsFile -Slurm $slurm

Submit-ArrayJob -ScriptPath $jobScript -Count $commands.Count -Throttle $Throttle -DryRun:$DryRun
