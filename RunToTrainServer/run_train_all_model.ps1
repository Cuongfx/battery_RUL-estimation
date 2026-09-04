# ============================================================
# run_train_all_model.ps1  (SLURM / RPTU Elwetritsch)
# Training script selection, submitted as one SLURM array job
# (one task per folder, one GPU each):
#   1 = train_clf_bml_V2.py             (default)
#   2 = train_clf_bml_transformer_V2.py
#   3 = train_clf_es_bml_V2.py          (sparse CMA-ES, with pretrain checkpoint)
#
# Requires content_bml/<folder> to already exist (run run_dataset.ps1 first).
#
# Usage (from the login node, run from anywhere):
#   pwsh ./RunToTrainServer/run_train_all_model.ps1
#       (no -TrainScript -> prompts interactively for 1, 2 or 3)
#   pwsh ./RunToTrainServer/run_train_all_model.ps1 -TrainScript 1
#   pwsh ./RunToTrainServer/run_train_all_model.ps1 -TrainScript 2
#   pwsh ./RunToTrainServer/run_train_all_model.ps1 -TrainScript 3
#       (TrainScript 3 needs a checkpoint already trained via -TrainScript 1
#        at ./checkpoints_clf_bml_<folder>/best_clf_bml.pt)
# ============================================================
param(
    [ValidateSet("1","2","3")]
    [string]$TrainScript = "",
    [string[]]$Folders   = @("HUST", "MATR", "LFP"),
    [string]$Account     = "default",
    [string]$Qos         = "u-rptu",
    [string]$Gpu         = "v100:1",
    [int]$Cpus           = 4,
    [string]$Mem         = "32G",
    [string]$Time        = "10:00:00",
    [string]$CondaEnv    = "battery_ml",
    [int]$Throttle       = 3,
    [switch]$DryRun
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_SlurmCommon.ps1")

if ($TrainScript -eq "") {
    Write-Host ""
    Write-Host "Select training script:" -ForegroundColor Cyan
    Write-Host "  1) train_clf_bml_V2.py             (classic BML)"  -ForegroundColor White
    Write-Host "  2) train_clf_bml_transformer_V2.py (transformer)"  -ForegroundColor White
    Write-Host "  3) train_clf_es_bml_V2.py          (sparse CMA-ES)" -ForegroundColor White
    $TrainScript = Read-Host "Enter 1, 2 or 3 [default: 1]"
    if ($TrainScript -eq "") { $TrainScript = "1" }
}

if ($TrainScript -eq "2") {
    $trainScriptName = "train_clf_bml_transformer_V2.py"
    $outSuffix = "_tf"
} elseif ($TrainScript -eq "3") {
    $trainScriptName = "train_clf_es_bml_V2.py"
    $outSuffix = "_es"
} else {
    $trainScriptName = "train_clf_bml_V2.py"
    $outSuffix = ""
}

$slurm = @{
    Account = $Account; Qos = $Qos; Gpu = $Gpu; Cpus = $Cpus
    Mem = $Mem; Time = $Time; CondaEnv = $CondaEnv
}

$commands = foreach ($folder in $Folders) {
    $cmd = "python -u $trainScriptName --content_dir `"./content_bml/$folder`"" +
           " --output_dir `"./checkpoints_clf_bml_${folder}${outSuffix}`""
    if ($TrainScript -eq "3") {
        $cmd += " --pretrain_ckpt `"./checkpoints_clf_bml_$folder/best_clf_bml.pt`""
    }
    $cmd
}

$jobDir     = New-JobDir -RepoRoot $RepoRoot -Name "train_model$outSuffix"
$paramsFile = Join-Path $jobDir "params.txt"
Write-LfFile -Path $paramsFile -Lines $commands

Show-JobPlan -Title "Training ($trainScriptName)" `
             -Count $commands.Count -Slurm $slurm -Throttle $Throttle -JobDir $jobDir
Write-Host ("Folders     : {0}" -f ($Folders -join ", ")) -ForegroundColor White

$jobScript = New-ArrayJobScript -JobName "bml_model$outSuffix" -RepoRoot $RepoRoot `
                                -JobDir $jobDir -ParamsFile $paramsFile -Slurm $slurm

Submit-ArrayJob -ScriptPath $jobScript -Count $commands.Count -Throttle $Throttle -DryRun:$DryRun
