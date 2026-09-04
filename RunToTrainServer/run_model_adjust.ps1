# ============================================================
# run_model_adjust.ps1  (SLURM / RPTU Elwetritsch)
# Grid search: cnn_dim x gru_dim x gru_layers  (4x4x3 = 48 combos)
# Each combo trained 3 times  ->  144 tasks, submitted as one SLURM
# array job and run -Throttle at a time.
#
# Usage (from the login node, run from anywhere):
#   pwsh ./RunToTrainServer/run_model_adjust.ps1
#   pwsh ./RunToTrainServer/run_model_adjust.ps1 -ContentDir "./content_bml/HUST"
#   pwsh ./RunToTrainServer/run_model_adjust.ps1 -Throttle 8 -DryRun
# ============================================================
param(
    [string]$ContentDir = "./content_bml/LFP",
    [string]$Account    = "default",
    [string]$Qos        = "u-rptu",
    [string]$Gpu        = "v100:1",
    [int]$Cpus          = 4,
    [string]$Mem        = "32G",
    [string]$Time       = "04:00:00",
    [string]$CondaEnv   = "battery_ml",
    [int]$Throttle      = 4,
    [switch]$DryRun
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_SlurmCommon.ps1")

$cnnDims     = @(8, 16, 32, 64)
$gruDims     = @(8, 16, 32, 64)
$gruLayers   = @(1, 2, 3)
$nRuns       = 3
$datasetName = Split-Path -Path $ContentDir -Leaf
$baseOut     = "./checkpoints_clf_bml_ModelAdjust_$datasetName"

$slurm = @{
    Account = $Account; Qos = $Qos; Gpu = $Gpu; Cpus = $Cpus
    Mem = $Mem; Time = $Time; CondaEnv = $CondaEnv
}

$commands = [System.Collections.Generic.List[string]]::new()
foreach ($cnn in $cnnDims) {
    foreach ($grud in $gruDims) {
        foreach ($grul in $gruLayers) {
            for ($r = 1; $r -le $nRuns; $r++) {
                $outDir = "$baseOut/CNN_GRUd_GRUl_${cnn}_${grud}_${grul}_run${r}"
                $commands.Add(
                    "python -u train_clf_bml_V2.py --content_dir `"$ContentDir`"" +
                    " --output_dir `"$outDir`"" +
                    " --cnn_dim $cnn --gru_dim $grud --gru_layers $grul")
            }
        }
    }
}

$jobDir     = New-JobDir -RepoRoot $RepoRoot -Name "model_adjust_$datasetName"
$paramsFile = Join-Path $jobDir "params.txt"
Write-LfFile -Path $paramsFile -Lines $commands

Show-JobPlan -Title "Model Grid Search ($datasetName)  --  $($commands.Count) runs total" `
             -Count $commands.Count -Slurm $slurm -Throttle $Throttle -JobDir $jobDir
Write-Host ("Content dir : {0}" -f $ContentDir) -ForegroundColor White
Write-Host ("Output base : {0}" -f $baseOut)    -ForegroundColor White

$jobScript = New-ArrayJobScript -JobName "bml_adjust_$datasetName" -RepoRoot $RepoRoot `
                                -JobDir $jobDir -ParamsFile $paramsFile -Slurm $slurm

Submit-ArrayJob -ScriptPath $jobScript -Count $commands.Count -Throttle $Throttle -DryRun:$DryRun
