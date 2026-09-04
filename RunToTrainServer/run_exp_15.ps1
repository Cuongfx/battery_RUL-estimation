# ============================================================
# run_exp_15.ps1  (SLURM / RPTU Elwetritsch)
# Grid search: N_EARLY x N_RANDOM  (5x5 = 25 combos)
# Each combo trained 5 times  ->  125 tasks, submitted as one SLURM
# array job and run -Throttle at a time.
#
# Usage (from the login node, run from anywhere):
#   pwsh ./RunToTrainServer/run_exp_15.ps1
#   pwsh ./RunToTrainServer/run_exp_15.ps1 -ContentDir "./content_bml/HUST"
#   pwsh ./RunToTrainServer/run_exp_15.ps1 -Throttle 8 -DryRun
# ============================================================
param(
    [string]$ContentDir = "./content_bml/MATR",
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

$earlyValues  = @(2, 4, 6, 8, 10)
$randomValues = @(2, 4, 6, 8, 10)
$nRuns        = 5
$datasetName  = Split-Path -Path $ContentDir -Leaf
$baseOut      = "./checkpoints_clf_bml_EXP_$datasetName"

$slurm = @{
    Account = $Account; Qos = $Qos; Gpu = $Gpu; Cpus = $Cpus
    Mem = $Mem; Time = $Time; CondaEnv = $CondaEnv
}

$commands = [System.Collections.Generic.List[string]]::new()
foreach ($early in $earlyValues) {
    foreach ($random in $randomValues) {
        for ($r = 1; $r -le $nRuns; $r++) {
            $outDir = "$baseOut/E_R_${early}_${random}_run${r}"
            $commands.Add(
                "python -u train_clf_bml_V2.py --content_dir `"$ContentDir`"" +
                " --output_dir `"$outDir`"" +
                " --n_early $early --n_random $random")
        }
    }
}

$jobDir     = New-JobDir -RepoRoot $RepoRoot -Name "exp_15_$datasetName"
$paramsFile = Join-Path $jobDir "params.txt"
Write-LfFile -Path $paramsFile -Lines $commands

Show-JobPlan -Title "EXP_$datasetName Grid Search  --  $($commands.Count) runs total" `
             -Count $commands.Count -Slurm $slurm -Throttle $Throttle -JobDir $jobDir
Write-Host ("Content dir : {0}" -f $ContentDir) -ForegroundColor White
Write-Host ("Output base : {0}" -f $baseOut)    -ForegroundColor White

$jobScript = New-ArrayJobScript -JobName "bml_exp15_$datasetName" -RepoRoot $RepoRoot `
                                -JobDir $jobDir -ParamsFile $paramsFile -Slurm $slurm

Submit-ArrayJob -ScriptPath $jobScript -Count $commands.Count -Throttle $Throttle -DryRun:$DryRun
