# ============================================================
# _SlurmCommon.ps1
# Shared SLURM helpers for the RunToTrainServer scripts. Dot-sourced
# by them, never run on its own.
#
# Every cluster-specific line lives in $JobPreamble below — change the
# module name or the conda bootstrap path here once, not in five files.
# ============================================================

$script:JobPreamble = @'
module load nvidia/latest
source ~/miniconda3/etc/profile.d/conda.sh
conda activate __CONDA_ENV__
'@

$script:JobBody = @'
set -euo pipefail
cd "__REPO_ROOT__"

__PREAMBLE__

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "__PARAMS_FILE__")
if [ -z "$CMD" ]; then
    echo "No command on line ${SLURM_ARRAY_TASK_ID} of __PARAMS_FILE__" >&2
    exit 1
fi
echo ">> [task ${SLURM_ARRAY_TASK_ID}] $CMD"
eval "$CMD"
'@

function New-JobDir {
    param(
        [Parameter(Mandatory)][string]$RepoRoot,
        [Parameter(Mandatory)][string]$Name
    )
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $dir = Join-Path $RepoRoot "jobs/${Name}_$stamp"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "logs") | Out-Null
    return $dir
}

function Write-LfFile {
    # sbatch rejects a job script with CRLF line endings, and Set-Content
    # follows the platform, so the bytes are written explicitly.
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][AllowEmptyCollection()][AllowEmptyString()][string[]]$Lines
    )
    [System.IO.File]::WriteAllText($Path, ($Lines -join "`n") + "`n")
}

function New-ArrayJobScript {
    param(
        [Parameter(Mandatory)][string]$JobName,
        [Parameter(Mandatory)][string]$RepoRoot,
        [Parameter(Mandatory)][string]$JobDir,
        [Parameter(Mandatory)][string]$ParamsFile,
        [Parameter(Mandatory)][hashtable]$Slurm
    )

    # Paths are embedded into a bash script, so they always use forward
    # slashes even when this is dry-run from Windows.
    $repo   = $RepoRoot.Replace('\', '/')
    $params = $ParamsFile.Replace('\', '/')

    $header = @(
        "#!/bin/bash"
        "#SBATCH --job-name=$JobName"
        "#SBATCH --account=$($Slurm.Account)"
        "#SBATCH --qos=$($Slurm.Qos)"
        "#SBATCH --ntasks=1"
        "#SBATCH --cpus-per-task=$($Slurm.Cpus)"
        "#SBATCH --mem=$($Slurm.Mem)"
        "#SBATCH --time=$($Slurm.Time)"
    )
    if ($Slurm.Gpu) { $header += "#SBATCH --gres=gpu:$($Slurm.Gpu)" }
    $header += @(
        "#SBATCH --output=$repo/logs/%x_%A_%a.out"
        "#SBATCH --error=$repo/logs/%x_%A_%a.err"
        ""
    )

    $preamble = $script:JobPreamble.Replace("__CONDA_ENV__", $Slurm.CondaEnv)
    $body = $script:JobBody.
        Replace("__REPO_ROOT__", $repo).
        Replace("__PREAMBLE__", $preamble).
        Replace("__PARAMS_FILE__", $params)

    $path = Join-Path $JobDir "job.sh"
    Write-LfFile -Path $path -Lines ($header + $body.Split("`n"))
    return $path
}

function Submit-ArrayJob {
    param(
        [Parameter(Mandatory)][string]$ScriptPath,
        [Parameter(Mandatory)][int]$Count,
        [int]$Throttle = 0,
        [switch]$DryRun
    )

    $spec = if ($Throttle -gt 0) { "1-$Count%$Throttle" } else { "1-$Count" }

    Write-Host ""
    Write-Host ">> sbatch --array=$spec $ScriptPath" -ForegroundColor Yellow

    if ($DryRun) {
        Write-Host "-DryRun: job script written, nothing submitted." -ForegroundColor DarkYellow
        return
    }

    & sbatch --array=$spec $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: sbatch failed (exit $LASTEXITCODE)" -ForegroundColor Red
        return
    }

    Write-Host ""
    Write-Host "Submitted. Track it with:" -ForegroundColor Green
    Write-Host "  squeue -u `$USER" -ForegroundColor White
    Write-Host "  tail -f logs/<job-name>_<jobid>_<task>.out" -ForegroundColor White
}

function Show-JobPlan {
    param(
        [Parameter(Mandatory)][string]$Title,
        [Parameter(Mandatory)][int]$Count,
        [Parameter(Mandatory)][hashtable]$Slurm,
        [Parameter(Mandatory)][int]$Throttle,
        [Parameter(Mandatory)][string]$JobDir
    )
    Write-Host ""
    Write-Host "############################################" -ForegroundColor Magenta
    Write-Host $Title -ForegroundColor Magenta
    Write-Host "############################################" -ForegroundColor Magenta
    Write-Host ("Array tasks : {0} (max {1} at a time)" -f $Count, $Throttle) -ForegroundColor White
    Write-Host ("Resources   : {0} cpus, {1}, {2}, gpu={3}" -f `
        $Slurm.Cpus, $Slurm.Mem, $Slurm.Time, ($(if ($Slurm.Gpu) { $Slurm.Gpu } else { "none" }))) -ForegroundColor White
    Write-Host ("Conda env   : {0}" -f $Slurm.CondaEnv) -ForegroundColor White
    Write-Host ("Job dir     : {0}" -f $JobDir) -ForegroundColor White
}
