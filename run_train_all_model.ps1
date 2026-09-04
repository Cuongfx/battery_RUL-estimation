# ============================================================
# run_train_all_model.ps1
# Training script selection
#   1 = train_clf_bml_V2.py             (default)
#   2 = train_clf_bml_transformer_V2.py
#   3 = train_clf_es_bml_V2.py          (sparse CMA-ES, with pretrain checkpoint)
#
# Requires content_bml/<folder> to already exist (run run_dataset.ps1 first).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_train_all_model.ps1
#       (no -TrainScript -> prompts interactively for 1, 2 or 3)
#   powershell -ExecutionPolicy Bypass -File .\run_train_all_model.ps1 -TrainScript 1
#   powershell -ExecutionPolicy Bypass -File .\run_train_all_model.ps1 -TrainScript 2
#   powershell -ExecutionPolicy Bypass -File .\run_train_all_model.ps1 -TrainScript 3
#       (TrainScript 3 needs a checkpoint already trained via -TrainScript 1
#        at ./checkpoints_clf_bml_<folder>/best_clf_bml.pt)
# ============================================================
param(
    [ValidateSet("1","2","3")]
    [string]$TrainScript = ""
)

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

Write-Host ""
Write-Host "Using training script: $trainScriptName" -ForegroundColor Green

# ============================================================
# Folders + benign exit codes
# ============================================================
# $folders = @("HUST", "ISU", "LFP", "MATR", "NMC", "Tongji", "ZN-coin")
$folders = @("HUST", "MATR", "LFP")

# Windows teardown crash codes that happen AFTER training finishes successfully.
$benignExitCodes = @(-1073740791, -1073741819)

$failed = @()

# ============================================================
# Per-folder loop: dataset build + training
# ============================================================
Write-Host ""
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "Training ($trainScriptName)" -ForegroundColor Magenta
Write-Host "############################################" -ForegroundColor Magenta
Write-Host ("Folders to train on ({0}):" -f $folders.Count) -ForegroundColor Magenta
foreach ($f in $folders) { Write-Host "  - $f" -ForegroundColor Magenta }

foreach ($folder in $folders) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Processing folder: $folder" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    if ($TrainScript -eq "3") {
        $trainCmd = "python $trainScriptName --content_dir ./content_bml/$folder --output_dir ./checkpoints_clf_bml_${folder}${outSuffix} --pretrain_ckpt ./checkpoints_clf_bml_$folder/best_clf_bml.pt"
    } else {
        $trainCmd = "python $trainScriptName --content_dir ./content_bml/$folder --output_dir ./checkpoints_clf_bml_${folder}${outSuffix}"
    }
    Write-Host ""
    Write-Host ">> $trainCmd" -ForegroundColor Yellow
    Invoke-Expression $trainCmd
    $trainExit = $LASTEXITCODE
    if ($trainExit -ne 0 -and -not ($benignExitCodes -contains $trainExit)) {
        Write-Host ("ERROR: train step failed for {0} (exit code {1}) - skipping to next folder" -f $folder, $trainExit) -ForegroundColor Red
        $failed += ("{0} [train, exit {1}]" -f $folder, $trainExit)
        continue
    }
    if ($benignExitCodes -contains $trainExit) {
        Write-Host "Note: train returned benign teardown exit code $trainExit - output was saved, continuing" -ForegroundColor DarkYellow
    }

    Write-Host ""
    Write-Host "Done: $folder" -ForegroundColor Green
}

# ── Final summary ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($failed.Count -eq 0) {
    Write-Host "All steps completed successfully." -ForegroundColor Green
} else {
    Write-Host "Completed with failures:" -ForegroundColor Yellow
    foreach ($f in $failed) { Write-Host "  - $f" -ForegroundColor Yellow }
}
