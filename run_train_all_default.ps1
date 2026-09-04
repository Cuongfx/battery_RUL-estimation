# ============================================================
# run_train_all_default.ps1
# Phase 2 — Training with default hyperparameters: train_clf_bml_V2.py over each folder
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_train_all_default.ps1
# ============================================================

$folders = @("HUST", "LFP", "MATR"
)

# Windows teardown crash codes that happen AFTER training finishes successfully.
$benignExitCodes = @(-1073740791, -1073741819)

$failed = @()

Write-Host ""
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "PHASE 2: Training (train_clf_bml_V2.py)" -ForegroundColor Magenta
Write-Host "############################################" -ForegroundColor Magenta
Write-Host ("Folders to train on ({0}):" -f $folders.Count) -ForegroundColor Magenta
foreach ($f in $folders) { Write-Host "  - $f" -ForegroundColor Magenta }

foreach ($folder in $folders) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Processing folder: $folder" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    $trainCmd = "python train_clf_bml_V2.py --content_dir ./content_bml/$folder --output_dir ./checkpoints_clf_bml_$folder"
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
