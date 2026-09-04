# ============================================================
# run_dataset.ps1
# Phase 1 — Feature generation: gen_feature_bml_v2.py over each subfolder of Raw/Raw_BML
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_dataset.ps1
# ============================================================

$rawFolders = @("HUST", "LFP", "MATR"
)

# Windows teardown crash codes that happen AFTER the step finishes successfully.
$benignExitCodes = @(-1073740791, -1073741819)

$failed = @()

Write-Host ""
Write-Host "############################################" -ForegroundColor Magenta
Write-Host "PHASE 1: Feature generation (gen_feature_bml_v2.py)" -ForegroundColor Magenta
Write-Host "############################################" -ForegroundColor Magenta
Write-Host ("Subfolders in Raw/Raw_BML to process ({0}):" -f $rawFolders.Count) -ForegroundColor Magenta
foreach ($rf in $rawFolders) { Write-Host "  - $rf" -ForegroundColor Magenta }

foreach ($folder in $rawFolders) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Generating features for: $folder" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    $genCmd = "python gen_feature_bml_v2.py --data_dir ./Raw/Raw_BML/$folder --out_dir ./content_bml/$folder"
    Write-Host ""
    Write-Host ">> $genCmd" -ForegroundColor Yellow
    Invoke-Expression $genCmd
    $genExit = $LASTEXITCODE
    if ($genExit -ne 0 -and -not ($benignExitCodes -contains $genExit)) {
        Write-Host ("ERROR: gen_feature step failed for {0} (exit code {1}) - skipping to next folder" -f $folder, $genExit) -ForegroundColor Red
        $failed += ("{0} [gen_feature, exit {1}]" -f $folder, $genExit)
        continue
    }
    if ($benignExitCodes -contains $genExit) {
        Write-Host "Note: gen_feature returned benign teardown exit code $genExit - treating as success" -ForegroundColor DarkYellow
    }

    Write-Host ""
    Write-Host "Done generating features: $folder" -ForegroundColor Green
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
