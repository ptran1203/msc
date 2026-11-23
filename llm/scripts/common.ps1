# Shared setup for the final-run scripts (dot-sourced). Run from PowerShell: g++ is needed
# for the compile/test signals.
$ErrorActionPreference = "Continue"  # python's stderr (warnings, tqdm) must not abort the run
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$LLM = Split-Path $PSScriptRoot -Parent
$SRC = Join-Path $LLM "src"
$OUT = Join-Path $LLM "outputs"
Set-Location $SRC

function Invoke-Py {
    Write-Host "`n>> python $args" -ForegroundColor Cyan
    & python -u @args
    if ($LASTEXITCODE -ne 0) { throw "step failed (exit $LASTEXITCODE): python $args" }
}
