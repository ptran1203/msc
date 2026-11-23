# Task 1 final: single-problem items from verified test outcomes (v4 test rule) + multi-problem
# items re-scored per problem by Qwen3-8B thinking [+ anchor]; compile failures by rule.
# Pick the variant that won task1_sample.ps1. ~3.5-4h (61 LLM items). Resumable.
param([ValidateSet("think-anchor", "think")][string]$Variant = "think-anchor")
. "$PSScriptRoot\common.ps1"
$TAG = "task1-v5-$Variant"
$extra = if ($Variant -eq "think-anchor") { @("--anchor") } else { @() }

Invoke-Py task1_multi.py --split public --base "$OUT\task1-v4-single-testrule\predictions_task1_public.json" `
    --single-test-rule --model Qwen/Qwen3-8B --thinking @extra --out-tag $TAG
Invoke-Py validate_submission.py 1 "$OUT\$TAG\predictions_task1_public.json" --split public
Write-Host "`nSUBMIT: $OUT\$TAG\predictions_task1_public.json" -ForegroundColor Green
