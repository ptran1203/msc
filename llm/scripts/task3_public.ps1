# Task 3 final: Qwen3-8B thinking, self-diagnosed labels + level-controlled feedback, then the
# grounded rules (L2 + real compile failure -> first g++ error; L1 + passes all tests -> "Tot." (praise)).
# -NoTestReport if task3_sample.ps1 shows the report doesn't help (variant A > B).
# ~14-16h. Resumable.
param([switch]$NoTestReport)
. "$PSScriptRoot\common.ps1"
$TAG = "qwen3-8b-thinking-fewshot" + $(if ($NoTestReport) { "-noreport" } else { "" })
$extra = if ($NoTestReport) { @("--no-test-report") } else { @() }

Invoke-Py run_prompting.py --split public --tasks 3 --model Qwen/Qwen3-8B --thinking --fewshot @extra --run-tag $TAG
Invoke-Py postprocess_task3.py "$OUT\$TAG\predictions_task3_public.json" --out-tag "$TAG-rules" --split public
Invoke-Py validate_submission.py 3 "$OUT\$TAG-rules\predictions_task3_public.json" --split public
Write-Host "`nSUBMIT: $OUT\$TAG-rules\predictions_task3_public.json" -ForegroundColor Green
