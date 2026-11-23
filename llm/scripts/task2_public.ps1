# Task 2 final: Qwen3-8B thinking (Task 2 prompt, few-shot + test_report; no Task 3 input)
# union feat-svc classifier + grounded compile/format rules. Sample: macro-F1 .340.
# ~13.5h on the 3060. Ctrl+C anytime; rerun this script to resume.
. "$PSScriptRoot\common.ps1"
$TAG = "qwen3-8b-thinking-fewshot"
$FINAL = "task2-final-think-u-svc"

Invoke-Py run_prompting.py --split public --tasks 2 --model Qwen/Qwen3-8B --thinking --fewshot --run-tag $TAG
Invoke-Py task2_from_task3.py build --t2 "$OUT\$TAG\predictions_task2_public.json" `
    --ml "$OUT\task2-feat-svc\predictions_task2_public.json" --combo "t2|ml+compile+format" --out-tag $FINAL
Invoke-Py validate_submission.py 2 "$OUT\$FINAL\predictions_task2_public.json" --split public
Write-Host "`nSUBMIT: $OUT\$FINAL\predictions_task2_public.json" -ForegroundColor Green
