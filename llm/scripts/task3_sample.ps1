# Task 3 check on the 32 sample items: Qwen3-8B thinking, self-diagnose prompt, without (A) and
# with (B) test_report, each +/- the grounded rules. Compare with the 7B rows in
# outputs\research_task3\testreport_ablation\summary.json. ~2h. Resumable per variant.
. "$PSScriptRoot\common.ps1"
Invoke-Py eval_testreport_sample.py --model Qwen/Qwen3-8B --thinking
Write-Host "`nResults: $OUT\research_task3\testreport_ablation\qwen3-8b-thinking\summary.json" -ForegroundColor Green
