# Task 1 check on the 15 EX01 multi-problem sample items (12 need the LLM): Qwen3-8B thinking
# with and without the teacher-graded anchor. Baseline (7B, task1-multi-v3): QWK .467.
# ~35-40 min per variant. Resumable.
. "$PSScriptRoot\common.ps1"
Invoke-Py task1_multi.py --split sample --model Qwen/Qwen3-8B --thinking --anchor --out-tag task1-v5-think-anchor
Invoke-Py task1_multi.py --split sample --model Qwen/Qwen3-8B --thinking --out-tag task1-v5-think
Write-Host "`nCompare the two 'sample multi (n=15): QWK' lines above with the 7B baseline .467" -ForegroundColor Green
