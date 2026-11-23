# Progress + ETA + GPU memory for a running final script, refreshed every minute (Ctrl+C to quit).
#   .\monitor.ps1 task2      .\monitor.ps1 task3      .\monitor.ps1 task1 [-Variant think]
param(
    [Parameter(Mandatory)][ValidateSet("task1", "task2", "task3")][string]$Run,
    [ValidateSet("think-anchor", "think")][string]$Variant = "think-anchor",
    [switch]$NoTestReport
)
$OUT = Join-Path (Split-Path $PSScriptRoot -Parent) "outputs"
$t3tag = "qwen3-8b-thinking-fewshot" + $(if ($NoTestReport) { "-noreport" } else { "" })
$ck, $total = switch ($Run) {
    "task2" { "$OUT\qwen3-8b-thinking-fewshot\predictions_task2_public.ckpt.jsonl", 328 }
    "task3" { "$OUT\$t3tag\predictions_task3_public.ckpt.jsonl", 328 }
    "task1" { "$OUT\task1-v5-$Variant\multi_public.ckpt.jsonl", 124 }
}
Write-Host "watching $ck"
$t0 = Get-Date; $n0 = $null
while ($true) {
    $n = if (Test-Path $ck) { (Get-Content $ck | Measure-Object -Line).Lines } else { 0 }
    if ($null -eq $n0) { $n0 = $n }
    $mins = ((Get-Date) - $t0).TotalMinutes
    $eta = if ($n -gt $n0 -and $mins -gt 0) { "ETA {0:N1} h" -f (($total - $n) / (($n - $n0) / $mins) / 60) } else { "ETA (measuring...)" }
    $gpu = nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    "{0}  {1}/{2} done  {3}  |  GPU {4}" -f (Get-Date -Format HH:mm), $n, $total, $eta, $gpu
    Start-Sleep 60
}
