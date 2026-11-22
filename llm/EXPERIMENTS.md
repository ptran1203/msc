# Experiment log

Every experiment gets a row: **run_id** (= `outputs/{run_id}/` folder or Kaggle kernel
version), **note** (what changed vs. the parent run), **local** score, **sub** (public
leaderboard score, or `—` if not submitted). Newest first within each task.

Local metrics:
- Task 1: QWK on the 32 `sample_dataset` items.
- Task 2: macro-F1 on the 32 `sample_dataset` items.
- Task 3 (research harness, forced self-diagnosis on the 32 items, `research_task3_ideas.py`):
  label macro-F1 / overlap_f1 / compliance. Public-only post-processing runs report
  compliance on the 328 public items (our checker) since there's no gold there.

Leaderboard: 2 submissions/day/team. Task 3 is scored by an LLM judge mainly against the
level definitions in `label_space.json` (instructor QA, `knowledge/QA.md`).

## Task 3

| date | run_id | note | local | sub |
|---|---|---|---|---|
| 2026-11-22 | `qwen7b-fewshot-testreport` (runner v10) | few-shot + compile_log (compile-only) + **test_report** in prompt, then both rules | running | — |
| 2026-11-22 | `qwen7b-fewshot-v2-rules` | `-compilefix` + rule 2: Level 1 + passes all verified tests → "Tốt.", labels [] (82 samples, 43 changed) | public compliance 1.0 | not yet |
| 2026-11-22 | `qwen7b-fewshot-v2-compilefix` | `qwen7b-fewshot-v2` + rule 1: Level 2 + real compile error → first g++ error line, + "Lỗi biên dịch" (70 samples). Source line dropped after it caused 4 compliance violations | public compliance 1.0 (0.988 with source line) | **0.30** (uncertain which of the two versions was uploaded) |
| 2026-11-22 | `ft_task3_lora_qwen2.5-coder-1.5b-instruct` | QLoRA 1.5B, 92 synth samples (7B-written targets), 3 epochs | macro .1114 / overlap .1298 / compl .906* (base 1.5B: .1014 / .0200 / 1.0) | — |
| 2026-11-21 | `research_task3_location_model` | RandomForest line-level bug-location model on synth | leave-one-problem-out top-1 .58–.85, but #1 pick on a real sample = empty line → no-go | — |
| 2026-11-21 | `qwen7b-fewshot-v2` | compile_log (link-based) + reasoning field + empty-feedback retry + **same-level RAG exemplar** | idea2 in spike: macro .2217 / overlap .1843 / 1.0 | **0.1784** |
| 2026-11-21 | research spike (5 ideas) | baseline static 3-exemplar: .2204 / .1799 · Task 2 hint .1391 / .1730 · compile_log guard .1271 / .1776 · 2 exemplars/level .1606 / .1748 · self-consistency x3 .1610 / .1768 | see note | — |
| 2026-11-21 | `qwen1.5b-fewshot` | 1.5B model, local | 87/328 empty feedback | rejected ("wrong shape") |
| 2026-11-19 | `qwen-rag` (runner v8) | RAG exemplars, link-based compile_log not yet added | — | 0.0800 |
| 2026-11-15 | `qwen2.5-coder-7b_fewshot` | brevity instruction + static few-shot | overlap .258 (given labels) | 0.1471 |
| 2026-11-08 | zero-shot baseline | 7B zero-shot | overlap .136 (given labels) | 0.1005 |

\* 2 of 3 violations were checker false positives from the quoted g++ source line.

## Task 2

| date | run_id | note | local | sub |
|---|---|---|---|---|
| 2026-11-22 | `qwen7b-fewshot-testreport` (runner v10) | few-shot + compile_log (compile-only) + test_report | running | — |
| 2026-11-20 | `submission_task2_ensemble` | `ensemble.py`: zero-shot + `qwen-rag`, label union | — | **0.1289** |
| 2026-11-19 | `qwen-rag` (runner v8) | RAG exemplars | — | 0.0421 |
| 2026-11-13 | few-shot (static) | 2 static exemplars | macro .075–.080 | — |
| 2026-11-08 | zero-shot baseline | 7B zero-shot | macro .183 | — |

## Task 1

| date | run_id | note | local | sub |
|---|---|---|---|---|
| 2026-11-22 | `qwen7b-fewshot-testreport` (runner v10) | few-shot + compile_log (compile-only) + test_report | running | — |
| 2026-11-19 | `qwen-rag` (runner v8) | RAG exemplars, no compile_log | — | -0.2171 |
| 2026-11-19 | `qwen-fewshot` | static few-shot, no compile_log | — | -0.2108 |
