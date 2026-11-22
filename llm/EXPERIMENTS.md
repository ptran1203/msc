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
| 2026-11-22 | `qwen3-8b-thinking` testreport ablation (Colab T4) | same as `testreport_ablation` but **Qwen3-8B, thinking mode** (`eval_testreport_sample.py --model Qwen/Qwen3-8B --thinking`) | running | — |
| 2026-11-22 | `testreport_ablation` (local, 32 items) | A: idea2 prompt + compile-only log, no test_report · B: + test_report (17/32 items, EX02) · A/B + rules 1+2 — `eval_testreport_sample.py` | macro / overlap / compl: **A** .2372 / .1986 / 1.0 · **B** .2190 / .1992 / 1.0 · **A+rules** .2372 / .2599 / 1.0 · **B+rules** .2190 / .2644 / 1.0 → test_report doesn't help labels (−.018 macro); rules give +.06 overlap | — |
| 2026-11-22 | `qwen7b-fewshot-testreport` (runner v11) | few-shot + compile_log (compile-only) + **test_report** in prompt, then both rules | running | — |
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
| 2026-11-23 | `task2-t3rules-u-feat-svc` | union of `task2-from-t3-rules` + `task2-feat-svc` | macro .3037 / micro .3830 (best local) | not yet |
| 2026-11-23 | `task2-feat-svc` | **no LLM**: LinearSVC on grounded run signals (g++ compile, test pass/format/timeout/crash fractions; synth gets them via oracle-built suites) + code stats; train 593 synth + real | macro .2883 / micro .3465 (beats LLM t3+rules .2782 on macro). Over-predicts: 2.69 labels/item vs gold 1.28 (public 2.65) | not yet |
| 2026-11-23 | `ml-task2-grid` (`ml_task2_grid.py`) | {MiniLM emb, run+stats feat, both} × {LogReg, LinearSVC, LightGBM}, LOO on 32 real | alone macro: emb .064/.080/.064 · **feat .254/.288/.212** · emb+feat .181/.227/.225 → embeddings hurt; SVC best; LGBM overfits small data | — |
| 2026-11-23 | `task2-from-t3-rules` | Task 2 labels = Task 3 self-diagnosis (`qwen7b-fewshot-v2` T3 public) + rules: real g++ fail → add/remove `Lỗi biên dịch`; test_report all-failures-format → add `Lỗi nhập/xuất` (`task2_from_task3.py`, combo `t3+compile+format`) | macro .2782 / micro .5135 (Task 2 prompt alone: .1612 / .4103; Task 2 prompt + rules .2106 / .4533) | **0.1634** (assumed this file; user reported score without naming it) |
| 2026-11-23 | `task2-union-t3-rules` | same, union of Task 3 + Task 2 (`submission_task2_ensemble`) labels + rules (`union+compile+format`) — more rare labels on public | macro .2820 / micro .4889 | **0.1649** (assumed this file) |
| 2026-11-23 | `ml-task2-logreg` (+ ensembles) | MiniLM embeddings + OvR LogReg on 593 synth + real; **leave-one-out on 32 real** (old .303 CV mixed synth into validation → inflated) | alone .1387 / .2297 · `t3∪ml`+rules .2759 / .3776 · 2-of-3 vote+rules .2539 / .4658 → no-go, doesn't beat t3+rules | — |
| 2026-11-22 | `qwen7b-fewshot-testreport-sample` vs `qwen7b-fewshot-noreport-sample` (local) | few-shot + compile-only log, with vs without test_report (17/32 items get one) | macro / micro: **with** .1521 / .3896 · **without** .1612 / .4103 → test_report slightly hurts | — |
| 2026-11-22 | `qwen7b-fewshot-testreport` (runner v11) | few-shot + compile_log (compile-only) + test_report | running | — |
| 2026-11-20 | `submission_task2_ensemble` | `ensemble.py`: zero-shot + `qwen-rag`, label union | — | **0.1289** |
| 2026-11-19 | `qwen-rag` (runner v8) | RAG exemplars | — | 0.0421 |
| 2026-11-13 | few-shot (static) | 2 static exemplars | macro .075–.080 | — |
| 2026-11-08 | zero-shot baseline | 7B zero-shot | macro .183 | — |

## Task 1

| date | run_id | note | local | sub |
|---|---|---|---|---|
| 2026-11-23 | `task1-multi-v3` | **multi-problem collapse**: every earlier file in this log scored public multi-problem ~0.7/10 (single ~5.6) — link-based compile log faked failures on function-only files + model applied the P1 'fail to read the file' prerequisite (QA: ignore it). Fix (`task1_multi.py`): compile rule (g++ fail → multi all 0 / single code_quality 1, 6/6 on sample; clean → compilable=1) + multi-problem scored **per problem, equal weights, no prerequisite**, dims averaged; singles from `qwen7b-fewshot-v2` + compile rule | sample multi (15) QWK .383 → **.467**; public multi mean 0.72 → 2.05 (63 compile-rule zeros, 21 compiling items ≥6) | **0.7331** (prev best 0.69, user-reported) |
| ? | unknown (user-reported) | Task 1 submission not logged here — file/run unknown | — | **0.69** |
| 2026-11-22 | `qwen7b-fewshot-testreport-sample` vs `qwen7b-fewshot-noreport-sample` (local) | few-shot + compile-only log, with vs without test_report (17/32 items get one); rubric clamped to range | QWK / MAE: **with** .5662 / 2.375 · **without** .5765 / 2.375 → no QWK gain; per-dim exact better with report (logic .66 vs .53, edge .47 vs .41, quality .56 vs .63) | — |
| 2026-11-22 | `qwen7b-fewshot-testreport` (runner v11) | few-shot + compile_log (compile-only) + test_report | running | — |
| 2026-11-19 | `qwen-rag` (runner v8) | RAG exemplars, no compile_log | — | -0.2171 |
| 2026-11-19 | `qwen-fewshot` | static few-shot, no compile_log | — | -0.2108 |
