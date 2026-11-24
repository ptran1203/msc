# Experiment log (thinh track)

Same columns as [../llm/EXPERIMENTS.md](../llm/EXPERIMENTS.md). All scores are **local, on the 32 `sample_dataset`
items** — this track has not submitted to the leaderboard (`sub` = `—`). "released / fixed" = released gold /
gold after the 3 teacher-confirmed `LABEL_FIXES`. n = 32 ⇒ QWK 95% CI ≈ ±0.2; gaps < 0.1 are not conclusive.

## Task 1

| date | run_id | note | local (QWK released / fixed, MAE) | sub |
|---|---|---|---|---|
| 2026-10-06 | `ex01-attempt+loop` (exploratory) | per-problem status, credit = attempted ∧ no infinite loop; chosen after seeing data | EX01 .815 · full .777 / .875, MAE 1.27 (EX01) | — |
| 2026-10-06 | `ex01-attempt` | per-problem status, credit = attempted; identical to the regex predictor (Opus ×2, Haiku ×1) | EX01 .726 · full .724 / .817 | — |
| 2026-10-06 | `ex01-runs` (pre-registered main) | credit only if program runs (correct/mostly/partial) | EX01 .605 · full .686 / .772 → **hypothesis rejected** | — |
| 2026-10-06 | `ex01-runs+prereq` | + statement rule "P1 fails ⇒ P2–P4 unscored" | EX01 .395 → gold does not apply the prerequisite | — |
| 2026-10-06 | `rule-weights-regex` | no LLM: compile gate + statement weights × function-has-body (EX01) + test rule (EX02) | .724 / .817, MAE .91, 28/32 within ±1 | — |
| 2026-10-06 | `rule-const` | no LLM: compile gate + EX01 = 6 + EX02 test rule | — / .76, MAE 1.41 | — |
| 2026-10-06 | `hybrid-v2` (headline) | compile gate + EX02 from tests + EX01 LLM total (prompt v2, Opus, whole exam in one context) | .703 / .803, MAE 1.19; EX01 .633, EX02 .95 (fixed) | — |
| 2026-10-06 | `hybrid-v2-per-sample` | same, one submission per call | .686 / .782; EX01 .587 | — |
| 2026-10-08 | `t1_basic` Qwen3-8B 0-shot | `eda/poc/t1_basic.py --model Qwen/Qwen3-8B`, compile gate, M5 MPS bf16 | .486, MAE 2.56 | — |
| 2026-10-07 | `t1_basic` Qwen2.5-VL-7B 0-shot | same prompt, text-only | .625, MAE 2.03 | — |
| 2026-10-07 | `t1_basic` Haiku 4-shot | same-exam exemplars, leave-one-out + near-dup exclusion | .594, MAE 1.97 | — |
| 2026-10-07 | `t1_basic` Haiku 0-shot | | .520, MAE 2.31 | — |
| 2026-10-07 | `t1_basic` Haiku (first run) | | .554, MAE 2.38 | — |

Reading: the deterministic part carries T1. A no-LLM predictor (.724 released) beats every pure-prompt run
(.486–.625); the LLM's only measured value on EX01 is catching infinite loops (3 problems, confirmed by hand).

## Task 2

| date | run_id | note | local (macro-F1 released / fixed) | sub |
|---|---|---|---|---|
| 2026-10-06 | `hybrid-v2` vote ≥ 2/4 | compile gate forces `Lỗi biên dịch`, LLM labels voted | .365 / .405 (thresholds: ≥1/4 .43, ≥3/4 .36) | — |
| 2026-10-06 | per-problem prompt labels | labels from the EX01 per-problem prompt, 1 run | .32 (fixed) → keep the vote | — |
| 2026-10-06 | Haiku capped 1–2 labels | shared T1/T2 config | .29 (vs .45 uncapped) → split T1/T2 configs | — |

## Task 3

Not run yet (design only, see METHOD §5).
