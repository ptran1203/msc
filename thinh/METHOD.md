# Method (thinh track)

Mirrors the layout of [../llm/METHOD.md](../llm/METHOD.md) so the two tracks can be read side by side.
Comparison with main: [COMPARE.md](COMPARE.md). Run log: [EXPERIMENTS.md](EXPERIMENTS.md).
Detailed Vietnamese reports live in [eda/plans/reports/](eda/plans/reports/) and [plans/reports/](plans/reports/).

Status: **analysis + POC on `sample_dataset` (32 labeled items) only.** No `public_test_release` run and no
leaderboard submission from this track yet. No fine-tuning yet.

## 1. Data

- Only `sample_dataset/` (32 items: EX01 15 `multi_problem`, EX02 17 `single_problem`). Put it at
  `thinh/sample_dataset/` locally; it is git-ignored (course privacy rule, the repo is public).
- Schema contracts for every dataset file and for the internal hand-offs: [eda/docs/contracts/](eda/docs/contracts/),
  checked by `eda/validate_contracts.py`.

### Data / label issues found (bonus-point candidates)
Scripts: `eda/meta_audit.py`, `eda/t1_rubric_eda.py`, `eda/t2_label_eda.py`, `eda/t3_feedback_eda.py`,
`eda/code_eda.py`, each with an independent `verify_*.py` re-implementation.
- **Three released labels disagree with the code's own build/test evidence** (`EX02-S214` whole file commented out yet 10/10;
  `S211`, `S216`). Kept as `LABEL_FIXES` in `eda/load_data.py` (teacher-confirmed). Every number is reported on
  **released** gold (primary) and **fixed** gold (ceiling).
- **EX01 is not graded by correctness.** Gold total ≈ Σ statement weights (1/4/2/3, read from the statement) over problems
  that have a function body. `exams.json problems[].max_score` says 2.5 each, contradicting the statement.
- **The written prerequisite rule ("P1 fails ⇒ P2–P4 unscored") is not applied by the gold**: applying it drops EX01 QWK
  0.605 → 0.395.
- **Infinite-loop rule is applied inconsistently**: S106/S107 (infinite loop) gold 2, S112 (infinite loop + all 4 problems
  broken) gold 9.
- **Rubric regularities**: EX02 `edge = logic/2`, `code_quality ≡ 1`; EX01 `complexity = code_quality = [logic > 0]`;
  failed build ⇒ EX01 all 0, EX02 `[0,0,0,0,0,1]` (6/6).
- **T2**: `Lỗi edge case` never in gold ⇒ macro-F1 ceiling 0.9; result depends on the scorer's `zero_division`
  (0.405 → 0.505 only with `zero_division=1` *and* never predicting edge case). `Lỗi thuật toán` vs `Lỗi logic` is
  a definition problem (4 gold, 0 hits).
- **T3**: target level vs. the level the feedback actually exhibits differ (`eda/out/t3/`), i.e. some gold feedback
  already violates its own level.

## 2. Evaluation harness (`eda/poc/score_predictions.py`)

- T1: QWK on a **fixed 0..10 axis** (= Lin's CCC, so never shrink predictions toward the mean), MAE, per-dim exact.
- T2: macro/micro-F1 over the 10 labels, `zero_division=0`, per-label F1.
- Always on released and fixed gold; paired bootstrap CIs in `eda/poc/analyze_results.py`.
- Blind inputs: `eda/poc/blind_inputs.py` strips rubric/taxonomy/feedback so a grader never sees gold.

## 3. Approach A — Signal-first hybrid prompting

Principle: **deterministic signals first, LLM only where no signal exists**, routed by `exam_type`.
Pipeline diagram: `eda/arch/grading-pipeline.html`.

1. **Compile gate** (`eda/signals.py`, `eda/code_toolchain.py`): real g++; only `file:line:col: error:` diagnostics fail the
   build — link errors (no `main` in a function-only file) do **not**. Failed build ⇒ exam's fixed rubric, no LLM call.
2. **EX02 / single_problem**: rubric from public test verdicts (strict vs lenient pass, crash, timeout, prompt-text noise).
3. **EX01 / multi_problem** (`eda/poc/problem_scoring.py`): the LLM never outputs a score. Per problem it returns
   `status ∈ {correct, mostly, partial, broken, missing}`, `infinite_loop`, label, one evidence line; code computes
   `total = Σ statement_weight × credit`, then splits into the 6 dims in a fixed order. Weights parsed from the statement
   (`signals.statement_weights`), not hard-coded.
4. **T2**: LLM labels, vote ≥ 2 of 4 runs; kept separate from T1 config (capping 1–2 labels helps T1 but cut Haiku
   macro-F1 0.45 → 0.29).
5. **Rubric constraints** (`eda/poc/rubric_constraints.py`): clamp to range, `total = sum(rubric)`.

Models: Claude Opus / Haiku through the local `claude` CLI for the POC graders; self-hosted `Qwen/Qwen3-8B`
(thinking off) and `Qwen/Qwen2.5-VL-7B-Instruct` (text-only), bf16, greedy, Apple M5 / 32 GB (MPS), via `eda/poc/local_llm.py`.
**Note:** the Claude runs are commercial API calls on real student code — fine for analysis only; the submission path must
move to the open-weight backend (main's constraint) and the API use must be reported.

## 4. Approach B — Fine-tuning (designed, not run)

Design in `eda/docs/03-technical-specification.md`: same backbone as Approach A (Qwen3-8B), QLoRA SFT for T1/T2 on
teacher-distilled per-problem status, T3 adapter with level control tokens `<L1>…<L4>` (+ optional DPO).
RL (GRPO/SPPO) explicitly deferred: 32 noisy labels are not enough.

## 5. Task 3 (designed, not run)

Diagnose → Render: structured diagnosis first, the renderer at L1/L2 never sees a fix. Level guard in 4 layers
(regex, tree-sitter syntax, quote-vs-new-code context, full-solution coverage), retry ×2, then a level-correct template.
Spec: `eda/docs/03-technical-specification.md` §T3, scenarios `eda/docs/04-scenarios.md`.

## 6. Open questions to the teacher

1. EX01: weights 1/4/2/3 for attempted problems, or graded by running? Why `max_score` 2.5?
2. Are hidden labels fixed like S214/S211/S216?
3. Official macro-F1: `zero_division`? per-label over the multilabel matrix?
4. Official definitions of logic / algorithm / loop / function / style.
