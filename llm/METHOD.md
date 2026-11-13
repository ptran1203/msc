# Method

Living document — updated as the approach evolves. See [README.md](README.md) for the task spec and [CLAUDE.md](CLAUDE.md) for constraints.

## 1. Data

Two sources under `data/public-test-cham-diem-phan-hoi-files/`:
- `sample_dataset/` — 32 labeled samples (EX01: 15, multi_problem; EX02: 17, single_problem). Not the official train/test split — for understanding format and few-shot exemplars only.
- `public_test_release/` — 328 unlabeled samples across 6 real exams (3 `multi_problem`, 3 `single_problem`). This is the actual dev set to predict against for the public leaderboard (2 submissions/day cap).

A separate Google Drive link initially offered as "training data" turned out to be a stale revision of `sample_dataset.zip` with 22/32 differing Task-1 labels — discarded in favor of the repo copy, which matches the real submission schema (`sample_submission_task3.json`). No separate full training set has been provided yet.

### Schema quirks found during EDA (`notebooks/01_eda.ipynb`, `src/eda.py`)
- `compile_log` is empty string for all 360 samples in both splits — this secondary signal isn't populated in the current data drop.
- Rubric dims and taxonomy labels are imbalanced; `Lỗi edge case` never appears in `sample_dataset`'s gold labels, and feedback Level 4 never appears in either split. Consequence: macro-F1 on `sample_dataset` has a <1.0 ceiling for a perfect predictor (absent label scores F1=0 by convention) — confirmed by `src/test_eval_sanity.py`.
- Prerequisite grading policy (e.g. EX01: P1 fail ⇒ P2–P4 unscored) is per-exam, read from `exams.json["grading_policy"]`, not uniform across exams.

## 2. Evaluation harness (`src/eval.py`)

Implemented before touching any model output, sanity-checked by scoring `sample_dataset`'s own gold labels against themselves (`src/test_eval_sanity.py`):
- Task 1: quadratic weighted kappa (QWK) on `total_score`, MAE, per-dimension exact-match.
- Task 2: per-label precision/recall → macro-F1 and micro-F1 over the 10-label taxonomy.
- Task 3: rule-based level-compliance check (Level 1/2 feedback must not contain code markers or explicit "sửa thành/lời giải/đáp án" language).

## 3. Approach A — Zero-shot prompting baseline (in progress)

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`, loaded 4-bit (bitsandbytes NF4) via `transformers` — fits the RTX 3060's 12GB VRAM. Chosen for code-task strength and because it doubles as the fine-tuning target later (Approach B), keeping the two tracks comparable.
- Deterministic decoding (`do_sample=False`), fixed seed, exact model revision logged per run — required for reproducibility.
- Prompts (`src/prompts.py`): per-task template injecting the exam statement + grading policy (from `exams.json`), student code, optional `compile_log`/`test_report`, and (Task 3 only) the given `taxonomy_error` + `target_feedback_level` with its definition/forbidden-content text from `label_space.json`.
- Output parsing (`src/llm_client.py:extract_json`): pulls the first balanced JSON object out of the raw generation (handles markdown code fences).
- Validated with a single-sample smoke test (`src/smoke_test.py`) across all 3 tasks before any full loop, per the no-heavy-compute-without-checking-first rule.

### Known limitation going in
No few-shot exemplar exists for `Lỗi edge case` or feedback Level 4 (absent from both data splits) — the baseline cannot be validated against real examples for these; flagged for the error-analysis section of the report.

### Task 3 schema fork (found via smoke test, 2026-11-08) — important
`sample_dataset` and `public_test_release` use **different Task 3 I/O contracts**, confirmed by inspecting both JSON files directly:
- `sample_dataset`: input **includes** `taxonomy_error` (given), output is `{feedback}` only. Matches that split's own README ("tách task này khỏi task 2").
- `public_test_release` (the real submission target): input has **no** `taxonomy_error` key — the model must self-diagnose it, and output must be `{taxonomy_error, feedback}`. This matches the top-level challenge README's original Task 3 spec, which the sample-set README had simplified away.

Consequence: a pipeline built only against `sample_dataset`'s format would silently produce wrong-shaped predictions for the actual submission. `src/prompts.py:task3_prompt` and `src/run_prompting.py` now branch on whether `taxonomy_error` is present in `sample["input"]`, building the right prompt/output-schema for each split. Local validation against `sample_dataset` therefore only checks the `{feedback}`-only path; the self-diagnosis path is untested against ground truth since no labeled example of it exists — flag as a reporting caveat.

### Task 2 prompt fix
First smoke-test run had the model occasionally return a bare `{}` for "no error" instead of `{"taxonomy_error": []}` — valid JSON, but missing the required key. Fixed by making the prompt explicit about the empty-list format with examples. `run_prompting.py` also now patches any missing *required* key with a schema-appropriate default (distinct from total parse failure) rather than letting malformed-but-parseable output through.

## 4. Approach B — Fine-tuned open-weight model (data blocker being addressed)

Only 32 labeled samples exist (`sample_dataset`) — no official training split has been released yet (the Drive link was a stale duplicate, not a larger set). QLoRA fine-tuning on 32 examples alone would overfit, so this track was blocked pending more labeled data. Addressing this with **synthetic data generation via controlled bug injection** (scope: Task 1 + Task 2 only — see below), rather than waiting indefinitely for an official release.

### Synthetic data generation (built, not yet run — needs GPU)

Key design principle: the LLM is used only to *write* buggy code from a precise instruction (something it's good at); it never *grades* its own output (something we measured it's bad at — zero-shot QWK 0.39, macro-F1 0.18). Labels come from an objective oracle, not self-report.

- **Reference solutions** (`synth/references/*.cpp`): hand-written by hand for the 3 single-problem types we have full statements for (`armstrong` = EX02/Class03-Midterm-1, `remove_duplicates` = Class03-Midterm-2, `earliest_login` = Class03-Final-1 "earliest login" puzzle). Verified against the worked examples in each problem statement via `g++` (through PowerShell — see Tooling note in [CLAUDE.md](CLAUDE.md), the sandboxed Bash tool blocks `cc1plus.exe` from spawning).
- **Bug taxonomy** (`src/synth_bugs.py`): one controlled instruction per official taxonomy label (all 10, including `Lỗi edge case`, which never appears in real data at all — directly fills that EDA-identified gap).
- **Objective oracle** (`src/synth_oracle.py`): generates seeded random + edge-case test inputs per problem, runs the reference solution as ground truth, compiles/runs the candidate, and reports raw vs. normalized (case/whitespace-insensitive) pass rates separately for general vs. edge cases — this is what distinguishes an `io_format` bug (raw fails, normalized passes) from a real `logic` bug (both fail). Self-check (reference graded against itself) is perfect on all 3 problems; verified it correctly catches both bug types on hand-crafted test cases before trusting it on LLM output.
- **Label derivation + QC** (`src/synth_label.py`): converts oracle signals into the official rubric schema (enforcing `total_score == sum(rubric)`), assigns `taxonomy_error` from injection intent (not re-diagnosed), and **discards** (does not mislabel) samples where the LLM didn't follow the injection instruction cleanly — e.g. an unintended compile break, a "style-only" edit that actually changed behavior (caught a real bug in my own first QC draft this way), or an injected bug with no measurable effect.
- **Generator** (`src/synth_generate.py`): orchestrates LLM mutation → oracle grading → QC → writes a `sample_dataset`-shaped output (`synth/generated/`: `exams.json`, `label_space.json`, `task1_grading.json`, `task2_error_taxonomy.json`, code files), loadable directly via `data.load_dataset()`. Has a `--smoke-test` mode (1 bug type × 1 variant × all 3 problems = 3 LLM calls) to run before the full loop, per the no-heavy-compute-without-checking-first rule.

**Deliberately out of scope: Task 3.** No objective oracle exists for feedback-*text* quality, so synthesizing it would reintroduce the same self-labeling problem this whole approach exists to avoid.

### Generation results (2026-11-09)

Ran `--smoke-test` (confirmed pipeline works end-to-end, including QC correctly discarding 3/3 `compile_error` attempts where the model refused to actually break compilation), then the full run: 90 candidate samples (3 problems × 10 bug types × 3 variants) → **57 kept, 33 discarded by QC** (63% yield). Verified `total_score == sum(rubric)` holds for all 57, and the output loads cleanly through the existing `data.load_dataset()`.

Per-label yield (`task2_error_taxonomy.json`):

| Label | Count | Note |
|---|---|---|
| Lỗi logic | 9 | |
| Lỗi style | 9 | |
| Lỗi mảng/chuỗi | 8 | |
| Lỗi thuật toán | 8 | |
| Lỗi nhập/xuất | 6 | |
| Lỗi hard-code | 6 | |
| Lỗi vòng lặp | 5 | |
| Lỗi hàm | 5 | |
| Lỗi edge case | **1** | the gap-filling label — low yield, but the one sample is a clean, genuine edge-case bug (special-cases `n<=9` incorrectly; caught by the dedicated edge test cases, missed by general random ones) |
| Lỗi biên dịch | 0 | model never cooperated with "break compilation" (9/9 attempts discarded) — not a problem, 6/32 real examples already cover this label |

Output at `synth/generated/` (schema-compatible with `sample_dataset`, loadable via `data.load_dataset()`). Could run more `edge_case_error` variants specifically (low yield, most valuable label) if more are wanted — straightforward since the pipeline already supports targeting individual bug types.

### Zero-shot baseline results on `sample_dataset` (32 labeled samples, see `notebooks/02_prompting_baseline.ipynb`)

| Task | Metric | Zero-shot | Trivial constant baseline |
|---|---|---|---|
| 1 (rubric) | QWK | 0.393 | 0.000 |
| 1 (rubric) | MAE | 3.19 | 3.16–3.47 |
| 2 (taxonomy) | macro-F1 | 0.183 | 0.055 (always predict most-common label) |
| 2 (taxonomy) | micro-F1 | 0.286 | 0.329 (always predict most-common label) |
| 3 (feedback) | level compliance | 1.0 (rule-based checker) | — |

Reading: zero-shot clearly beats the trivial baseline on the metrics the leaderboard actually ranks by (QWK, macro-F1) — real signal, not noise — but Task 1's MAE is barely better than guessing a constant, meaning the model's score *ordering* is informative but its *calibration* against the teacher's scale is off. Task 3's perfect compliance score is against a rule-based regex checker that only catches blatant code/solution leakage, not subtler level violations (e.g. suggesting a fix approach in prose at Level 2) — flagged as a checker limitation, not a guarantee.

### Few-shot variant (built, not yet run — waiting on GPU availability)
Added `--fewshot` to `run_prompting.py`; `prompts.py` now accepts an optional `fewshot_ds` and prepends a hand-picked exemplar block (`FEWSHOT_IDS`), excluding the current sample from its own exemplar set when it happens to be one of the picks. Exemplars chosen for diversity, not at random:
- Task 1: 4 exemplars spanning the full `total_score` range (0, 3, 6, 10) across both exam types — targets the calibration gap identified above, not comprehension.
- Task 2: 4 exemplars — one empty-label case, one rare label (`Lỗi mảng/chuỗi`, 1/32 occurrence), two common labels.
- Task 3: 3 exemplars, one per level actually present in the data (1, 2, 3). For the self-diagnosis branch (public test), the same given-label exemplars are repurposed to demonstrate the `{taxonomy_error, feedback}` output shape by hiding the label from the shown input. Still no Level 4 exemplar exists anywhere in the released data.

Sanity-checked prompt construction with pure-Python tests (no model load): exemplar text is present, self-exclusion works, prompt sizes are ~7-15K chars (well within context). Not yet run against the model — holding until GPU is free.

### Bug found and fixed: `extract_json` wasn't string-aware
The first public run (328 samples × 3 tasks) hit exactly one parse failure, on `Class01-Test1-S18` (Task 3). Root cause: the model's feedback text legitimately contained a literal `'}'` character (explaining a missing closing brace in the student's C++ struct) inside the JSON string value. `llm_client.py:extract_json`'s brace-depth counter wasn't aware it was inside a string literal, so it treated that `}` as closing the JSON object early, truncating mid-string — which reliably produces an unterminated-string `JSONDecodeError`, not silently-wrong valid JSON. Fixed by tracking in-string/escape state while scanning. Re-verified Task 1/2 had 0 parse failures throughout (their value types rarely contain literal braces), so this was an isolated case; patched by regenerating that one sample (`src/patch_sample.py`) rather than rerunning all 328.

### Submission rejected: `total_score` must exactly equal the rubric sum
The course's actual grader rejected `predictions_task1_public.json` outright (error on `Class01-Test1-S10`: `'total' (2) does not match the sum of the six dimensions (3.0)`). Checked and confirmed: in all 32 `sample_dataset` gold labels, `total_score` is *always* exactly `sum(rubric.values())` — it's not an independent holistic judgment, just the sum. The model, however, frequently computed `total_score` independently of its own rubric breakdown: **179/328 (55%) of Task 1 predictions were inconsistent**. Fixed in two places: `run_prompting.py` now overrides `total_score = sum(rubric.values())` unconditionally after parsing (never trusts the model's own total), and `validate_submission.py` now checks this invariant so it's caught locally before ever touching the leaderboard again. Already-generated files were patched in place (pure JSON edit, no model rerun needed).

## 5. Submission pipeline

`src/run_prompting.py` generates `predictions.json` per task; `src/validate_submission.py` checks schema/value-range/sample-id-coverage before anything would be submitted externally (no official checker script was provided in this data drop, so this is our own reconstruction of the schema). First real submission files for `public_test_release` (328 samples) generated via `python run_prompting.py --split public --tasks 1,2,3`.


## Few shot prompt + RAG on train data