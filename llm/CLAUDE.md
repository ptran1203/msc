# llm/ project instructions

Course project: auto-grading + feedback for student C++ submissions (see [README.md](README.md) for the full spec). 3 tasks: rubric grading (QWK/MAE), multi-label error taxonomy (macro/micro-F1), controlled-level Vietnamese feedback (level compliance).

## Constraints

- **Open-source/open-weight models only.** No commercial LLM APIs (no OpenAI/Anthropic/Gemini keys) — the whole pipeline must run on local open-weight models.
- **No heavy training jobs on this machine without explicit approval.** Before any real training run, always fit a single batch first to confirm the setup works, then stop and check in before launching the full run.
- Hardware: RTX 3060 12GB VRAM (see root [CLAUDE.md](../CLAUDE.md)) — caps fine-tuning to ≤7B models, 4-bit QLoRA.
- Fix seeds, log exact model name/revision for every run — required for reproducibility grading.
- Never use the `feedback` field as input to Task 1 or Task 2 (label leakage, explicitly forbidden by the rules).
- Public leaderboard allows only 2 submissions/day/team — validate predictions.json locally (format + sanity metrics on `sample_dataset`) before ever submitting.

## Current goals (as of 2026-11-08)

1. EDA notebooks under `notebooks/` (not just scripts) covering both `sample_dataset` (32, labeled) and `public_test_release` (328, unlabeled — the real dev set).
2. A working eval harness (`src/eval.py`), sanity-checked against ground truth before trusting it on model output.
3. First submission per task: a zero-shot open-weight prompting baseline run over all 328 public-test samples, producing a valid `predictions.json` per task.
4. Document the approach as it evolves in [METHOD.md](METHOD.md) — update it incrementally, not just at the end.

## Tooling note

- **VRAM cliff for Qwen2.5-Coder-7B-Instruct (4-bit) on the RTX 3060's 12GB**: somewhere between ~4241 and ~4799 input tokens (model weights ~5.9GB + KV cache/activations for the rest), the process's GPU working set exceeds the card's 12GB and Windows WDDM silently spills into "shared GPU memory" (system RAM over PCIe) instead of raising an OOM error. This is ~15-20x slower (3.8s → 72s observed for a near-identical call, same output), not a crash, so it's easy to mistake for a content-specific bug rather than a memory ceiling. Diagnostic: `torch.cuda.memory_reserved()` reporting *more* than 12GB confirms it's happening. Fix: keep total prompt+generation token budget with real margin below ~4000 tokens (few-shot exemplar count was the lever here); there's no code-side error, the only symptom is wall-clock time.
- **Always run Python with `python -u`** (unbuffered stdout) when piping to a log file for live monitoring (`... | Tee-Object -FilePath log.txt`). Without `-u`, Python block-buffers stdout when it's not a TTY, so `print()` progress lines sit invisible in the buffer until it fills or the process exits — the log looks frozen even though the run is progressing fine.


- `g++` compile/run commands must go through the **PowerShell** tool, not Bash — the sandboxed Bash tool silently blocks `cc1plus.exe` (g++'s subprocess) from spawning, failing with exit 1 and no error output. PowerShell has no such restriction.
- When piping multi-line/non-ASCII input to a compiled `.exe` from PowerShell, `|` pipes corrupt stdin (encoding mismatch). Write input to a file with `Out-File -Encoding ascii -NoNewline` and redirect via `cmd /c "exe.exe < input.txt"` instead.

## Key data facts learned so far (see METHOD.md for details)

- `compile_log` is empty for all 360 samples currently available (both splits) — the secondary signal README describes isn't populated in this data drop.
- `Lỗi edge case` taxonomy label and feedback Level 4 never appear in either split — no few-shot exemplars for them, and macro-F1 has a <1.0 ceiling even for an oracle on `sample_dataset` because of this.
- Prerequisite grading policy differs per exam and must be read from `exams.json["grading_policy"]`, never hard-coded.
