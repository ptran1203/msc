# Final runs

PowerShell scripts for the final submissions. Everything runs locally on the RTX 3060 (Qwen3-8B 4-bit, thinking mode), except the fine-tune, which runs on Colab.

- **Run them from PowerShell.** The compile and test signals need `g++`.
- **One GPU job at a time.** Each job needs about 8 GB of the 12 GB, so two jobs together spill into shared memory and run 15–20× slower.
- **Stop and resume:** press `Ctrl+C` at any time, then rerun the same script to continue from the last finished item.
- **Monitor** a running job from a second window with `.\monitor.ps1 task1|task2|task3`. It shows items done, the ETA, and GPU memory, which should stay near 8 GB.

| Order | Script | Time | Output (submit / compare) |
|---|---|---|---|
| 1 | `.\task2_public.ps1` | ~13.5 h | `outputs\task2-final-think-u-svc\predictions_task2_public.json` |
| 2 | `.\task1_sample.ps1` | ~1.2 h | QWK on 15 EX01 items vs 7B baseline .467 |
| 3 | `.\task1_public.ps1 [-Variant think\|think-anchor]` | ~3.5–4 h | `outputs\task1-v5-<variant>\predictions_task1_public.json` |
| 4 | `.\task3_sample.ps1` | ~2 h | `outputs\research_task3\testreport_ablation\qwen3-8b-thinking\summary.json` vs the 7B rows |
| 5 | `.\task3_public.ps1 [-NoTestReport]` | ~14–16 h | `outputs\qwen3-8b-thinking-fewshot[-noreport]-rules\predictions_task3_public.json` |

When to run steps 3 and 5:
- **Step 3:** only if one of the step 2 variants beats .467. Pass the winning variant with `-Variant`.
- **Step 5:** only if Qwen3-8B thinking beats the 7B on the step 4 sample check. If variant A (no test report) beats B, add `-NoTestReport`.

Every script validates its output before printing `SUBMIT: <path>`. The leaderboard allows only 2 submissions per day, so log each run in `EXPERIMENTS.md`.

## Methods
- **Task 1**
  - Single-problem items are scored from verified test outcomes (`test_rule`).
  - Multi-problem items are scored per problem by the LLM, with equal weights.
  - A real `g++` failure is scored by the compile rule.
- **Task 2:** Task 2 prompt labels (with thinking) ∪ the `feat-svc` classifier, plus the compile and format rules.
  - It uses no Task 3 output.
  - It never uses the `feedback` field.
- **Task 3:** labels self-diagnosed by the model, plus feedback at the target level, plus two grounded rules:
  - Level 2 + a real compile failure → the first `g++` error;
  - Level 1 + all tests passed → praise.

## Fine-tune (course requirement)
The fine-tune runs on Colab: `colab\colab_finetune_<model>.ipynb`, built by `src\build_colab_finetune_notebook.py`. Checkpoints, the adapter, the eval results and the predictions all go to Google Drive under `MyDrive/llm_ft/`. Each notebook embeds the Kaggle token, so never commit or share those notebooks.
