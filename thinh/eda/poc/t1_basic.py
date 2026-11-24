"""Basic Task 1 POC: one prompt per submission -> rubric scores, then compare with teacher labels.

Two additions on top of zero-shot:
- Compile gate: a real compiler error in compile_log skips the LLM and returns the exam's fixed rubric.
- Few-shot (--shots K): K teacher-graded submissions of the same exam are shown before the target.
  Leave-one-out: the target and its near-duplicates are never used as examples.

Usage: python eda/poc/t1_basic.py [--shots 4] [--limit 5] [--exam EX02] [--model haiku]
Self-hosted: <venv with torch+transformers>/bin/python eda/poc/t1_basic.py --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "sample_dataset"
OUT_DIR = Path(__file__).resolve().parent
from configs import DIMS, Rubric

# "file:line:col: error:" diagnostics only; link errors (no file:line:col) do not trigger the gate.
COMPILE_ERROR_RE = re.compile(r"^\S+:\d+:\d+: (fatal )?error:", re.M)
# Rubric the teacher gives on a failed build, as observed on every released build failure (6/6).
COMPILE_FAIL_RUBRIC = {
    "EX01": dict.fromkeys(DIMS, 0),
    "EX02": {**dict.fromkeys(DIMS, 0), "code_quality": 1},
}
# Near-identical submissions; using one as an example for another would leak its score.
NEAR_DUP_GROUPS = [{"EX02-S203", "EX02-S204", "EX02-S205", "EX02-S206"}]
# Labels that would teach the wrong thing as examples (whole file commented out, yet graded 10/10).
NOT_AN_EXAMPLE = {"EX02-S214"}

PROMPT = """You are grading a student's C++ submission.

## Problem statement
{statement}
{examples}
## Student code to grade
```cpp
{code}
```

## Compile log (g++ -std=c++11; may differ from the teacher's compiler)
{compile_log}

## Public test results
{tests}

Score each rubric dimension (integer, max in brackets):
compilable [1], io_format [1], logic [4], edge_case [2], complexity [1], code_quality [1].
{calibration}
Reply with ONLY a JSON object, e.g.
{{"compilable": 1, "io_format": 1, "logic": 3, "edge_case": 1, "complexity": 1, "code_quality": 1}}
"""

EXAMPLE = """
### Example {i} (teacher score: {total}/10)
```cpp
{code}
```
Compile log: {compile_log}
Public tests: {tests}
Teacher rubric: {rubric}
"""


def call_llm(prompt: str, model: str) -> str:
    """Send one prompt to Claude via the local `claude` CLI, or to a self-hosted HF model when
    `model` is a repo id like "Qwen/Qwen3-8B"; return the raw text reply."""
    if "/" in model:
        from local_llm import generate
        return generate(prompt, model)
    result = subprocess.run(["claude", "-p", "--model", model], input=prompt,
                            capture_output=True, text=True, check=True)
    return result.stdout


def load_samples() -> list[dict]:
    """Task 1 samples joined with their exam statement, code and gold rubric."""
    exams = {e["exam_id"]: e for e in json.loads((DATA_ROOT / "exams.json").read_text())["exams"]}
    samples = json.loads((DATA_ROOT / "task1_grading.json").read_text())["samples"]
    for s in samples:
        inp = s["input"]
        s["statement"] = exams[inp["exam_id"]]["statement"]
        s["code"] = (DATA_ROOT / inp["code_file"]).read_text(errors="replace")
    return samples


def has_compile_error(sample: dict) -> bool:
    return bool(COMPILE_ERROR_RE.search(sample["input"].get("compile_log") or ""))


def format_tests(tests: list[dict] | None, detailed: bool = True) -> str:
    if not tests:
        return "(none)"
    passed = sum(t["passed"] for t in tests)
    if not detailed:
        return f"{passed}/{len(tests)} passed"
    lines = [f"input={t['input']!r} expected={t['expected']!r} actual={t['actual']!r} passed={t['passed']}"
             for t in tests]
    return f"{passed}/{len(tests)} passed\n" + "\n".join(lines)


def pick_examples(target: dict, pool: list[dict], k: int) -> list[dict]:
    """K same-exam, buildable examples spread evenly over the score range, excluding the target's group."""
    sid = target["sample_id"]
    blocked = {sid, *NOT_AN_EXAMPLE}.union(*(g for g in NEAR_DUP_GROUPS if sid in g))
    cands = sorted((s for s in pool if s["input"]["exam_id"] == target["input"]["exam_id"]
                    and s["sample_id"] not in blocked and not has_compile_error(s)),
                   key=lambda s: (s["output"]["total_score"], s["sample_id"]))
    if k <= 0 or not cands:
        return []
    if k == 1:
        return [cands[len(cands) // 2]]
    if k >= len(cands):
        return cands
    return [cands[round(i * (len(cands) - 1) / (k - 1))] for i in range(k)]


def build_prompt(sample: dict, examples: list[dict]) -> str:
    inp = sample["input"]
    shots = "".join(
        EXAMPLE.format(i=i, total=ex["output"]["total_score"], code=ex["code"],
                       compile_log=ex["input"].get("compile_log") or "(empty)",
                       tests=format_tests(ex["input"].get("test_report"), detailed=False),
                       rubric=json.dumps(ex["output"]["rubric"]))
        for i, ex in enumerate(examples, 1))
    if shots:
        shots = "\n## Submissions already graded by the same teacher\n" + shots
    calibration = ("Grade the way this teacher does in the examples above "
                   "(same leniency, same use of each dimension).\n") if examples else ""
    return PROMPT.format(statement=sample["statement"], examples=shots, code=sample["code"],
                         compile_log=inp.get("compile_log") or "(empty)",
                         tests=format_tests(inp.get("test_report")), calibration=calibration)


def grade(sample: dict, pool: list[dict], shots: int, model: str) -> dict[str, int]:
    """Compile gate first; otherwise prompt the LLM and parse its JSON reply into clamped scores."""
    if has_compile_error(sample):
        return dict(COMPILE_FAIL_RUBRIC[sample["input"]["exam_id"]])
    reply = call_llm(build_prompt(sample, pick_examples(sample, pool, shots)), model)
    # The model may add prose or C++ snippets with braces after the JSON, so only take a flat
    # object that contains the rubric keys.
    match = re.search(r'\{[^{}]*"compilable"[^{}]*\}', reply)
    if not match:
        raise ValueError(f"{sample['sample_id']}: no rubric JSON in reply:\n{reply[:500]}")
    return Rubric.from_dict(json.loads(match.group(0))).to_dict()


def qwk(gold: list[int], pred: list[int], n: int = 11) -> float:
    """Quadratic weighted kappa over integer totals 0..10."""
    obs = [[0] * n for _ in range(n)]
    for g, p in zip(gold, pred):
        obs[g][p] += 1
    total = len(gold)
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(n)) for j in range(n)]
    num = sum((i - j) ** 2 * obs[i][j] for i in range(n) for j in range(n))
    den = sum((i - j) ** 2 * row[i] * col[j] / total for i in range(n) for j in range(n))
    return 1 - num / den if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=0, help="few-shot examples per prompt (0 = zero-shot)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--exam")
    ap.add_argument("--model", default="haiku")
    args = ap.parse_args()

    pool = load_samples()
    samples = [s for s in pool if not args.exam or s["input"]["exam_id"] == args.exam][: args.limit]
    # Local models get their own file so they don't overwrite the Claude predictions.
    model_tag = "_" + args.model.split("/")[-1] if "/" in args.model else ""
    out_file = OUT_DIR / f"t1_basic_preds_shots{args.shots}{model_tag}.jsonl"
    gold, pred = [], []
    with out_file.open("w", encoding="utf-8") as f:
        for s in samples:
            rubric = grade(s, pool, args.shots, args.model)
            total, gold_total = sum(rubric.values()), s["output"]["total_score"]
            gold.append(gold_total)
            pred.append(total)
            f.write(json.dumps({"sample_id": s["sample_id"], "rubric": rubric, "total_score": total}) + "\n")
            gate = " [compile gate]" if has_compile_error(s) else ""
            print(f"{s['sample_id']}  pred={total:2d}  gold={gold_total:2d}  {rubric}{gate}", flush=True)

    mae = sum(abs(g - p) for g, p in zip(gold, pred)) / len(gold)
    exact = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    print(f"\nn={len(gold)}  QWK={qwk(gold, pred):.3f}  MAE={mae:.2f}  exact_total={exact:.0%}  -> {out_file}")


if __name__ == "__main__":
    main()
