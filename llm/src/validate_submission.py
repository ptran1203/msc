"""Validate a predictions.json against the schema inferred from the released data
(no official checker script has been provided in this data drop — see METHOD.md).

Usage: python validate_submission.py <task_num> <predictions.json> [--split sample|public]
"""
import argparse
import json
import sys

from data import load_public_test, load_sample_dataset

RUBRIC_RANGES = {
    "compilable": (0, 1), "io_format": (0, 1), "logic": (0, 4),
    "edge_case": (0, 2), "complexity": (0, 1), "code_quality": (0, 1),
}


def validate_task1(output, label_space):
    errors = []
    rubric = output.get("rubric")
    if not isinstance(rubric, dict):
        return ["rubric missing or not an object"]
    for dim, (lo, hi) in RUBRIC_RANGES.items():
        if dim not in rubric:
            errors.append(f"rubric missing dim {dim}")
        elif not isinstance(rubric[dim], int) or not (lo <= rubric[dim] <= hi):
            errors.append(f"rubric.{dim}={rubric.get(dim)} out of range [{lo},{hi}]")
    total = output.get("total_score")
    if not isinstance(total, (int, float)) or not (0 <= total <= 10):
        errors.append(f"total_score={total} out of range [0,10]")
    elif isinstance(rubric, dict) and not errors:
        rubric_sum = sum(rubric.values())
        if total != rubric_sum:
            errors.append(f"total_score={total} does not match sum of rubric dims ({rubric_sum})")
    return errors


def validate_task2(output, label_space):
    errors = []
    labels = output.get("taxonomy_error")
    if not isinstance(labels, list):
        return ["taxonomy_error missing or not a list"]
    valid = set(label_space["taxonomy_error"])
    bad = [l for l in labels if l not in valid]
    if bad:
        errors.append(f"unknown labels: {bad}")
    return errors


def validate_task3(output, label_space, self_diagnose):
    errors = []
    if self_diagnose:
        labels = output.get("taxonomy_error")
        if not isinstance(labels, list):
            errors.append("taxonomy_error missing or not a list")
        else:
            valid = set(label_space["taxonomy_error"])
            bad = [l for l in labels if l not in valid]
            if bad:
                errors.append(f"unknown labels: {bad}")
    fb = output.get("feedback")
    if not isinstance(fb, str) or not fb.strip():
        errors.append("feedback missing or empty")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_num", type=int, choices=[1, 2, 3])
    ap.add_argument("pred_path")
    ap.add_argument("--split", choices=["sample", "public"], default="public")
    args = ap.parse_args()

    ds = load_sample_dataset() if args.split == "sample" else load_public_test()
    expected_ids = {s["sample_id"] for s in ds.tasks[args.task_num]}
    self_diagnose = "taxonomy_error" not in ds.tasks[3][0]["input"] if args.task_num == 3 else False

    preds = json.loads(open(args.pred_path, encoding="utf-8").read())
    if not isinstance(preds, list):
        print("FAIL: top-level JSON must be a list"); sys.exit(1)

    seen_ids = set()
    total_errors = 0
    for i, p in enumerate(preds):
        if "sample_id" not in p or "output" not in p:
            print(f"FAIL [{i}]: missing sample_id/output"); total_errors += 1; continue
        sid = p["sample_id"]
        seen_ids.add(sid)
        if args.task_num == 1:
            errs = validate_task1(p["output"], ds.label_space)
        elif args.task_num == 2:
            errs = validate_task2(p["output"], ds.label_space)
        else:
            errs = validate_task3(p["output"], ds.label_space, self_diagnose)
        for e in errs:
            print(f"FAIL [{sid}]: {e}")
            total_errors += 1

    missing = expected_ids - seen_ids
    extra = seen_ids - expected_ids
    if missing:
        print(f"FAIL: missing {len(missing)} sample_id(s), e.g. {list(missing)[:5]}"); total_errors += len(missing)
    if extra:
        print(f"FAIL: {len(extra)} unexpected sample_id(s), e.g. {list(extra)[:5]}"); total_errors += len(extra)

    if total_errors == 0:
        print(f"OK: {len(preds)} predictions, all {len(expected_ids)} expected sample_ids present, schema valid.")
    else:
        print(f"\n{total_errors} total error(s) — DO NOT submit.")
        sys.exit(1)


if __name__ == "__main__":
    main()
