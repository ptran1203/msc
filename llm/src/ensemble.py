"""Combine predictions from multiple already-generated runs into one ensembled
predictions file -- no GPU needed, works on whatever's already in outputs/.

Task 1 (rubric): average each rubric dimension across inputs (rounded, clipped to its
    valid range), total_score recomputed as the sum (same invariant run_prompting.py
    enforces).
Task 2 (taxonomy_error): per-label vote -- a label is kept if it appears in at least
    --vote-threshold (default 0.5) of the inputs. With 2 inputs this defaults to union
    (favors recall); raise --vote-threshold toward 1.0 for intersection-like behavior
    (favors precision).
Task 3 (feedback): free text can't be averaged -- feedback is taken as-is from the
    FIRST input path (the "primary" run). Only taxonomy_error (if the split requires
    self-diagnosis) is combined by the same per-label vote as Task 2.

Usage:
    python ensemble.py 1 ensembled.json runA.json runB.json [runC.json ...]
    python ensemble.py 2 ensembled.json runA.json runB.json [runC.json ...] --vote-threshold 0.5
    python ensemble.py 3 ensembled.json primary.json runB.json [runC.json ...] --vote-threshold 0.5
"""
import argparse
import json

from validate_submission import RUBRIC_RANGES


def load(path):
    preds = json.load(open(path, encoding="utf-8"))
    return {p["sample_id"]: p["output"] for p in preds}


def vote_labels(label_lists, threshold):
    n = len(label_lists)
    counts = {}
    for labels in label_lists:
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
    return [l for l, c in counts.items() if c / n >= threshold]


def ensemble_task1(outputs_by_file, sample_ids):
    merged = {}
    for sid in sample_ids:
        rubric = {}
        for dim, (lo, hi) in RUBRIC_RANGES.items():
            vals = [o[sid]["rubric"][dim] for o in outputs_by_file]
            avg = round(sum(vals) / len(vals))
            rubric[dim] = max(lo, min(hi, avg))
        merged[sid] = {"rubric": rubric, "total_score": sum(rubric.values())}
    return merged


def ensemble_task2(outputs_by_file, sample_ids, threshold):
    merged = {}
    for sid in sample_ids:
        label_lists = [o[sid]["taxonomy_error"] for o in outputs_by_file]
        merged[sid] = {"taxonomy_error": vote_labels(label_lists, threshold)}
    return merged


def ensemble_task3(outputs_by_file, sample_ids, threshold):
    merged = {}
    primary = outputs_by_file[0]
    self_diagnose = "taxonomy_error" in primary[sample_ids[0]]
    for sid in sample_ids:
        out = {"feedback": primary[sid]["feedback"]}
        if self_diagnose:
            label_lists = [o[sid]["taxonomy_error"] for o in outputs_by_file]
            out = {"taxonomy_error": vote_labels(label_lists, threshold), **out}
        merged[sid] = out
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_num", type=int, choices=[1, 2, 3])
    ap.add_argument("out_path")
    ap.add_argument("inputs", nargs="+", help="2+ predictions.json files; for task 3 the FIRST is the feedback source")
    ap.add_argument("--vote-threshold", type=float, default=0.5, help="task 2/3: keep a label if it appears in >= this fraction of inputs (default 0.5)")
    args = ap.parse_args()
    if len(args.inputs) < 2:
        raise SystemExit("need at least 2 input files to ensemble")

    outputs_by_file = [load(p) for p in args.inputs]
    sample_ids = sorted(outputs_by_file[0].keys())
    for p, o in zip(args.inputs, outputs_by_file):
        if set(o.keys()) != set(sample_ids):
            raise SystemExit(f"{p} has a different sample_id set than {args.inputs[0]} -- can't ensemble")

    if args.task_num == 1:
        merged = ensemble_task1(outputs_by_file, sample_ids)
    elif args.task_num == 2:
        merged = ensemble_task2(outputs_by_file, sample_ids, args.vote_threshold)
    else:
        merged = ensemble_task3(outputs_by_file, sample_ids, args.vote_threshold)

    predictions = [{"sample_id": sid, "output": merged[sid]} for sid in sample_ids]
    json.dump(predictions, open(args.out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {args.out_path} ({len(predictions)} samples, ensembled from {len(args.inputs)} inputs)")


if __name__ == "__main__":
    main()
