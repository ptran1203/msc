"""EDA on sample_dataset (labeled) and public_test (unlabeled inputs only)."""
import json
import re
from collections import Counter
from pathlib import Path

from data import REPO_ROOT, load_public_test, load_sample_dataset, read_code

RUBRIC_DIMS = ["compilable", "io_format", "logic", "edge_case", "complexity", "code_quality"]


def rubric_distribution(tasks1):
    dist = {d: Counter() for d in RUBRIC_DIMS}
    totals = Counter()
    for s in tasks1:
        r = s["output"]["rubric"]
        for d in RUBRIC_DIMS:
            dist[d][r[d]] += 1
        totals[s["output"]["total_score"]] += 1
    return dist, totals


def taxonomy_distribution(tasks2):
    label_count = Counter()
    n_labels_per_sample = Counter()
    for s in tasks2:
        labels = s["output"]["taxonomy_error"]
        n_labels_per_sample[len(labels)] += 1
        for l in labels:
            label_count[l] += 1
    return label_count, n_labels_per_sample


def feedback_level_distribution(tasks3):
    return Counter(s["input"]["target_feedback_level"] for s in tasks3)


def compile_log_emptiness(tasks1):
    """compile_log is documented as a secondary signal; check how often it's actually populated."""
    nonempty = [s["sample_id"] for s in tasks1 if (s["input"].get("compile_log") or "").strip()]
    return nonempty


def exam_type_breakdown(ds, task_samples):
    c = Counter()
    for s in task_samples:
        exam = ds.exams[s["input"]["exam_id"]]
        c[(exam["exam_id"], exam["exam_type"])] += 1
    return c


def feedback_level_leak_check(tasks3):
    """Check for code-like content in Level 1/2 feedback (should be prose only)."""
    code_markers = re.compile(r"(\bint\s+\w+\s*\(|#include|for\s*\(|while\s*\(|\{\s*\n|;\s*\n)")
    violations = []
    for s in tasks3:
        level = s["input"]["target_feedback_level"]
        if level.startswith("Level 1") or level.startswith("Level 2"):
            fb = s["output"]["feedback"]
            if code_markers.search(fb):
                violations.append((s["sample_id"], level, fb[:100]))
    return violations


def run(ds, name):
    print(f"\n=== {name} ===")
    print(f"n_samples: {len(ds.tasks[1])}")
    print("exam breakdown (task1):", dict(exam_type_breakdown(ds, ds.tasks[1])))

    dist, totals = rubric_distribution(ds.tasks[1])
    print("\ntotal_score distribution:", dict(sorted(totals.items())))
    for d in RUBRIC_DIMS:
        print(f"  {d}: {dict(sorted(dist[d].items()))}")

    label_count, n_per_sample = taxonomy_distribution(ds.tasks[2])
    print("\ntaxonomy_error label frequency:", dict(label_count.most_common()))
    print("n_labels per sample:", dict(sorted(n_per_sample.items())))
    empty_labels = n_per_sample.get(0, 0)
    print(f"  -> {empty_labels}/{len(ds.tasks[2])} samples have NO error label ([])")

    level_dist = feedback_level_distribution(ds.tasks[3])
    print("\ntarget_feedback_level distribution:", dict(level_dist))

    nonempty_logs = compile_log_emptiness(ds.tasks[1])
    print(f"\ncompile_log non-empty: {len(nonempty_logs)}/{len(ds.tasks[1])} (secondary signal appears unpopulated if 0)")

    violations = feedback_level_leak_check(ds.tasks[3])
    print(f"\nLevel 1/2 feedback containing code-like markers: {len(violations)}")
    for sid, level, fb in violations[:5]:
        print(f"  {sid} ({level}): {fb!r}")

    return {
        "n_samples": len(ds.tasks[1]),
        "total_score_dist": dict(sorted(totals.items())),
        "rubric_dist": {d: dict(sorted(dist[d].items())) for d in RUBRIC_DIMS},
        "taxonomy_label_freq": dict(label_count.most_common()),
        "n_labels_per_sample": dict(sorted(n_per_sample.items())),
        "feedback_level_dist": dict(level_dist),
        "compile_log_nonempty_count": len(nonempty_logs),
        "level_leak_violations": len(violations),
    }


if __name__ == "__main__":
    ds_sample = load_sample_dataset()
    ds_public = load_public_test()

    summary_sample = run(ds_sample, "sample_dataset (32, LABELED)")

    # public_test has no labels -> only input-side stats
    print("\n\n=== public_test (328, UNLABELED) ===")
    print("exam breakdown (task1 inputs):", dict(exam_type_breakdown(ds_public, ds_public.tasks[1])))
    level_dist_pub = feedback_level_distribution(ds_public.tasks[3])
    print("target_feedback_level distribution (requested, input-side):", dict(level_dist_pub))
    nonempty_logs_pub = compile_log_emptiness(ds_public.tasks[1])
    print(f"compile_log non-empty: {len(nonempty_logs_pub)}/{len(ds_public.tasks[1])}")

    out_path = REPO_ROOT / "reports/eda_summary.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_dataset": summary_sample,
                "public_test": {
                    "n_samples": len(ds_public.tasks[1]),
                    "exam_breakdown": {f"{k[0]}|{k[1]}": v for k, v in exam_type_breakdown(ds_public, ds_public.tasks[1]).items()},
                    "feedback_level_dist_requested": dict(level_dist_pub),
                    "compile_log_nonempty_count": len(nonempty_logs_pub),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nWrote {out_path}")
