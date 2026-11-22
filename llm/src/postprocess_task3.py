"""Rule-based Task 3 post-steps, each grounded in running the student's own code (never in
gold labels) -- instructor QA allows rule-based / hybrid rule+LLM.

1. Level 2 + real g++ compile failure -> feedback = the first g++ error line (how teachers
   write it), and "Lỗi biên dịch" in the labels. Error line only, no source line: code in
   Level 1/2 feedback is a level-compliance violation.
2. Level 1 + passes EVERY verified LLM-generated test (test_report.py) -> "Tốt." with no
   error labels -- teachers give Level 1 to correct code, and Level 1 = signal/praise only.

Usage: python postprocess_task3.py <predictions_task3.json> --out-tag <run_tag> [--split public]
"""
import argparse
import json
import re

from compile_check import attach_real_compile_log, first_compile_error
from data import REPO_ROOT, load_public_test, load_sample_dataset
from test_report import attach_test_report

COMPILE_LABEL = "Lỗi biên dịch"


def passes_all_tests(test_report) -> bool:
    m = re.search(r"đúng (\d+)/(\d+)\.", test_report or "")
    return bool(m) and m.group(1) == m.group(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_path")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--split", choices=["sample", "public"], default="public")
    args = ap.parse_args()

    ds = load_public_test() if args.split == "public" else load_sample_dataset()
    attach_real_compile_log(ds, 3)
    attach_test_report(ds, 3)
    by_id = {s["sample_id"]: s for s in ds.tasks[3]}

    preds = json.load(open(args.pred_path, encoding="utf-8"))
    n_compile = n_label = n_good = n_good_changed = 0
    for p in preds:
        inp = by_id[p["sample_id"]]["input"]
        level, out = inp["target_feedback_level"], p["output"]
        if level.startswith("Level 2") and inp["compile_log"].startswith("Lỗi biên dịch"):
            out["feedback"] = first_compile_error(inp["compile_log"])
            n_compile += 1
            if out.get("taxonomy_error") is not None and COMPILE_LABEL not in out["taxonomy_error"]:
                out["taxonomy_error"].append(COMPILE_LABEL)
                n_label += 1
        elif level.startswith("Level 1") and passes_all_tests(inp.get("test_report")):
            n_good += 1
            if out["feedback"] != "Tốt." or out.get("taxonomy_error"):
                n_good_changed += 1
            out["feedback"] = "Tốt."
            if "taxonomy_error" in out:
                out["taxonomy_error"] = []

    out_path = REPO_ROOT / "outputs" / args.out_tag / f"predictions_task3_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"rule 1 (L2 compile error): {n_compile} feedbacks replaced, '{COMPILE_LABEL}' added to {n_label}")
    print(f"rule 2 (L1 passes all tests): {n_good} samples, {n_good_changed} actually changed")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
