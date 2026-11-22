"""Rule-based Task 3 post-step: for Level 2 (Chẩn đoán) samples whose real g++ compile
fails, replace the feedback with the first g++ error line -- exactly how teachers write
it in the gold data ("student.cpp:51:26: error: expected primary-expression ..."), and
make sure "Lỗi biên dịch" is in the self-diagnosed labels (all gold compile-error L2
samples carry it). Grounded in the real compile, never in gold labels.

Usage: python postprocess_task3_compile.py <predictions_task3.json> --out-tag <run_tag> [--split public]
"""
import argparse
import json

from compile_check import attach_real_compile_log, first_compile_error
from data import REPO_ROOT, load_public_test, load_sample_dataset

COMPILE_LABEL = "Lỗi biên dịch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_path")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--split", choices=["sample", "public"], default="public")
    args = ap.parse_args()

    ds = load_public_test() if args.split == "public" else load_sample_dataset()
    attach_real_compile_log(ds, 3)
    by_id = {s["sample_id"]: s for s in ds.tasks[3]}

    preds = json.load(open(args.pred_path, encoding="utf-8"))
    n_fb = n_label = 0
    for p in preds:
        inp = by_id[p["sample_id"]]["input"]
        if not inp["target_feedback_level"].startswith("Level 2") or not inp["compile_log"].startswith("Lỗi biên dịch"):
            continue
        p["output"]["feedback"] = first_compile_error(inp["compile_log"])
        n_fb += 1
        labels = p["output"].get("taxonomy_error")
        if labels is not None and COMPILE_LABEL not in labels:
            labels.append(COMPILE_LABEL)
            n_label += 1

    out_path = REPO_ROOT / "outputs" / args.out_tag / f"predictions_task3_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"{n_fb}/{len(preds)} Level-2 compile-error feedbacks replaced, '{COMPILE_LABEL}' added to {n_label} -> {out_path}")


if __name__ == "__main__":
    main()
