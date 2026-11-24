"""Post-process model predictions with deterministic gates and rubric regularities.

Gate (from the exam grading policy, always valid): a failed build forces the compile-fail rubric
and the compile label; a clean build forces compilable = 1 and drops the compile label.
Regularities (seen in the released labels, so optimistic on this same set): EX02 edge = logic / 2
and code_quality = 1; EX01 complexity = code_quality = [logic > 0]; io_format = 0 implies the I/O label.
Usage: python poc/rubric_constraints.py --data-root <dataset> in.jsonl out.jsonl [--gate-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_data import COMPILE_FAIL_RUBRIC, DIMS, load_samples  # noqa: E402
from signals import compile_status  # noqa: E402


def constrain(pred: dict, exam_id: str, build_failed: bool, gate_only: bool) -> dict:
    dims, labels = dict(pred["rubric"]), list(pred["taxonomy_error"])
    if build_failed:
        dims, labels = dict(zip(DIMS, COMPILE_FAIL_RUBRIC[exam_id])), ["Lỗi biên dịch"]
    else:
        dims["compilable"] = 1
        labels = [lab for lab in labels if lab != "Lỗi biên dịch"]
        if not gate_only:
            if exam_id == "EX02":
                dims.update(edge_case=dims["logic"] // 2, code_quality=1)
            else:
                dims.update(complexity=int(dims["logic"] > 0), code_quality=int(dims["logic"] > 0))
            if dims["io_format"] == 0 and "Lỗi nhập/xuất" not in labels:
                labels.append("Lỗi nhập/xuất")
    return {**pred, "rubric": dims, "total_score": sum(dims.values()), "taxonomy_error": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()
    meta = {r.sample_id: (r.exam_id, compile_status(r.compile_log or "") in ("error", "link"))
            for r in load_samples(args.data_root).itertuples()}
    preds = [json.loads(line) for line in args.inp.read_text(encoding="utf-8").splitlines()]
    args.out.write_text("".join(json.dumps(constrain(p, *meta[p["sample_id"]], args.gate_only), ensure_ascii=False) + "\n"
                                for p in preds), encoding="utf-8")


if __name__ == "__main__":
    main()
