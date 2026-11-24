"""LLM-free baselines, written as predictions so they are scored exactly like model output.

B0 prior: leave-one-out per-exam majority for every dimension and the most frequent label set.
B1 signals: compile gate (exam policy) + public-test verdicts for EX02, B0 prior for the rest.
Usage: python poc/rule_baseline.py --data-root <dataset> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_data import COMPILE_FAIL_RUBRIC, DIMS, load_samples  # noqa: E402
from signals import compile_status, test_summary  # noqa: E402



def prior(rows: list, exclude: str) -> tuple[dict, list]:
    rest = [r for r in rows if r.sample_id != exclude]
    dims = {d: Counter(getattr(r, d) for r in rest).most_common(1)[0][0] for d in DIMS}
    labels = Counter(tuple(sorted(r.labels)) for r in rest).most_common(1)[0][0]
    return {d: int(v) for d, v in dims.items()}, list(labels)


def predict(r, rows: list, use_signals: bool) -> dict:
    dims, labels = prior([x for x in rows if x.exam_id == r.exam_id], r.sample_id)
    if use_signals:
        if compile_status(r.compile_log or "") in ("error", "link"):
            dims = dict(zip(DIMS, COMPILE_FAIL_RUBRIC[r.exam_id]))
            labels = ["Lỗi biên dịch"]
        elif (ts := test_summary(r.test_report if isinstance(r.test_report, list) else None)):
            logic = 4 if ts["lenient_pass"] == ts["n"] else (2 if ts["lenient_pass"] > 0 else 0)
            dims.update(compilable=1, io_format=int(ts["strict_pass"] > 0), logic=logic, edge_case=logic // 2,
                        code_quality=1)
            labels = (["Lỗi nhập/xuất"] if ts["strict_pass"] == 0 else []) + (["Lỗi logic"] if logic < 4 else [])
    return {"sample_id": r.sample_id, "rubric": dims, "total_score": sum(dims.values()), "taxonomy_error": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    rows = list(load_samples(args.data_root).sort_values("sample_id").itertuples())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, use_signals in (("b0-prior", False), ("b1-signals", True)):
        out = args.out_dir / f"{name}.jsonl"
        out.write_text("".join(json.dumps(predict(r, rows, use_signals), ensure_ascii=False) + "\n" for r in rows),
                       encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
