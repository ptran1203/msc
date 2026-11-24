"""Score multi_problem submissions from per-problem LLM statuses times the statement's point weights.

The LLM never outputs a score: it classifies each problem (status below) and the total is computed
here, so the scale cannot be compressed by the model. Mappings were fixed BEFORE the first run and
are all reported; PRIMARY_MAPPING was chosen from the statement wording ("Hàm nào không làm thì để
thân hàm rỗng", "Không có điểm cho ý tưởng, chỉ chấm chương trình chạy được"), not from results.

Exploratory mappings added after the first scored run, both taken from the statement rule
"Nhận 0 điểm nếu ... tạo vòng lặp vô hạn": attempt+loop drops a problem that can loop forever,
attempt+loop0 zeroes the whole submission. They were chosen after seeing results; treat as hypotheses.

Statuses: correct | mostly (minor bug) | partial (some cases or fields right) |
          broken (real code that cannot produce a usable result: crash, infinite loop, wrong source) |
          missing (absent, declared only, or empty/trivial body)

Usage: python poc/problem_scoring.py --data-root <dataset> --problem-preds pp.jsonl --base base.jsonl \
           --mapping runs --out preds.jsonl
  base.jsonl supplies the rows this method does not grade (single_problem exams).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_data import COMPILE_FAIL_RUBRIC, DIMS, load_meta, load_samples
from signals import compile_status, statement_weights

STATUSES = ["correct", "mostly", "partial", "broken", "missing"]
MAPPINGS = {
    "attempt": {"correct": 1, "mostly": 1, "partial": 1, "broken": 1, "missing": 0},
    "runs": {"correct": 1, "mostly": 1, "partial": 1, "broken": 0, "missing": 0},
    "graded": {"correct": 1, "mostly": 1, "partial": 0.5, "broken": 0, "missing": 0},
    # "runs" plus the written prerequisite: P1 earns nothing -> P2..P4 are not graded.
    "runs+prereq": {"correct": 1, "mostly": 1, "partial": 1, "broken": 0, "missing": 0},
}
MAPPINGS["attempt+loop"] = MAPPINGS["attempt+loop0"] = MAPPINGS["attempt"]
PRIMARY_MAPPING = "runs"
PREREQ_MAPPINGS = {"runs+prereq"}


def total_from_problems(problems: list[dict], weights: dict[str, float], mapping: str) -> int:
    by_id = {p["problem"]: p for p in problems}
    if mapping == "attempt+loop0" and any(p.get("infinite_loop") for p in problems):
        return 0
    credit = {}
    for k in weights:
        p = by_id.get(k, {"status": "missing"})
        credit[k] = 0 if mapping == "attempt+loop" and p.get("infinite_loop") else MAPPINGS[mapping][p["status"]]
    if mapping in PREREQ_MAPPINGS and credit.get("prob1", 0) == 0:
        return 0
    return int(sum(weights[k] * c for k, c in credit.items()) + 0.5)  # round half up


def dims_from_total(total: int) -> dict[str, int]:
    """Spread a built multi_problem total over the 6 dims with a fixed fill order.

    compilable first, then complexity + code_quality together (released EX01 has both equal to
    [logic > 0]), then io_format, logic (max 4), edge_case (max 2). Sums back to `total` for 1..10.
    """
    dims = dict.fromkeys(DIMS, 0)
    if total <= 0:
        return dims
    dims["compilable"], rest = 1, total - 1
    if rest >= 3:
        dims["complexity"] = dims["code_quality"] = 1
        rest -= 2
    dims["io_format"] = min(1, rest)
    rest -= dims["io_format"]
    dims["logic"] = min(4, rest)
    rest -= dims["logic"]
    dims["edge_case"] = min(2, rest)
    return dims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--problem-preds", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--mapping", choices=sorted(MAPPINGS), default=PRIMARY_MAPPING)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    _, exams, _ = load_meta(args.data_root)
    rows = {r.sample_id: r for r in load_samples(args.data_root).itertuples()}
    base = {p["sample_id"]: p for p in map(json.loads, args.base.read_text(encoding="utf-8").splitlines())}
    graded = {p["sample_id"]: p for p in map(json.loads, args.problem_preds.read_text(encoding="utf-8").splitlines())}
    out = []
    for sid in sorted(base):
        r = rows[sid]
        if r.exam_type != "multi_problem":
            out.append(base[sid])
            continue
        if compile_status(r.compile_log or "") in ("error", "link"):
            dims, labels = dict(zip(DIMS, COMPILE_FAIL_RUBRIC[r.exam_id])), ["Lỗi biên dịch"]
        else:
            g = graded[sid]
            weights = statement_weights(exams[r.exam_id]["statement"])
            dims = dims_from_total(total_from_problems(g["problems"], weights, args.mapping))
            labels = [lab for lab in g["taxonomy_error"] if lab != "Lỗi biên dịch"]
        out.append({"sample_id": sid, "rubric": dims, "total_score": sum(dims.values()), "taxonomy_error": labels})
    args.out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in out), encoding="utf-8")
    print(f"{args.out} ({len(out)} rows, mapping={args.mapping})")


if __name__ == "__main__":
    main()
