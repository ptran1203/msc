"""Score prediction files with the challenge metrics, on released and on corrected gold.

T1: QWK on total_score (fixed 0..10 axis), MAE, exact match per dimension.
T2: macro-F1 and micro-F1 over the 10 taxonomy labels (zero_division=0), plus per-label F1.
Corrected gold applies LABEL_FIXES (a build-failure fix takes the exam's compile-fail rubric), recomputes the
total and adds the implied taxonomy label.
Usage: python poc/score_predictions.py --data-root <dataset> pred1.jsonl [pred2.jsonl ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_data import COMPILE_FAIL_RUBRIC, DIMS, FIX_IMPLIED_LABEL, LABEL_FIXES, load_meta, load_samples  # noqa: E402


def gold_table(root: Path, fixed: bool) -> dict[str, dict]:
    gold = {}
    for r in load_samples(root).itertuples():
        dims = {d: int(getattr(r, d)) for d in DIMS}
        labels = list(r.labels)
        if fixed:
            for d, v in LABEL_FIXES.get(r.sample_id, {}).items():
                dims[d] = v
                if d == "compilable" and v == 0:  # a build failure takes the exam's compile-fail rubric
                    dims = dict(zip(DIMS, COMPILE_FAIL_RUBRIC[r.exam_id]))
                if v == 0 and FIX_IMPLIED_LABEL[d] not in labels:
                    labels.append(FIX_IMPLIED_LABEL[d])
        total = sum(dims.values()) if fixed and r.sample_id in LABEL_FIXES else int(r.total_score)
        gold[r.sample_id] = {"exam": r.exam_id, "rubric": dims, "total": total, "labels": labels}
    return gold


def score(gold: dict, preds: dict, taxonomy: list[str], ids: list[str]) -> dict:
    y = np.array([gold[i]["total"] for i in ids])
    p = np.array([int(preds[i]["total_score"]) for i in ids])
    out = {
        "n": len(ids),
        "qwk_total": round(float(cohen_kappa_score(y, p, weights="quadratic", labels=list(range(11)))), 3),
        "mae_total": round(float(np.abs(y - p).mean()), 2),
        "exact": {d: round(float(np.mean([gold[i]["rubric"][d] == preds[i]["rubric"][d] for i in ids])), 3) for d in DIMS},
    }
    Y = np.array([[lab in gold[i]["labels"] for lab in taxonomy] for i in ids], dtype=int)
    P = np.array([[lab in preds[i]["taxonomy_error"] for lab in taxonomy] for i in ids], dtype=int)
    out["macro_f1"] = round(float(f1_score(Y, P, average="macro", zero_division=0)), 3)
    out["micro_f1"] = round(float(f1_score(Y, P, average="micro", zero_division=0)), 3)
    out["per_label_f1"] = dict(zip(taxonomy, [round(float(v), 3) for v in f1_score(Y, P, average=None, zero_division=0)]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("preds", type=Path, nargs="+")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    label_space, _, _ = load_meta(args.data_root)
    taxonomy = label_space["taxonomy_error"]
    report = {}
    print(f"{'run':<26}{'gold':<7}{'scope':<6}{'QWK':>7}{'MAE':>7}{'macroF1':>9}{'microF1':>9}")
    for path in args.preds:
        preds = {p["sample_id"]: p for p in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
        for fixed in (False, True):
            gold = gold_table(args.data_root, fixed)
            for scope in ("all", "EX01", "EX02"):
                ids = sorted(i for i in preds if scope == "all" or gold[i]["exam"] == scope)
                s = score(gold, preds, taxonomy, ids)
                key = f"{path.stem}|{'fixed' if fixed else 'released'}|{scope}"
                report[key] = s
                print(f"{path.stem:<26}{'fixed' if fixed else 'rel':<7}{scope:<6}{s['qwk_total']:>7}{s['mae_total']:>7}"
                      f"{s['macro_f1']:>9}{s['micro_f1']:>9}")
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
