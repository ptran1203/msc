"""Build the input + 6-dimension rubric EDA page from a dataset root.

All numbers in the page are computed here from the data (nothing typed by hand), except the
EX01 per-problem reading, which comes from the verified EDA report (manual code + feedback
reading) and is labelled as such in the page.
Usage: python viz/build_input_rubric_viz.py --data-root <dataset dir> [--out viz/input-rubric-eda.html]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from scipy.stats import spearmanr

EDA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDA_DIR))
from load_data import DIM_MAX, DIMS, LABEL_FIXES, load_meta, load_samples  # noqa: E402
from signals import classify_test, compile_status, ex01_functions, strip_comments  # noqa: E402

TEMPLATE = Path(__file__).with_name("input-rubric-template.html")
CHARS_PER_TOKEN = 3.5  # rough lower bound for code + Vietnamese text
LOG_CAP = 2000  # compile_log is cut at exactly this many characters in the release

# EX01 per-problem status from the verified EDA report (§2.4): manual reading of code + feedback.
# Feedback is used only for this EDA view, never as a model input. Policy = written rule
# (weights 1/4/2/3, P1 wrong -> P2..P4 not graded, compile failure / infinite loop -> 0).
EX01_POLICY = {
    "EX01-S101": {"p": ["ok", "ok", "none", "wrong"], "policy": 5},
    "EX01-S102": {"p": ["ok", "wrong", "ok", "ok"], "policy": 6},
    "EX01-S103": {"p": ["ce", "ce", "ce", "ce"], "policy": 0},
    "EX01-S104": {"p": ["ok", "ok", "wrong", "ok"], "policy": 8},
    "EX01-S105": {"p": ["wrong", "gated", "gated", "gated"], "policy": 0},
    "EX01-S106": {"p": ["unk", "unk", "loop", "unk"], "policy": 0},
    "EX01-S107": {"p": ["unk", "unk", "loop", "unk"], "policy": 0},
    "EX01-S108": {"p": ["wrong", "gated", "none", "none"], "policy": 0},
    "EX01-S109": {"p": ["ce", "ce", "ce", "ce"], "policy": 0},
    "EX01-S110": {"p": ["wrong", "gated", "none", "gated"], "policy": 0},
    "EX01-S111": {"p": ["ce", "ce", "ce", "ce"], "policy": 0},
    "EX01-S112": {"p": ["wrong", "wrong", "gated", "wrong"], "policy": 0},
    "EX01-S113": {"p": ["wrong", "gated", "gated", "gated"], "policy": 0},
    "EX01-S114": {"p": ["wrong", "gated", "gated", "gated"], "policy": 0},
    "EX01-S115": {"p": ["ok", "ok", "ok", "ok"], "policy": 10},
}


def build(root: Path) -> dict:
    df = load_samples(root)
    _, exams, _ = load_meta(root)
    rows = []
    for r in df.sort_values("sample_id").itertuples():
        code = r.code or ""
        eff_lines = [ln for ln in strip_comments(code).splitlines() if ln.strip()]
        tests = r.test_report if isinstance(r.test_report, list) else None
        verdicts = [classify_test(t) for t in tests] if tests is not None else None
        log = r.compile_log or ""
        statement = exams[r.exam_id]["statement"]
        prompt_chars = len(statement) + len(code) + len(log) + (len(json.dumps(tests, ensure_ascii=False)) if tests else 0)
        row = {
            "id": r.sample_id,
            "exam": r.exam_id,
            "type": r.exam_type,
            "dims": {d: int(getattr(r, d)) for d in DIMS},
            "total": int(r.total_score),
            "labels": list(r.labels),
            "level": int(r.level_num),
            "compile": compile_status(log),
            "log_chars": len(log),
            "log_truncated": len(log) == LOG_CAP,
            "code_lines": int(r.code_lines),
            "eff_lines": len(eff_lines),
            "commented_out": len(eff_lines) == 0,
            "prompt_tokens": round(prompt_chars / CHARS_PER_TOKEN),
            "tests": None,
        }
        if tests is not None:
            row["tests"] = [
                {"input": t["input"], "expected": t["expected"], "actual": t["actual"], "verdict": v}
                for t, v in zip(tests, verdicts)
            ]
            row["strict_pass"] = sum(v == "pass" for v in verdicts)
            row["lenient_pass"] = sum(v in ("pass", "prompt") for v in verdicts)
        if r.exam_id == "EX01":
            row["functions"] = ex01_functions(code)
            row.update(EX01_POLICY.get(r.sample_id, {}))
        rows.append(row)

    def holds(pred, sel):
        sub = [x for x in rows if sel(x)]
        return {"holds": sum(pred(x) for x in sub), "n": len(sub)}

    checks = [
        {"name": "total = Σ 6 chiều", "scope": "cả hai đề", **holds(lambda x: x["total"] == sum(x["dims"].values()), lambda x: True)},
        {"name": "edge_case = logic / 2", "scope": "EX02", **holds(lambda x: x["dims"]["edge_case"] * 2 == x["dims"]["logic"], lambda x: x["exam"] == "EX02")},
        {"name": "code_quality = 1", "scope": "EX02", **holds(lambda x: x["dims"]["code_quality"] == 1, lambda x: x["exam"] == "EX02")},
        {"name": "logic ∈ {0, 2, 4}", "scope": "EX02", **holds(lambda x: x["dims"]["logic"] in (0, 2, 4), lambda x: x["exam"] == "EX02")},
        {"name": "complexity = code_quality = [logic > 0]", "scope": "EX01", **holds(
            lambda x: x["dims"]["complexity"] == x["dims"]["code_quality"] == int(x["dims"]["logic"] > 0), lambda x: x["exam"] == "EX01")},
        {"name": "biên dịch lỗi ⇒ 000000", "scope": "EX01 (compilable = 0)", **holds(
            lambda x: sum(x["dims"].values()) == 0, lambda x: x["exam"] == "EX01" and x["dims"]["compilable"] == 0)},
        {"name": "biên dịch lỗi ⇒ 000001 (code_quality vẫn 1)", "scope": "EX02 (compilable = 0)", **holds(
            lambda x: [x["dims"][d] for d in DIMS] == [0, 0, 0, 0, 0, 1], lambda x: x["exam"] == "EX02" and x["dims"]["compilable"] == 0)},
        {"name": "log có lỗi build ⇔ compilable = 0", "scope": "cả hai đề", **holds(
            lambda x: (x["compile"] in ("error", "link")) == (x["dims"]["compilable"] == 0), lambda x: True)},
    ]
    # Compile-failure detectors vs the label (positive = "does not compile", i.e. compilable == 0).
    gcc_path = Path(__file__).with_name("gcc-results.json")
    gcc = json.loads(gcc_path.read_text(encoding="utf-8")) if gcc_path.exists() else None
    detectors = {"compile_log (BTC, g++ 13)": lambda x: x["compile"] in ("error", "link")}
    if gcc:
        for x in rows:
            x["gcc_fail"] = not gcc["results"][x["id"]]["ok"]
        detectors[f"Biên dịch lại: {gcc['compiler'].split(' (')[0]} {gcc['compiler'].split()[-1]}"] = lambda x: x["gcc_fail"]
    def confusion(detectors: dict, label, scope) -> list[dict]:
        out = []
        sub = [x for x in rows if scope(x)]
        for name, pred in detectors.items():
            cell = {"tp": [], "fp": [], "fn": [], "tn": []}
            for x in sub:
                p, y = bool(pred(x)), bool(label(x))
                cell[("t" if p == y else "f") + ("p" if p else "n")].append(x["id"])
            tp, fp, fn = len(cell["tp"]), len(cell["fp"]), len(cell["fn"])
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            out.append({"name": name, **cell, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)})
        return out

    def gold(x, dim, fixed=True):
        return LABEL_FIXES.get(x["id"], {}).get(dim, x["dims"][dim]) if fixed else x["dims"][dim]

    compile_conf = confusion(detectors, lambda x: gold(x, "compilable") == 0, lambda x: True)
    compile_orig = confusion(detectors, lambda x: gold(x, "compilable", False) == 0, lambda x: True)
    # io_format: positive = io_format == 0; only EX02 samples whose test_report has real tests.
    io_detectors = {
        "test_report: có ≥ 1 test fail (so khớp chính xác)": lambda x: x["strict_pass"] < len(x["tests"]),
        "test_report: có test fail chỉ vì định dạng (dòng nhắc)": lambda x: any(t["verdict"] in ("prompt", "prompt_wrong") for t in x["tests"]),
        "test_report: fail toàn bộ test": lambda x: x["strict_pass"] == 0,
    }
    io_conf = confusion(io_detectors, lambda x: gold(x, "io_format") == 0, lambda x: bool(x["tests"]))
    io_orig = confusion(io_detectors, lambda x: gold(x, "io_format", False) == 0, lambda x: bool(x["tests"]))

    with_tests = [x for x in rows if x["tests"]]
    totals = [x["total"] for x in with_tests]
    corr = {
        "n": len(with_tests),
        "strict": round(float(spearmanr([x["strict_pass"] for x in with_tests], totals)[0]), 2),
        "lenient": round(float(spearmanr([x["lenient_pass"] for x in with_tests], totals)[0]), 2),
        "expected_true": sum(t["expected"] == "True" for t in with_tests[0]["tests"]) if with_tests else 0,
        "n_public": len(with_tests[0]["tests"]) if with_tests else 0,
    }
    return {
        "generated_from": str(root.name),
        "ex02_corr": corr,
        "compile_confusion": compile_conf,
        "io_confusion": io_conf,
        "compile_confusion_original": compile_orig,
        "io_confusion_original": io_orig,
        "label_fixes": LABEL_FIXES,
        "n": len(rows),
        "dims": DIMS,
        "dim_max": DIM_MAX,
        "exams": {k: {"type": v["exam_type"], "course": v.get("course"), "has_hidden_tests": v.get("has_hidden_testcases")} for k, v in exams.items()},
        "samples": rows,
        "checks": checks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("input-rubric-eda.html"))
    ap.add_argument("--dump-json", type=Path, help="also write the computed data as JSON (for checking)")
    args = ap.parse_args()
    data = build(args.data_root)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
    args.out.write_text(html, encoding="utf-8")
    if args.dump_json:
        args.dump_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({len(html):,} bytes, {data['n']} samples)")


if __name__ == "__main__":
    main()
