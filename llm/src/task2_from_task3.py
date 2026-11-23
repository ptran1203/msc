"""Task 2 labels from Task 3 self-diagnosis (+ the Task 2 prompt's own labels) + rules grounded
in running the student's code. All three tasks share the same submissions (same sample_ids),
so Task 3's self-diagnosed taxonomy_error is a prediction for the Task 2 item too.

Rules (never use gold or the teacher's feedback):
  compile  real g++ compile failure -> add "Lỗi biên dịch"; compiles fine -> remove it
  format   test_report: every failing test is a format mismatch -> add "Lỗi nhập/xuất"

  python task2_from_task3.py eval [--ml <loo json>]     # sample: score combinations on the 32 gold items
  python task2_from_task3.py build --t2 <public t2 json> --t3 <public t3 json> --combo <name> --out-tag <tag>
"""
import argparse
import json
import re
from pathlib import Path

from compile_check import attach_real_compile_log
from data import REPO_ROOT, load_public_test, load_sample_dataset
from eval import task2_metrics
from test_report import attach_test_report

COMPILE = "Lỗi biên dịch"
IO = "Lỗi nhập/xuất"
OUT = REPO_ROOT / "outputs"


def load_labels(path) -> dict:
    """{sample_id: [labels]} from a predictions file or a research-harness record file."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if rows and "pred_labels" in rows[0]:
        return {r["sample_id"]: list(r["pred_labels"]) for r in rows}
    return {r["sample_id"]: list(r["output"].get("taxonomy_error", [])) for r in rows}


def all_failures_format(report: str | None) -> bool:
    """True if the code ran and every failing test was a pure format mismatch."""
    if not report or "SAI ĐỊNH DẠNG" not in report:
        return False
    m = re.search(r"đúng (\d+)/(\d+)\.", report)
    if not m or m.group(1) == m.group(2):
        return False
    n_fail = int(m.group(2)) - int(m.group(1))
    n_fmt = sum(int(k) for k in re.findall(r"- (\d+) test: nội dung đúng nhưng SAI ĐỊNH DẠNG", report))
    return n_fmt == n_fail


def apply_rules(labels: list, inp: dict, rules: set) -> list:
    labels = list(labels)
    if "compile" in rules and inp.get("compile_log"):
        failed = inp["compile_log"].startswith("Lỗi biên dịch")
        if failed and COMPILE not in labels:
            labels.append(COMPILE)
        if not failed and COMPILE in labels:
            labels.remove(COMPILE)
    if "format" in rules and all_failures_format(inp.get("test_report")) and IO not in labels:
        labels.append(IO)
    return labels


def combine(t2: list, t3: list, how: str, ml: list | None = None) -> list:
    if how == "t3|ml":
        return list(dict.fromkeys(t3 + ml))
    if how == "t2|ml":  # no Task 3 involved: Task 2 prompt labels + classifier
        return list(dict.fromkeys(t2 + ml))
    if how == "vote":  # label kept if >=2 of the 3 sources (t2, t3, ml) predict it
        return [l for l in dict.fromkeys(t3 + t2 + ml) if (l in t2) + (l in t3) + (l in ml) >= 2]
    if how == "t2":
        return list(t2)
    if how == "t3":
        return list(t3)
    if how == "union":
        return list(dict.fromkeys(t3 + t2))
    if how == "intersect":
        return [l for l in t3 if l in t2]
    raise ValueError(how)


COMBOS = {f"{how}{'+' + '+'.join(r) if r else ''}": (how, set(r))
          for how in ("t2", "t3", "union", "intersect", "t3|ml", "t2|ml", "vote")
          for r in ((), ("compile",), ("format",), ("compile", "format"))}


def predict(ds, t2_by_id, t3_by_id, combo, ml_by_id=None):
    how, rules = COMBOS[combo]
    inp_by_id = {s["sample_id"]: s["input"] for s in ds.tasks[2]}
    ml = lambda sid: (ml_by_id or {}).get(sid, [])
    return {sid: apply_rules(combine(t2_by_id.get(sid, []), t3_by_id.get(sid, []), how, ml(sid)), inp, rules)
            for sid, inp in inp_by_id.items()}


def prep(ds):
    attach_real_compile_log(ds, 2)
    attach_test_report(ds, 2)
    return ds


def cmd_eval(args):
    ds = prep(load_sample_dataset())
    label_space = ds.label_space["taxonomy_error"]
    t2 = load_labels(args.t2)
    t3 = load_labels(args.t3)
    ml = load_labels(args.ml) if args.ml else None
    print(f"Task 2 sample (n={len(ds.tasks[2])}); t2={args.t2}\n  t3={args.t3}\n")
    print(f"{'combo':<26} {'macro_f1':>9} {'micro_f1':>9}")
    rows = []
    for combo in COMBOS:
        if ml is None and COMBOS[combo][0] in ("t3|ml", "t2|ml", "vote"):
            continue
        pred = predict(ds, t2, t3, combo, ml)
        m = task2_metrics(ds.tasks[2], pred, label_space)
        rows.append((combo, m["macro_f1"], m["micro_f1"]))
        print(f"{combo:<26} {m['macro_f1']:>9.4f} {m['micro_f1']:>9.4f}")
    best = max(rows, key=lambda r: r[1])
    print(f"\nbest macro: {best[0]} ({best[1]:.4f})")


def cmd_build(args):
    ds = prep(load_public_test())
    pred = predict(ds, load_labels(args.t2), load_labels(args.t3) if args.t3 else {}, args.combo, load_labels(args.ml) if args.ml else None)
    out = [{"sample_id": s["sample_id"], "output": {"taxonomy_error": pred[s["sample_id"]]}} for s in ds.tasks[2]]
    path = OUT / args.out_tag / "predictions_task2_public.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path} ({len(out)} items, combo={args.combo})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--t2", default=str(OUT / "qwen7b-fewshot-noreport-sample" / "predictions_task2_sample.json"))
    e.add_argument("--t3", default=str(OUT / "research_task3" / "testreport_ablation" / "A_no_report.json"))
    e.add_argument("--ml", default=None, help="3rd source, e.g. outputs/ml-task2-logreg/predictions_task2_sample_loo.json")
    b = sub.add_parser("build")
    b.add_argument("--ml", default=None)
    b.add_argument("--t2", required=True)
    b.add_argument("--t3", default=None, help="not needed for t2 / t2|ml combos")
    b.add_argument("--combo", required=True, choices=list(COMBOS))
    b.add_argument("--out-tag", required=True)
    args = ap.parse_args()
    {"eval": cmd_eval, "build": cmd_build}[args.cmd](args)
