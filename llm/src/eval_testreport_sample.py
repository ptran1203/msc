"""Ablation on the 32 gold sample items: what does test_report (and the Task 3 rules) add?

Task 3 uses the research harness setup (forced self-diagnosis + same-level RAG exemplar =
production prompt, see research_task3_ideas.py), with compile-only compile_log:
  A  no test_report
  B  + test_report (same block prompts.task3_prompt inserts)
  A+rules / B+rules  postprocess_task3.py's rules 1+2 applied to A / B
Tasks 1/2 are measured with run_prompting.py --split sample [--no-test-report].

Usage: python -u eval_testreport_sample.py [--model HF_ID] [--thinking] [--limit N]
"""
import argparse
import json

import numpy as np

import research_task3_ideas as R
from compile_check import attach_real_compile_log, first_compile_error
from data import load_sample_dataset, read_code
from llm_client import MODEL_NAME, LocalLLM
from postprocess_task3 import COMPILE_LABEL, passes_all_tests
from rag import _get_model
from test_report import attach_test_report



def with_report(prompt, sample):
    report = sample["input"].get("test_report")
    if not report:
        return prompt
    log = sample["input"].get("compile_log") or "(không có)"
    anchor = f"## compile_log\n{log}\n"
    assert anchor in prompt
    return prompt.replace(anchor, f"{anchor}\n## test_report (chạy code trên test sinh tự động)\n{report}\n", 1)


def apply_rules(records, ds):
    by_id = {s["sample_id"]: s["input"] for s in ds.tasks[3]}
    out = []
    for r in records:
        r = dict(r, pred_labels=list(r["pred_labels"]))
        inp = by_id[r["sample_id"]]
        level = inp["target_feedback_level"]
        if level.startswith("Level 2") and inp["compile_log"].startswith("Lỗi biên dịch"):
            r["pred_feedback"] = first_compile_error(inp["compile_log"])
            if COMPILE_LABEL not in r["pred_labels"]:
                r["pred_labels"].append(COMPILE_LABEL)
        elif level.startswith("Level 1") and passes_all_tests(inp.get("test_report")):
            r["pred_feedback"], r["pred_labels"] = "Tốt.", []
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--thinking", action="store_true", help="Qwen3-style thinking mode (see LocalLLM)")
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke test)")
    args = ap.parse_args()
    # 7B default keeps the original folder; other models/modes get their own
    tag = "" if args.model == MODEL_NAME and not args.thinking else args.model.split("/")[-1].lower() + ("-thinking" if args.thinking else "")
    out = R.OUT_DIR / "testreport_ablation" / tag if tag else R.OUT_DIR / "testreport_ablation"
    out.mkdir(parents=True, exist_ok=True)
    R.OUT_DIR = out  # run_variant writes its per-variant json there too

    ds = load_sample_dataset()
    if args.limit:
        ds.tasks[3] = ds.tasks[3][: args.limit]
    attach_real_compile_log(ds, 3)
    attach_test_report(ds, 3)
    valid = set(ds.label_space["taxonomy_error"])

    emb = _get_model().encode([read_code(ds, s) for s in ds.tasks[3]], normalize_embeddings=True)

    def rag_fewshot(sample):  # identical to research_task3_ideas idea2
        idx = ds.tasks[3].index(sample)
        for i in np.argsort(-(emb @ emb[idx])):
            c = ds.tasks[3][i]
            if c["sample_id"] != sample["sample_id"] and c["input"]["target_feedback_level"] == sample["input"]["target_feedback_level"]:
                return [(ds, c)]
        return []

    print(f"model={args.model} thinking={args.thinking} n={len(ds.tasks[3])} -> {out}")
    llm = LocalLLM(model_name=args.model, thinking=args.thinking)
    base = lambda s: R.build_self_diagnose_prompt(ds, s, fewshot_items=rag_fewshot(s))
    runs = {
        "A_no_report": R.run_variant("A_no_report", llm, ds, valid, base),
        "B_test_report": R.run_variant("B_test_report", llm, ds, valid, lambda s: with_report(base(s), s)),
    }
    runs["A+rules"] = apply_rules(runs["A_no_report"], ds)
    runs["B+rules"] = apply_rules(runs["B_test_report"], ds)

    results = []
    for name, recs in runs.items():
        (out / f"{name}.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(R.score(name, ds, recs))
    print(f"\n=== Task 3 ablation (n={len(ds.tasks[3])}, {sum(1 for s in ds.tasks[3] if s['input'].get('test_report'))} with test_report) ===")
    print(f"{'variant':<16} {'macro_f1':>9} {'micro_f1':>9} {'jaccard':>9} {'compliance':>11} {'overlap_f1':>11}")
    for r in results:
        print(f"{r['name']:<16} {r['macro_f1']:>9} {r['micro_f1']:>9} {r['mean_jaccard']:>9} {r['compliance_rate']:>11} {r['overlap_f1']:>11}")
    (out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
