"""Task 2 classifier grid: features x heads, scored leave-one-out on the 32 real items.

Features
  emb    all-MiniLM-L6-v2 code embedding (sees only the first 256 tokens)
  feat   grounded run signals (real g++ compile, test pass/format/timeout/crash fractions,
         no-code) + code statistics (loops, nesting, arrays, strings, functions, literals...)
Heads: LogReg, LinearSVC, LightGBM (one-vs-rest, class-balanced).
Training = 593 synth items (oracle labels) + the other 31 real items.

Synth items have no compile_log / test_report in the data, so they get the same signals by
running them through the same machinery: compile_check + test_report with suites built from
synth_oracle's generators and reference solutions (written to synth/work/llm_tests, never to
outputs/llm_tests, which is the real-exam suite set shipped to Kaggle).

Must run from PowerShell (g++). Usage: python -u ml_task2_grid.py [build]
"""
import json
import re
import warnings

import numpy as np
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import LinearSVC

import synth_oracle
import test_report as TR
from compile_check import attach_real_compile_log
from data import REPO_ROOT, load_dataset, load_public_test, load_sample_dataset, read_code
from eval import task2_metrics
from rag import EMBED_MODEL, SYNTH_DIR
from synth_generate import strip_cpp_comments
from task2_from_task3 import COMBOS, load_labels, predict

warnings.filterwarnings("ignore")
SEED = 42
OUT = REPO_ROOT / "outputs" / "ml-task2-grid"
SYNTH_SUITES = REPO_ROOT / "synth" / "work" / "llm_tests"
SYN_PROBLEM = {"SYN-ARMSTRONG": "armstrong", "SYN-DEDUP": "remove_duplicates", "SYN-LOGIN": "earliest_login",
               "SYN-BINSEARCH": "binary_search", "SYN-PALINDROME": "palindrome_check"}


def build_synth_suites():
    SYNTH_SUITES.mkdir(parents=True, exist_ok=True)
    for exam_id, problem in SYN_PROBLEM.items():
        path = SYNTH_SUITES / f"{exam_id}.json"
        if not path.exists():
            cases = synth_oracle.build_oracle(problem)
            tests = [{"input": c["input"], "expected": c["expected"], "is_example": False} for c in cases]
            path.write_text(json.dumps({"exam_id": exam_id, "tests": tests}, ensure_ascii=False), encoding="utf-8")


def attach_signals(ds, suites_dir=None):
    attach_real_compile_log(ds, 2)
    if suites_dir is None:
        TR.attach_test_report(ds, 2)
        return
    saved = TR.TESTS_DIR
    TR.TESTS_DIR = suites_dir
    try:
        TR.attach_test_report(ds, 2)
    finally:
        TR.TESTS_DIR = saved


def report_feats(rep):
    """[has_report, no_code, pass, format, timeout, crash, wrong] (fractions of tests run)."""
    if not rep:
        return [0, 0, 0, 0, 0, 0, 0]
    if rep.startswith("Bài nộp không có code"):
        return [1, 1, 0, 0, 0, 0, 0]
    if rep.startswith("Không chạy được test"):
        return [1, 0, 0, 0, 0, 0, 0]
    m = re.search(r"đúng (\d+)/(\d+)\.", rep)
    ok, n = int(m.group(1)), max(int(m.group(2)), 1)
    num = lambda pat: sum(int(x) for x in re.findall(pat, rep))
    fmt = num(r"- (\d+) test: nội dung đúng nhưng SAI ĐỊNH DẠNG")
    to = num(r"- (\d+) test quá thời gian")
    cr = num(r"- (\d+) test chương trình bị lỗi")
    wrong = max(n - ok - fmt - to - cr, 0)
    return [1, 0, ok / n, fmt / n, to / n, cr / n, wrong / n]


def code_feats(code):
    c = strip_cpp_comments(code)
    lines = [l for l in c.splitlines() if l.strip()]
    cnt = lambda pat: len(re.findall(pat, c))
    depth = max_depth = 0
    loop_depth = []  # crude loop nesting: track brace depth at each loop keyword
    for tok in re.finditer(r"\b(for|while)\b|\{|\}", c):
        t = tok.group(0)
        if t == "{":
            depth += 1
        elif t == "}":
            depth -= 1
            loop_depth = [d for d in loop_depth if d <= depth]
        else:
            loop_depth.append(depth)
            max_depth = max(max_depth, len(loop_depth))
    n_funcs = cnt(r"^\s*(?:int|void|bool|double|float|long|string|char|vector<[^>]*>)\s+\**\w+\s*\([^;]*\)\s*\{")
    return [
        np.log1p(len(lines)), cnt(r"\bfor\b"), cnt(r"\bwhile\b"), max_depth, cnt(r"\bif\b"),
        max(n_funcs - 1, 0), cnt(r"\w\s*\["), cnt(r"\bstring\b|\bchar\b|strlen|getline"),
        cnt(r"\bvector\b"), cnt(r"\bcout\b|\bprintf\b"), cnt(r"\bcin\b|\bscanf\b"),
        cnt(r'cout\s*<<\s*"[^"]*"\s*;'), cnt(r"\b\d{2,}\b"), cnt(r"\bpow\s*\("),
        cnt(r"\breturn\b"), int(not c.strip()), cnt(r"#include"),
    ]


def features(ds):
    rows = []
    for s in ds.tasks[2]:
        inp = s["input"]
        compile_fail = int((inp.get("compile_log") or "").startswith("Lỗi biên dịch"))
        rows.append([compile_fail, int(inp["exam_type"] != "single_problem")]
                    + report_feats(inp.get("test_report")) + code_feats(read_code(ds, s)))
    return np.array(rows, dtype=float)


HEADS = {
    "logreg": lambda: make_pipeline(StandardScaler(), OneVsRestClassifier(
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))),
    "svc": lambda: make_pipeline(StandardScaler(), OneVsRestClassifier(
        LinearSVC(class_weight="balanced", C=0.1, random_state=SEED))),
    "lgbm": lambda: OneVsRestClassifier(LGBMClassifier(
        n_estimators=200, num_leaves=15, min_child_samples=5, learning_rate=0.05,
        class_weight="balanced", random_state=SEED, verbose=-1)),
}


def main():
    build_synth_suites()
    real, synth = load_sample_dataset(), load_dataset(SYNTH_DIR)
    print("attaching compile/test signals (g++, cached)...")
    attach_signals(real)
    attach_signals(synth, SYNTH_SUITES)
    print(f"synth with test_report: {sum(1 for s in synth.tasks[2] if s['input'].get('test_report'))}/{len(synth.tasks[2])}")

    label_space = real.label_space["taxonomy_error"]
    mlb = MultiLabelBinarizer(classes=label_space)
    mlb.fit([label_space])
    Ys = mlb.transform([s["output"]["taxonomy_error"] for s in synth.tasks[2]])
    Yr = mlb.transform([s["output"]["taxonomy_error"] for s in real.tasks[2]])

    enc = SentenceTransformer(EMBED_MODEL)
    emb = lambda ds: enc.encode([read_code(ds, s) for s in ds.tasks[2]], normalize_embeddings=True)
    X = {"emb": (emb(synth), emb(real)), "feat": (features(synth), features(real))}
    X["emb+feat"] = (np.hstack([X["emb"][0], X["feat"][0]]), np.hstack([X["emb"][1], X["feat"][1]]))

    # best LLM combo so far, as the base for unions
    t2 = load_labels(OUT.parent / "qwen7b-fewshot-noreport-sample" / "predictions_task2_sample.json")
    t3 = load_labels(OUT.parent / "research_task3" / "testreport_ablation" / "A_no_report.json")
    base = predict(real, t2, t3, "t3+compile+format")
    mb = task2_metrics(real.tasks[2], base, label_space)
    print(f"\nbase t3+compile+format: macro {mb['macro_f1']:.4f} micro {mb['micro_f1']:.4f}\n")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'features':<10} {'head':<7} {'alone macro':>11} {'micro':>7} {'| ∪ base macro':>15} {'micro':>7}")
    results = []
    for fname, (Xs, Xr) in X.items():
        for hname, make in HEADS.items():
            pred = {}
            for i, s in enumerate(real.tasks[2]):
                keep = np.arange(len(Xr)) != i
                clf = make().fit(np.vstack([Xs, Xr[keep]]), np.vstack([Ys, Yr[keep]]))
                pred[s["sample_id"]] = list(mlb.inverse_transform(clf.predict(Xr[i:i + 1]))[0])
            m = task2_metrics(real.tasks[2], pred, label_space)
            uni = {sid: list(dict.fromkeys(base[sid] + pred[sid])) for sid in base}
            mu = task2_metrics(real.tasks[2], uni, label_space)
            results.append({"features": fname, "head": hname, "macro": m["macro_f1"], "micro": m["micro_f1"],
                            "union_macro": mu["macro_f1"], "union_micro": mu["micro_f1"]})
            print(f"{fname:<10} {hname:<7} {m['macro_f1']:>11.4f} {m['micro_f1']:>7.4f} {mu['macro_f1']:>15.4f} {mu['micro_f1']:>7.4f}", flush=True)
            (OUT / f"loo_{fname}_{hname}.json").write_text(json.dumps(
                [{"sample_id": k, "output": {"taxonomy_error": v}} for k, v in pred.items()], ensure_ascii=False), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")




def build_public(fname="feat", hname="svc"):
    """Train on synth + all 32 real, predict public; also write the union with the
    Task 3-labels+rules file (task2-from-t3-rules). Usage: python ml_task2_grid.py build"""
    assert fname == "feat", "only the feat set is wired for public"
    real, synth, public = load_sample_dataset(), load_dataset(SYNTH_DIR), load_public_test()
    attach_signals(real)
    attach_signals(synth, SYNTH_SUITES)
    attach_signals(public)
    label_space = real.label_space["taxonomy_error"]
    mlb = MultiLabelBinarizer(classes=label_space)
    mlb.fit([label_space])
    Y = mlb.transform([s["output"]["taxonomy_error"] for s in synth.tasks[2] + real.tasks[2]])
    clf = HEADS[hname]().fit(np.vstack([features(synth), features(real)]), Y)
    pred = {s["sample_id"]: list(l) for s, l in zip(public.tasks[2], mlb.inverse_transform(clf.predict(features(public))))}
    base = load_labels(OUT.parent / "task2-from-t3-rules" / "predictions_task2_public.json")
    for tag, labels in ((f"task2-{fname}-{hname}", pred),
                        (f"task2-t3rules-u-{fname}-{hname}", {sid: list(dict.fromkeys(base[sid] + pred[sid])) for sid in base})):
        path = OUT.parent / tag / "predictions_task2_public.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([{"sample_id": s["sample_id"], "output": {"taxonomy_error": labels[s["sample_id"]]}}
                                    for s in public.tasks[2]], ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    import sys
    build_public() if sys.argv[1:] == ["build"] else main()
