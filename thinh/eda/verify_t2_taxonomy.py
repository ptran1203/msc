"""Independent verification of the Task 2 (error taxonomy) EDA findings.

Reads the raw JSON files directly (does not reuse load_data.load_samples nor the
analyst's t2_label_eda.py) and recomputes every number the analyst reported, plus a
few extra checks. Prints markdown tables.

Run:
    .venv/bin/python verify_t2_taxonomy.py --data-root /path/to/sample_dataset [--compile] [--run]

--compile  local Apple clang (macOS `g++` alias, NOT GNU g++ 13 used for compile_log):
           EX01 -fsyntax-only, EX02 full build into a temp dir. Nothing executed.
--run      EX02 only: builds each submission (and the uncommented EX02-S214) and runs
           the 8 public tests in a temp dir, 2 s timeout per test, only when a static
           grep finds no dangerous call. Compares with test_report.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom, fisher_exact
from sklearn.metrics import cohen_kappa_score, f1_score

DIMS = ["compilable", "io_format", "logic", "edge_case", "complexity", "code_quality"]
DANGEROUS_RE = re.compile(r"system\s*\(|popen|fork\s*\(|\bexec\w*\s*\(|remove\s*\(|rename\s*\(|ofstream|fopen|socket")
GCC_DIAG_ERROR_RE = re.compile(r"^\S+:\d+:\d+: (?:fatal )?error:", re.M)
LINKER_RE = re.compile(r"undefined reference|collect2: error|ld returned")
ZD = {"0": 0.0, "1": 1.0, "nan": np.nan}


# ------------------------------------------------------------------ io helpers
def md(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "(empty)"
    cols = cols or list(rows[0].keys())

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1e-3 and v != 0 else f"{v:.3f}"
        return str(v).replace("|", "\\|")

    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def head(title: str) -> None:
    print(f"\n## {title}\n")


def load(root: Path) -> tuple[list[str], list[dict]]:
    """Return (label order from label_space.json, one merged record per sample_id)."""
    labels = json.loads((root / "label_space.json").read_text(encoding="utf-8"))["taxonomy_error"]
    raw = {k: json.loads((root / f).read_text(encoding="utf-8"))["samples"]
           for k, f in {"t1": "task1_grading.json", "t2": "task2_error_taxonomy.json", "t3": "task3_feedback.json"}.items()}
    recs: dict[str, dict] = {}
    for key, samples in raw.items():
        for s in samples:
            r = recs.setdefault(s["sample_id"], {"sid": s["sample_id"]})
            r[f"{key}_in"] = s["input"]
            r[f"{key}_out"] = s["output"]
    out = []
    for r in recs.values():
        i1 = r["t1_in"]
        r.update({"exam": i1["exam_id"], "log": i1.get("compile_log") or "", "tests": i1.get("test_report"),
                  "labels": list(r["t2_out"]["taxonomy_error"]), "t3_labels": list(r["t3_in"]["taxonomy_error"]),
                  "total": r["t1_out"]["total_score"], "feedback": r["t3_out"]["feedback"],
                  "code": (root / i1["code_file"]).read_text(encoding="utf-8", errors="replace")})
        r.update({d: r["t1_out"]["rubric"][d] for d in DIMS})
        r["lset"] = frozenset(r["labels"])
        out.append(r)
    return labels, sorted(out, key=lambda x: x["sid"])


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def crosstab(name: str, recs: list[dict], lab: callable, sig: callable) -> dict:
    a = [bool(lab(r)) for r in recs]
    b = [bool(sig(r)) for r in recs]
    k = kappa(a, b)
    if not np.isnan(k):
        assert np.isclose(k, cohen_kappa_score(a, b)), name
    return {"pair": name, "n": len(recs), "L1S1": sum(x and y for x, y in zip(a, b)),
            "L1S0": sum(x and not y for x, y in zip(a, b)), "L0S1": sum(y and not x for x, y in zip(a, b)),
            "L0S0": sum(not x and not y for x, y in zip(a, b)), "kappa": k,
            "L_only": ",".join(r["sid"] for r, x, y in zip(recs, a, b) if x and not y) or "-",
            "S_only": ",".join(r["sid"] for r, x, y in zip(recs, a, b) if y and not x) or "-"}


# ------------------------------------------------------------------ V01 prevalence
def v01_prevalence(labels: list[str], recs: list[dict]) -> None:
    head("V01. Prevalence, cardinality, combinations, per-exam Fisher test")
    exams = sorted({r["exam"] for r in recs})
    rows = []
    for lab in labels:
        cnt = {ex: sum(lab in r["lset"] for r in recs if r["exam"] == ex) for ex in exams}
        nex = {ex: sum(r["exam"] == ex for r in recs) for ex in exams}
        table = [[cnt[ex], nex[ex] - cnt[ex]] for ex in exams]
        p = fisher_exact(table)[1] if sum(cnt.values()) else float("nan")
        rows.append({"label": lab, "n": sum(cnt.values()), "pct": 100 * sum(cnt.values()) / len(recs),
                     **{f"{ex}/{nex[ex]}": cnt[ex] for ex in exams}, "fisher_p": p})
    print(md(rows))
    occ = sum(len(r["labels"]) for r in recs)
    card = Counter(len(r["labels"]) for r in recs)
    print(f"\n- label occurrences: {occ}; cardinality: {dict(sorted(card.items()))}; mean={occ / len(recs):.3f}")
    for ex in exams:
        sub = [r for r in recs if r["exam"] == ex]
        print(f"- {ex}: mean={sum(len(r['labels']) for r in sub) / len(sub):.3f}; absent="
              f"{[lab for lab in labels if not any(lab in r['lset'] for r in sub)]}")
    print(f"- empty: {[r['sid'] for r in recs if not r['labels']]}")
    sets = Counter(r["lset"] for r in recs)
    ordered = Counter(tuple(r["labels"]) for r in recs)
    print(f"- distinct combos as SET: {len(sets)}, seen once: {sum(v == 1 for v in sets.values())}; "
          f"as ORDERED list: {len(ordered)}")
    print(f"- min Fisher p over labels: {min(x['fisher_p'] for x in rows if not np.isnan(x['fisher_p'])):.3f}")


# ------------------------------------------------------------------ V02 edge case
def v02_edge(recs: list[dict]) -> None:
    head("V02. 'Lỗi edge case' vs edge_case dimension; where do edge_case<2 come from?")
    low = [r for r in recs if r["edge_case"] < 2]
    comp_fail = [r for r in low if r["compilable"] == 0]
    logic0 = [r for r in low if r["compilable"] == 1 and r["logic"] == 0]
    rest = [r for r in low if r["compilable"] == 1 and r["logic"] > 0]
    ok = [r for r in recs if r["compilable"] == 1]
    print(f"- edge_case<2: {len(low)}/{len(recs)}; compilable=1: {sum(r['edge_case'] < 2 for r in ok)}/{len(ok)}; "
          f"label 'Lỗi edge case': {sum('Lỗi edge case' in r['lset'] for r in recs)}")
    print(f"- decomposition: compile-fail {len(comp_fail)}, compilable&logic=0 {len(logic0)}, "
          f"compilable&logic>0 {len(rest)} -> {[(r['sid'], r['logic'], r['edge_case'], r['labels']) for r in rest]}")
    for ex in ("EX01", "EX02"):
        pairs = Counter((r["logic"], r["edge_case"]) for r in recs if r["exam"] == ex)
        by_logic: dict[int, set] = {}
        for (lg, ed), _ in pairs.items():
            by_logic.setdefault(lg, set()).add(ed)
        det = all(len(v) == 1 for v in by_logic.values())
        print(f"- {ex} (logic, edge)->count: {dict(sorted(pairs.items()))}; edge is a function of logic: {det}")
    for sid in ("EX01-S102", "EX01-S104"):
        r = next(x for x in recs if x["sid"] == sid)
        print(f"- {sid}: logic={r['logic']} edge_case={r['edge_case']} labels={r['labels']} feedback_chars={len(r['feedback'])}")


# ------------------------------------------------------------------ V03 metric
def f1_vec(y: np.ndarray, p: np.ndarray, zd: float) -> np.ndarray:
    tp = (y & p).sum(0)
    fp = (~y & p).sum(0)
    fn = (y & ~p).sum(0)
    den = 2 * tp + fp + fn
    return np.array([2 * t / d if d else zd for t, d in zip(tp, den)], dtype=float)


def macro(y: np.ndarray, p: np.ndarray, zd: float) -> float:
    v = f1_vec(y, p, zd)
    return float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")


def mat(sets: list[frozenset], labels: list[str]) -> np.ndarray:
    return np.array([[lab in s for lab in labels] for s in sets], dtype=bool)


def predictors(labels: list[str], recs: list[dict]) -> dict[str, list[frozenset]]:
    observed = frozenset(lab for lab in labels if any(lab in r["lset"] for r in recs))
    comp = [bool(GCC_DIAG_ERROR_RE.search(r["log"])) for r in recs]
    return {
        "always []": [frozenset()] * len(recs),
        "always {logic}": [frozenset({"Lỗi logic"})] * len(recs),
        "all 10": [frozenset(labels)] * len(recs),
        "all 9 observed (no edge)": [observed] * len(recs),
        "log-error rule": [frozenset({"Lỗi biên dịch"}) if c else frozenset() for c in comp],
        "log-error rule else {logic}": [frozenset({"Lỗi biên dịch"}) if c else frozenset({"Lỗi logic"}) for c in comp],
        "T3 input lookup": [frozenset(r["t3_labels"]) for r in recs],
        "oracle": [r["lset"] for r in recs],
    }


def v03_metric(labels: list[str], recs: list[dict]) -> None:
    head("V03. Macro-F1 under zero_division conventions (own implementation, cross-checked with sklearn)")
    y = mat([r["lset"] for r in recs], labels)
    rows = []
    for name, pred in predictors(labels, recs).items():
        p = mat(pred, labels)
        row: dict[str, object] = {"predictor": name}
        for tag, zd in ZD.items():
            row[f"macro zd={tag}"] = macro(y, p, zd)
            ref = f1_score(y, p, average="macro", zero_division=zd)
            assert np.isclose(row[f"macro zd={tag}"], ref, equal_nan=True), (name, tag, ref)
        row["micro"] = f1_score(y, p, average="micro", zero_division=0)
        row["samples zd0"] = f1_score(y, p, average="samples", zero_division=0)
        row["samples zd1"] = f1_score(y, p, average="samples", zero_division=1)
        row["exact"] = float(np.mean([a == b for a, b in zip(pred, [r["lset"] for r in recs])]))
        rows.append(row)
    print(md(rows))
    print("\nPer-exam oracle macro (zd=0 / zd=1):")
    for ex in ("EX01", "EX02"):
        ys = mat([r["lset"] for r in recs if r["exam"] == ex], labels)
        print(f"- {ex}: {macro(ys, ys, 0.0):.3f} / {macro(ys, ys, 1.0):.3f}")
    # all-positive predictor closed form: F1 = 2p/(1+p)
    prev = y.mean(0)
    print(f"\n- all-positive closed form mean(2p/(1+p)) over 10 labels = {np.mean(2 * prev / (1 + prev)):.3f}")


# ------------------------------------------------------------------ V04 small-test instability
def v04_instability(labels: list[str], recs: list[dict], n_boot: int, seed: int) -> None:
    head("V04. P(no positive) for test size N (binomial) and bootstrap spread")
    n = len(recs)
    rows = []
    for lab in labels:
        k = sum(lab in r["lset"] for r in recs)
        p_hat = k / n
        p_use = p_hat if k else (k + 0.5) / (n + 1)
        row: dict[str, object] = {"label": lab, "k": k, "p_used": p_use,
                                  "CP95_hi": float(beta.ppf(0.975, k + 1, n - k))}
        for size in (36, 121, 100, 300):
            row[f"P0 N={size}"] = (1 - p_use) ** size
        row["P<=2 N=100"] = float(binom.cdf(2, 100, p_use))
        row["P<=2 N=36"] = float(binom.cdf(2, 36, p_use))
        rows.append(row)
    print(md(rows))
    rng = np.random.default_rng(seed)
    y = mat([r["lset"] for r in recs], labels)
    p = mat(predictors(labels, recs)["log-error rule else {logic}"], labels)
    out = []
    for size in (32, 36, 100):
        res = {t: [] for t in ZD}
        for _ in range(n_boot):
            idx = rng.integers(0, n, size)
            for t, zd in ZD.items():
                res[t].append(macro(y[idx], p[idx], zd))
        out.append({"N": size, **{f"std zd={t}": float(np.std(v)) for t, v in res.items()}})
    print("\nBootstrap std of macro-F1, predictor 'log-error rule else {logic}':")
    print(md(out))


# ------------------------------------------------------------------ V05 leakage
def v05_leakage(recs: list[dict]) -> None:
    head("V05. Cross-task leakage")
    n = len(recs)
    ordered = sum(r["labels"] == r["t3_labels"] for r in recs)
    as_set = sum(r["lset"] == frozenset(r["t3_labels"]) for r in recs)
    t12 = sum(r["t1_in"] == r["t2_in"] for r in recs)
    shared = [k for k in recs[0]["t1_in"] if k in recs[0]["t3_in"]]
    t13 = sum(all(r["t1_in"][k] == r["t3_in"][k] for k in shared) for r in recs)
    print(f"- T2 out == T3 in taxonomy_error: ordered {ordered}/{n}, set {as_set}/{n}")
    print(f"- T1 input == T2 input: {t12}/{n}; T1==T3 on shared keys {shared}: {t13}/{n}")
    print(f"- T3-only input keys: {sorted(set(recs[0]['t3_in']) - set(recs[0]['t1_in']))}; "
          f"T1 keys missing from T3: {sorted(set(recs[0]['t1_in']) - set(recs[0]['t3_in']))}")


# ------------------------------------------------------------------ V06/V07 signals vs labels
def fully_commented(code: str) -> bool:
    body = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return all(not ln.strip() or ln.strip().startswith("//") for ln in body.splitlines())


def v06_consistency(recs: list[dict]) -> None:
    head("V06. Label vs rubric / signal crosstabs")
    ok = [r for r in recs if r["compilable"] == 1]
    ex2 = [r for r in recs if r["exam"] == "EX02"]
    ex2t = [r for r in ex2 if r["tests"]]
    has = lambda lab: (lambda r: lab in r["lset"])  # noqa: E731
    diag_err = lambda r: bool(GCC_DIAG_ERROR_RE.search(r["log"]))  # noqa: E731
    any_err = lambda r: diag_err(r) or bool(LINKER_RE.search(r["log"]))  # noqa: E731
    allpass = lambda r: all(t["passed"] for t in r["tests"])  # noqa: E731
    specs = [
        ("biên dịch ~ compilable=0", recs, has("Lỗi biên dịch"), lambda r: r["compilable"] == 0),
        ("biên dịch ~ diag error regex", recs, has("Lỗi biên dịch"), diag_err),
        ("biên dịch ~ diag or linker", recs, has("Lỗi biên dịch"), any_err),
        ("[] ~ total=10", recs, lambda r: not r["labels"], lambda r: r["total"] == 10),
        ("[] ~ 8/8 pass (EX02 tests>0)", ex2t, lambda r: not r["labels"], allpass),
        ("nhập/xuất ~ io_format=0", recs, has("Lỗi nhập/xuất"), lambda r: r["io_format"] == 0),
        ("style ~ code_quality=0", recs, has("Lỗi style"), lambda r: r["code_quality"] == 0),
        ("thuật toán ~ complexity=0", recs, has("Lỗi thuật toán"), lambda r: r["complexity"] == 0),
        ("logic ~ logic<4 (compilable)", ok, has("Lỗi logic"), lambda r: r["logic"] < 4),
        ("vòng lặp ~ TIMEOUT (EX02)", ex2, has("Lỗi vòng lặp"),
         lambda r: any("TIMEOUT" in t["actual"] for t in (r["tests"] or []))),
        ("hàm ~ io_format=0 & compilable (EX01)", [r for r in recs if r["exam"] == "EX01"], has("Lỗi hàm"),
         lambda r: r["io_format"] == 0 and r["compilable"] == 1),
        ("{hàm|nhập/xuất} ~ io_format=0 & compilable", recs,
         lambda r: bool(r["lset"] & {"Lỗi hàm", "Lỗi nhập/xuất"}), lambda r: r["io_format"] == 0 and r["compilable"] == 1),
        ("label set non-empty ~ total<10", recs, lambda r: bool(r["labels"]), lambda r: r["total"] < 10),
    ]
    print(md([crosstab(*s) for s in specs]))
    cq0 = [r for r in recs if r["code_quality"] == 0]
    cf = [r for r in recs if r["compilable"] == 0]
    print(f"\n- code_quality=0: {len(cq0)} -> exams {Counter(r['exam'] for r in cq0)}")
    print(f"- compile-fail rubric by exam: {[(r['sid'], [r[d] for d in DIMS]) for r in cf]}")
    print(f"- mean code_quality among compile-fail: {np.mean([r['code_quality'] for r in cf]):.2f}")
    for lab in ("Lỗi style", "Lỗi thuật toán"):
        sub = [r for r in recs if lab in r["lset"]]
        print(f"- {lab}: n={len(sub)} mean total={np.mean([r['total'] for r in sub]):.2f} "
              f"complexity={[r['complexity'] for r in sub]} code_quality={[r['code_quality'] for r in sub]}")
    print(f"- [] totals: {[r['total'] for r in recs if not r['labels']]}")

    head("V07. Output-format behaviour (EX02) from test_report and source")
    prompt_re = re.compile(r'cout\s*<<\s*"[^"]*:\s*"')
    for r in ex2:
        tests = r["tests"] or []
        prompt = bool(prompt_re.search(r["code"]))
        lead_space = bool(re.search(r'cout\s*<<\s*" (True|False)', r["code"]))
        upper = bool(re.search(r'"(TRUE|FALSE)"', r["code"]))
        extra = [t["actual"] for t in tests if not t["passed"] and t["expected"] in t["actual"] and t["actual"] != t["expected"]]
        if prompt or lead_space or upper or extra or "Lỗi nhập/xuất" in r["lset"]:
            print(f"- {r['sid']}: prompt_in_code={prompt} leading_space={lead_space} upper_case={upper} "
                  f"pass={sum(t['passed'] for t in tests)}/{len(tests)} empty_actual={sum(t['actual'] == '' for t in tests)} "
                  f"io_format={r['io_format']} total={r['total']} labels={r['labels']}")
    print(f"- EX02 with test_report == []: {[r['sid'] for r in ex2 if r['tests'] == []]}; "
          f"test_report is None: {[r['sid'] for r in ex2 if r['tests'] is None]}")
    print(f"- fully commented sources: {[r['sid'] for r in recs if fully_commented(r['code'])]}")
    print(f"- linker-only logs: {[r['sid'] for r in recs if LINKER_RE.search(r['log']) and not GCC_DIAG_ERROR_RE.search(r['log'])]}")
    wret = [r for r in recs if "-Wreturn-type" in r["log"]]
    print(f"- logs with -Wreturn-type: {[(r['sid'], r['labels'], r['total']) for r in wret]}")


# ------------------------------------------------------------------ V08 co-occurrence / strings / provenance
def v08_misc(labels: list[str], recs: list[dict], root: Path) -> None:
    head("V08. Co-occurrence, label strings, compile_log provenance")
    pairs = Counter(p for r in recs for p in combinations(sorted(r["lset"], key=labels.index), 2))
    print(f"- pairs: {[(a.replace('Lỗi ', ''), b.replace('Lỗi ', ''), c) for (a, b), c in pairs.most_common()]}")
    n_loop = sum("Lỗi vòng lặp" in r["lset"] for r in recs)
    n_ll = sum({"Lỗi vòng lặp", "Lỗi logic"} <= r["lset"] for r in recs)
    print(f"- P(logic | loop) = {n_ll}/{n_loop}")
    print(f"- compile-label samples with extra labels: {[(r['sid'], r['labels']) for r in recs if 'Lỗi biên dịch' in r['lset'] and len(r['lset']) > 1]}")
    occ = [x for r in recs for x in r["labels"]] + [x for r in recs for x in r["t3_labels"]]
    print(f"- occurrences (labels+t3_labels): {len(occ)}; in label_space: {sum(x in labels for x in occ)}; "
          f"NFC: {sum(unicodedata.is_normalized('NFC', x) for x in occ)}; stripped: {sum(x == x.strip() for x in occ)}; "
          f"dups: {sum(len(r['labels']) != len(set(r['labels'])) for r in recs)}")
    print(f"- labels whose NFD != NFC: {sum(unicodedata.normalize('NFD', x) != x for x in labels)}/{len(labels)}")
    order = {lab: i for i, lab in enumerate(labels)}
    print(f"- lists not in label_space order: {[r['sid'] for r in recs if [order[x] for x in r['labels']] != sorted(order[x] for x in r['labels'])]}")
    hdr = json.loads((root / "task2_error_taxonomy.json").read_text(encoding="utf-8"))["label_space"]
    print(f"- task2 header label_space == label_space.json: {hdr == labels}")
    logs = [r for r in recs if r["log"]]
    print(f"- non-empty logs: {len(logs)}; with '/tmp/tmp*/student.cpp': {sum(bool(re.search(r'/tmp/tmp[^/]+/student.cpp', r['log'])) for r in logs)}; "
          f"with 'x86_64-linux-gnu/13': {sum('x86_64-linux-gnu/13' in r['log'] for r in logs)}; "
          f"GCC-style quotes ‘’: {sum('‘' in r['log'] for r in logs)}")
    r109 = next(r for r in recs if r["sid"] == "EX01-S109")
    print(f"- EX01-S109 log loc: {re.findall(r'student\.cpp:(\d+:\d+): error', r109['log'])}; "
          f"feedback loc: {re.findall(r'student\.h:(\d+:\d+): error', r109['feedback'])}")


# ------------------------------------------------------------------ V09 local compiler / run
def compiler() -> str | None:
    return shutil.which("g++") or shutil.which("clang++")


def v09_compile(recs: list[dict], root: Path) -> None:
    head("V09. Local compile (Apple clang via `g++`; compile_log came from GNU g++ 13/Linux)")
    cc = compiler()
    if not cc:
        print("no compiler")
        return
    res = []
    with tempfile.TemporaryDirectory() as tmp:
        for r in recs:
            src = Path(tmp) / "student.cpp"
            shutil.copyfile(root / r["t1_in"]["code_file"], src)
            cmd = [cc, "-std=c++11", str(src)] + (["-fsyntax-only"] if r["exam"] == "EX01" else ["-o", str(Path(tmp) / "a.out")])
            res.append(subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=tmp).returncode != 0)
    for r, fail in zip(recs, res):
        r["clang_fail"] = fail
    any_err = lambda r: bool(GCC_DIAG_ERROR_RE.search(r["log"]) or LINKER_RE.search(r["log"]))  # noqa: E731
    print(md([crosstab("log diag|linker ~ clang fail", recs, any_err, lambda r: r["clang_fail"]),
              crosstab("biên dịch ~ clang fail", recs, lambda r: "Lỗi biên dịch" in r["lset"], lambda r: r["clang_fail"]),
              crosstab("compilable=0 ~ clang fail", recs, lambda r: r["compilable"] == 0, lambda r: r["clang_fail"])]))


def run_tests(code: str, tests: list[dict]) -> list[str] | str:
    if DANGEROUS_RE.search(code):
        return "skipped: dangerous pattern"
    cc = compiler()
    with tempfile.TemporaryDirectory() as tmp:
        src, exe = Path(tmp) / "s.cpp", Path(tmp) / "a.out"
        src.write_text(code, encoding="utf-8")
        if subprocess.run([cc, "-std=c++11", str(src), "-o", str(exe)], capture_output=True, timeout=60).returncode:
            return "build failed"
        outs = []
        for t in tests:
            try:
                p = subprocess.run([str(exe)], input=t["input"] + "\n", capture_output=True, text=True, timeout=2, cwd=tmp)
                outs.append(p.stdout if p.returncode == 0 else f"{p.stdout}<rc={p.returncode}>")
            except subprocess.TimeoutExpired:
                outs.append("<TIMEOUT>")
        return outs


def v10_run(recs: list[dict]) -> None:
    head("V10. Run EX02 public tests locally (sandboxed: temp dir, 2 s timeout, danger grep)")
    ex2 = [r for r in recs if r["exam"] == "EX02"]
    ref = next(r["tests"] for r in ex2 if r["tests"])
    same_tests = all([(t["input"], t["expected"]) for t in r["tests"]] == [(t["input"], t["expected"]) for t in ref]
                     for r in ex2 if r["tests"])
    print(f"- all non-empty EX02 test_reports share the same 8 (input, expected): {same_tests}")
    rows = []
    for r in ex2:
        code = r["code"]
        tag = r["sid"]
        if fully_commented(code):
            code = re.sub(r"^(\s*)//", r"\1", code, flags=re.M)
            tag += " (uncommented)"
        outs = run_tests(code, ref)
        if isinstance(outs, str):
            rows.append({"sample": tag, "local": outs})
            continue
        local_pass = sum(o.strip() == t["expected"] for o, t in zip(outs, ref))
        # Pass count if a leading "<prompt>: " were removed - separates prompt failures from logic failures.
        no_prompt = sum(re.sub(r"^[^:\n]*:\s*", "", o.strip()) == t["expected"] for o, t in zip(outs, ref))
        rep = r["tests"] or []
        match = sum(o.strip() == t["actual"] for o, t in zip(outs, rep)) if rep else "-"
        fails = [t["input"] for o, t in zip(outs, ref) if o.strip() != t["expected"]]
        rows.append({"sample": tag, "local": f"{local_pass}/8", "no_prompt": f"{no_prompt}/8",
                     "report": f"{sum(t['passed'] for t in rep)}/{len(rep)}",
                     "actual==local(stripped)": match, "failed_inputs": ",".join(fails) or "-",
                     "raw_first": repr(outs[0][:25]), "labels": ",".join(x.replace("Lỗi ", "") for x in r["labels"]) or "[]"})
    print(md(rows))


def v11_extra(labels: list[str], recs: list[dict]) -> None:
    head("V11. Extra checks (candidate missed findings)")
    import warnings

    y = mat([r["lset"] for r in recs], labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"- sklearn default zero_division ('warn') oracle macro-F1: {f1_score(y, y, average='macro'):.3f}")
    ex2 = [r for r in recs if r["exam"] == "EX02"]
    extra = lambda r: any(not t["passed"] and t["expected"] in t["actual"] and t["actual"] != t["expected"]  # noqa: E731
                          for t in (r["tests"] or []))
    print(md([crosstab("nhập/xuất ~ extra text in actual (EX02)", ex2, lambda r: "Lỗi nhập/xuất" in r["lset"], extra)]))
    gate_re = re.compile(r"tiên quyết")
    gate = [r for r in recs if gate_re.search(r["feedback"])]
    print("\nEX01 feedback stating the P1 prerequisite gate ('tiên quyết'):")
    print(md([{"sid": r["sid"], "skip_rest_phrase": "không chấm" in r["feedback"], "logic": r["logic"],
               "edge": r["edge_case"], "total": r["total"], "labels": ",".join(r["labels"])} for r in gate]))
    print("\nEX01 zeroing causes per feedback keywords (P1 gate / infinite loop / compile):")
    ex1 = [r for r in recs if r["exam"] == "EX01"]
    print(md([{"sid": r["sid"], "compilable": r["compilable"], "P1_gate_or_P1_wrong": bool(re.search(r"tiên quyết|P1 sai|P1 đọc file sai", r["feedback"])),
               "infinite_loop": "vô tận" in r["feedback"], "code_quality": r["code_quality"], "logic": r["logic"],
               "total": r["total"], "labels": ",".join(x.replace("Lỗi ", "") for x in r["labels"]) or "[]"}
              for r in ex1 if r["code_quality"] == 0 or re.search(r"tiên quyết|P1 sai|P1 đọc file sai|vô tận", r["feedback"])]))
    io0 = [r for r in recs if r["compilable"] == 1 and r["io_format"] == 0]
    print(f"\n- compilable=1 & io_format=0: {[(r['sid'], r['labels']) for r in io0]}")
    io0_all = [r for r in recs if r["io_format"] == 0]
    print(f"- io_format=0 total {len(io0_all)}: compile-fail {sum(r['compilable'] == 0 for r in io0_all)}, "
          f"compilable {len(io0)}")
    lab_low = [r for r in recs if r["compilable"] == 1 and r["logic"] < 4]
    print(f"- compilable & logic<4 with >=1 label: {sum(bool(r['labels']) for r in lab_low)}/{len(lab_low)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    labels, recs = load(args.data_root)
    print(f"# verify_t2 — n={len(recs)} {Counter(r['exam'] for r in recs)}")
    v01_prevalence(labels, recs)
    v02_edge(recs)
    v03_metric(labels, recs)
    v04_instability(labels, recs, args.n_boot, args.seed)
    v05_leakage(recs)
    v06_consistency(recs)
    v08_misc(labels, recs, args.data_root)
    v11_extra(labels, recs)
    if args.compile:
        v09_compile(recs, args.data_root)
    if args.run:
        v10_run(recs)


if __name__ == "__main__":
    main()
