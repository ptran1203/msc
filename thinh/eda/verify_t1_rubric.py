"""Independent re-computation of the Task-1 (rubric) EDA findings, written from scratch.

Re-derives every number from the raw JSON / .cpp files (does not import the analyst's script) and adds
checks the analyst did not run. Output is markdown, one section per finding id (V01..V15, M1..M5).

Safety: EX02 binaries are executed only after a grep for dangerous calls, inside a temp dir, with a
2 s timeout and the test input on stdin. EX01 is never executed (no main / missing data file).
Local compiler is whatever `g++` resolves to (Apple clang on macOS) -- NOT the GNU g++ 13 that produced
compile_log, so local results are indicative only.

Usage:
  python verify_t1_rubric.py --data-root /path/to/sample_dataset [--no-exec] [--bootstrap 2000]
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from load_data import COMPILE_ERROR_RE, DIMS, load_samples

SCORE_AXIS = list(range(11))
RUN_TIMEOUT_S = 2.0
TIMEOUT = "<TIMEOUT>"
UNSAFE_RE = re.compile(
    r"\bsystem\s*\(|\bpopen\b|\bfork\s*\(|\bexec\w*\s*\(|\bremove\s*\(|\brename\s*\(|\bofstream\b|\bfopen\b|\bsocket\b"
)
# Any toolchain failure line: gcc/clang "file:l:c: error", linker "undefined reference", collect2/ld exit status.
ANY_FAIL_RE = re.compile(r"(^\S+:\d+:\d+: (fatal )?error:)|undefined reference|ld returned \d+ exit status", re.M)
PROMPT_TOKEN_RE = re.compile(r"(\S+)\s*$")
# Armstrong numbers in range not covered by the public tests + two non-Armstrong upper-range values.
PROBES = ["2", "407", "4210818", "9926315", "912985153", "999999999", "1000000000"]


# ----------------------------------------------------------------------------- small helpers
def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else (f"{v:.3f}" if not v.is_integer() else str(int(v))))
            else:
                cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def h(title: str) -> None:
    print(f"\n## {title}\n")


def short(sample_id: str) -> str:
    return sample_id.split("-")[1]


def nfc(text: str | None) -> str:
    return unicodedata.normalize("NFC", text or "")


def drop_comments(code: str) -> str:
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", code, flags=re.S))


def body_of(code: str, name: str) -> str | None:
    """Brace-matched body of the first definition of `name(...) {`."""
    m = re.search(rf"\b{name}\s*\(([^)]*)\)\s*\{{", code)
    if not m:
        return None
    depth = 0
    for i in range(m.end() - 1, len(code)):
        depth += {"{": 1, "}": -1}.get(code[i], 0)
        if depth == 0:
            return code[m.end() : i]
    return None


def vec(row: pd.Series) -> str:
    return "".join(str(int(row[d])) for d in DIMS)


def qwk_index(y: np.ndarray, p: np.ndarray, axis: list[int]) -> float:
    """Own QWK: weights (i-j)^2 on the *position* in `axis` (what sklearn does with its labels)."""
    pos = {v: i for i, v in enumerate(axis)}
    k = len(axis)
    obs = np.zeros((k, k))
    for a, b in zip(y, p):
        obs[pos[a], pos[b]] += 1
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2
    den = (w * exp).sum()
    return float("nan") if den == 0 else float(1 - (w * obs).sum() / den)


def qwk(y: np.ndarray, p: np.ndarray, fixed_axis: bool) -> float:
    axis = SCORE_AXIS if fixed_axis else sorted(set(y.tolist()) | set(p.tolist()))
    return qwk_index(np.asarray(y), np.asarray(p), axis)


def sk_qwk(y: np.ndarray, p: np.ndarray, fixed_axis: bool) -> float:
    return float(cohen_kappa_score(y, p, weights="quadratic", labels=SCORE_AXIS if fixed_axis else None))


def spear(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    rho, pv = spearmanr(x.astype(float), y.astype(float))
    return float(rho), float(pv), len(x)


def lenient_eq(actual: str, expected: str) -> bool:
    """Accept output whose last whitespace token equals the expected answer (drops prompts / spaces)."""
    m = PROMPT_TOKEN_RE.search(actual or "")
    return bool(m) and m.group(1) == expected


def armstrong(n: int) -> str:
    s = str(n)
    return "True" if sum(int(c) ** len(s) for c in s) == n else "False"


# ----------------------------------------------------------------------------- local toolchain
def compile_src(src: Path, out_dir: Path, link: bool) -> tuple[bool, str, Path | None]:
    binary = out_dir / f"{src.stem}.bin"
    cmd = ["g++", "-std=c++11", "-w", str(src)] + (["-o", str(binary)] if link else ["-fsyntax-only"])
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60, cwd=out_dir)
    first = next((ln for ln in proc.stderr.splitlines() if "error" in ln), "")
    return proc.returncode == 0, first[-90:], binary if (link and proc.returncode == 0) else None


def run_bin(binary: Path, stdin: str, cwd: Path) -> tuple[str, int | None]:
    try:
        proc = subprocess.run([str(binary)], input=stdin + "\n", capture_output=True, text=True,
                              errors="replace", timeout=RUN_TIMEOUT_S, cwd=cwd)
        return proc.stdout, proc.returncode
    except subprocess.TimeoutExpired:
        return TIMEOUT, None


# ----------------------------------------------------------------------------- data
def load(root: Path) -> tuple[pd.DataFrame, dict]:
    df = load_samples(root)
    exams = {e["exam_id"]: e for e in json.loads((root / "exams.json").read_text(encoding="utf-8"))["exams"]}
    raw_t1 = json.loads((root / "task1_grading.json").read_text(encoding="utf-8"))
    df["input_keys"] = [",".join(sorted(s["input"].keys())) for s in raw_t1["samples"]]
    df["vector"] = df.apply(vec, axis=1)
    df["any_fail_log"] = df["compile_log"].fillna("").map(lambda s: bool(ANY_FAIL_RE.search(s)))
    df["link_fail_log"] = df["compile_log"].fillna("").str.contains("undefined reference|ld returned", regex=True)
    df["n_pass0"] = df["n_passed"].fillna(0).astype(int)
    df["n_lenient"] = df["test_report"].map(
        lambda r: sum(lenient_eq(t["actual"], t["expected"]) for t in r) if isinstance(r, list) else 0)
    return df, exams


# ----------------------------------------------------------------------------- V01 S214
def v01(df: pd.DataFrame, root: Path, exec_ok: bool, work: Path) -> dict:
    h("V01 (T1-01) EX02-S214: file comment toàn bộ")
    r = df.set_index("sample_id").loc["EX02-S214"]
    lines = [ln for ln in r["code"].splitlines() if ln.strip()]
    n_comment = sum(ln.lstrip().startswith("//") for ln in lines)
    print(f"- non-empty lines: {len(lines)}, starting with '//': {n_comment}; code after stripping comments: "
          f"{len(drop_comments(r['code']).strip())} chars")
    print(f"- gold vector={r['vector']} total={r['total_score']}; test_report={r['test_report']!r}; "
          f"loader log_has_error={r['log_has_error']}; link-fail regex={r['link_fail_log']}")
    print(f"- T2 labels={r['labels']}; feedback mentions 'comment': {'comment' in nfc(r['feedback'])}")
    result: dict = {}
    if not exec_ok:
        return result
    ok, err, _ = compile_src(root / r["code_file"], work, link=True)
    print(f"- clang link of original file: ok={ok} ({err})")
    unc = work / "S214_uncommented.cpp"
    unc.write_text(re.sub(r"^(\s*)//", r"\1", r["code"], flags=re.M), encoding="utf-8")
    assert not UNSAFE_RE.search(unc.read_text())
    ok, err, binary = compile_src(unc, work, link=True)
    public = [("1634", "True"), ("123", "False"), ("153", "True"), ("9474", "True"), ("9", "True"),
              ("10", "False"), ("1", "True"), ("370", "True")]
    outs = [(i, e, run_bin(binary, i, work)[0].strip()) for i, e in public]
    fails = [i for i, e, o in outs if o != e]
    probe = [(p, run_bin(binary, p, work)[0].strip(), armstrong(int(p))) for p in PROBES]
    print(f"- uncommented: build ok={ok}; public pass {len(public) - len(fails)}/8, fails on {fails}; "
          f"probes pass {sum(o == ref for _, o, ref in probe)}/7")
    result["S214_public"] = len(public) - len(fails)
    return result


# ----------------------------------------------------------------------------- EX01 manual reading
# Independent reading of code + teacher feedback (EDA only). p1_fb = teacher verdict on P1 in feedback.
# P2..P4 status: C correct, W wrong, N not done (absent/declared only), G behind gate, ? unknown.
EX01_READING: dict[str, dict] = {
    "EX01-S101": {"p1_fb": "correct", "P2": "C", "P3": "N", "P4": "W", "inf": False},
    "EX01-S102": {"p1_fb": "correct", "P2": "W", "P3": "C", "P4": "C", "inf": False},
    "EX01-S103": {"p1_fb": "n/a (compile)", "P2": "N", "P3": "N", "P4": "N", "inf": False},
    "EX01-S104": {"p1_fb": "correct", "P2": "C", "P3": "W", "P4": "C", "inf": False},
    "EX01-S105": {"p1_fb": "wrong", "P2": "W", "P3": "N", "P4": "N", "inf": False},
    "EX01-S106": {"p1_fb": "unmentioned", "P2": "W", "P3": "W", "P4": "?", "inf": True},
    "EX01-S107": {"p1_fb": "unmentioned", "P2": "?", "P3": "W", "P4": "?", "inf": True},
    "EX01-S108": {"p1_fb": "wrong", "P2": "G", "P3": "N", "P4": "N", "inf": False},
    "EX01-S109": {"p1_fb": "n/a (compile)", "P2": "?", "P3": "N", "P4": "N", "inf": False},
    "EX01-S110": {"p1_fb": "wrong", "P2": "G", "P3": "N", "P4": "G", "inf": False},
    "EX01-S111": {"p1_fb": "n/a (compile)", "P2": "N", "P3": "N", "P4": "N", "inf": False},
    "EX01-S112": {"p1_fb": "wrong", "P2": "G", "P3": "G", "P4": "G", "inf": False},
    "EX01-S113": {"p1_fb": "wrong", "P2": "N", "P3": "N", "P4": "N", "inf": False},
    "EX01-S114": {"p1_fb": "wrong", "P2": "N", "P3": "N", "P4": "N", "inf": False},
    "EX01-S115": {"p1_fb": "correct", "P2": "C", "P3": "C", "P4": "C", "inf": False},
}
P1_WRONG_FB_RE = re.compile(r"P1[^;]*?(sai|lỗi|gán cứng|hardcode)", re.I)
GATE_FB_RE = re.compile(r"tiên quyết", re.I)


def p1_static(code: str) -> dict:
    """Static signals in prob1: allocation sized by caller n, n written anywhere (incl. by-ref helper)."""
    src = drop_comments(code)
    body = body_of(src, "prob1") or ""
    alloc_n = bool(re.search(r"new\s+General\s*\[\s*n\s*\]|General\s+\w+\s*\[\s*n\s*\]", body))
    writes_n = bool(re.search(r"(?<![\w.>])n\s*(=(?!=)|\+\+|\+=)|\+\+\s*n\b", body))
    helper_writes = False
    for call in re.finditer(r"\b(\w+)\s*\(([^()]*)\)", body):
        args = [a.strip() for a in call.group(2).split(",")]
        if "n" not in args:
            continue
        d = re.search(rf"\b{call.group(1)}\s*\(([^)]*)\)\s*\{{", src)
        if not d:
            continue
        params = [p.strip() for p in d.group(1).split(",")]
        idx = args.index("n")
        if idx < len(params) and "&" in params[idx]:
            pname = re.findall(r"\w+", params[idx])[-1]
            hb = body_of(src, call.group(1)) or ""
            helper_writes |= bool(re.search(rf"(?<![\w.>]){pname}\s*(=(?!=)|\+\+|\+=)|\+\+\s*{pname}\b", hb))
    return {
        "alloc_caller_n": alloc_n and not (writes_n or helper_writes),
        "reads_cin": bool(re.search(r"getline\s*\(\s*cin|cin\s*>>", body)),
        "opens_literal": bool(re.search(r"open\s*\(\s*\"|stream\s+\w+\s*\(\s*\"", body)),
        "has_main": bool(re.search(r"\bint\s+main\s*\(", src)),
    }


def v02_v03_v04_v10(df: pd.DataFrame, exams: dict) -> pd.DataFrame:
    ex1 = df[df["exam_id"] == "EX01"].copy()
    rows = []
    for r in ex1.itertuples():
        fb = nfc(r.feedback)
        st = p1_static(r.code)
        rows.append({"sample_id": r.sample_id, "vector": r.vector, "total": r.total_score, "logic": r.logic,
                     "compilable": r.compilable, "io": r.io_format,
                     "fb_P1_wrong": bool(P1_WRONG_FB_RE.search(fb)), "fb_gate": bool(GATE_FB_RE.search(fb)),
                     **st, **EX01_READING[r.sample_id]})
    a = pd.DataFrame(rows)

    h("V02 (T1-02) EX01: P1 sai nhưng logic>0")
    built = a[a["compilable"] == 1]
    p1w = built[built["fb_P1_wrong"]]
    viol = p1w[p1w["logic"] > 0]
    print(f"- compilable=1: {len(built)}/15. Feedback nói P1 sai (regex `P1…sai|lỗi|gán cứng`): {len(p1w)} → "
          f"{[short(s) for s in p1w['sample_id']]}")
    print(f"- trong đó logic>0: {len(viol)}/{len(p1w)} → "
          f"{viol[['sample_id', 'vector', 'total', 'fb_gate']].values.tolist()}")
    print(f"- P1 sai + logic=0: {p1w[p1w['logic'] == 0][['sample_id', 'vector', 'total']].values.tolist()}")
    print(f"- feedback nhắc 'tiên quyết': {[short(s) for s in a.loc[a['fb_gate'], 'sample_id']]}")
    print("- Nếu tính cả 3 bài compile fail (P1 coi như sai): mẫu số 9 hoặc 8 tuỳ cách đếm S109 "
          "(S109 lỗi compile ở prob2, không phải prob1).")

    h("V03 (T1-03) prob1 cấp phát theo n của caller, không gán n")
    flag = a[a["alloc_caller_n"]]
    print(md(flag[["sample_id", "p1_fb", "vector", "total", "compilable"]]))
    print("\n- S101 vs S110: cùng vector? "
          f"{a.set_index('sample_id').loc['EX01-S101', 'vector'] == a.set_index('sample_id').loc['EX01-S110', 'vector']}")

    h("V04 (T1-04) trọng số câu EX01")
    exam = exams["EX01"]
    stmt = {f"P{k}": float(v) for k, v in re.findall(r"Problem\s+(\d)\s*\((\d+)\s*điểm\)", nfc(exam["statement"]))}
    meta = {p["pid"]: p["max_score"] for p in exam["problems"]}
    print(f"- statement weights {stmt} (Σ={sum(stmt.values())}); problems[].max_score {meta} (Σ={sum(meta.values())})")
    s102 = df.set_index("sample_id").loc["EX01-S102", "code"]
    print(f"- S102 code comments carry statement weights: {re.findall(r'Problem\s+(\d)\s*\((\d)\)', s102)}")
    print(f"- S102 feedback weights in parentheses: {re.findall(r'P(\d) đúng \((\d+)\)', nfc(df.set_index('sample_id').loc['EX01-S102', 'feedback']))}")

    def policy(row: pd.Series, w: dict[str, float]) -> float:
        if row["compilable"] == 0 or row["inf"] or row["p1_fb"] != "correct":
            return 0.0
        return w["P1"] + sum(w[p] for p in ["P2", "P3", "P4"] if row[p] == "C")

    a["pol_1423"] = a.apply(policy, axis=1, w=stmt)
    a["pol_2.5"] = a.apply(policy, axis=1, w=meta)
    a["pol_1423_plus_ci"] = a["pol_1423"] + a["compilable"] + a["io"]
    print(md(a[["sample_id", "total", "logic", "p1_fb", "P2", "P3", "P4", "inf", "pol_1423", "pol_2.5"]]))
    for col in ["pol_1423", "pol_2.5"]:
        rho, pv, n = spear(a[col], a["total"])
        print(f"- total vs {col}: Spearman ρ={rho:.3f} (p={pv:.3f}, n={n}); MAE={np.abs(a[col] - a['total']).mean():.2f}")
    print(f"- số mẫu mà hai policy khác nhau: {int((a['pol_1423'] != a['pol_2.5']).sum())} → "
          f"{a.loc[a['pol_1423'] != a['pol_2.5'], 'sample_id'].tolist()}")
    nonzero = a[(a["pol_1423"] == 0) & (a["total"] > 0)]
    print(f"- policy=0 nhưng total>0: {len(nonzero)} (compilable+io offset chiếm {int((nonzero['total'] <= 2).sum())})")
    chk = a[a["compilable"] == 1]
    formula = 2 + chk["logic"] + df.set_index("sample_id").loc[chk["sample_id"], "edge_case"].to_numpy() + 2 * (chk["logic"] > 0)
    print(f"- total == 2+logic+edge+2·[logic>0] (compilable=1): {int((formula.to_numpy() == chk['total'].to_numpy()).sum())}/{len(chk)}")

    h("V10 (T1-10) quy định 0 điểm, hàm main")
    inf = a[a["inf"]]
    print(f"- vòng lặp vô hạn (đọc code + feedback): {inf[['sample_id', 'vector', 'total']].values.tolist()}")
    print(f"- int main ngoài comment: {int(a['has_main'].sum())}/15 → {[short(s) for s in a.loc[a['has_main'], 'sample_id']]}")
    print(f"- có main & total=10: {a.loc[a['has_main'] & (a['total'] == 10), 'sample_id'].tolist()}")
    print(f"- exams note: {exam['note']!r}")
    return a


# ----------------------------------------------------------------------------- V05/V06 structure
def v05_v06(df: pd.DataFrame) -> None:
    h("V05 (T1-05) compile fail theo exam")
    cf = df[df["compilable"] == 0]
    print(md(cf.groupby(["exam_id", "vector"]).agg(n=("sample_id", "size"),
                                                    ids=("sample_id", lambda s: ",".join(map(short, s)))).reset_index()))
    for ex, sub in df.groupby("exam_id"):
        c0 = sub[sub["compilable"] == 0]
        print(f"- {ex}: compilable=0 ⇒ quality=0 holds {int((c0['code_quality'] == 0).sum())}/{len(c0)} (non-vacuous); "
              f"code_quality value counts {sub['code_quality'].value_counts().to_dict()}")

    h("V06 (T1-06) ràng buộc giữa các chiều")
    rules = {
        "edge == logic/2": lambda s: s["edge_case"] * 2 == s["logic"],
        "quality == 1": lambda s: s["code_quality"] == 1,
        "complexity == quality == [logic>0]": lambda s: (s["complexity"] == s["code_quality"])
        & (s["code_quality"] == (s["logic"] > 0).astype(int)),
        "total == Σdims": lambda s: s["total_score"] == s[DIMS].sum(axis=1),
    }
    rows = []
    for name, f in rules.items():
        rows.append({"rule": name, **{ex: f"{int(f(sub).sum())}/{len(sub)}" for ex, sub in df.groupby("exam_id")}})
    print(md(pd.DataFrame(rows)))
    for ex, sub in df.groupby("exam_id"):
        print(f"- {ex}: logic values {sorted(sub['logic'].unique().tolist())}; #vectors {sub['vector'].nunique()}; "
              f"totals {sorted(sub['total_score'].unique().tolist())}")


# ----------------------------------------------------------------------------- EX02 tests
def ex02_local(df: pd.DataFrame, root: Path, work: Path) -> pd.DataFrame:
    rows = []
    for r in df[df["exam_id"] == "EX02"].itertuples():
        src = root / r.code_file
        if UNSAFE_RE.search(r.code):
            rows.append({"sample_id": r.sample_id, "status": "unsafe-skip"})
            continue
        if not drop_comments(r.code).strip():
            src = work / f"{r.sample_id}_unc.cpp"
            src.write_text(re.sub(r"^(\s*)//", r"\1", r.code, flags=re.M), encoding="utf-8")
        ok, err, binary = compile_src(src, work, link=True)
        row = {"sample_id": r.sample_id, "status": "built" if ok else "build_fail", "counterfactual": src.parent == work}
        if binary is None:
            rows.append(row)
            continue
        tests = r.test_report if r.test_report else [{"input": i, "expected": armstrong(int(i))}
                                                      for i in ["1634", "123", "153", "9474", "9", "10", "1", "370"]]
        outs = [run_bin(binary, t["input"], work) for t in tests]
        row.update({
            "loc_strict": sum(o.strip() == t["expected"] for (o, _), t in zip(outs, tests)),
            "loc_lenient": sum(lenient_eq(o, t["expected"]) for (o, _), t in zip(outs, tests)),
            "loc_lead_ws": sum(bool(o) and o != TIMEOUT and o[0].isspace() for o, _ in outs),
            "loc_timeout": sum(o == TIMEOUT for o, _ in outs),
            "loc_first_out": repr(outs[0][0][:20]), "loc_first_rc": outs[0][1],
            "loc_eq_report": sum((o.strip() if o != TIMEOUT else o) == t.get("actual") for (o, _), t in zip(outs, tests))
            if r.test_report else None,
        })
        if row["loc_timeout"] < 2:
            row["probe_pass"] = sum(lenient_eq(run_bin(binary, p, work)[0], armstrong(int(p))) for p in PROBES)
        rows.append(row)
    return pd.DataFrame(rows)


def v07_v08_v09_v12(df: pd.DataFrame, local: pd.DataFrame | None) -> pd.DataFrame:
    ex2 = df[df["exam_id"] == "EX02"].copy()
    ex2["has_prompt"] = ex2["test_report"].map(lambda r: any(":" in t["actual"] for t in r) if r else False)
    ex2["n_empty"] = ex2["test_report"].map(lambda r: sum(t["actual"] == "" for t in r) if r else 0)
    ex2["n_timeout"] = ex2["test_report"].map(lambda r: sum(t["actual"] == TIMEOUT for t in r) if r else 0)
    if local is not None:
        ex2 = ex2.merge(local, on="sample_id", how="left")

    h("V07 (T1-07) io_format vs test_report")
    cols = ["sample_id", "vector", "total_score", "io_format", "n_pass0", "n_lenient", "has_prompt"]
    cols += [c for c in ["loc_strict", "loc_lenient", "loc_lead_ws", "loc_eq_report"] if c in ex2]
    print(md(ex2[cols]))
    print(f"\n- io=1 & prompt: {ex2.loc[(ex2['io_format'] == 1) & ex2['has_prompt'], 'sample_id'].tolist()}")
    print(f"- compilable=1 & io=0: {ex2.loc[(ex2['compilable'] == 1) & (ex2['io_format'] == 0), ['sample_id', 'n_pass0']].values.tolist()}")

    h("V08 (T1-08) Spearman total vs số test pass (EX02)")
    rows = []
    for x in ["n_pass0", "n_lenient"]:
        for scope, sub in [("all n=17", ex2), ("n_tests>0", ex2[ex2["n_tests"] > 0])]:
            rho, pv, n = spear(sub[x], sub["total_score"])
            rows.append({"x": x, "scope": scope, "n": n, "rho": rho, "p": pv})
    print(md(pd.DataFrame(rows)))
    print(f"\n- total≥7 & n_pass0=0: {ex2.loc[(ex2['total_score'] >= 7) & (ex2['n_pass0'] == 0), 'sample_id'].tolist()}")
    print(f"- logic=2 n_pass0: {sorted(ex2.loc[ex2['logic'] == 2, 'n_pass0'].tolist())}")
    print(f"- n_pass0 by logic: {ex2.groupby('logic')['n_pass0'].apply(list).to_dict()}")
    print(f"- n_lenient by logic: {ex2.groupby('logic')['n_lenient'].apply(list).to_dict()}")

    h("V09 (T1-09) test_report thiếu trạng thái thực thi")
    keys = {k for r in ex2["test_report"] if r for t in r for k in t}
    print(f"- test keys: {sorted(keys)}; n_tests value counts: {ex2['n_tests'].value_counts().to_dict()}")
    print(f"- n_tests=0: {ex2.loc[ex2['n_tests'] == 0, ['sample_id', 'compilable']].values.tolist()}; "
          f"pass_rate NaN: {int(ex2['pass_rate'].isna().sum())}")
    cols = ["sample_id", "vector", "n_empty", "n_timeout"] + [c for c in ["loc_first_out", "loc_first_rc", "loc_timeout"] if c in ex2]
    print(md(ex2.loc[(ex2["n_empty"] > 0) | (ex2["n_timeout"] > 0), cols]))

    h("V12 (T1-12) phủ miền của test công khai + probe")
    inputs = sorted({t["input"] for r in ex2["test_report"] if r for t in r}, key=int)
    print(f"- public inputs: {inputs} (max digits {max(map(len, inputs))})")
    if "probe_pass" in ex2:
        print(md(ex2[["sample_id", "logic", "n_pass0", "n_lenient", "status", "counterfactual", "probe_pass"]]))
    return ex2


# ----------------------------------------------------------------------------- V11 toolchain
def v11(df: pd.DataFrame, root: Path, exec_ok: bool, work: Path) -> None:
    h("V11 (T1-11) compiles_locally & clang vs compile_log")
    keys = df["input_keys"].value_counts().to_dict()
    print(f"- T1 input key sets: {keys}")
    ct = pd.crosstab([df["exam_id"], df["any_fail_log"]], df["compilable"])
    print(md(ct.reset_index().rename(columns={0: "compilable=0", 1: "compilable=1"})))
    if not exec_ok:
        return
    agree = []
    for r in df.itertuples():
        ok, err, _ = compile_src(root / r.code_file, work, link=(r.exam_id == "EX02"))
        agree.append({"sample_id": r.sample_id, "clang_ok": ok, "log_fail": r.any_fail_log, "agree": ok != r.any_fail_log,
                      "err": err})
    ag = pd.DataFrame(agree)
    print(f"\n- clang outcome == compile_log outcome: {int(ag['agree'].sum())}/{len(ag)}")
    if (~ag["agree"]).any():
        print(md(ag[~ag["agree"]]))


# ----------------------------------------------------------------------------- QWK
def baselines(df: pd.DataFrame) -> dict[str, np.ndarray]:
    is2 = (df["exam_id"] == "EX02").to_numpy()
    fail = df["any_fail_log"].to_numpy()
    nt = df["n_tests"].fillna(0).to_numpy()
    rate = np.where(nt > 0, df["n_pass0"] / np.where(nt > 0, nt, 1), 0.0)
    rate_l = np.where(nt > 0, df["n_lenient"] / np.where(nt > 0, nt, 1), 0.0)
    comp = np.where(fail, 0, 10)
    return {
        "exam_median": df.groupby("exam_id")["total_score"].transform("median").round().astype(int).to_numpy(),
        "compile_loader": np.where(df["log_has_error"], 0, 10),
        "compile_strict": comp,
        "pass_strict": np.where(is2, np.where(fail, 0, np.rint(10 * rate)), comp).astype(int),
        "pass_lenient": np.where(is2, np.where(fail, 0, np.rint(10 * rate_l)), comp).astype(int),
        "rubric_prior": np.where(is2, np.where(fail, 0, 2 + np.rint(8 * rate_l)), comp).astype(int),
    }


def v13_v14_v15(df: pd.DataFrame, n_boot: int) -> None:
    y = df["total_score"].to_numpy(int)
    preds = baselines(df)
    h("V13–V15 (T1-13/14/15) QWK baseline (own QWK, cross-checked with sklearn)")
    rows = []
    for name, p in preds.items():
        row = {"baseline": name, "pooled_None": qwk(y, p, False), "pooled_0..10": qwk(y, p, True)}
        assert np.isclose(row["pooled_None"], sk_qwk(y, p, False)) and np.isclose(row["pooled_0..10"], sk_qwk(y, p, True))
        for ex in ["EX01", "EX02"]:
            m = (df["exam_id"] == ex).to_numpy()
            row[f"{ex}_None"] = qwk(y[m], p[m], False)
            row[f"{ex}_0..10"] = qwk(y[m], p[m], True)
        row["MAE"] = float(np.abs(y - p).mean())
        rows.append(row)
    print(md(pd.DataFrame(rows)))

    print("\n**labels=None vs 0..10 — nhiễu loạn trên gold**\n")
    for ex, hi, lo in [("EX02", 7, 3), ("EX01", 6, 2)]:
        m = (df["exam_id"] == ex).to_numpy()
        yt = y[m]
        pt = np.where(yt == hi, lo, yt)
        print(f"- {ex}: {hi}→{lo} on {int((yt == hi).sum())} samples: None={qwk(yt, pt, False):.3f}, "
              f"0..10={qwk(yt, pt, True):.3f}")
    toy_y, toy_p = np.array([0, 0, 10, 10]), np.array([1, 1, 9, 9])
    print(f"- toy y={toy_y.tolist()} p={toy_p.tolist()}: None={qwk(toy_y, toy_p, False):.3f}, "
          f"0..10={qwk(toy_y, toy_p, True):.3f} (labels=None có thể làm GIẢM QWK)")
    n_lower = sum(qwk(y, p, False) < qwk(y, p, True) - 1e-9 for p in preds.values())
    print(f"- pooled: labels=None < labels=0..10 ở {n_lower}/{len(preds)} baseline")

    print("\n**Bootstrap & độ nhạy**\n")
    rng = np.random.default_rng(20261003)
    for name in ["compile_loader", "rubric_prior"]:
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            v = qwk(y[idx], preds[name][idx], True)
            if np.isfinite(v):
                vals.append(v)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"- {name}: QWK={qwk(y, preds[name], True):.3f}, bootstrap {len(vals)}/{n_boot} CI95 [{lo:.3f}, {hi:.3f}]")
    for name in ["compile_loader", "rubric_prior"]:
        p = preds[name]
        jack = [(qwk(np.delete(y, i), np.delete(p, i), True), df["sample_id"].iloc[i]) for i in range(len(y))]
        lo, hi = min(jack), max(jack)
        print(f"- jackknife (bỏ 1 mẫu) {name}: QWK từ {lo[0]:.3f} (bỏ {short(lo[1])}) đến {hi[0]:.3f} (bỏ {short(hi[1])})")
    out_of_range = sum(int(((df[d] < 0) | (df[d] > mx)).sum()) for d, mx in
                       zip(DIMS, [1, 1, 4, 2, 1, 1]))
    print(f"- integrity: out-of-range dim values {out_of_range}; non-integer "
          f"{int(sum((df[d] % 1 != 0).sum() for d in DIMS))}; missing {int(df[DIMS + ['total_score']].isna().sum().sum())}")
    s214 = (df["sample_id"] == "EX02-S214").to_numpy()
    print(f"- S214: loader pred={preds['compile_loader'][s214][0]}, strict pred={preds['compile_strict'][s214][0]}, "
          f"gold={y[s214][0]}; chỉ khác nhau ở {int((preds['compile_loader'] != preds['compile_strict']).sum())} mẫu")
    suspects = ["EX02-S214", "EX01-S108", "EX01-S110", "EX01-S112"]
    keep = ~df["sample_id"].isin(suspects).to_numpy()
    for name in ["rubric_prior", "compile_loader"]:
        p = preds[name]
        print(f"- {name} bỏ {suspects}: QWK {qwk(y, p, True):.3f} → {qwk(y[keep], p[keep], True):.3f}; "
              f"MAE {np.abs(y - p).mean():.2f} → {np.abs(y[keep] - p[keep]).mean():.2f}")
    for ex in ["EX01", "EX02"]:
        sub = df[df["exam_id"] == ex]
        print(f"- {ex} exact-match of per-exam mode: " + ", ".join(
            f"{d}={(sub[d] == sub[d].mode().iloc[0]).mean():.3f}" for d in DIMS))


# ----------------------------------------------------------------------------- missed findings
def missed(df: pd.DataFrame, a: pd.DataFrame) -> None:
    h("M1 EX01: io_format không phản ánh lỗi đọc file")
    ex1 = a.copy()
    print(md(ex1[ex1["compilable"] == 1][["sample_id", "io", "logic", "total", "reads_cin", "opens_literal", "p1_fb"]]))
    lab = df.set_index("sample_id")["labels"]
    print(f"\n- 'Lỗi nhập/xuất' trong T2 của EX01: {sum('Lỗi nhập/xuất' in lab[s] for s in ex1['sample_id'])}/15")
    print(f"- io=0 & compilable=1: {ex1.loc[(ex1['io'] == 0) & (ex1['compilable'] == 1), 'sample_id'].tolist()} "
          f"→ T2 {[lab[s] for s in ex1.loc[(ex1['io'] == 0) & (ex1['compilable'] == 1), 'sample_id']]}")

    h("M2 Nhất quán chéo T1 ↔ T2")
    has = lambda s, lbl: lbl in s  # noqa: E731
    ce = df["labels"].map(lambda s: has(s, "Lỗi biên dịch"))
    print(f"- 'Lỗi biên dịch' ⇔ compilable=0: {int((ce == (df['compilable'] == 0)).sum())}/{len(df)}")
    empty = df["labels"].map(len) == 0
    print(f"- labels=[] → total: {df.loc[empty, 'total_score'].tolist()}; total=10 → labels: "
          f"{df.loc[df['total_score'] == 10, ['sample_id', 'labels']].values.tolist()}")
    io_lbl = df["labels"].map(lambda s: has(s, "Lỗi nhập/xuất"))
    built = df["compilable"] == 1
    print(f"- 'Lỗi nhập/xuất' → io=0: {int((io_lbl & (df['io_format'] == 0)).sum())}/{int(io_lbl.sum())}; "
          f"io=0 & compilable=1 có nhãn này: {int((io_lbl & built & (df['io_format'] == 0)).sum())}/"
          f"{int((built & (df['io_format'] == 0)).sum())}")

    h("M3 EX01-S112 prob3: vòng lặp thứ hai không tiến con trỏ ở 2/3 nhánh")
    code = df.set_index("sample_id").loc["EX01-S112", "code"]
    body = body_of(drop_comments(code), "prob3") or ""
    loops = [m.start() for m in re.finditer(r"while\s*\(", body)]
    second = body[loops[1]:]
    branches = re.split(r"\}\s*else", second)
    print(f"- #while in prob3: {len(loops)}; second loop branches: {len(branches)}; "
          f"branches advancing temp (`temp=temp->next`): {[bool(re.search(r'temp\s*=\s*temp->next', b)) for b in branches]}")
    for sid in ["EX01-S106", "EX01-S107"]:
        b = body_of(drop_comments(df.set_index("sample_id").loc[sid, "code"]), "prob3") or ""
        print(f"- {sid} prob3 has `->next` advance inside while: {bool(re.search(r'(cur|head)\s*=\s*(cur|head)->next', b.split('while', 1)[-1]))}")
    print(f"- vectors: S106 {a.set_index('sample_id').loc['EX01-S106', 'vector']}, "
          f"S107 {a.set_index('sample_id').loc['EX01-S107', 'vector']}, S112 {a.set_index('sample_id').loc['EX01-S112', 'vector']}")

    h("M4 Bài gần trùng lặp (code đã bỏ comment + khoảng trắng)")
    norm = {r.sample_id: re.sub(r"\s+", "", drop_comments(r.code)) for r in df.itertuples()}
    sims = sorted(((difflib.SequenceMatcher(None, norm[x], norm[y], autojunk=False).ratio(), x, y)
                   for x, y in itertools.combinations(norm, 2) if x[:4] == y[:4]), reverse=True)
    print(f"- top-3 pairs: {[(round(s, 3), short(x), short(y)) for s, x, y in sims[:3]]}; "
          f"median pair sim {np.median([s for s, _, _ in sims]):.3f} over {len(sims)} pairs")
    top = sims[0]
    g = df.set_index("sample_id")
    print(f"- {short(top[1])}: vector {g.loc[top[1], 'vector']} T2 {g.loc[top[1], 'labels']}; "
          f"{short(top[2])}: vector {g.loc[top[2], 'vector']} T2 {g.loc[top[2], 'labels']}")

    h("M5 compile_log: cắt cụt và lệch dòng so với feedback")
    lens = df["compile_log"].fillna("").str.len()
    print(f"- max len {lens.max()} ({df.loc[lens.idxmax(), 'sample_id']}); logs với len==max: {int((lens == lens.max()).sum())}; "
          f"tail={df.loc[lens.idxmax(), 'compile_log'][-12:]!r}")
    for sid in ["EX01-S103", "EX01-S109", "EX01-S111"]:
        log_loc = re.findall(r"student\.(?:cpp|h):(\d+):(\d+): (?:fatal )?error", g.loc[sid, "compile_log"])[:1]
        fb_loc = re.findall(r"student\.(cpp|h):(\d+):(\d+)", nfc(g.loc[sid, "feedback"]))[:1]
        print(f"- {sid}: first error in compile_log {log_loc}; in feedback {fb_loc}")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--no-exec", action="store_true", help="skip local compile/run")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    root = args.data_root.resolve()
    df, exams = load(root)
    print(f"# Verify T1 — n={len(df)}; {df['exam_id'].value_counts().sort_index().to_dict()}")
    exec_ok = not args.no_exec
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        v01(df, root, exec_ok, work)
        a = v02_v03_v04_v10(df, exams)
        v05_v06(df)
        local = ex02_local(df, root, work) if exec_ok else None
        v07_v08_v09_v12(df, local)
        v11(df, root, exec_ok, work)
    v13_v14_v15(df, args.bootstrap)
    missed(df, a)


if __name__ == "__main__":
    main()
