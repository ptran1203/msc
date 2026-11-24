"""Independent re-check of the meta_audit findings (M-01..M-16) plus extra checks.

Written from scratch: it only reuses the shared loader (load_data.py), never the
analyst's meta_audit.py. Every table is recomputed from the raw dataset files.

Run:
    .venv/bin/python verify_meta_audit.py --data-root /path/to/sample_dataset [--no-compile] [--no-docker]

Privacy: prints sample_ids, counts and at most one short line of feedback/code per item.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import DIM_MAX, DIMS, load_meta, load_samples  # noqa: E402

CHARS_PER_TOKEN = 3.5
LOG_TRUNCATION_LEN = 2000
REQUIRED_EX01 = ["prob1", "prob2", "prob3", "prob4"]
ALLOWED_EX01_HEADERS = {"iostream", "fstream", "sstream", "string", "cstring", "cmath"}
DANGEROUS_RE = re.compile(r"\b(system|popen|fork|exec\w*|remove|rename|socket)\s*\(|ofstream|fopen")
# Any diagnostic that makes the toolchain fail: file:line:col errors OR linker failures.
ANY_BUILD_ERROR_RE = re.compile(r"(^\S+:\d+:\d+: (fatal )?error:)|(collect2: error)|(undefined reference to)|(ld returned)|(^ld: )|(linker command failed)", re.M)
LOADER_ERROR_RE = re.compile(r"^\S+:\d+:\d+: (fatal )?error:", re.M)  # same rule as load_data.COMPILE_ERROR_RE


# ----------------------------------------------------------------------------- helpers
def md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row.tolist():
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if not np.isnan(v) else "")
            else:
                cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def header(title: str) -> None:
    print(f"\n## {title}\n")


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else f"{k}/0"


def strip_comments(code: str) -> str:
    """Remove // and /* */ comments while keeping string/char literals intact."""
    out: list[str] = []
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = code.find("\n", i)
            i = n if j == -1 else j
        elif c == "/" and nxt == "*":
            j = code.find("*/", i + 2)
            seg = code[i: n if j == -1 else j + 2]
            out.append("\n" * seg.count("\n"))
            i = n if j == -1 else j + 2
        elif c in ("'", '"'):
            j = i + 1
            while j < n and code[j] != c and code[j] != "\n":
                j += 2 if code[j] == "\\" else 1
            out.append(code[i: j + 1])
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def strip_strings(code_only: str) -> str:
    return re.sub(r'"(\\.|[^"\\\n])*"', '""', code_only)


def defines(code_only: str, fname: str) -> bool:
    return re.search(rf"\b{fname}\s*\([^;{{}}]*\)\s*(const\s*)?\{{", code_only) is not None


def has_main(code_only: str) -> bool:
    return re.search(r"\b(int|void)\s+main\s*\(", code_only) is not None


def est_tokens(text: str | None) -> float:
    return len(text or "") / CHARS_PER_TOKEN


# ----------------------------------------------------------------------------- M-01
def check_m01(exams: dict, df: pd.DataFrame) -> None:
    header("M-01 Trọng số EX01: statement vs problems[].max_score")
    stmt = exams["EX01"]["statement"]
    parsed = {f"P{k}": float(v) for k, v in re.findall(r"Problem (\d) \((\d+(?:\.\d+)?) điểm\)", stmt)}
    json_w = {p["pid"]: p["max_score"] for p in exams["EX01"]["problems"]}
    print(md(pd.DataFrame([{"source": "statement", **parsed, "sum": sum(parsed.values())},
                           {"source": "problems[].max_score", **json_w, "sum": sum(json_w.values())}])))
    rows = []
    for _, r in df[df.exam_id == "EX01"].iterrows():
        hits = re.findall(r"Problem (\d) \((\d)\)", r.code or "")
        if hits:
            rows.append({"sample_id": r.sample_id, "skeleton_weights": ",".join(f"P{a}={b}" for a, b in hits)})
    print("\nKhung đề còn sót trong code (pattern `Problem k (w)`):\n")
    print(md(pd.DataFrame(rows)))
    print(f"\nREADME nhắc 1/4/2/3: {'1/4/2/3' in (ROOT / 'README.md').read_text(encoding='utf-8')}")


# ----------------------------------------------------------------------------- M-02 / M-11 feedback parsing
P1_WRONG_RE = re.compile(r"P1[^;.]*?(sai|lỗi|hardcode|gán cứng)|P1 SV hardcode", re.I)


def p1_status(fb: str) -> str:
    if re.search(r"P1 đúng", fb):
        return "ok"
    if P1_WRONG_RE.search(fb):
        return "wrong"
    return "unknown"


def bounds(status: str, weights: dict[str, float], fb: str) -> tuple[float, float]:
    """Upper bound on total implied by the prerequisite policy, when only P1 status is trusted."""
    if status == "wrong":
        return 0.0, weights["P1"]  # P1 at most partial credit, P2-P4 not graded
    return 0.0, sum(weights.values())


def check_m02(exams: dict, df: pd.DataFrame) -> None:
    header("M-02 EX01: total vs luật tiên quyết P1")
    stmt_w = {f"P{k}": float(v) for k, v in re.findall(r"Problem (\d) \((\d+) điểm\)", exams["EX01"]["statement"])}
    json_w = {p["pid"]: float(p["max_score"]) for p in exams["EX01"]["problems"]}
    rows = []
    for _, r in df[df.exam_id == "EX01"].iterrows():
        fb = r.feedback
        st = p1_status(fb)
        ub_s = bounds(st, stmt_w, fb)[1]
        ub_j = bounds(st, json_w, fb)[1]
        # S101-like: P3 'không làm' gives a tighter bound when P1,P2 ok
        not_done = re.findall(r"(P\d)[^;]*không làm", fb)
        not_done_all = set(re.findall(r"P(\d)", " ".join(not_done)))
        if st == "ok" and not_done:
            ub_s = sum(w for p, w in stmt_w.items() if p[1:] not in not_done_all)
            ub_j = sum(w for p, w in json_w.items() if p[1:] not in not_done_all)
        rows.append({"sample_id": r.sample_id, "compilable": r.compilable, "P1": st,
                     "tien_quyet_in_fb": "tiên quyết" in fb, "not_done": ",".join(sorted("P" + x for x in not_done_all)),
                     "ub_stmt_1423": ub_s, "ub_json_2.5": ub_j, "total": r.total_score,
                     "over_stmt": r.total_score > ub_s, "over_json": r.total_score > ub_j})
    t = pd.DataFrame(rows)
    print(md(t))
    n = len(t)
    print(f"\nP1 sai (parse): {pct(int((t.P1 == 'wrong').sum()), n)}; có chữ 'tiên quyết': {pct(int(t.tien_quyet_in_fb.sum()), n)}")
    print(f"Vượt cận theo 1/4/2/3: {pct(int(t.over_stmt.sum()), n)} -> {list(t[t.over_stmt].sample_id)}")
    print(f"Vượt cận theo 2.5x4: {pct(int(t.over_json.sum()), n)} -> {list(t[t.over_json].sample_id)}")
    pw = t[t.P1 == "wrong"][["sample_id", "tien_quyet_in_fb", "total"]]
    print("\nCùng tình huống 'P1 sai' nhưng total khác nhau (nội bộ, không phụ thuộc trọng số):\n")
    print(md(pw))
    print("\nCặp cùng lỗi P1 'assume n là số dòng': ",
          [(r.sample_id, r.total_score) for _, r in df.iterrows() if r.exam_id == "EX01" and "assume n" in r.feedback])


# ----------------------------------------------------------------------------- M-03
def check_m03(df: pd.DataFrame) -> None:
    header("M-03 EX02-S214 và tam giác compilable / log / nhãn CE")
    rows = []
    for _, r in df.iterrows():
        code_only = strip_comments(r.code)
        toks = re.findall(r"[A-Za-z_]\w*|\d+|\S", code_only)
        log = r.compile_log or ""
        rows.append({"sample_id": r.sample_id, "code_tokens": len(toks),
                     "any_build_error": bool(ANY_BUILD_ERROR_RE.search(log)),
                     "loader_log_has_error": bool(LOADER_ERROR_RE.search(log)),
                     "compilable": r.compilable, "CE_label": "Lỗi biên dịch" in r.labels,
                     "void_main": bool(re.search(r"\bvoid\s+main\s*\(", code_only)),
                     "pow_call": bool(re.search(r"\bpow\s*\(", strip_strings(code_only))),
                     "user_pow": bool(re.search(r"\b(int|long|double|float)\s+pow\s*\([^)]*\)\s*\{", code_only)),
                     "cmath": "cmath" in code_only or "math.h" in code_only})
    t = pd.DataFrame(rows)
    mism = t[(t.any_build_error & (t.compilable == 1)) | (~t.any_build_error & (t.compilable == 0))]
    print("Bài lệch giữa log-có-lỗi (tính cả linker) và compilable:\n")
    print(md(mism))
    print(f"\nLoader bỏ sót lỗi linker: {list(t[t.any_build_error & ~t.loader_log_has_error].sample_id)}")
    agree_ce = ((t.compilable == 0) == t.CE_label).sum()
    print(f"compilable==0 <-> nhãn 'Lỗi biên dịch': {pct(int(agree_ce), len(t))}")
    print(f"Tam giác log/compilable/nhãn nhất quán: {pct(int(((t.any_build_error == (t.compilable == 0)) & ((t.compilable == 0) == t.CE_label)).sum()), len(t))}")
    print(f"void main: {pct(int(t.void_main.sum()), len(t))}; gọi pow không include cmath: "
          f"{pct(int((t.pow_call & ~t.cmath).sum()), len(t))} (trong đó tự định nghĩa pow: {list(t[t.pow_call & ~t.cmath & t.user_pow].sample_id)})")
    s = df.set_index("sample_id").loc["EX02-S214"]
    print(f"\nS214: code_tokens(after strip comments)={int(t.set_index('sample_id').loc['EX02-S214','code_tokens'])}, "
          f"n_lines={s.code.count(chr(10)) + 1}, test_report={s.test_report}, rubric={[s[d] for d in DIMS]}, "
          f"total={s.total_score}, labels={s.labels}, level={s.level}")


def check_m03_uncommented(df: pd.DataFrame, compile_ok: bool) -> None:
    """Uncomment EX02-S214 in a temp dir and run the 8 public inputs: is the gold 10/10 about the commented code?"""
    if not compile_ok or not shutil.which("clang++"):
        return
    s = df.set_index("sample_id").loc["EX02-S214"]
    code = re.sub(r"(?m)^(\s*)//", r"\1", s.code)
    if DANGEROUS_RE.search(code):
        print("S214 uncommented: unsafe, skipped")
        return
    inputs = [tt["input"] for tt in df[df.n_tests == 8].iloc[0].test_report]
    expected = [tt["expected"] for tt in df[df.n_tests == 8].iloc[0].test_report]
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "u.cpp"
        src.write_text(code, encoding="utf-8")
        runs = run_with_clang(src, "-O0", inputs)
    passed = sum(o == e for (o, _), e in zip(runs, expected))
    print(f"S214 sau khi bỏ '//' đầu dòng: build+chạy 8 test công khai (clang) -> đậu {passed}/8")


# ----------------------------------------------------------------------------- M-04
def check_m04(tasks: dict, root: Path) -> None:
    header("M-04 Input giữa T1/T2/T3")
    by = {k: {s["sample_id"]: s for s in v["samples"]} for k, v in tasks.items()}
    order_same = [s["sample_id"] for s in tasks["t1"]["samples"]] == [s["sample_id"] for s in tasks["t2"]["samples"]] == [s["sample_id"] for s in tasks["t3"]["samples"]]
    ids = list(by["t1"])
    t12 = sum(by["t1"][i]["input"] == by["t2"][i]["input"] for i in ids)
    only1, only3, diff_common = set(), set(), 0
    for i in ids:
        a, c = by["t1"][i]["input"], by["t3"][i]["input"]
        only1 |= set(a) - set(c)
        only3 |= set(c) - set(a)
        diff_common += any(a[k] != c[k] for k in set(a) & set(c))
    tax_eq = sum(by["t3"][i]["input"]["taxonomy_error"] == by["t2"][i]["output"]["taxonomy_error"] for i in ids)
    tax_set_eq = sum(set(by["t3"][i]["input"]["taxonomy_error"]) == set(by["t2"][i]["output"]["taxonomy_error"]) for i in ids)
    ds = sorted(str(p.relative_to(root)) for p in root.rglob(".DS_Store"))
    print(md(pd.DataFrame([
        {"check": "cùng thứ tự sample_id 3 file", "value": order_same},
        {"check": "T1.input == T2.input", "value": pct(t12, len(ids))},
        {"check": "key chỉ ở T1", "value": sorted(only1)},
        {"check": "key chỉ ở T3", "value": sorted(only3)},
        {"check": "key chung khác giá trị", "value": diff_common},
        {"check": "T3.taxonomy_error == T2.output (cả thứ tự)", "value": pct(tax_eq, len(ids))},
        {"check": "... (như tập hợp)", "value": pct(tax_set_eq, len(ids))},
        {"check": ".DS_Store", "value": ds},
    ])))
    loader_src = (Path(__file__).resolve().parent / "load_data.py").read_text(encoding="utf-8")
    print(f"\nload_data.py có duyệt thư mục (glob/iterdir/rglob)? {bool(re.search(r'glob|iterdir|listdir', loader_src))}")


# ----------------------------------------------------------------------------- M-05
def check_m05(exams: dict, df: pd.DataFrame, root: Path, compile_ok: bool) -> None:
    header("M-05 EX01: main, hàm bắt buộc, lệnh chấm -o id")
    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="vm05_"))
    for _, r in df[df.exam_id == "EX01"].iterrows():
        co = strip_comments(r.code)
        missing = [f for f in REQUIRED_EX01 if not defines(co, f)]
        link = None
        if compile_ok and shutil.which("clang++"):
            p = subprocess.run(["clang++", "-std=c++11", "-w", str(root / r.code_file), "-o", str(tmp / "a.out")],
                               capture_output=True, text=True, timeout=60)
            link = "ok" if p.returncode == 0 else ("link_err" if "ld:" in p.stderr or "linker" in p.stderr else "compile_err")
        rows.append({"sample_id": r.sample_id, "main": has_main(co), "missing": ",".join(missing),
                     "compilable": r.compilable, "total": r.total_score, "clang_full_build": link})
    shutil.rmtree(tmp, ignore_errors=True)
    t = pd.DataFrame(rows)
    print(md(t))
    m = t[t["main"]]
    miss = t[t.missing != ""]
    print(f"\nCó main: {pct(len(m), len(t))}; trong đó total>0: {pct(int((m.total > 0).sum()), len(m))}")
    print(f"Thiếu >=1 hàm bắt buộc: {pct(len(miss), len(t))}; trong đó compilable=1: {pct(int((miss.compilable == 1).sum()), len(miss))}, "
          f"total range {miss[miss.compilable == 1].total.min()}-{miss[miss.compilable == 1].total.max()}")
    if compile_ok:
        print(f"Build đầy đủ (clang, có link) thành công: {pct(int((t.clang_full_build == 'ok').sum()), len(t))}")
    stmt = exams["EX01"]["statement"]
    print(f"\nStatement chứa '-o id': {'-o id' in stmt}; chứa 'xoá trước khi nộp': {'xoá trước khi nộp' in stmt}; note: {exams['EX01']['note'][:60]}")


# ----------------------------------------------------------------------------- M-06
def check_m06(exams: dict, df: pd.DataFrame) -> None:
    header("M-06 Luật 0 điểm: EX01 vs EX02")
    ce = df[df.compilable == 0][["sample_id", "exam_id", *DIMS, "total_score"]]
    print(md(ce))
    for ex in ("EX01", "EX02"):
        st = exams[ex]["statement"]
        print(f"\n{ex} statement có luật '0 điểm': {'0 điểm' in st}; có 'vòng lặp vô hạn': {'vòng lặp vô hạn' in st}")
    inf = df[df.feedback.str.contains("vô tận|vô hạn", regex=True)][["sample_id", "total_score"]]
    print("\nFeedback nhắc lặp vô tận:\n")
    print(md(inf))
    to = df[df.test_report.map(lambda r: isinstance(r, list) and any(t.get("actual") == "<TIMEOUT>" for t in r))]
    print("\nCó test TIMEOUT:", [(r.sample_id, sum(t["actual"] == "<TIMEOUT>" for t in r.test_report), r.total_score) for _, r in to.iterrows()])


# ----------------------------------------------------------------------------- M-07
# Verifier's own manual reading of every L1/L2/L3 gold (not the analyst's). Codes:
#   FIX     = states the concrete change to make
#   REVEAL  = names the exact missing statement (cause == fix)
#   QUOTE   = pastes compiler output that quotes the student's own code line
#   CAUSE   = (L1 only) names a specific cause
#   FIXDIR  = (L3) gives an explicit fix direction;  IMPLICIT = (L3) diagnosis that implies the fix
#   NOFIX   = (L3) no fix direction (bare diagnosis or generic study advice)
#   ok      = within level
MANUAL_LEVEL = {
    "EX01-S104": "FIX", "EX01-S110": "FIX", "EX01-S106": "REVEAL", "EX01-S107": "REVEAL",
    "EX01-S109": "QUOTE", "EX01-S111": "QUOTE", "EX02-S214": "CAUSE",
    "EX02-S206": "FIXDIR", "EX02-S207": "FIXDIR", "EX02-S205": "IMPLICIT", "EX02-S209": "IMPLICIT",
    "EX02-S212": "IMPLICIT", "EX02-S203": "NOFIX", "EX02-S204": "NOFIX", "EX02-S208": "NOFIX",
    "EX02-S215": "NOFIX", "EX02-S216": "NOFIX",
}
FIX_CUE_RE = re.compile(r"chỉ cần|phải |nên |bỏ |sửa|thay vì|chứ không phải|là được|cần ")
CODE_CUE_RE = re.compile(r"\w+\s*=\s*\w+->\w+|\w+\([^)]*\)\s*;|\babs\(|\breturn\s*;|\bgetline\(")


def check_m07(df: pd.DataFrame) -> None:
    header("M-07 Gold feedback vs điều cấm của level")
    rows = []
    for _, r in df.iterrows():
        rows.append({"sample_id": r.sample_id, "L": r.level_num, "fix_cue": bool(FIX_CUE_RE.search(r.feedback)),
                     "code_cue": bool(CODE_CUE_RE.search(r.feedback)), "manual": MANUAL_LEVEL.get(r.sample_id, "ok")})
    t = pd.DataFrame(rows)
    print(md(t[(t.fix_cue | t.code_cue | (t.manual != "ok"))]))
    l12 = t[t.L.isin([1, 2])]
    strict = l12.manual.isin(["FIX"]).sum()
    loose = l12.manual.isin(["FIX", "REVEAL", "QUOTE", "CAUSE"]).sum()
    print(f"\nL1/L2 gold: n={len(l12)}; vi phạm rõ (FIX): {pct(int(strict), len(l12))}; "
          f"tính cả ranh giới (REVEAL/QUOTE/CAUSE): {pct(int(loose), len(l12))}")
    print("Analyst list {S104,S106,S107,S109,S111,S214} thiếu:", sorted(set(l12[l12.manual != 'ok'].sample_id) - {"EX01-S104", "EX01-S106", "EX01-S107", "EX01-S109", "EX01-S111", "EX02-S214"}))
    l3 = t[t.L == 3]
    print(f"L3 gold: n={len(l3)}; hướng sửa rõ: {pct(int((l3.manual == 'FIXDIR').sum()), len(l3))} {list(l3[l3.manual == 'FIXDIR'].sample_id)}; "
          f"chỉ ngầm qua chẩn đoán: {pct(int((l3.manual == 'IMPLICIT').sum()), len(l3))} {list(l3[l3.manual == 'IMPLICIT'].sample_id)}; "
          f"không có: {pct(int((l3.manual == 'NOFIX').sum()), len(l3))} {list(l3[l3.manual == 'NOFIX'].sample_id)}")
    short = df[df.feedback.str.len() <= 25][["sample_id", "level", "feedback"]]
    print("\nGold rất ngắn (<=25 ký tự):\n")
    print(md(short))


def check_m07_s207_fix(df: pd.DataFrame, compile_ok: bool) -> None:
    """L3 gold of EX02-S207 says 'remove int at line 9': does that single edit fully solve the public tests?"""
    if not compile_ok or not shutil.which("clang++"):
        return
    s = df.set_index("sample_id").loc["EX02-S207"]
    ref = df[df.n_tests == 8].iloc[0].test_report
    inputs, expected = [t["input"] for t in ref], [t["expected"] for t in ref]
    lines = s.code.splitlines()
    fixed = lines.copy()
    fixed[8] = re.sub(r"\bint\s+", "", fixed[8], count=1)
    with tempfile.TemporaryDirectory() as td:
        res = {}
        for name, text in (("as_submitted", s.code), ("line9_without_int", "\n".join(fixed))):
            src = Path(td) / f"{name}.cpp"
            src.write_text(text, encoding="utf-8")
            res[name] = sum(o == e for (o, _), e in zip(run_with_clang(src, "-O0", inputs), expected))
    print(f"\nS207 (L3): đậu test công khai {res} -> gold đưa đúng thao tác sửa hoàn chỉnh")


# ----------------------------------------------------------------------------- M-08
def check_m08(df: pd.DataFrame) -> None:
    header("M-08 Level vs chất lượng bài")
    print(md(pd.crosstab([df.exam_id, df.level_num], df.total_score == 10).reset_index()))
    ex2 = df[df.exam_id == "EX02"]
    print("\n" + md(ex2.groupby("level_num").agg(n=("sample_id", "size"), mean_total=("total_score", "mean"),
                                                   n_total10=("total_score", lambda s: int((s == 10).sum())),
                                                   n_empty_labels=("labels", lambda s: int(s.map(len).eq(0).sum()))).reset_index()))
    a = int(((ex2.level_num == 1) & (ex2.total_score == 10)).sum())
    b = int(((ex2.level_num == 1) & (ex2.total_score < 10)).sum())
    c = int(((ex2.level_num != 1) & (ex2.total_score == 10)).sum())
    d = int(((ex2.level_num != 1) & (ex2.total_score < 10)).sum())
    print(f"\nEX02 Fisher exact L1 vs total==10: table={[[a, b], [c, d]]}, p={fisher_exact([[a, b], [c, d]])[1]:.5f}")
    print(f"Level 4: {pct(int((df.level_num == 4).sum()), len(df))}; EX01 all L2: {(df[df.exam_id == 'EX01'].level_num == 2).all()}")


# ----------------------------------------------------------------------------- M-09
def check_m09(exams: dict, df: pd.DataFrame) -> None:
    header("M-09 test_report vs io_format")
    ex2 = df[df.exam_id == "EX02"]
    acts = [t["actual"] for r in ex2.test_report for t in r]
    print(f"Số test result: {len(acts)}; actual có khoảng trắng đầu/cuối: {sum(a != a.strip() for a in acts)}; "
          f"actual rỗng '': {sum(a == '' for a in acts)}; '<TIMEOUT>': {sum(a == '<TIMEOUT>' for a in acts)}")
    inputs = sorted({t["input"] for r in ex2.test_report for t in r}, key=int)
    print(f"Tập input test (hợp nhất): {inputs}; max={max(map(int, inputs))}; n_tests phân phối: {ex2.n_tests.value_counts().to_dict()}")
    stmt_examples = re.findall(r"\| `(\d+)` \|", exams["EX02"]["statement"])
    print(f"Ví dụ trong đề: {stmt_examples} -> nằm trong test công khai: {[x in inputs for x in stmt_examples]}")
    rows = []
    for _, r in ex2.iterrows():
        co = strip_comments(r.code)
        lits = re.findall(r'cout\s*<<\s*"([^"]*)"', co)
        prompt = [x for x in lits if x.strip() and x.strip() not in ("True", "False", "TRUE", "FALSE")]
        padded = [x for x in lits if x.strip() in ("True", "False", "TRUE", "FALSE") and x != x.strip()]
        rows.append({"sample_id": r.sample_id, "extra_literals": prompt[:2], "padded_TF": padded,
                     "upper_TF": any(x in ("TRUE", "FALSE") for x in lits),
                     "tests": f"{r.n_passed}/{r.n_tests}" if r.n_tests else "[]",
                     "io_format": r.io_format, "total": r.total_score, "labels": r.labels})
    t = pd.DataFrame(rows)
    print("\n" + md(t[(t.extra_literals.map(len) > 0) | (t.padded_TF.map(len) > 0) | t.upper_TF | (t.io_format == 0)]))
    print(f"\nhas_hidden_testcases EX02={exams['EX02']['has_hidden_testcases']}; có trường hidden trong test_report: "
          f"{any('hidden' in json.dumps(r) for r in ex2.test_report)}")


# ----------------------------------------------------------------------------- M-10
def check_m10(df: pd.DataFrame, root: Path, compile_ok: bool) -> None:
    header("M-10 Toolchain của compile_log + đối chiếu clang local")
    logs = df[df.log_nonempty]
    fp = {
        "/usr/include/c++/13": logs.compile_log.str.contains("/usr/include/c++/13", regex=False).sum(),
        "x86_64-linux-gnu/13": logs.compile_log.str.contains("x86_64-linux-gnu/13", regex=False).sum(),
        "/tmp/tmp*/student.cpp": logs.compile_log.str.contains(r"/tmp/tmp\w+/student\.cpp", regex=True).sum(),
        "unicode quotes ‘’": logs.compile_log.str.contains("‘", regex=False).sum(),
        "-fpermissive": logs.compile_log.str.contains("-fpermissive", regex=False).sum(),
        "collect2": logs.compile_log.str.contains("collect2", regex=False).sum(),
    }
    print(md(pd.DataFrame([{"fingerprint": k, "n_logs": int(v), "of_nonempty": len(logs)} for k, v in fp.items()])))
    s109 = df.set_index("sample_id").loc["EX01-S109"]
    fb_ref = re.findall(r"(student\.\w+):(\d+)", s109.feedback)
    log_ref = re.findall(r"(student\.\w+):(\d+)", s109.compile_log)
    code_lines = s109.code.splitlines()
    ret_lines = [i + 1 for i, ln in enumerate(code_lines) if re.match(r"\s*return\s*;", ln)]
    print(f"\nS109: feedback refs {sorted(set(fb_ref))}; log refs {sorted(set(log_ref))}; dòng 'return;' trong file: {ret_lines}; "
          f"feedback dùng nháy ASCII: {chr(39) in s109.feedback and '‘' not in s109.feedback}")
    bom = [r.sample_id for _, r in df.iterrows() if (root / r.code_file).read_bytes().startswith(b"\xef\xbb\xbf")]
    crlf = [r.sample_id for _, r in df.iterrows() if b"\r\n" in (root / r.code_file).read_bytes()]
    vs = [r.sample_id for _, r in df.iterrows() if "This file contains the 'main' function" in r.code]
    winpath = [r.sample_id for _, r in df.iterrows() if re.search(r'"[A-Za-z]:\\', r.code)]
    print(f"BOM UTF-8: {bom}; CRLF: {pct(len(crlf), len(df))}; VS template header: {vs}; đường dẫn ổ Windows trong chuỗi: {winpath}")
    if not compile_ok or not shutil.which("clang++"):
        return
    rows = []
    for _, r in df.iterrows():
        src = root / r.code_file
        with tempfile.TemporaryDirectory() as td:
            cmd = ["clang++", "-std=c++11", "-fsyntax-only", str(src)] if r.exam_id == "EX01" else \
                  ["clang++", "-std=c++11", str(src), "-o", str(Path(td) / "a.out")]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=td)
        clang_err = p.returncode != 0
        rows.append({"sample_id": r.sample_id, "log_error": bool(ANY_BUILD_ERROR_RE.search(r.compile_log or "")),
                     "clang_error": clang_err})
    t = pd.DataFrame(rows)
    agree = int((t.log_error == t.clang_error).sum())
    print(f"\nApple clang ({subprocess.run(['clang++', '--version'], capture_output=True, text=True).stdout.splitlines()[0]}) "
          f"trạng thái lỗi khớp log: {pct(agree, len(t))}; lệch: {list(t[t.log_error != t.clang_error].sample_id)}")


# ----------------------------------------------------------------------------- M-11
def check_m11(df: pd.DataFrame, tasks: dict) -> None:
    header("M-11 total == Σ6 chiều; có điểm từng câu không")
    in_range = all(((df[d] >= 0) & (df[d] <= DIM_MAX[d])).all() for d in DIMS)
    ints = all(float(x).is_integer() for d in DIMS + ["total_score"] for x in df[d])
    print(f"total==sum_dims: {pct(int((df.total_score == df.sum_dims).sum()), len(df))}; nguyên: {ints}; trong khoảng: {in_range}")
    keys = {tuple(sorted(s["output"].keys())) for s in tasks["t1"]["samples"]}
    print(f"Key output T1: {keys}")
    ex1 = df[df.exam_id == "EX01"]
    full4 = [r.sample_id for _, r in ex1.iterrows() if all(re.search(rf"P{k}\b", r.feedback) for k in range(1, 5))]
    tq = list(ex1[ex1.feedback.str.contains("tiên quyết")].sample_id)
    p1w = [r.sample_id for _, r in ex1.iterrows() if p1_status(r.feedback) == "wrong"]
    print(f"Feedback nhắc đủ P1..P4: {pct(len(full4), len(ex1))} {full4}")
    print(f"P1 sai: {pct(len(p1w), len(ex1))} {p1w}; có chữ 'tiên quyết': {pct(len(tq), len(ex1))} {tq}")


# ----------------------------------------------------------------------------- M-12
def check_m12(df: pd.DataFrame, label_space: dict) -> None:
    header("M-12 Nhãn 'Lỗi edge case' và macro-F1")
    labels = label_space["taxonomy_error"]
    freq = {l: int(df.labels.map(lambda xs: l in xs).sum()) for l in labels}
    print(md(pd.DataFrame([{"label": k, "support": v} for k, v in freq.items()])))
    print(f"\nedge_case value_counts: {df.edge_case.value_counts().sort_index().to_dict()}; edge_case<2: {pct(int((df.edge_case < 2).sum()), len(df))}")
    iso = df[(df.edge_case < 2) & (df.logic == DIM_MAX["logic"])]
    print(f"edge_case<2 nhưng logic=4 (trừ riêng edge case): {pct(len(iso), len(df))} {list(iso.sample_id)} labels={list(iso.labels)}")
    print(f"edge_case<2 và logic<4 (trừ dây chuyền): {pct(int(((df.edge_case < 2) & (df.logic < 4)).sum()), len(df))}")
    print(f"corr(edge_case, logic) Spearman={df.edge_case.corr(df.logic, method='spearman'):.3f}")
    mlb = MultiLabelBinarizer(classes=labels)
    y = mlb.fit_transform(df.labels)
    res = []
    for zd in (0, 1):
        res.append({"setting": f"perfect pred, labels=10, zero_division={zd}",
                    "macro_f1": f1_score(y, y, average="macro", zero_division=zd)})
    fp = y.copy()
    fp[0, labels.index("Lỗi edge case")] = 1
    for zd in (0, 1):
        res.append({"setting": f"perfect + 1 FP 'edge case', zero_division={zd}",
                    "macro_f1": f1_score(y, fp, average="macro", zero_division=zd)})
    present = [i for i, l in enumerate(labels) if freq[l] > 0]
    res.append({"setting": "perfect pred, chỉ nhãn có support (9)", "macro_f1": f1_score(y[:, present], y[:, present], average="macro", zero_division=0)})
    print("\n" + md(pd.DataFrame(res)))


# ----------------------------------------------------------------------------- M-13
def check_m13(exams: dict, df: pd.DataFrame, root: Path) -> None:
    header("M-13 generals.csv, whitelist thư viện, pow")
    csv_files = [str(p.relative_to(root)) for p in root.rglob("*.csv")]
    print(f"File .csv trong dataset: {csv_files}; statement nhắc generals.csv: {'generals.csv' in exams['EX01']['statement']}")
    names: dict[str, list[str]] = {}
    for _, r in df[df.exam_id == "EX01"].iterrows():
        for nm in set(re.findall(r'"([^"\n]*\.csv)"', strip_comments(r.code))):
            names.setdefault(re.sub(r"^[A-Za-z]:\\", lambda _m: "<drive>/", nm), []).append(r.sample_id)
    print("\nTên file csv trong code (đã bỏ comment):\n")
    print(md(pd.DataFrame([{"csv_name": k, "n": len(v), "samples": v} for k, v in names.items()])))
    rows = []
    for _, r in df[df.exam_id == "EX01"].iterrows():
        inc = re.findall(r'#\s*include\s*[<"]([^>"]+)[>"]', strip_comments(r.code))
        bad = [h for h in inc if h not in ALLOWED_EX01_HEADERS]
        if bad:
            rows.append({"sample_id": r.sample_id, "non_whitelist": bad, "compilable": r.compilable, "total": r.total_score})
    print("\nInclude ngoài whitelist EX01:\n")
    print(md(pd.DataFrame(rows)))
    rows = []
    for _, r in df[df.exam_id == "EX02"].iterrows():
        co = strip_strings(strip_comments(r.code))
        calls = len(re.findall(r"\bpow\s*\(", co))
        if calls:
            rows.append({"sample_id": r.sample_id, "pow_tokens": calls,
                         "user_defined_pow": bool(re.search(r"\b(int|long|double|float)\s+pow\s*\([^)]*\)\s*\{", co)),
                         "includes_cmath": "cmath" in co, "total": r.total_score})
    print("\nEX02 dùng pow:\n")
    print(md(pd.DataFrame(rows)))
    print(f"\nEX01 has_hidden_testcases={exams['EX01']['has_hidden_testcases']}; EX01 test_report null: {df[df.exam_id=='EX01'].test_report.isna().sum()}/15")


# ----------------------------------------------------------------------------- M-14
def check_m14(exams: dict, df: pd.DataFrame) -> None:
    header(f"M-14 Ước lượng token (chars/{CHARS_PER_TOKEN})")
    rows = []
    for ex, g in df.groupby("exam_id"):
        st = est_tokens(exams[ex]["statement"])
        prompt = g.apply(lambda r: est_tokens(exams[ex]["statement"]) + est_tokens(r.code) + est_tokens(r.compile_log)
                         + est_tokens(json.dumps(r.test_report, ensure_ascii=False) if isinstance(r.test_report, list) else ""), axis=1)
        code = g.code.map(est_tokens)
        fb = g.feedback.map(est_tokens)
        rows.append({"exam": ex, "statement": round(st), "code_p50": round(code.quantile(.5)), "code_p95": round(code.quantile(.95)),
                     "prompt_p50": round(prompt.quantile(.5)), "prompt_p95": round(prompt.quantile(.95)), "prompt_max": round(prompt.max()),
                     "fb_p50": round(fb.quantile(.5)), "fb_p95": round(fb.quantile(.95))})
    print(md(pd.DataFrame(rows)))


# ----------------------------------------------------------------------------- M-15
def check_m15(df: pd.DataFrame) -> None:
    header("M-15 Dấu vết ẩn danh (chỉ đếm)")
    pats = {
        "REDACTED token": r"_REDACTED\]",
        "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
        ">=7 chữ số liền": r"(?<!\d)\d{7,}(?!\d)",
        "MSSV/Họ tên/Lớp/Student ID/Name:": r"(?i)mssv|họ\s*(và)?\s*tên|\blớp\s*:|student\s*id|\bname\s*:",
        "VS template header": r"This file contains the 'main' function",
    }
    rows = []
    for k, p in pats.items():
        hit = [r.sample_id for _, r in df.iterrows()
               if re.search(p, (r.code or "") + "\n" + (r.feedback or "") + "\n" + (r.compile_log or ""))]
        rows.append({"pattern": k, "n_samples": len(hit), "samples": hit})
    print(md(pd.DataFrame(rows)))
    hits = []
    for _, r in df.iterrows():
        code_only = strip_strings(strip_comments(r.code))
        non_code = (r.feedback or "") + "\n" + (r.compile_log or "") + "\n" + "\n".join(re.findall(r"//[^\n]*|/\*.*?\*/|\"[^\"\n]*\"", r.code, flags=re.S))
        for m in re.findall(r"(?<!\d)\d{7,}(?!\d)", r.code):
            hits.append({"sample_id": r.sample_id, "n_digits": len(m), "power_of_10": bool(re.fullmatch(r"10*", m)),
                         "in_code_only": m in code_only, "in_comment_string_fb_log": m in non_code})
    print("\nChuỗi >=7 chữ số (không in giá trị):\n")
    print(md(pd.DataFrame(hits)))


# ----------------------------------------------------------------------------- M-16
def check_m16(root: Path, tasks: dict) -> None:
    header("M-16 Metric/định dạng nộp trong README và task file")
    readme = (root / "README.md").read_text(encoding="utf-8")
    for k, v in tasks.items():
        print(f"- {k} metric: {v['metric']}")
    m = re.search(r"metric: \(a\)[^\n]*\n[^\n]*", readme)
    print(f"- README T3 metric: {m.group(0) if m else None}")
    blob = readme + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'samples'} for k, v in tasks.items()}, ensure_ascii=False)
    for kw in ("prediction", "predictions", "dự đoán", "submission", "định dạng nộp", "file nộp", "leaderboard"):
        print(f"  keyword '{kw}': {len(re.findall(re.escape(kw), blob, flags=re.I))}")
    print(f"- T3 input_fields có test_report: {'test_report' in ' '.join(tasks['t3']['input_fields'])}")


# ----------------------------------------------------------------------------- extra checks (missed findings)
def check_extra_truncation(df: pd.DataFrame) -> None:
    header("X1 compile_log bị cắt ở 2000 ký tự")
    lens = df.compile_log.fillna("").str.len()
    print(md(df.assign(log_len=lens)[lens > 0][["sample_id", "log_len"]].sort_values("log_len")))
    for _, r in df[lens == LOG_TRUNCATION_LEN].iterrows():
        print(f"\n{r.sample_id}: len={len(r.compile_log)}; 30 ký tự cuối={r.compile_log[-30:]!r}; "
              f"số 'error:' còn lại={r.compile_log.count('error:')}")


def check_extra_full_size(root: Path) -> None:
    header("X2 Quy mô tập đầy đủ suy ra từ README")
    readme = (root / "README.md").read_text(encoding="utf-8")
    m = re.findall(r"(\d+)/(\d+) bài `(EX0\d)`", readme)
    tot = {ex: int(d) for _, d, ex in m}
    print(f"README: {m} -> tổng tập đầy đủ ≈ {sum(tot.values())}; mẫu = 32 = {32 / sum(tot.values()):.1%}; "
          f"EX01 15/{tot.get('EX01')} , EX02 17/{tot.get('EX02')}")


def qwk(y: np.ndarray, p: np.ndarray) -> float:
    return cohen_kappa_score(y, p, weights="quadratic", labels=list(range(11)))


def check_extra_qwk(df: pd.DataFrame, seed: int = 0, n_boot: int = 2000) -> None:
    header("X3 QWK trên total: baseline thô chỉ dùng tín hiệu build")
    y = df.total_score.to_numpy()
    build_err = df.compile_log.fillna("").map(lambda s: bool(ANY_BUILD_ERROR_RE.search(s))).to_numpy()
    ok_med = int(np.median(y[~build_err]))
    preds = {
        f"build_err?0:{ok_med} (2 giá trị)": np.where(build_err, 0, ok_med),
        "constant 7": np.full_like(y, 7),
    }
    ex2 = df.exam_id.eq("EX02").to_numpy()
    pr = df.pass_rate.fillna(0).to_numpy()
    preds["EX02 dùng pass_rate*10, EX01 build?0:8"] = np.where(ex2, np.where(build_err, 0, np.rint(pr * 10)), np.where(build_err, 0, 8)).astype(int)
    rng = np.random.default_rng(seed)
    rows = []
    for name, p in preds.items():
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if len(set(y[idx])) > 1 and len(set(p[idx])) > 1:
                boots.append(qwk(y[idx], p[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        rows.append({"predictor": name, "QWK": qwk(y, p), "MAE": float(np.abs(y - p).mean()), "boot95_lo": lo, "boot95_hi": hi})
    print(md(pd.DataFrame(rows)))
    print(f"\nPhân phối total: {df.total_score.value_counts().sort_index().to_dict()}")


def run_in_gcc_container(src: Path, opt: str, inputs: list[str], image: str) -> list[str]:
    """Compile+run one EX02 file inside an existing local GNU image (no pull, no network), 2 s per run."""
    loop = " ; ".join(f"printf '%s\\n' '{x}' | timeout 2 /tmp/s; echo \"[rc=$?]\"" for x in inputs)
    script = f"cp /src/s.cpp /tmp/s.cpp && g++ -std=c++11 {opt} /tmp/s.cpp -o /tmp/s 2>/dev/null && {{ {loop} ; }}"
    p = subprocess.run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "sh",
                        "-v", f"{src}:/src/s.cpp:ro", image, "-c", script], capture_output=True, text=True, timeout=180)
    return re.findall(r"(.*?)\[rc=(\d+)\]", p.stdout.replace("\n", ""))


def run_with_clang(src: Path, opt: str, inputs: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as td:
        exe = Path(td) / "a.out"
        subprocess.run(["clang++", "-std=c++11", opt, "-w", str(src), "-o", str(exe)], capture_output=True, timeout=60)
        for x in inputs:
            try:
                p = subprocess.run([str(exe)], input=x + "\n", capture_output=True, text=True, timeout=2, cwd=td)
                out.append((p.stdout.strip(), str(p.returncode)))
            except subprocess.TimeoutExpired:
                out.append(("<TIMEOUT>", "timeout"))
    return out


def summarize_runs(runs: list[tuple[str, str]]) -> str:
    counts: dict[str, int] = {}
    for stdout, rc in runs:
        key = f"{stdout.strip() or chr(39) * 2}[rc={rc}]"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))


def check_extra_empty_actual(df: pd.DataFrame, root: Path, use_docker: bool, image: str) -> None:
    header("X4 actual='' trong test_report: crash/UB không được ghi nhận")
    ex2 = df[df.exam_id == "EX02"]
    rows = []
    for _, r in ex2.iterrows():
        if not isinstance(r.test_report, list) or not r.test_report:
            continue
        n_empty = sum(t["actual"] == "" for t in r.test_report)
        if n_empty:
            co = strip_comments(r.code)
            rows.append({"sample_id": r.sample_id, "empty_actual": f"{n_empty}/{len(r.test_report)}",
                         "cout_TF_literals": len(re.findall(r'cout\s*<<\s*"(True|False|TRUE|FALSE)"', co)),
                         "log_warning": (r.compile_log or "").split("warning:")[-1].strip().splitlines()[0][:60] if "warning" in (r.compile_log or "") else "",
                         "safe_to_run": not DANGEROUS_RE.search(co)})
    t = pd.DataFrame(rows)
    print(md(t))
    keys = set().union(*[set(tt.keys()) for r in ex2.test_report if isinstance(r, list) for tt in r])
    print(f"\nKey của mỗi test: {sorted(keys)} (không có exit code / stderr)")
    inputs = [tt["input"] for tt in ex2[ex2.n_tests == 8].iloc[0].test_report]
    res = []
    for _, r in t.iterrows():
        if not r.safe_to_run:
            continue
        src = (root / f"submissions/EX02/{r.sample_id}.cpp").resolve()
        if shutil.which("clang++"):
            for opt in ("-O0", "-O2"):
                res.append({"sample_id": r.sample_id, "toolchain": f"Apple clang {opt}", "8 inputs ->": summarize_runs(run_with_clang(src, opt, inputs))})
        if use_docker and shutil.which("docker"):
            for opt in ("-O0", "-O2"):
                res.append({"sample_id": r.sample_id, "toolchain": f"GNU g++ ({image}) {opt}",
                            "8 inputs ->": summarize_runs(run_in_gcc_container(src, opt, inputs, image))})
    print("\nChạy lại 8 input công khai (timeout 2s/lần; rc=124 là timeout của GNU timeout):\n")
    print(md(pd.DataFrame(res)))


def as_report_value(stdout: str, rc: str) -> str:
    """Map a raw run to the test_report vocabulary: timeouts -> <TIMEOUT>, otherwise stripped stdout."""
    return "<TIMEOUT>" if rc in ("124", "timeout") else stdout.strip()


def check_extra_reproduce_tests(df: pd.DataFrame, root: Path, use_docker: bool, image: str) -> None:
    header("X5 Tái lập test_report EX02 theo toolchain / mức tối ưu")
    ex2 = df[(df.exam_id == "EX02") & (df.n_tests == 8)]
    configs: list[tuple[str, str]] = [("clang", "-O0"), ("clang", "-O2")]
    if use_docker and shutil.which("docker"):
        configs += [("gcc", "-O0"), ("gcc", "-O2")]
    rows = []
    for _, r in ex2.iterrows():
        if DANGEROUS_RE.search(strip_comments(r.code)):
            rows.append({"sample_id": r.sample_id, "skipped": "unsafe"})
            continue
        src = (root / r.code_file).resolve()
        inputs = [tt["input"] for tt in r.test_report]
        recorded = [tt["actual"] for tt in r.test_report]
        row: dict[str, object] = {"sample_id": r.sample_id}
        for tool, opt in configs:
            runs = run_with_clang(src, opt, inputs) if tool == "clang" else run_in_gcc_container(src, opt, inputs, image)
            got = [as_report_value(o, rc) for o, rc in runs]
            row[f"{tool}{opt}"] = sum(a == b for a, b in zip(got, recorded))
        rows.append(row)
    t = pd.DataFrame(rows)
    print(md(t))
    for c in [c for c in t.columns if c != "sample_id" and c != "skipped"]:
        print(f"- {c}: khớp {int(t[c].sum())}/{8 * len(t)} actual; khớp đủ 8/8 ở {pct(int((t[c] == 8).sum()), len(t))} bài")


def check_extra_degenerate_dims(df: pd.DataFrame) -> None:
    header("X6 Chiều rubric suy biến -> exact-match từng chiều dễ 'ăn' bằng hằng số/luật")
    rows = []
    for ex, g in df.groupby("exam_id"):
        for d in DIMS:
            vc = g[d].value_counts()
            rows.append({"exam": ex, "dim": d, "mode": int(vc.index[0]), "mode_share": pct(int(vc.iloc[0]), len(g))})
    print(md(pd.DataFrame(rows)))
    e1 = df[df.exam_id == "EX01"]
    rule = (e1.logic > 0).astype(int)
    print(f"\nEX01 complexity==code_quality==(logic>0): {pct(int(((e1.complexity == rule) & (e1.code_quality == rule)).sum()), len(e1))}")
    print(f"EX01 io_format==compilable: {pct(int((e1.io_format == e1.compilable).sum()), len(e1))}; ngoại lệ: {list(e1[e1.io_format != e1.compilable].sample_id)}")
    e2 = df[df.exam_id == "EX02"]
    print(f"EX02 code_quality=1 kể cả bài CE: {pct(int((e2.code_quality == 1).sum()), len(e2))}")
    vec = df[DIMS].astype(str).agg(",".join, axis=1)
    print("Vector rubric phổ biến nhất mỗi exam:", {ex: vec[df.exam_id == ex].value_counts().head(2).to_dict() for ex in ("EX01", "EX02")})


# ----------------------------------------------------------------------------- main
ROOT = Path(".")


def main() -> None:
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--no-compile", action="store_true", help="skip local clang builds")
    ap.add_argument("--no-docker", action="store_true", help="skip GNU g++ re-runs inside an existing docker image")
    ap.add_argument("--gcc-image", default="ghcr.io/open-webui/open-webui:main", help="existing local image that has g++")
    args = ap.parse_args()
    ROOT = args.data_root
    label_space, exams, tasks = load_meta(ROOT)
    df = load_samples(ROOT)
    print(f"# verify_meta_audit — n={len(df)} (EX01={int((df.exam_id == 'EX01').sum())}, EX02={int((df.exam_id == 'EX02').sum())})")
    check_m01(exams, df)
    check_m02(exams, df)
    check_m03(df)
    check_m03_uncommented(df, not args.no_compile)
    check_m04(tasks, ROOT)
    check_m05(exams, df, ROOT, not args.no_compile)
    check_m06(exams, df)
    check_m07(df)
    check_m07_s207_fix(df, not args.no_compile)
    check_m08(df)
    check_m09(exams, df)
    check_m10(df, ROOT, not args.no_compile)
    check_m11(df, tasks)
    check_m12(df, label_space)
    check_m13(exams, df, ROOT)
    check_m14(exams, df)
    check_m15(df)
    check_m16(ROOT, tasks)
    check_extra_truncation(df)
    check_extra_full_size(ROOT)
    check_extra_qwk(df)
    check_extra_degenerate_dims(df)
    check_extra_empty_actual(df, ROOT, not args.no_docker, args.gcc_image)
    check_extra_reproduce_tests(df, ROOT, not args.no_docker, args.gcc_image)


if __name__ == "__main__":
    main()
