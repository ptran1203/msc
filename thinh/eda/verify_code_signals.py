"""Independent re-computation of the code_signals findings (C-01..C-15) plus extra checks.

Usage:
    python verify_code_signals.py --data-root <dataset dir> [--workdir DIR] [--cxx clang++]
                                  [--gnu-image IMAGE] [--no-run]

Everything is recomputed from the raw files; the analyst's helper modules are NOT imported.
Student code is untrusted: only EX02 (stdin/stdout) programs are executed, only after a
source grep for dangerous calls, inside a temp dir, with a <=2 s timeout and the test input
on stdin. --gnu-image runs the same EX02 programs with GNU g++ inside an already-present
local Docker image (network disabled, `timeout 2`), never pulling anything.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import hypergeom, spearmanr

from load_data import DIMS, load_meta, load_samples

RUN_TIMEOUT_S = 2.0
DANGEROUS_RE = re.compile(r"system\s*\(|popen|\bfork\s*\(|\bexec\w*\s*\(|\bremove\s*\(|\brename\s*\(|ofstream|fopen|socket")
LINK_ERROR_RE = re.compile(r"collect2: error|undefined reference|ld returned|linker command failed|Undefined symbols")
DIAG_ERROR_RE = re.compile(r"^\S+:\d+:\d+: (?:fatal )?error:", re.M)
DIAG_WARN_RE = re.compile(r"^\S+:\d+:\d+: warning:.*\[(-W[\w-]+)\]", re.M)
GNU_MARKERS = {
    "unicode_quotes": re.compile(r"[‘’]"),
    "in_function": re.compile(r"In function ‘"),
    "gcc13_path": re.compile(r"/c\+\+/13/|/gcc/x86_64-linux-gnu/13/"),
    "caret_gutter": re.compile(r"^\s*\d+ \| ", re.M),
    "collect2": re.compile(r"collect2"),
    "compilation_terminated": re.compile(r"compilation terminated"),
}
CPP_KEYWORDS = set(
    """alignas alignof and asm auto bool break case catch char char16_t char32_t class const constexpr
    const_cast continue decltype default delete do double dynamic_cast else enum explicit export extern
    false float for friend goto if inline int long mutable namespace new noexcept not nullptr operator or
    private protected public register reinterpret_cast return short signed sizeof static static_assert
    static_cast struct switch template this thread_local throw true try typedef typeid typename union
    unsigned using virtual void volatile wchar_t while include define std string cout cin endl getline
    ifstream fstream stringstream istringstream main NULL size_t stoi stof abs""".split()
)
ARMSTRONG_EXTRA = [2, 5, 7, 371, 407, 8208, 54748, 548834, 9800817, 24678050, 146511208, 912985153]
NON_ARMSTRONG_EXTRA = [100, 999, 9475, 100000, 999999999, 1000000000]


# ------------------------------------------------------------------------------ helpers


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if math.isnan(v) else f"{v:.2f}")
            elif isinstance(v, (list, tuple, set)):
                cells.append(", ".join(map(str, v)))
            else:
                cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def frac(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.0%})" if n else "0/0"


def header(title: str) -> None:
    print(f"\n## {title}\n")


@dataclass
class Stripped:
    code: str  # comments replaced by spaces, newlines kept
    comment_line_nos: set[int]
    comments: list[tuple[int, str]]  # (start line, text)


def strip_comments(src: str) -> Stripped:
    """Small C++ lexer: removes // and /* */ comments, respects string and char literals."""
    out: list[str] = []
    comments: list[tuple[int, str]] = []
    comment_lines: set[int] = set()
    i, line, n = 0, 1, len(src)
    state = "code"
    buf: list[str] = []
    start = 0
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state, buf, start = "line", [], line
                i += 2
                continue
            if c == "/" and nxt == "*":
                state, buf, start = "block", [], line
                i += 2
                continue
            if c in "\"'":
                q = c
                out.append(c)
                i += 1
                while i < n and src[i] != q and src[i] != "\n":
                    if src[i] == "\\" and i + 1 < n:
                        out.append(src[i : i + 2])
                        i += 2
                        continue
                    out.append(src[i])
                    i += 1
                if i < n and src[i] == q:
                    out.append(q)
                    i += 1
                continue
            out.append(c)
        elif state == "line":
            comment_lines.add(line)
            if c == "\n":
                comments.append((start, "".join(buf)))
                state = "code"
                out.append(c)
            else:
                buf.append(c)
        else:  # block
            comment_lines.add(line)
            if c == "*" and nxt == "/":
                comments.append((start, "".join(buf)))
                state = "code"
                i += 2
                continue
            buf.append(c)
            if c == "\n":
                out.append(c)
        if c == "\n":
            line += 1
        i += 1
    if state == "line":
        comments.append((start, "".join(buf)))
    return Stripped("".join(out), comment_lines, comments)


def effective_lines(src: str) -> int:
    return sum(1 for ln in strip_comments(src).code.splitlines() if ln.strip())


def looks_like_code(text: str) -> bool:
    t = text.strip()
    return bool(t) and bool(re.search(r"[;{}]\s*$|^#\s*include|\b(?:if|for|while|return|int|cout|cin)\b", t))


def is_armstrong(n: int) -> bool:
    s = str(n)
    return sum(int(ch) ** len(s) for ch in s) == n


def run(cmd: list[str], cwd: Path, stdin: str | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True, timeout=timeout)


def safe_to_run(code: str) -> bool:
    return not DANGEROUS_RE.search(code)


def run_binary(binary: Path, stdin: str) -> tuple[str, int | str]:
    try:
        res = subprocess.run([str(binary)], cwd=binary.parent, input=stdin, capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
        return res.stdout, res.returncode
    except subprocess.TimeoutExpired:
        return "<TIMEOUT>", "timeout"


def lenient_norm(s: str) -> str:
    s = s.strip()
    if ":" in s:
        s = s.rsplit(":", 1)[1]
    return s.strip().casefold()


# ------------------------------------------------------------------------------ sections


def c01_s214(df: pd.DataFrame, cxx: str, work: Path, do_run: bool) -> None:
    header("C-01 EX02-S214: file comment toàn bộ")
    r = df.set_index("sample_id").loc["EX02-S214"]
    code = r.code
    st = strip_comments(code)
    lines = code.splitlines()
    nonblank = [i for i, ln in enumerate(lines, 1) if ln.strip()]
    comment_nonblank = [i for i in nonblank if i in st.comment_line_nos]
    code_like = [i for i in nonblank if looks_like_code(lines[i - 1].lstrip("/").strip())]
    print(md(pd.DataFrame([{
        "lines": len(lines), "nonblank": len(nonblank), "comment_lines": len(comment_nonblank),
        "effective_code_lines": effective_lines(code), "code_like_comment_lines": len(code_like),
        "log_link_error": bool(LINK_ERROR_RE.search(r.compile_log or "")), "loader_log_has_error": r.log_has_error,
        "test_report": r.test_report, "compilable": r.compilable, "total": r.total_score,
        "logic": r.logic, "edge_case": r.edge_case, "labels": r.labels,
    }])))
    if not do_run:
        return
    uncommented = "\n".join(re.sub(r"^//", "", ln) for ln in lines) + "\n"
    if not safe_to_run(uncommented):
        print("- bỏ qua chạy: nguồn chứa lời gọi nguy hiểm")
        return
    wd = work / "s214_uncommented"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "main.cpp").write_text(uncommented)
    comp = run([cxx, "-std=c++11", "-O0", "-w", "main.cpp", "-o", "a.out"], wd)
    rows = []
    for t in public_tests(df):
        out, rc = run_binary(wd / "a.out", t["input"] + "\n")
        rows.append({"input": t["input"], "expected": t["expected"], "stdout": out.strip(), "pass": out.strip() == t["expected"]})
    res = pd.DataFrame(rows)
    print(f"\n- Bỏ `//` rồi biên dịch ({cxx}): rc={comp.returncode}; pass {frac(int(res['pass'].sum()), len(res))} test công khai")
    print(md(res))


def public_tests(df: pd.DataFrame) -> list[dict]:
    seen: dict[str, dict] = {}
    for rep in df[df.exam_id == "EX02"].test_report:
        if isinstance(rep, list):
            for t in rep:
                seen.setdefault(t["input"], {"input": t["input"], "expected": t["expected"]})
    return list(seen.values())


def compile_signals(df: pd.DataFrame, cxx: str, work: Path) -> pd.DataFrame:
    """Local syntax check (EX01: -fsyntax-only as in exams.json note; EX02: full compile+link)."""
    rows = []
    for r in df.itertuples():
        wd = work / "compile" / r.sample_id
        wd.mkdir(parents=True, exist_ok=True)
        src = wd / "student.cpp"
        src.write_text(r.code)
        if r.exam_id == "EX01":
            res = run([cxx, "-std=c++11", "-fsyntax-only", "student.cpp"], wd)
            alone = run([cxx, "-std=c++11", "student.cpp", "-o", "id"], wd)
        else:
            res = run([cxx, "-std=c++11", "-O0", "student.cpp", "-o", "a.out"], wd)
            alone = res
        log = r.compile_log or ""
        rows.append({
            "sample_id": r.sample_id, "exam_id": r.exam_id, "compilable": r.compilable, "total": r.total_score,
            "log_error": bool(DIAG_ERROR_RE.search(log) or LINK_ERROR_RE.search(log)),
            "log_link_error": bool(LINK_ERROR_RE.search(log)),
            "loader_log_has_error": bool(r.log_has_error),
            "local_error": res.returncode != 0,
            "local_n_error": len(re.findall(r": error:", res.stderr)),
            "log_n_error": len(re.findall(r": (?:fatal )?error:", log)),
            "statement_cmd_link_ok": alone.returncode == 0,
            "local_warnings": sorted(set(re.findall(r"\[(-W[\w-]+)\]", res.stderr))),
        })
    return pd.DataFrame(rows)


def c02_c10_compile(df: pd.DataFrame, comp: pd.DataFrame, readme: str) -> None:
    header("C-02 / C-10 compilable vs tín hiệu biên dịch")
    cross = comp.groupby(["exam_id", "compilable", "log_error", "local_error"]).size().reset_index(name="n")
    print(md(cross))
    void_main = df.code.map(lambda c: bool(re.search(r"\bvoid\s+main\s*\(", strip_comments(c).code)))
    pow_call = df.code.map(lambda c: len(re.findall(r"(?<![\w.])pow\s*\(", strip_comments(c).code)))
    pow_def = df.code.map(lambda c: len(re.findall(r"\bint\s+pow\s*\(", strip_comments(c).code)))
    cmath = df.code.map(lambda c: bool(re.search(r"#\s*include\s*<(?:cmath|math\.h)>", strip_comments(c).code)))
    pow_no_cmath = df[(pow_call - pow_def > 0) & ~cmath].sample_id.tolist()
    print(f"\n- void main: {frac(int(void_main.sum()), len(df))}; file có `pow(` (gồm cả định nghĩa): {df[pow_call > 0].sample_id.tolist()}; "
          f"có định nghĩa `int pow(`: {df[pow_def > 0].sample_id.tolist()}; có lời gọi `pow(` (kể cả tới hàm tự định nghĩa) mà không include cmath: {pow_no_cmath}")
    bad = comp[(comp.compilable == 1) & (comp.log_error | comp.local_error)]
    print(f"- compilable=1 nhưng log gốc hoặc clang báo lỗi: {bad.sample_id.tolist()}")
    ex01 = comp[comp.exam_id == "EX01"]
    fail_link = ex01[(ex01.compilable == 1) & ~ex01.statement_cmd_link_ok]
    print(f"- EX01 với đúng lệnh `g++ id.cpp -o id` (link): ok {frac(int(ex01.statement_cmd_link_ok.sum()), len(ex01))}; FAIL mà compilable=1: {fail_link.sample_id.tolist()}")
    m = re.search(r"(\d+)/(\d+) bài `EX02` và (\d+)/(\d+) bài `EX01`", readme)
    if m:
        k2, n2, k1, n1 = map(int, m.groups())
        s1, s2 = int((df.exam_id == "EX01").sum()), int((df.exam_id == "EX02").sum())
        p0_ex01 = hypergeom(n1, k1, s1).pmf(0)
        p0_ex02 = hypergeom(n2, k2, s2).pmf(0)
        print(f"- README: {k2}/{n2} EX02, {k1}/{n1} EX01 có compiles_locally=false mà compilable=1. Nếu mẫu là tập con ngẫu nhiên: "
              f"P(0 ca trong {s1} bài EX01)={p0_ex01:.2f}; P(0 ca trong {s2} bài EX02)={p0_ex02:.2f} -> vắng mặt trên mẫu không bác bỏ README.")
    miss = comp[comp.log_error != comp.loader_log_has_error]
    print(f"- C-10 loader.log_has_error khác phát hiện lỗi (gồm link): {miss.sample_id.tolist()}; "
          f"lỗi link trong log: {comp[comp.log_link_error].sample_id.tolist()}")
    ex02 = df[df.exam_id == "EX02"]
    empty_rep = ex02[ex02.n_tests.fillna(0) == 0].sample_id.tolist()
    log_err02 = comp[(comp.exam_id == "EX02") & comp.log_error].sample_id.tolist()
    print(f"- EX02: test_report rỗng ở {empty_rep}; log có lỗi (gồm link) ở {log_err02}; trùng khớp: {sorted(empty_rep) == sorted(log_err02)}")
    lab = df.labels.map(lambda xs: "Lỗi biên dịch" in xs)
    print(f"- Nhãn 'Lỗi biên dịch' ⇔ compilable==0: {int((lab == (df.compilable == 0)).sum())}/{len(df)}; "
          f"log có lỗi (gồm link) mà không có nhãn 'Lỗi biên dịch': {comp[comp.log_error & ~lab.values].sample_id.tolist()}")


def statement_code_blocks(statement: str) -> str:
    return "\n".join(re.findall(r"```cpp\n(.*?)```", statement, re.S))


def parse_proto(text: str) -> tuple[str, str, tuple[str, ...]]:
    m = re.match(r"\s*(.+?)\b(\w+)\s*\(([^)]*)\)", text)
    assert m, text
    ret = re.sub(r"\s+", "", m.group(1))
    params = []
    for p in [x for x in m.group(3).split(",") if x.strip()]:
        pm = re.match(r"^(.*?)(\w+)\s*$", p.strip())
        params.append(re.sub(r"\s+", "", pm.group(1) if pm and pm.group(1).strip() else p))
    return ret, m.group(2), tuple(params)


def c03_c04_ex01(df: pd.DataFrame, exams: dict, readme: str, comp: pd.DataFrame, cxx: str, work: Path) -> None:
    header("C-03 / C-04 EX01: quy định đề, prototype, harness, độ dài")
    ex = exams["EX01"]
    st = ex["statement"]
    stated = [float(x) for x in re.findall(r"\*\*Problem \d+ \((\d+(?:\.\d+)?) điểm\)", st)]
    meta = [p["max_score"] for p in ex["problems"]]
    readme_w = re.search(r"trọng số (\d+/\d+/\d+/\d+)", readme)
    print(f"- Trọng số: statement {stated} (tổng {sum(stated)}); exams.json max_score {meta} (tổng {sum(meta)}); README {readme_w.group(1) if readme_w else None}")
    allowed = {x.strip() for x in re.search(r"thư viện: `([^`]+)`", st).group(1).split(",")}
    print(f"- Thư viện cho phép: {sorted(allowed)}; đề nói 'phải xoá main', 'Nhận 0 điểm nếu: vi phạm quy định'")
    protos = [parse_proto(p["prototype"]) for p in ex["problems"]]
    g = df[df.exam_id == "EX01"]
    rows = []
    for r in g.itertuples():
        code = strip_comments(r.code).code
        incs = re.findall(r"#\s*include\s*[<\"]([^>\"]+)[>\"]", code)
        rec = {
            "sample_id": r.sample_id, "total": r.total_score, "compilable": r.compilable,
            "has_main": bool(re.search(r"\bint\s+main\s*\(", code)),
            "outside_allowed": sorted(set(incs) - allowed),
            "eff_lines": effective_lines(r.code), "lines": r.code_lines,
        }
        for (ret, name, params), pid in zip(protos, ["P1", "P2", "P3", "P4"]):
            defs = re.findall(rf"([\w:<>\*&\s]+?)\b{name}\s*\(([^)]*)\)\s*\{{", code)
            decls = re.findall(rf"\b{name}\s*\([^)]*\)\s*;", code)
            if defs:
                d_ret = re.sub(r"\s+", "", defs[0][0].strip().split("\n")[-1])
                d_params = parse_proto(f"{defs[0][0].strip().splitlines()[-1]} {name}({defs[0][1]})")[2]
                rec[pid] = "match" if (d_ret, d_params) == (ret, params) else f"mismatch {d_ret} {d_params}"
            else:
                rec[pid] = "declared_only" if decls else "missing"
        rows.append(rec)
    t = pd.DataFrame(rows)
    pids = ["P1", "P2", "P3", "P4"]
    t["all4_match"] = t[pids].apply(lambda s: all(v == "match" for v in s), axis=1)
    print(md(t))
    print(f"\n- has_main: {frac(int(t.has_main.sum()), len(t))}; total của bài có main: {dict(zip(t[t.has_main].sample_id, t[t.has_main].total))}")
    print(f"- Đủ 4 hàm đúng prototype: {frac(int(t.all4_match.sum()), len(t))}; trạng thái: { {p: dict(Counter(t[p].str.split(' ').str[0])) for p in pids} }")
    print(f"- Header ngoài danh sách: {[(a, b, c) for a, b, c in zip(t.sample_id, t.outside_allowed, t.total) if b]}")
    print(f"- Trung bình total: có main {t[t.has_main].total.mean():.2f} (n={int(t.has_main.sum())}) vs không main {t[~t.has_main].total.mean():.2f} (n={int((~t.has_main).sum())})")
    t["n_impl"] = t[pids].apply(lambda s: sum(v == "match" for v in s), axis=1)
    for a, b in [("n_impl", "total"), ("eff_lines", "n_impl")]:
        rho, p = spearmanr(t[a], t[b])
        print(f"- Spearman {a}~{b} (EX01, n={len(t)}): rho={rho:.2f} p={p:.3f}")
    ok = t[t.compilable == 1]
    rho, p = spearmanr(ok.n_impl, ok.total)
    print(f"- Spearman n_impl~total (compilable==1, n={len(ok)}): rho={rho:.2f} p={p:.3f}")

    # Harness link simulation (compile + link only, never executed).
    blocks = re.findall(r"```cpp\n(.*?)```", st, re.S)
    struct_src = "\n".join(b for b in blocks if "struct" in b)
    decls = "\n".join(p["prototype"] for p in ex["problems"])
    stub_bodies = {
        "prob1": "General* prob1(string filename, int& n){ return 0; }",
        "prob2": "Node* prob2(General* generals, int n){ return 0; }",
        "prob3": "General prob3(Node* head){ return General(); }",
        "prob4": "void prob4(Node*& head, string nationality){}",
    }
    main_src = 'int main(){ int n=0; General* g=prob1("x", n); Node* h=prob2(g,n); General r=prob3(h); prob4(h,"x"); (void)r; return 0; }'
    base = f"#include <string>\nusing namespace std;\n{struct_src}\n{decls}\n"
    harness = {
        "strict": base + main_src + "\n",
        "weak_stubs": base + "\n".join("__attribute__((weak)) " + b for b in stub_bodies.values()) + "\n" + main_src + "\n",
    }
    hr = []
    for r in g.itertuples():
        wd = work / "harness" / r.sample_id
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "student.cpp").write_text(r.code)
        obj = run([cxx, "-std=c++11", "-w", "-c", "student.cpp", "-Dmain=student_main_renamed", "-o", "student.o"], wd)
        rec = {"sample_id": r.sample_id, "compilable": r.compilable}
        for mode, src in harness.items():
            (wd / f"h_{mode}.cpp").write_text(src)
            if obj.returncode != 0:
                rec[mode] = "compile_fail"
                continue
            lk = run([cxx, "-std=c++11", "-w", f"h_{mode}.cpp", "student.o", "-o", f"bin_{mode}"], wd)
            rec[mode] = "ok" if lk.returncode == 0 else "link_fail"
        hr.append(rec)
    h = pd.DataFrame(hr)
    for mode in harness:
        ok = h[mode] == "ok"
        print(f"- harness {mode}: link ok {frac(int(ok.sum()), len(h))}; trùng tập compilable=1: {sorted(h[ok].sample_id) == sorted(h[h.compilable == 1].sample_id)}")
    link_alone = comp[comp.exam_id == "EX01"].set_index("sample_id").statement_cmd_link_ok
    print(f"- lệnh đề `g++ id.cpp -o id`: ok {frac(int(link_alone.sum()), len(link_alone))}")

    sp = []
    for exam_id, gg in df.groupby("exam_id"):
        eff = gg.code.map(effective_lines)
        for name, sub in [("all", gg.index), ("compilable==1", gg[gg.compilable == 1].index)]:
            rho, p = spearmanr(eff.loc[sub], gg.loc[sub, "total_score"])
            sp.append({"exam": exam_id, "subset": name, "n": len(sub), "rho_effective_lines_total": rho, "p": p})
    print("\n**Spearman số dòng code hiệu dụng ~ total**\n")
    print(md(pd.DataFrame(sp)))


def build_ex02(df: pd.DataFrame, cxx: str, work: Path) -> dict[str, dict[str, Path]]:
    variants = {"O0": ["-O0"], "O0_trap_missing_return": ["-O0", "-fsanitize=return", "-fsanitize-trap=return"]}
    bins: dict[str, dict[str, Path]] = {}
    for r in df[(df.exam_id == "EX02")].itertuples():
        if not safe_to_run(r.code):
            print(f"- bỏ qua {r.sample_id}: nguồn có lời gọi nguy hiểm")
            continue
        for v, flags in variants.items():
            wd = work / "run" / v / r.sample_id
            wd.mkdir(parents=True, exist_ok=True)
            (wd / "main.cpp").write_text(r.code)
            res = run([cxx, "-std=c++11", "-w", *flags, "main.cpp", "-o", "a.out"], wd)
            if res.returncode == 0:
                bins.setdefault(r.sample_id, {})[v] = wd / "a.out"
    return bins


def c05_rerun(df: pd.DataFrame, bins: dict[str, dict[str, Path]]) -> pd.DataFrame:
    header("C-05 Chạy lại binary EX02 trên 8 test công khai (clang cục bộ)")
    rows = []
    for r in df[(df.exam_id == "EX02") & (df.n_tests.fillna(0) > 0)].itertuples():
        for v, b in bins.get(r.sample_id, {}).items():
            for t in r.test_report:
                out, rc = run_binary(b, t["input"] + "\n")
                rows.append({"sample_id": r.sample_id, "variant": v, "input": t["input"], "actual": t["actual"],
                             "local_raw": out, "local": out.strip(), "rc": rc, "same": out.strip() == t["actual"].strip()})
    res = pd.DataFrame(rows)
    summ = res.groupby("variant").agg(tests=("same", "size"), reproduced=("same", "sum"))
    per = res.groupby(["variant", "sample_id"]).same.all().groupby("variant").agg(["sum", "size"])
    print(md(summ.join(per.rename(columns={"sum": "samples_100pct", "size": "samples"})).reset_index()))
    diff = res[~res.same].groupby(["variant", "sample_id"]).agg(local=("local", lambda s: sorted(set(s))), rc=("rc", lambda s: sorted(set(map(str, s))))).reset_index()
    print("\n**Không tái lập**\n")
    print(md(diff) if len(diff) else "_không có_")
    empty = sum(1 for rep in df.test_report if isinstance(rep, list) for t in rep if t["actual"] == "")
    total = sum(len(rep) for rep in df.test_report if isinstance(rep, list))
    print(f"\n- actual=='' trong test_report: {frac(empty, total)}; sample: {sorted({r.sample_id for r in df.itertuples() if isinstance(r.test_report, list) for t in r.test_report if t['actual'] == ''})}")
    trap = res[(res.variant == "O0_trap_missing_return") & res.sample_id.isin(["EX02-S206", "EX02-S210"])]
    print(f"- rc của S206/S210 với trap: {dict(Counter(map(str, trap.rc)))}")
    return res


def gnu_docker(df: pd.DataFrame, image: str, work: Path, extra_inputs: list[int]) -> pd.DataFrame | None:
    header(f"GNU g++ trong image cục bộ `{image}` (không pull, --network none)")
    if shutil.which("docker") is None:
        print("_không có docker_")
        return None
    d = work / "gnu"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    tests = [t["input"] for t in public_tests(df)] + [str(x) for x in extra_inputs]
    ids = []
    for r in df[df.exam_id == "EX02"].itertuples():
        if r.compilable is not None and safe_to_run(r.code):
            (d / f"{r.sample_id}.cpp").write_text(r.code)
            ids.append(r.sample_id)
    lines = ["set -u", "cd /w", "g++ --version | head -1",
             "echo 'int f(){}' > probe.cpp; g++ -c -O0 -funreachable-traps probe.cpp -o probe.o >/dev/null 2>&1 && echo FLAG_UNREACHABLE_TRAPS=yes || echo FLAG_UNREACHABLE_TRAPS=no"]
    for sid in ids:
        lines.append(f"g++ -std=c++11 -O0 -w {sid}.cpp -o {sid}.bin 2>/dev/null || echo 'RES\t{sid}\tCOMPILE_FAIL'")
        for inp in tests:
            lines.append(f"[ -x {sid}.bin ] && {{ out=$(echo {inp} | timeout 2 ./{sid}.bin 2>/dev/null); rc=$?; printf 'RES\\t{sid}\\t{inp}\\t%s\\t%s\\n' \"$rc\" \"$(echo $out | tr -d '\\n')\"; }}")
    for sid in [x for x in ("EX02-S206", "EX02-S210") if x in ids]:
        for opt in ("O1", "O2"):
            lines.append(f"g++ -std=c++11 -{opt} -w {sid}.cpp -o {sid}.{opt} && {{ out=$(echo 1634 | timeout 2 ./{sid}.{opt} 2>/dev/null); "
                         f"printf 'OPT {sid} -{opt} input=1634 rc=%s stdout=[%s]\\n' \"$?\" \"$out\"; }}")
    (d / "run.sh").write_text("\n".join(lines) + "\n")
    try:
        res = subprocess.run(["docker", "run", "--rm", "--network", "none", "-v", f"{d}:/w", "--entrypoint", "sh", image, "/w/run.sh"],
                             capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("_docker timeout_")
        return None
    head = [ln for ln in res.stdout.splitlines() if not ln.startswith("RES")]
    print("- " + "; ".join(head))
    rows = []
    for ln in res.stdout.splitlines():
        parts = ln.split("\t")
        if parts[0] == "RES" and len(parts) == 5:
            rows.append({"sample_id": parts[1], "input": parts[2], "rc": int(parts[3]), "stdout": parts[4]})
    return pd.DataFrame(rows)


def c05_gnu_compare(df: pd.DataFrame, gnu: pd.DataFrame | None) -> None:
    if gnu is None or gnu.empty:
        return
    rows = []
    for r in df[(df.exam_id == "EX02") & (df.n_tests.fillna(0) > 0)].itertuples():
        for t in r.test_report:
            m = gnu[(gnu.sample_id == r.sample_id) & (gnu.input == t["input"])]
            if m.empty:
                continue
            g = m.iloc[0]
            local = "<TIMEOUT>" if g.rc == 124 else g.stdout.strip()
            rows.append({"sample_id": r.sample_id, "same": local == t["actual"].strip(), "rc": g.rc, "gnu": local})
    res = pd.DataFrame(rows)
    print(f"- GNU -O0 tái lập actual: {frac(int(res.same.sum()), len(res))}; sample không khớp: "
          f"{res[~res.same].groupby('sample_id').agg(gnu=('gnu', lambda s: sorted(set(s))), rc=('rc', lambda s: sorted(set(s)))).reset_index().to_dict('records')}")


def c06_c07_c08_tests(df: pd.DataFrame, exams: dict, runs: pd.DataFrame | None) -> None:
    header("C-06 / C-07 / C-08 test_report vs điểm")
    g = df[(df.exam_id == "EX02") & (df.n_tests.fillna(0) > 0)].copy()
    g["strict"] = g.test_report.map(lambda rep: sum(t["passed"] for t in rep))
    g["strict_recheck"] = g.test_report.map(lambda rep: sum(t["actual"].strip() == t["expected"].strip() for t in rep))
    g["lenient"] = g.test_report.map(lambda rep: sum(lenient_norm(t["actual"]) == lenient_norm(t["expected"]) for t in rep))
    print(md(g[["sample_id", "strict", "strict_recheck", "lenient", "total_score", "io_format", "logic", "edge_case"]]))
    for col in ["strict", "lenient"]:
        rho, p = spearmanr(g[col], g.total_score)
        print(f"- n={len(g)}: Spearman {col}~total rho={rho:.2f} p={p:.3f}")
    all17 = df[df.exam_id == "EX02"].copy()
    all17["strict"] = all17.test_report.map(lambda rep: sum(t["passed"] for t in rep) if isinstance(rep, list) else 0)
    all17["lenient"] = all17.test_report.map(lambda rep: sum(lenient_norm(t["actual"]) == lenient_norm(t["expected"]) for t in rep) if isinstance(rep, list) else 0)
    for col in ["strict", "lenient"]:
        rho, p = spearmanr(all17[col], all17.total_score)
        print(f"- n=17 (test_report rỗng tính 0 pass): Spearman {col}~total rho={rho:.2f} p={p:.3f}")
    zero = g[g.strict == 0]
    print(f"- 0/8 strict: {dict(zip(zero.sample_id, zero.total_score))}; trong đó total>=7: {frac(int((zero.total_score >= 7).sum()), len(g))}")
    case_only = sum(1 for rep in g.test_report for t in rep if t["actual"] != t["expected"] and t["actual"].casefold() == t["expected"].casefold())
    ws = sum(1 for rep in g.test_report for t in rep if t["actual"] != t["actual"].strip())
    print(f"- C-07 test chỉ khác hoa/thường trong test_report: {case_only} (=> không quan sát được harness có phân biệt hoa/thường); actual có khoảng trắng đầu/cuối: {ws}")
    if runs is not None and not runs.empty:
        s205 = runs[(runs.sample_id == "EX02-S205") & (runs.variant == "O0")].local_raw.iloc[0]
        print(f"- stdout thô S205 (clang): {s205!r} vs actual {g.set_index('sample_id').loc['EX02-S205'].test_report[0]['actual']!r} -> test_report đã trim")
        s206 = runs[(runs.sample_id == "EX02-S206") & (runs.variant == "O0")].local.unique().tolist()
        print(f"- S206 stdout clang -O0 (không trap): {s206}")
    io = df[df.exam_id == "EX02"][["sample_id", "io_format", "total_score"]].set_index("sample_id")
    print(f"- io_format: S205={io.loc['EX02-S205'].io_format}, S206={io.loc['EX02-S206'].io_format}, S211={io.loc['EX02-S211'].io_format}, S216={io.loc['EX02-S216'].io_format}")

    tests = public_tests(df)
    sets = {tuple(sorted((t["input"], t["expected"]) for t in rep)) for rep in g.test_report}
    exp = Counter(t["expected"] for t in tests)
    const_true = sum(t["expected"] == "True" for t in tests)
    print(f"\n- C-08 số bộ test khác nhau giữa các bài: {len(sets)}; expected {dict(exp)}; in hằng 'True' pass {const_true}/{len(tests)}; "
          f"chữ số dài nhất {max(len(t['input']) for t in tests)}; ràng buộc: {exams['EX02']['problems'][0].get('constraints')}; "
          f"has_hidden_testcases EX02={exams['EX02'].get('has_hidden_testcases')}")
    print(f"- expected đúng theo định nghĩa Armstrong: {sum((t['expected'] == 'True') == is_armstrong(int(t['input'])) for t in tests)}/{len(tests)}")
    if runs is not None and not runs.empty:
        s208 = runs[(runs.sample_id == "EX02-S208") & (runs.variant == "O0")]
        print(f"- S208 stdout trên 8 input: {sorted(set(s208.local))}; total={io.loc['EX02-S208'].total_score}")
    for exam_id, gg in df.groupby("exam_id"):
        pairs = Counter(zip(gg.logic, gg.edge_case))
        logic_to_edge = gg.groupby("logic").edge_case.nunique()
        print(f"- {exam_id}: cặp (logic, edge_case) {dict(sorted(pairs.items()))}; edge_case là hàm của logic: {bool((logic_to_edge == 1).all())}")


def c05_extended(df: pd.DataFrame, bins: dict[str, dict[str, Path]], gnu: pd.DataFrame | None) -> None:
    header("Bổ sung: test mở rộng (edge case) vs nhãn edge_case/logic")
    inputs = ARMSTRONG_EXTRA + NON_ARMSTRONG_EXTRA
    rows = []
    for r in df[(df.exam_id == "EX02") & (df.compilable == 1)].itertuples():
        rec = {"sample_id": r.sample_id, "edge_case": r.edge_case, "logic": r.logic, "total": r.total_score}
        for src in ["clang_O0", "gnu_O0"]:
            ok = 0
            fails = []
            for x in inputs:
                exp = "True" if is_armstrong(x) else "False"
                if src == "clang_O0":
                    b = bins.get(r.sample_id, {}).get("O0")
                    if b is None:
                        ok = None
                        break
                    out, _ = run_binary(b, f"{x}\n")
                elif gnu is not None and not gnu.empty:
                    m = gnu[(gnu.sample_id == r.sample_id) & (gnu.input == str(x))]
                    if m.empty:
                        ok = None
                        break
                    out = "<TIMEOUT>" if m.iloc[0].rc == 124 else m.iloc[0].stdout
                else:
                    ok = None
                    break
                good = lenient_norm(out) == exp.casefold()
                ok += good
                if not good:
                    fails.append(x)
            rec[f"{src}_pass_lenient"] = f"{ok}/{len(inputs)}" if ok is not None else ""
            rec[f"{src}_fails_ge_1e5"] = [f for f in fails if f >= 10**5] if ok is not None else ""
        rows.append(rec)
    print(f"- {len(inputs)} input thêm: Armstrong {ARMSTRONG_EXTRA}; không Armstrong {NON_ARMSTRONG_EXTRA}; so sánh lenient (bỏ prompt, không phân biệt hoa/thường)\n")
    print(md(pd.DataFrame(rows)))


def c09_logs(df: pd.DataFrame, comp: pd.DataFrame) -> None:
    header("C-09 compile_log: định dạng, độ dài, khớp nguồn; so với lỗi trích trong feedback")
    rows = []
    hits = total = 0
    for r in df.itertuples():
        log = r.compile_log or ""
        if not log.strip():
            continue
        lines = r.code.splitlines()
        cur_student = False
        for ln in log.splitlines():
            dm = re.match(r"^(\S+?):\d+:\d+: ", ln)
            if dm:
                cur_student = dm.group(1).endswith("student.cpp")
                continue
            em = re.match(r"^\s*(\d+) \| (.*)$", ln)
            if em and cur_student:
                k = int(em.group(1))
                total += 1
                hits += 0 < k <= len(lines) and re.sub(r"\s+", "", lines[k - 1]) == re.sub(r"\s+", "", em.group(2))
        rows.append({"sample_id": r.sample_id, "chars": len(log),
                     "errors": len(re.findall(r": (?:fatal )?error:", log)) + bool(LINK_ERROR_RE.search(log)),
                     "warnings": sorted(set(re.findall(r"\[(-W[\w-]+)\]", log))),
                     **{k: bool(p.search(log)) for k, p in GNU_MARKERS.items()}})
    t = pd.DataFrame(rows)
    print(md(t))
    n = len(df)
    err = t.errors > 0
    print(f"\n- log rỗng {frac(n - len(t), n)}; có error (gồm link) {frac(int(err.sum()), n)}; chỉ warning {frac(int((~err).sum()), n)}; "
          f"dài đúng 2000 ký tự: {t[t.chars == 2000].sample_id.tolist()}; dòng nguồn trích trong log khớp .cpp: {hits}/{total}")
    print(f"- log có dấu hiệu phiên bản 13: {t[t.gcc13_path].sample_id.tolist()} ({int(t.gcc13_path.sum())}/{len(t)})")
    agree = int((comp.log_error == comp.local_error).sum())
    print(f"- clang cục bộ đồng ý error/không error với log gốc: {agree}/{len(comp)}; số error log vs clang: "
          f"{[(a, b, c) for a, b, c in zip(comp.sample_id, comp.log_n_error, comp.local_n_error) if b != c and (b or c)]}")
    fb = []
    for r in df.itertuples():
        f = r.feedback or ""
        m = re.search(r"(student\.\w+):(\d+):\d+:", f)
        if not m:
            continue
        log = r.compile_log or ""
        lm = re.search(r"(student\.\w+):(\d+):\d+: (?:fatal )?error", log)
        fb.append({"sample_id": r.sample_id, "fb_file": m.group(1), "fb_line": int(m.group(2)),
                   "log_file": lm.group(1) if lm else "", "log_line": int(lm.group(2)) if lm else None,
                   "fb_ascii_quotes": bool(re.search(r"'[^']+'", f)), "log_unicode_quotes": bool(re.search(r"[‘’]", log))})
    print("\n**Feedback giảng viên trích lỗi biên dịch vs compile_log phát hành**\n")
    print(md(pd.DataFrame(fb)))


def top_level_functions(code: str) -> list[tuple[str, str]]:
    """(name, body) for function definitions whose header starts at column 0; brace-matched."""
    out = []
    for m in re.finditer(r"^[A-Za-z_][\w:<>\*&\s]*?\b(\w+)\s*\([^;{)]*\)\s*\{", code, re.M):
        depth, i = 1, m.end()
        while i < len(code) and depth:
            depth += {"{": 1, "}": -1}.get(code[i], 0)
            i += 1
        out.append((m.group(1), code[m.end() : i - 1]))
    return out


def c11_hardcode(df: pd.DataFrame, exams: dict) -> None:
    header("C-11 'Lỗi hard-code' vs luật tĩnh")
    lab = df.labels.map(lambda xs: "Lỗi hard-code" in xs)
    print(f"- Nhãn hard-code: {frac(int(lab.sum()), len(df))}: {df[lab].sample_id.tolist()}")
    prob_names = {"prob1", "prob2", "prob3", "prob4"}
    rows = []
    tests = {int(t["input"]) for t in public_tests(df)}
    for r in df.itertuples():
        code = strip_comments(r.code).code
        funcs = top_level_functions(code)
        non_main = {name: body for name, body in funcs if name != "main"}
        body = "\n".join(non_main.values())
        rule = {
            "include_data_file": len(re.findall(r"#\s*include\s*\"[^\"]+\.(?:csv|txt)\"", code)),
            "literal_path_outside_main": len(re.findall(r"(?:\.open\s*\(|[io]?fstream\s+\w+\s*\()\s*\"", body)),
            "new_fixed_ge10": len([x for x in re.findall(r"\bnew\s+\w+\s*\[\s*(\d+)\s*\]", body) if int(x) >= 10]),
            "cin_in_probN": sum(1 for name, b in non_main.items() if name in prob_names and re.search(r"\bcin\b", b)),
        }
        magic = re.findall(r"[=!]=\s*(\d{2,})\b|\b(\d{2,})\s*[=!]=", code)
        mvals = {int(a or b) for a, b in magic}
        rows.append({"sample_id": r.sample_id, "label": bool(lab.loc[r.Index]), **rule, "rule_any": any(rule.values()),
                     "magic_cmp_armstrong_or_test": sorted(v for v in mvals if is_armstrong(v) or v in tests)})
    t = pd.DataFrame(rows)
    tp = int((t.label & t.rule_any).sum())
    fp = int((~t.label & t.rule_any).sum())
    fn = int((t.label & ~t.rule_any).sum())
    print(md(t[t.label | t.rule_any | t.magic_cmp_armstrong_or_test.map(bool)]))
    print(f"\n- TP={tp} FP={fp} FN={fn} (luật được dựng sau khi xem chính 3 mẫu dương -> chỉ là fit trong mẫu, chưa phải validation)")
    print(f"- So sánh với số Armstrong/test input: {frac(int(t.magic_cmp_armstrong_or_test.map(bool).sum()), len(t))}")


def c12_prereq(df: pd.DataFrame) -> None:
    header("C-12 EX01: P1 tiên quyết vs total (feedback chỉ dùng cho EDA)")
    g = df[df.exam_id == "EX01"]
    rows = []
    for r in g.itertuples():
        f = r.feedback or ""
        code = strip_comments(r.code).code
        m = re.search(r"General\s*\*\s*prob1\s*\(([^)]*)\)\s*\{(.*?)\n\}", code, re.S)
        sets_n = None
        if m:
            pname = re.findall(r"(\w+)\s*$", m.group(1).split(",")[-1])[0]
            sets_n = bool(re.search(rf"\b{pname}\s*(?:=(?!=)|\+=|\+\+)|\+\+\s*{pname}\b", m.group(2)))
            uses_new_n = bool(re.search(rf"new\s+General\s*\[\s*{pname}\s*\]", m.group(2)))
        else:
            uses_new_n = None
        rows.append({"sample_id": r.sample_id, "total": r.total_score, "compilable": r.compilable,
                     "fb_P1_ok": bool(re.search(r"P1 đúng", f)),
                     "fb_P1_wrong": bool(re.search(r"P1 sai|P1 đọc file sai|P1 lỗi|P1: lỗi|P1 SV hardcode|P1 gán cứng", f)),
                     "fb_tien_quyet": "tiên quyết" in f,
                     "prob1_sets_n": sets_n, "prob1_new_General_n": uses_new_n})
    t = pd.DataFrame(rows)
    print(md(t))
    w = t[t.fb_P1_wrong]
    print(f"\n- Feedback nói P1 sai: {frac(len(w), len(t))}: {dict(zip(w.sample_id, w.total))}; có chữ 'tiên quyết': {frac(int(t.fb_tien_quyet.sum()), len(t))}")
    print(f"- P1 sai mà total>2: {frac(int((w.total > 2).sum()), len(w))}")
    print(f"- total == tổng 6 chiều rubric (cả 2 exam): {int((df.sum_dims == df.total_score).sum())}/{len(df)} -> total EX01 không phải điểm theo câu P1–P4")


def tokens(code: str) -> list[str]:
    return re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|==|!=|<=|>=|\+\+|--|&&|\|\||<<|>>|->|::|\S", code)


def norm_tokens(code: str) -> list[str]:
    out = []
    for t in tokens(code):
        if t.startswith(("\"", "'")):
            out.append("STR")
        elif t[0].isdigit():
            out.append("NUM")
        elif re.match(r"[A-Za-z_]", t) and t not in CPP_KEYWORDS:
            out.append("ID")
        else:
            out.append(t)
    return out


def c13_similarity(df: pd.DataFrame, exams: dict) -> None:
    header("C-13 Tương đồng code (5-gram token chuẩn hoá định danh, bỏ shingle có trong code mẫu của đề)")
    k = 5
    rows = []
    for exam_id, g in df.groupby("exam_id"):
        tmpl_toks = norm_tokens(statement_code_blocks(exams[exam_id]["statement"]))
        tmpl = {tuple(tmpl_toks[i : i + k]) for i in range(len(tmpl_toks) - k + 1)}
        sh, ids = {}, {}
        for r in g.itertuples():
            code = strip_comments(r.code).code
            toks = norm_tokens(code)
            sh[r.sample_id] = {tuple(toks[i : i + k]) for i in range(len(toks) - k + 1)} - tmpl
            ids[r.sample_id] = {t for t in tokens(code) if re.match(r"[A-Za-z_]\w*$", t) and t not in CPP_KEYWORDS}
        for a, b in combinations(sorted(sh), 2):
            inter = len(sh[a] & sh[b])
            rows.append({"exam": exam_id, "a": a, "b": b, "jaccard": inter / max(1, len(sh[a] | sh[b])),
                         "containment": inter / max(1, min(len(sh[a]), len(sh[b]))),
                         "id_jaccard": len(ids[a] & ids[b]) / max(1, len(ids[a] | ids[b])), "template_shingles": len(tmpl)})
    t = pd.DataFrame(rows)
    summ = t.groupby("exam").jaccard.describe(percentiles=[0.5, 0.9])[["count", "50%", "90%", "max"]].reset_index()
    summ["pairs_ge_0.7"] = t.groupby("exam").jaccard.apply(lambda s: int((s >= 0.7).sum())).values
    print(md(summ))
    print("\n**Top 4 cặp theo jaccard mỗi exam**\n")
    print(md(t.sort_values("jaccard", ascending=False).groupby("exam").head(4)))
    bug = re.compile(r"res\s*=\s*res\s*\+\s*\(n\s*%\s*10\)\s*\*\s*dem")
    print(f"- File chứa đúng biểu thức lỗi `res = res + (n % 10) * dem`: {df[df.code.map(lambda c: bool(bug.search(c)))].sample_id.tolist()}")
    print("\n**Top 3 cặp theo containment mỗi exam**\n")
    print(md(t.sort_values("containment", ascending=False).groupby("exam").head(3)))


def c14_c15_profile(df: pd.DataFrame, exams: dict) -> None:
    header("C-14 / C-15 PII (chỉ đếm) + hồ sơ code")
    pats = {
        "redacted_placeholder": r"\[[A-Z_]+_REDACTED\]",
        "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
        "digits_8_10": r"(?<!\d)\d{8,10}(?!\d)",
        "mssv_or_name_kw": r"(?i)\bmssv\b|student\s*id|họ\s*(?:và\s*)?tên|\bho\s*ten\b",
        "user_path": r"(?i)[A-Z]:\\\\|/Users/|/home/",
        "anon_id_token": r"\bS[12]\d{2}\b",
        "injection_kw": r"(?i)ignore (?:previous|all)|bỏ qua|chấm|điểm|\bgrader\b|\bAI\b|10/10",
    }
    rows = []
    for r in df.itertuples():
        st = strip_comments(r.code)
        comments = "\n".join(c for _, c in st.comments)
        strings = " ".join(re.findall(r"\"[^\"\n]*\"", st.code))
        rec = {"sample_id": r.sample_id}
        for k, p in pats.items():
            rec[k] = len(re.findall(p, r.code))
            rec[k + "_in_comment_or_string"] = len(re.findall(p, comments + "\n" + strings))
        rows.append(rec)
    t = pd.DataFrame(rows)
    cols = [c for c in t.columns if c != "sample_id"]
    print(md(pd.DataFrame([{"pattern": c, "files": int((t[c] > 0).sum()), "samples": t[t[c] > 0].sample_id.tolist()} for c in cols])))

    vi = re.compile(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.I)
    prof = []
    for exam_id, g in df.groupby("exam_id"):
        com = g.code.map(lambda c: strip_comments(c).comments)
        prose = com.map(lambda cs: sum(1 for _, c in cs if c.strip() and not looks_like_code(c)))
        prof.append({
            "exam": exam_id, "n": len(g), "chars_median": g.code_chars.median(), "tok_est_median": round(g.code_chars.median() / 3.5),
            "tok_est_max": round(g.code_chars.max() / 3.5), "statement_tok_est": round(len(exams[exam_id]["statement"]) / 3.5),
            "bits_stdc++": int(g.code.str.contains("bits/stdc++").sum()), "conio": int(g.code.str.contains("conio").sum()),
            "system_pause": int(g.code.str.contains(r"system\s*\(\s*\"pause", regex=True).sum()),
            "files_comment_vi_diacritic": int(com.map(lambda cs: any(vi.search(c) for _, c in cs)).sum()),
            "files_with_prose_comment": int((prose > 0).sum()),
            "bom": g.code.map(lambda c: c.startswith("\ufeff")).pipe(lambda s: g[s].sample_id.tolist()),
        })
    print()
    print(md(pd.DataFrame(prof)))
    cc = []
    for r in df.itertuples():
        st = strip_comments(r.code)
        lines = r.code.splitlines()
        n_cc = sum(1 for i in st.comment_line_nos if i <= len(lines) and looks_like_code(re.sub(r"^\s*(//|/\*|\*)", "", lines[i - 1])))
        cc.append((r.sample_id, n_cc, len(st.comment_line_nos)))
    cc.sort(key=lambda x: -x[1])
    print(f"\n- dòng comment trông như code (top 5): {cc[:5]}")
    ex01 = df[df.exam_id == "EX01"]
    print(f"- EX01 có comment template 'AD or BC' chép từ đề: {frac(int(ex01.code.str.contains('AD or BC').sum()), len(ex01))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--cxx", default="clang++")
    ap.add_argument("--gnu-image", default=None, help="local Docker image that already contains GNU g++")
    ap.add_argument("--no-run", action="store_true")
    args = ap.parse_args()

    df = load_samples(args.data_root)
    _, exams, _ = load_meta(args.data_root)
    readme = (args.data_root / "README.md").read_text(encoding="utf-8")
    tmp = tempfile.TemporaryDirectory(dir=args.workdir) if args.workdir else tempfile.TemporaryDirectory()
    work = Path(tmp.name)
    try:
        version = run([args.cxx, "--version"], work).stdout.splitlines()[0]
        print(f"# Verify code_signals (n={len(df)}; compiler cục bộ: {version})")
        c01_s214(df, args.cxx, work, not args.no_run)
        comp = compile_signals(df, args.cxx, work)
        c02_c10_compile(df, comp, readme)
        c03_c04_ex01(df, exams, readme, comp, args.cxx, work)
        bins, runs, gnu = {}, None, None
        if not args.no_run:
            bins = build_ex02(df, args.cxx, work)
            runs = c05_rerun(df, bins)
            if args.gnu_image:
                gnu = gnu_docker(df, args.gnu_image, work, ARMSTRONG_EXTRA + NON_ARMSTRONG_EXTRA)
                c05_gnu_compare(df, gnu)
        c06_c07_c08_tests(df, exams, runs)
        if not args.no_run:
            c05_extended(df, bins, gnu)
        c09_logs(df, comp)
        c11_hardcode(df, exams)
        c12_prereq(df)
        c13_similarity(df, exams)
        c14_c15_profile(df, exams)
    finally:
        tmp.cleanup()


if __name__ == "__main__":
    main()
