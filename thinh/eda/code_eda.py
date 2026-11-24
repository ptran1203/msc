"""EDA of student source code and the auxiliary signals (compile_log, test_report).

Usage:
    python code_eda.py --data-root <dataset dir> [--cxx clang++] [--no-compile] [--no-run]

Prints markdown tables to stdout; writes per-sample features (CSV) and up to two
figures to <eda>/out/code/. Works on the 32-sample set and on the full release.
Student code is untrusted: see code_toolchain.py for the run sandbox.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from code_cpp_text import (
    CPP_KEYWORDS,
    STD_NAMES,
    VI_DIACRITIC_RE,
    extract_functions,
    jaccard,
    looks_like_code,
    normalize_tokens,
    normalize_type,
    param_types,
    parse_prototype,
    shingles,
    strip_comments,
    tokenize,
)
from code_toolchain import (
    TIMEOUT_MARK,
    build_harness,
    compile_source,
    link_with_harness,
    compiler_version,
    is_safe_to_run,
    run_binary,
    summarize_log,
)
from load_data import DIMS, load_meta, load_samples

EDA_DIR = Path(__file__).resolve().parent
CHARS_PER_TOKEN = 3.5
TRAP_ON_MISSING_RETURN = ["-fsanitize=return", "-fsanitize-trap=return"]
ARMSTRONG_NUMBERS = {153, 370, 371, 407, 1634, 8208, 9474, 54748, 92727, 93084, 548834, 1741725, 4210818, 9800817, 9926315, 24678050, 24678051, 88593477, 146511208, 472335975, 534494836, 912985153}
INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", re.M)
MAIN_RE = re.compile(r"\b(int|void)\s+main\s*\(")
MSVC_PATTERNS: dict[str, re.Pattern[str]] = {
    "void_main": re.compile(r"\bvoid\s+main\s*\("),
    "stdafx_h": re.compile(r"#\s*include\s*[<\"](?:stdafx|pch)\.h[>\"]"),
    "conio_h": re.compile(r"#\s*include\s*[<\"]conio\.h[>\"]"),
    "system_pause": re.compile(r"system\s*\(\s*\"pause\"\s*\)"),
    "getch": re.compile(r"\b_?getch\s*\("),
    "secure_crt": re.compile(r"_CRT_SECURE_NO_WARNINGS|\b(?:scanf_s|printf_s|fopen_s|strcpy_s|gets_s)\b"),
    "pragma_vs": re.compile(r"#\s*pragma\s+(?:once|warning)"),
    "vs_template_comment": re.compile(r"This file contains the 'main' function"),
    "windows_path_literal": re.compile(r"\"[A-Za-z]:\\"),
    "int64_ms": re.compile(r"\b__int64\b"),
    "windows_h": re.compile(r"#\s*include\s*[<\"]windows\.h[>\"]"),
}
POW_CALL_RE = re.compile(r"(?<![\w.])(?:std::)?pow\s*\(")
OPEN_LITERAL_RE = re.compile(r"(?:\.open\s*\(|\b[io]?fstream\s+\w+\s*[({]|\bfopen\s*\()\s*\"([^\"]*)\"")
FIXED_ALLOC_RE = re.compile(r"\bnew\s+\w+\s*\[\s*(\d+)\s*\]")
MAGIC_COMPARE_RE = re.compile(r"(?:[=!]=\s*(\d{2,})\b|\b(\d{2,})\s*[=!]=|\bcase\s+(\d{2,})\s*:)")
INCLUDE_DATA_RE = re.compile(r"#\s*include\s*\"[^\"]+\.(?:csv|txt|dat)\"", re.I)
ALLOWED_LIBS_RE = re.compile(r"Chỉ được dùng các thư viện:\s*`([^`]+)`")
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "redacted_name": re.compile(r"\[NAME_REDACTED\]"),
    "redacted_class": re.compile(r"\[CLASS_REDACTED\]"),
    "redacted_other": re.compile(r"\[(?!NAME_|CLASS_)[A-Z_]+_REDACTED\]"),
    "digits_8_10": re.compile(r"(?<!\d)\d{8,10}(?!\d)"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "mssv_kw": re.compile(r"\bmssv\b|student\s*id|mã\s*số\s*sinh\s*viên", re.I),
    "name_kw": re.compile(r"họ\s*(?:và\s*)?tên|\bho\s*ten\b|\bfull\s*name\b", re.I),
    "class_kw": re.compile(r"\blớp\s*[:\-]|\bclass\s*:", re.I),
    "user_path": re.compile(r"[A-Za-z]:\\\\?Users\\\\?|/Users/|/home/", re.I),
    "anon_id_token": re.compile(r"\bS[12]\d{2}\b"),
}
INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "cham": re.compile(r"chấm|\bcham\b", re.I),
    "diem": re.compile(r"điểm|\bdiem\b", re.I),
    "ignore": re.compile(r"\bignore\b|bỏ qua|\bbo qua\b", re.I),
    "grader_ai": re.compile(r"\bgrader\b|\bllm\b|\bchatgpt\b|\bassistant\b|\bAI\b|giảng viên|\bgiang vien\b", re.I),
    "score_words": re.compile(r"\b10/10\b|full\s*(?:điểm|score|marks)|\bscore\b", re.I),
}


# ------------------------------------------------------------------------------ utils


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    def fmt(v: object) -> str:
        if v is pd.NA or v is None:
            return ""
        if isinstance(v, float):
            return "" if np.isnan(v) else floatfmt.format(v)
        if isinstance(v, (list, tuple, set)):
            return ", ".join(map(str, v))
        return str(v).replace("|", "\\|").replace("\n", " ")

    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def section(title: str) -> None:
    print(f"\n## {title}\n")


def rate(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.0%})" if n else f"{k}/0"


# -------------------------------------------------------------------- static features


def allowed_libraries(exam: dict) -> set[str] | None:
    m = ALLOWED_LIBS_RE.search(exam.get("statement", ""))
    return {x.strip() for x in m.group(1).split(",")} if m else None


def static_features(code: str, exam: dict) -> dict:
    raw = code
    stripped = strip_comments(raw)
    funcs = extract_functions(stripped)
    func_names = {f.name for f in funcs}
    lines = raw.splitlines()
    nonblank = [i + 1 for i, line in enumerate(lines) if line.strip()]
    code_lines = [i + 1 for i, line in enumerate(stripped.code.splitlines()) if line.strip()]
    comment_lines = stripped.comment_lines & set(nonblank)
    commented_code = {
        ln + k
        for ln, text in stripped.comments
        for k, part in enumerate(text.split("\n"))
        if looks_like_code(part)
    }
    includes = INCLUDE_RE.findall(stripped.code)
    allowed = allowed_libraries(exam)
    pow_defs = sum(1 for f in funcs if f.name == "pow")
    pow_calls = len(POW_CALL_RE.findall(stripped.code)) - pow_defs
    comment_text = [t for _, t in stripped.comments]
    feats: dict = {
        "lines": len(lines),
        "chars": len(raw),
        "tokens_est": round(len(raw) / CHARS_PER_TOKEN),
        "nonblank_lines": len(nonblank),
        "effective_code_lines": len(code_lines),
        "comment_lines": len(comment_lines),
        "comment_ratio": len(comment_lines) / len(nonblank) if nonblank else 0.0,
        "commented_code_lines": len(commented_code & comment_lines),
        "n_functions": len(funcs),
        "has_main": bool(MAIN_RE.search(stripped.code)),
        "void_main": bool(MSVC_PATTERNS["void_main"].search(stripped.code)),
        "includes": includes,
        "bits_stdcpp": "bits/stdc++.h" in includes,
        "using_ns_std": bool(re.search(r"using\s+namespace\s+std\s*;", stripped.code)),
        "pow_calls": max(pow_calls, 0),
        "user_defines_pow": pow_defs > 0,
        "user_power_func": sorted(n for n in func_names if re.search(r"pow|luythua|luythuy|^mu$|tinhtich", n, re.I) and n != "pow"),
        "has_cmath": any(h in {"cmath", "math.h"} for h in includes),
        "vi_diacritic_comments": sum(bool(VI_DIACRITIC_RE.search(t)) for t in comment_text),
        "vi_diacritic_strings": len([s for s in re.findall(r"\"([^\"\n]*)\"", stripped.code) if VI_DIACRITIC_RE.search(s)]),
        "disallowed_includes": sorted(set(includes) - allowed) if allowed is not None else [],
        "bom": raw.startswith("\ufeff"),
        "crlf": "\r\n" in raw,
        "tab_indent_lines": sum(1 for line in lines if line.startswith("\t")),
        "non_ascii_chars": sum(1 for ch in raw if ord(ch) > 127),
    }
    for name, pattern in MSVC_PATTERNS.items():
        target = raw if name == "vs_template_comment" else stripped.code
        feats[f"msvc_{name}"] = len(pattern.findall(target))
    feats["msvc_any"] = sum(feats[f"msvc_{n}"] for n in MSVC_PATTERNS) > 0
    return feats


def prototype_check(code: str, exam: dict) -> dict[str, str]:
    """Per problem: match | mismatch:<found> | declared_only | missing."""
    stripped = strip_comments(code)
    funcs = {f.name: f for f in extract_functions(stripped)}
    result: dict[str, str] = {}
    for prob in exam.get("problems", []):
        proto = prob.get("prototype")
        if not proto:
            continue
        ret, name, ptypes = parse_prototype(proto)
        f = funcs.get(name)
        if f is None:
            declared = re.search(rf"\b{name}\s*\([^)]*\)\s*;", stripped.code)
            result[prob["pid"]] = "declared_only" if declared else "missing"
            continue
        found = (normalize_type(f.ret), param_types(f.params))
        result[prob["pid"]] = "match" if found == (ret, ptypes) else f"mismatch:{found[0]} {name}({', '.join(found[1])})"
    return result


def prob1_sets_count(code: str, exam: dict) -> bool | None:
    """Does the file-reading function write its int& count parameter? (None if not applicable)."""
    protos = [p.get("prototype", "") for p in exam.get("problems", [])]
    if not protos or "int&" not in normalize_type(protos[0]):
        return None
    funcs = {f.name: f for f in extract_functions(strip_comments(code))}
    name = parse_prototype(protos[0])[1]
    f = funcs.get(name)
    if f is None:
        return None
    pname = re.findall(r"(\w+)\s*$", f.params.split(",")[-1])
    if not pname:
        return None
    p = re.escape(pname[0])
    return bool(re.search(rf"\b{p}\s*(?:=(?!=)|\+=|-=|\+\+|--)|(?:\+\+|--)\s*{p}\b", f.body))


def hardcode_flags(code: str, exam: dict, public_inputs: set[int]) -> dict[str, int]:
    stripped = strip_comments(code)
    funcs = [f for f in extract_functions(stripped) if f.name != "main"]
    multi = exam.get("exam_type") == "multi_problem"
    prob_names = {parse_prototype(p["prototype"])[1] for p in exam.get("problems", []) if p.get("prototype")}
    magic = public_inputs | ARMSTRONG_NUMBERS
    flags = {
        "hc_include_data_file": len(INCLUDE_DATA_RE.findall(stripped.code)),
        "hc_literal_path_in_func": sum(len(OPEN_LITERAL_RE.findall(f.body)) for f in funcs),
        "hc_fixed_size_alloc": sum(1 for f in funcs for n in FIXED_ALLOC_RE.findall(f.body) if int(n) >= 10),
        "hc_stdin_in_file_task": sum(1 for f in funcs if multi and f.name in prob_names and re.search(r"\bcin\b", f.body)),
        "hc_magic_compare": 0,
    }
    for f in [*funcs, *[g for g in extract_functions(stripped) if g.name == "main"]]:
        for m in MAGIC_COMPARE_RE.finditer(f.body):
            value = int(next(g for g in m.groups() if g))
            flags["hc_magic_compare"] += value in magic
    return flags


# ------------------------------------------------------------------------ test report


def classify_failure(expected: str, actual: str) -> str:
    exp, act = expected.strip(), actual.strip()
    if act == TIMEOUT_MARK:
        return "timeout"
    if act == "":
        return "empty_output"
    if act.casefold() == exp.casefold():
        return "case_only"
    if re.sub(r"\s+", "", act) == re.sub(r"\s+", "", exp):
        return "whitespace_only"
    tail = act.rsplit(":", 1)[-1].strip() if ":" in act else act.split()[-1]
    if tail != act:
        return "prompt_prefix_only" if tail.casefold() == exp.casefold() else "prompt_prefix+wrong"
    return "wrong_answer"


def lenient_equal(expected: str, actual: str) -> bool:
    act = actual.strip()
    if ":" in act:
        act = act.rsplit(":", 1)[-1]
    return act.strip().casefold() == expected.strip().casefold()


def public_tests_by_exam(df: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    tests: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in df.itertuples():
        if isinstance(row.test_report, list):
            for t in row.test_report:
                tests[row.exam_id][t["input"]].add(t["expected"])
    return tests


# --------------------------------------------------------------------------- sections


def report_code_stats(df: pd.DataFrame, feats: pd.DataFrame, exams: dict) -> None:
    section("1. Thống kê mã nguồn theo exam")
    agg_rows = []
    for exam_id, g in feats.groupby("exam_id"):
        n = len(g)
        agg_rows.append(
            {
                "exam": exam_id,
                "n": n,
                "lines med [min-max]": f"{g.lines.median():.0f} [{g.lines.min()}-{g.lines.max()}]",
                "chars med [min-max]": f"{g.chars.median():.0f} [{g.chars.min()}-{g.chars.max()}]",
                "tokens_est med/max": f"{g.tokens_est.median():.0f}/{g.tokens_est.max()}",
                "has_main": rate(int(g.has_main.sum()), n),
                "void_main": rate(int(g.void_main.sum()), n),
                "using ns std": rate(int(g.using_ns_std.sum()), n),
                "bits/stdc++": rate(int(g.bits_stdcpp.sum()), n),
                "conio.h": rate(int((g.msvc_conio_h > 0).sum()), n),
                "stdafx.h": rate(int((g.msvc_stdafx_h > 0).sum()), n),
                "system(pause)": rate(int((g.msvc_system_pause > 0).sum()), n),
                "pow( call": rate(int((g.pow_calls > 0).sum()), n),
                "user fn named pow": rate(int(g.user_defines_pow.sum()), n),
                "has cmath/math.h": rate(int(g.has_cmath.sum()), n),
                "VN-diacritic comment": rate(int((g.vi_diacritic_comments > 0).sum()), n),
                "comment ratio mean": f"{g.comment_ratio.mean():.2f}",
                "any MSVC trace": rate(int(g.msvc_any.sum()), n),
                "BOM": rate(int(g.bom.sum()), n),
                "CRLF": rate(int(g.crlf.sum()), n),
            }
        )
    print(md_table(pd.DataFrame(agg_rows).set_index("exam").T.reset_index().rename(columns={"index": "metric"})))
    for exam_id, exam in exams.items():
        st = exam.get("statement", "")
        print(f"- Statement {exam_id}: {len(st)} ký tự ≈ {len(st) / CHARS_PER_TOKEN:.0f} token (chars/3.5; tiếng Việt có dấu thường tốn token hơn)")

    print("\n**Đếm #include (số file có header, theo exam)**\n")
    inc_rows = []
    for exam_id, g in feats.groupby("exam_id"):
        c = Counter(h for hs in g.includes for h in set(hs))
        for h, k in c.most_common():
            inc_rows.append({"exam": exam_id, "header": h, "files": rate(k, len(g))})
    print(md_table(pd.DataFrame(inc_rows)))

    print("\n**Header ngoài danh sách cho phép trong statement (chỉ áp dụng nếu statement có danh sách)**\n")
    bad = feats[feats.disallowed_includes.map(len) > 0][["sample_id", "disallowed_includes", "total_score"]]
    print(md_table(bad) if len(bad) else "_không có_")

    print("\n**MSVC/Visual Studio traces (sample có ≥1 dấu vết)**\n")
    msvc_cols = [c for c in feats.columns if c.startswith("msvc_") and c != "msvc_any"]
    tr = feats[feats.msvc_any][["sample_id", *msvc_cols]]
    tr = tr.loc[:, (tr != 0).any(axis=0)]
    print(md_table(tr) if len(tr) else "_không có_")

    print("\n**Code bị comment-out / code hiệu dụng (top theo commented_code_lines)**\n")
    cc = feats.sort_values("commented_code_lines", ascending=False).head(6)[
        ["sample_id", "nonblank_lines", "effective_code_lines", "comment_lines", "commented_code_lines", "total_score"]
    ]
    print(md_table(cc))

    print("\n**Tương quan Spearman: độ dài code vs total_score (theo exam)**\n")
    rows = []
    for exam_id, g in feats.groupby("exam_id"):
        for col in ["lines", "effective_code_lines", "n_functions"]:
            rho, p = spearmanr(g[col], g.total_score)
            rows.append({"exam": exam_id, "feature": col, "n": len(g), "rho": rho, "p": p})
    print(md_table(pd.DataFrame(rows)))


def report_prototypes(df: pd.DataFrame, exams: dict) -> pd.DataFrame:
    section("1b. Prototype hàm (đề multi_problem) so với exams.json")
    rows = []
    for row in df.itertuples():
        exam = exams.get(row.exam_id, {})
        if not any(p.get("prototype") for p in exam.get("problems", [])):
            continue
        res = prototype_check(row.code, exam)
        rows.append(
            {
                "sample_id": row.sample_id,
                **res,
                "p1_sets_count_param": prob1_sets_count(row.code, exam),
                "has_main": bool(MAIN_RE.search(strip_comments(row.code).code)),
                # EDA-only cross-check; feedback must never be a model input for Task 1/2.
                "fb_says_prereq_failed": "tiên quyết" in (row.feedback or "").casefold(),
                "total": row.total_score,
                "logic": row.logic,
            }
        )
    proto = pd.DataFrame(rows)
    if proto.empty:
        print("_không có đề có prototype_")
        return proto
    for exam_id, exam in exams.items():
        stated = re.findall(r"\*\*Problem (\d+) \((\d+(?:\.\d+)?) điểm\)", exam.get("statement", ""))
        meta = [p.get("max_score") for p in exam.get("problems", []) if p.get("prototype")]
        if stated:
            print(f"- {exam_id}: điểm từng câu trong statement {[float(s) for _, s in stated]} vs exams.json max_score {meta}\n")
    print(md_table(proto))
    pre = proto[proto.fb_says_prereq_failed]
    print(f"\n- Feedback nói P1 (tiên quyết) sai: {rate(len(pre), len(proto))}; total của các bài này: {dict(zip(pre.sample_id, pre.total))}")
    pids = [c for c in proto.columns if c.startswith("P")]
    summary = {pid: dict(Counter(proto[pid].map(lambda s: s.split(":")[0]))) for pid in pids}
    print("\n**Tóm tắt trạng thái prototype**\n")
    print(md_table(pd.DataFrame(summary).fillna(0).astype(int).reset_index().rename(columns={"index": "status"})))
    complete = proto[pids].apply(lambda r: all(v == "match" for v in r), axis=1)
    print(f"\nBài có đủ cả {len(pids)} hàm đúng prototype: {rate(int(complete.sum()), len(proto))}")
    return proto


def report_compile(df: pd.DataFrame, cxx: str, do_compile: bool, workroot: Path) -> pd.DataFrame:
    section("2. compile_log: phân loại + biên dịch lại cục bộ")
    rows = []
    for row in df.itertuples():
        s = summarize_log(row.compile_log)
        rows.append(
            {
                "sample_id": row.sample_id,
                "exam_id": row.exam_id,
                "compilable": row.compilable,
                "total": row.total_score,
                "log_chars": len(row.compile_log or ""),
                "log_family": s.compiler_family,
                "log_err": s.n_error + s.n_fatal,
                "log_warn": s.n_warning,
                "log_link_err": s.linker_error,
                "log_has_error": s.has_error,
                "loader_log_has_error": row.log_has_error,
                "log_truncated": s.truncated,
                "log_first_error": s.first_error_family,
                "log_families": sorted(set(s.families)),
                "excerpt_match": excerpt_consistency(row.compile_log or "", row.code),
            }
        )
    comp = pd.DataFrame(rows)
    print("**compile_log theo sample (chỉ log khác rỗng)**\n")
    print(md_table(comp[comp.log_chars > 0].drop(columns=["exam_id"])))
    n = len(comp)
    print(
        f"\n- Log rỗng: {rate(int((comp.log_chars == 0).sum()), n)}; log có error (gồm link): {rate(int(comp.log_has_error.sum()), n)}; "
        f"chỉ warning: {rate(int(((comp.log_warn > 0) & ~comp.log_has_error).sum()), n)}; bị cắt ở 2000 ký tự: {rate(int(comp.log_truncated.sum()), n)}"
    )
    print(f"- Họ compiler sinh log: {dict(Counter(comp.log_family))}")
    ex = comp.excerpt_match[comp.excerpt_match != ""].map(lambda s: tuple(map(int, s.split("/"))))
    print(f"- Trích dòng nguồn trong log khớp dòng tương ứng của file .cpp: {rate(sum(a for a, _ in ex), sum(b for _, b in ex))} (kiểm tra file .cpp đúng phiên bản đã biên dịch)")
    miss = comp[comp.log_has_error != comp.loader_log_has_error]
    print(f"- loader.log_has_error bỏ sót lỗi link: {list(miss.sample_id)}")

    if do_compile:
        print(f"\n**Biên dịch lại cục bộ bằng `{cxx}` ({compiler_version(cxx)})** — multi_problem: `-std=c++11 -fsyntax-only`; single_problem: `-std=c++11 -O0` + link\n")
        local = []
        for row in df.itertuples():
            wd = workroot / f"compile_{row.sample_id}"
            wd.mkdir(parents=True, exist_ok=True)
            res = compile_source(Path(row.abs_path), cxx, row.exam_type == "multi_problem", wd)
            s = summarize_log(res.stderr)
            local.append(
                {
                    "sample_id": row.sample_id,
                    "local_rc": res.returncode,
                    "local_has_error": s.has_error or res.returncode != 0,
                    "local_err": s.n_error + s.n_fatal + int(s.linker_error),
                    "local_warn": s.n_warning,
                    "local_first_error": s.first_error_family,
                    "local_warn_families": sorted({f.split(":", 1)[1] for f in s.families if f.startswith("warning")}),
                    "binary": str(res.binary) if res.binary else "",
                }
            )
        comp = comp.merge(pd.DataFrame(local), on="sample_id")
        cross = comp.groupby(["compilable", "log_has_error", "local_has_error"]).size().reset_index(name="n")
        print(md_table(cross))
        agree = int((comp.log_has_error == comp.local_has_error).sum())
        print(f"\n- Đồng ý error/không error giữa log gốc và {cxx}: {rate(agree, n)}")
        incons = comp[(comp.compilable == 1) & (comp.log_has_error | comp.local_has_error) | (comp.compilable == 0) & ~comp.log_has_error & ~comp.local_has_error]
        print("\n**Mâu thuẫn nhãn compilable vs tín hiệu biên dịch**\n")
        print(md_table(incons[["sample_id", "compilable", "total", "log_first_error", "local_first_error"]]) if len(incons) else "_không có_")
        extra = comp[comp.local_warn_families.map(lambda xs: any(x not in {"missing_return_warning", "multichar_warning"} for x in xs))]
        print("\n**Cảnh báo chỉ clang nêu (không có trong log gốc)**\n")
        print(md_table(extra[["sample_id", "local_warn_families", "log_warn", "total"]]) if len(extra) else "_không có_")
    return comp


def report_harness(df: pd.DataFrame, exams: dict, cxx: str, workroot: Path) -> None:
    """For function-only exams: does the file link (a) alone as the statement says, (b) with a harness main?"""
    section("2b. Đề chỉ nộp hàm: mô phỏng link (chỉ biên dịch + link, KHÔNG chạy)")
    rows = []
    for exam_id, g in df.groupby("exam_id"):
        exam = exams.get(exam_id, {})
        protos = [p["prototype"] for p in exam.get("problems", []) if p.get("prototype")]
        if not protos:
            continue
        harness = {
            "harness_strict": build_harness(statement_code(exam), protos, weak_stubs=False),
            "harness_weak_stubs": build_harness(statement_code(exam), protos, weak_stubs=True),
        }
        for row in g.itertuples():
            rec: dict = {"sample_id": row.sample_id, "compilable": row.compilable, "total": row.total_score}
            wd = workroot / f"link_{row.sample_id}"
            wd.mkdir(parents=True, exist_ok=True)
            alone = compile_source(Path(row.abs_path), cxx, False, wd)
            rec["statement_cmd_link_alone"] = "ok" if alone.returncode == 0 else "FAIL:" + summarize_log(alone.stderr).first_error_family
            for mode, src in harness.items():
                mwd = workroot / f"{mode}_{row.sample_id}"
                mwd.mkdir(parents=True, exist_ok=True)
                ok, why = link_with_harness(Path(row.abs_path), src, cxx, mwd)
                rec[mode] = "ok" if ok else f"FAIL:{why}"
            rows.append(rec)
    if not rows:
        print("_không có đề function-only_")
        return
    hr = pd.DataFrame(rows)
    print(md_table(hr))
    n = len(hr)
    for mode in ["statement_cmd_link_alone", "harness_strict", "harness_weak_stubs"]:
        ok = hr[mode] == "ok"
        print(f"- {mode}: link ok {rate(int(ok.sum()), n)}; FAIL nhưng compilable=1: {list(hr[~ok & (hr.compilable == 1)].sample_id)}")


def excerpt_consistency(log: str, code: str) -> str:
    """Do GCC source excerpts ('  90 |   return;') match the shipped .cpp line? -> 'k/m'."""
    lines = code.splitlines()
    hits = total = 0
    in_student = False
    for line in log.splitlines():
        diag = re.match(r"^(\S+?):\d+:\d+: ", line)
        if diag:
            in_student = diag.group(1).endswith("student.cpp")
            continue
        m = re.match(r"^\s*(\d+) \| (.*)$", line)
        if m and in_student:
            ln = int(m.group(1))
            total += 1
            hits += 0 < ln <= len(lines) and "".join(lines[ln - 1].split()) == "".join(m.group(2).split())
    return f"{hits}/{total}" if total else ""


def report_tests(df: pd.DataFrame, comp: pd.DataFrame, cxx: str, do_run: bool, timeout_s: float, workroot: Path) -> pd.DataFrame:
    section("3. test_report: phân loại test fail + chạy lại")
    tests = public_tests_by_exam(df)
    for exam_id, mapping in tests.items():
        exp_counts = Counter(next(iter(v)) for v in mapping.values())
        conflicting = [k for k, v in mapping.items() if len(v) > 1]
        max_digits = max(len(k.strip()) for k in mapping)
        always = {e: sum(1 for v in mapping.values() if e in v) for e in exp_counts}
        print(
            f"- {exam_id}: {len(mapping)} input công khai khác nhau; phân bố expected {dict(exp_counts)}; input mâu thuẫn expected: {conflicting}; "
            f"số chữ số lớn nhất của input: {max_digits}; baseline in hằng: {always}"
        )
    tr_rows = df[df.test_report.map(lambda r: isinstance(r, list))]
    sets = tr_rows.test_report.map(lambda r: tuple(t["input"] for t in r))
    print(f"- Bài có test_report rỗng `[]`: {list(tr_rows[sets.map(len) == 0].sample_id)} (compilable={list(tr_rows[sets.map(len) == 0].compilable)})")
    print(f"- Các danh sách input khác nhau (bỏ rỗng): {sets[sets.map(len) > 0].nunique()}")

    fail_rows, per_sample = [], []
    for row in tr_rows.itertuples():
        report = row.test_report
        cats = Counter(classify_failure(t["expected"], t["actual"]) for t in report if not t["passed"])
        for t in report:
            if not t["passed"]:
                fail_rows.append({"sample_id": row.sample_id, "category": classify_failure(t["expected"], t["actual"])})
        lenient = sum(lenient_equal(t["expected"], t["actual"]) for t in report)
        per_sample.append(
            {
                "sample_id": row.sample_id,
                "n_tests": len(report),
                "passed": sum(bool(t["passed"]) for t in report),
                "lenient_pass": lenient,
                "fail_categories": dict(cats),
                "total": row.total_score,
                "io_format": row.io_format,
                "logic": row.logic,
                "edge_case": row.edge_case,
            }
        )
    ps = pd.DataFrame(per_sample)
    if ps.empty:
        print("_không có test_report_")
        return ps
    print("\n**Theo sample**\n")
    print(md_table(ps))
    fr = pd.DataFrame(fail_rows)
    if not fr.empty:
        print("\n**Phân loại tất cả test fail**\n")
        cat = fr.groupby("category").agg(n_tests=("sample_id", "size"), samples=("sample_id", lambda s: sorted(set(s)))).reset_index()
        cat["share"] = cat.n_tests / cat.n_tests.sum()
        print(md_table(cat))
    ran = ps[ps.n_tests > 0]
    if len(ran) >= 3:
        print("\n**Spearman (bài có test, n=%d)**\n" % len(ran))
        rows = []
        for metric in ["passed", "lenient_pass"]:
            for target in ["total", "logic", "io_format"]:
                rho, p = spearmanr(ran[metric], ran[target])
                rows.append({"x": metric, "y": target, "rho": rho, "p": p})
        print(md_table(pd.DataFrame(rows)))
    zero_hi = ps[(ps.n_tests > 0) & (ps.passed == 0) & (ps.total >= 7)]
    print(f"\n- 0 test pass nhưng total ≥7: {rate(len(zero_hi), int((ps.n_tests > 0).sum()))} → {list(zero_hi.sample_id)}")
    full_low = ps[(ps.n_tests > 0) & (ps.passed == ps.n_tests) & (ps.total < 10)]
    print(f"- pass hết nhưng total <10: {list(full_low.sample_id)} (io_format={list(full_low.io_format)})")

    if do_run:
        print(f"\n**Chạy lại binary ({cxx}) trên input công khai; timeout {timeout_s}s; chỉ bài qua deny-list**\n")
        rerun_rows = []
        for row in tr_rows.itertuples():
            safe, hits = is_safe_to_run(row.code)
            info = comp.set_index("sample_id").loc[row.sample_id]
            if not safe:
                rerun_rows.append({"sample_id": row.sample_id, "status": f"skipped_unsafe:{hits}"})
                continue
            if not info.get("binary"):
                rerun_rows.append({"sample_id": row.sample_id, "status": "no_binary"})
                continue
            report = row.test_report or [{"input": k, "expected": next(iter(v)), "actual": None} for k, v in tests[row.exam_id].items()]
            variants = {"O0": Path(info["binary"])}
            trap_dir = workroot / f"trap_{row.sample_id}"
            trap_dir.mkdir(exist_ok=True)
            # Emulates GCC>=13 at -O0 (-funreachable-traps): flowing off a non-void function traps.
            trap = compile_source(Path(row.abs_path), cxx, False, trap_dir, extra_flags=TRAP_ON_MISSING_RETURN)
            if trap.binary:
                variants["O0+trap_on_missing_return"] = trap.binary
            for vname, binary in variants.items():
                outs = [run_binary(binary, t["input"] + "\n", timeout_s) for t in report]
                same = sum(o.as_report_actual == (t["actual"] or "").strip() for o, t in zip(outs, report) if t["actual"] is not None)
                crashes = sum(1 for o in outs if o.returncode is not None and o.returncode < 0)
                rerun_rows.append(
                    {
                        "sample_id": row.sample_id,
                        "status": vname,
                        "reproduced": f"{same}/{len(report)}",
                        "local_actual_first3": [o.as_report_actual or f"''(rc={o.returncode})" for o in outs[:3]],
                        "report_actual_first3": [t["actual"] for t in report[:3]],
                        "crashes": crashes,
                    }
                )
        rr = pd.DataFrame(rerun_rows)
        if "crashes" in rr:
            rr["crashes"] = rr["crashes"].astype("Int64")
        print(md_table(rr))
        for v in ["O0", "O0+trap_on_missing_return"]:
            sub = rr[rr.status == v]
            if len(sub):
                k = sub.reproduced.map(lambda s: int(s.split("/")[0])).sum()
                m = sub.reproduced.map(lambda s: int(s.split("/")[1])).sum()
                full = int((sub.reproduced.map(lambda s: s.split("/")[0] == s.split("/")[1])).sum())
                print(f"- {v}: tái lập {rate(int(k), int(m))} kết quả test; {rate(full, len(sub))} bài khớp 100%")
    return ps


def report_hardcode(df: pd.DataFrame, exams: dict) -> pd.DataFrame:
    section("4. Hard-code: phát hiện tĩnh vs nhãn 'Lỗi hard-code'")
    tests = public_tests_by_exam(df)
    rows = []
    for row in df.itertuples():
        pub = {int(k) for k in tests.get(row.exam_id, {}) if k.strip().isdigit() and int(k) >= 10}
        flags = hardcode_flags(row.code, exams.get(row.exam_id, {}), pub)
        label = "Lỗi hard-code" in (row.labels or [])
        rows.append({"sample_id": row.sample_id, **flags, "pred": sum(flags.values()) > 0, "label_hardcode": label, "total": row.total_score})
    hc = pd.DataFrame(rows)
    flagged = hc[hc.pred | hc.label_hardcode]
    print(md_table(flagged) if len(flagged) else "_không có_")
    tp = int((hc.pred & hc.label_hardcode).sum())
    fp = int((hc.pred & ~hc.label_hardcode).sum())
    fn = int((~hc.pred & hc.label_hardcode).sum())
    print(f"\n- Rule tĩnh vs nhãn: TP={tp}, FP={fp}, FN={fn}, nhãn dương={int(hc.label_hardcode.sum())}/{len(hc)}")
    print(f"- So sánh với literal test công khai/số Armstrong (`n == 153`…): {int((hc.hc_magic_compare > 0).sum())}/{len(hc)} bài")
    return hc


def statement_code(exam: dict) -> str:
    blocks = re.findall(r"```(?:cpp|c\+\+)?\n(.*?)```", exam.get("statement", ""), re.S)
    return strip_comments("\n".join(blocks)).code


def statement_template_shingles(exam: dict, k: int) -> set:
    return shingles(normalize_tokens(tokenize(statement_code(exam))), k)


def statement_template_identifiers(exam: dict) -> set[str]:
    return {text.lower() for kind, text in tokenize(statement_code(exam)) if kind == "id"}


def report_similarity(df: pd.DataFrame, exams: dict, k: int, threshold: float, out_dir: Path) -> pd.DataFrame:
    section(f"5. Trùng lặp: Jaccard {k}-gram token chuẩn hoá (cùng exam)")
    toks = {r.sample_id: tokenize(strip_comments(r.code).code) for r in df.itertuples()}
    norm = {sid: shingles(normalize_tokens(t), k) for sid, t in toks.items()}
    idents = {sid: {text.lower() for kind, text in t if kind == "id" and text not in CPP_KEYWORDS and text not in STD_NAMES} for sid, t in toks.items()}
    rows = []
    mats: dict[str, tuple[list[str], np.ndarray]] = {}
    for exam_id, g in df.groupby("exam_id"):
        tmpl = statement_template_shingles(exams.get(exam_id, {}), k)
        tmpl_ids = statement_template_identifiers(exams.get(exam_id, {}))
        ids = sorted(g.sample_id)
        mat = np.eye(len(ids))
        for (i, a), (j, b) in combinations(enumerate(ids), 2):
            sa, sb = norm[a] - tmpl, norm[b] - tmpl
            excl = jaccard(sa, sb)
            mat[i, j] = mat[j, i] = excl
            rows.append(
                {
                    "exam": exam_id,
                    "a": a,
                    "b": b,
                    "jaccard": jaccard(norm[a], norm[b]),
                    "jaccard_no_template": excl,
                    "containment": len(sa & sb) / min(len(sa), len(sb)) if sa and sb else 0.0,
                    "ident_jaccard": jaccard(idents[a] - tmpl_ids, idents[b] - tmpl_ids),
                }
            )
        mats[exam_id] = (ids, mat)
    sim = pd.DataFrame(rows)
    lab = df.set_index("sample_id")
    for col in ["total_score", "labels", "compilable"]:
        sim[f"a_{col}"] = sim.a.map(lab[col])
        sim[f"b_{col}"] = sim.b.map(lab[col])
    print("**Phân bố Jaccard (sau khi bỏ shingle của code mẫu trong đề)**\n")
    desc = sim.groupby("exam").jaccard_no_template.describe(percentiles=[0.5, 0.9]).reset_index()
    print(md_table(desc))
    hi = sim[(sim[["jaccard", "jaccard_no_template", "containment"]] >= threshold).any(axis=1)]
    print(f"\n**Cặp có jaccard hoặc containment ≥ {threshold}**\n")
    print(md_table(hi) if len(hi) else "_không có_")
    print("\n**Top-3 cặp mỗi exam**\n")
    print(md_table(sim.sort_values("jaccard_no_template", ascending=False).groupby("exam").head(3)))
    plot_similarity(mats, out_dir / "similarity_heatmap.png")
    return sim


def report_privacy(df: pd.DataFrame) -> None:
    section("6. Ẩn danh / PII / prompt injection (CHỈ ĐẾM)")
    rows = []
    for row in df.itertuples():
        stripped = strip_comments(row.code)
        comment_text = "\n".join(t for _, t in stripped.comments)
        literals = "\n".join(re.findall(r"\"([^\"\n]*)\"", stripped.code))
        free_text = comment_text + "\n" + literals
        counts = {f"pii_{k}": len(p.findall(row.code)) for k, p in PII_PATTERNS.items()}
        counts["pii_digits_8_10_in_comment_or_string"] = len(PII_PATTERNS["digits_8_10"].findall(free_text))
        counts["pii_anon_id_is_own"] = sum(1 for m in PII_PATTERNS["anon_id_token"].findall(row.code) if m == row.sample_id.split("-")[1])
        counts.update({f"inj_{k}": len(p.findall(comment_text)) for k, p in INJECTION_PATTERNS.items()})
        rows.append({"sample_id": row.sample_id, **counts})
    pv = pd.DataFrame(rows)
    summary = [{"pattern": c, "files_with_hit": rate(int((pv[c] > 0).sum()), len(pv)), "total_hits": int(pv[c].sum()), "sample_ids": list(pv[pv[c] > 0].sample_id)} for c in pv.columns if c != "sample_id"]
    print(md_table(pd.DataFrame(summary)))


# ------------------------------------------------------------------------------ plots


def plot_similarity(mats: dict[str, tuple[list[str], np.ndarray]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(mats), figsize=(6.5 * len(mats), 5.8), facecolor="#fcfcfb")
    axes = np.atleast_1d(axes)
    for ax, (exam_id, (ids, mat)) in zip(axes, mats.items()):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        short = [s.split("-")[1] for s in ids]
        ax.set_xticks(range(len(ids)), short, rotation=90, fontsize=7, color="#52514e")
        ax.set_yticks(range(len(ids)), short, fontsize=7, color="#52514e")
        ax.set_title(f"{exam_id}: Jaccard 5-gram (template removed)", fontsize=10, color="#0b0b0b")
        for i, j in zip(*np.where(np.triu(mat, 1) >= 0.5)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im, ax=list(axes), shrink=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_pass_vs_score(ps: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ran = ps[ps.n_tests > 0]
    if ran.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    # Spread samples that share the same (strict pass, score) point.
    dup_rank = ran.groupby(["passed", "total"]).cumcount()
    dup_size = ran.groupby(["passed", "total"]).sample_id.transform("size")
    jitter = (dup_rank - (dup_size - 1) / 2) * 0.18
    ax.scatter(ran.passed + jitter, ran.total, s=48, color="#2a78d6", label="strict pass (as in test_report)", edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    moved = ran.lenient_pass != ran.passed
    for r, jx in zip(ran[moved].itertuples(), jitter[moved]):
        ax.annotate("", xy=(r.lenient_pass + jx, r.total), xytext=(r.passed + jx, r.total), arrowprops={"arrowstyle": "->", "color": "#eb6834", "lw": 1.4}, zorder=2)
    ax.scatter(ran.lenient_pass[moved] + jitter[moved], ran.total[moved], s=48, color="#eb6834", marker="D", label="lenient pass (prompt stripped, case ignored)", edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    for r, jx, rank in zip(ran.itertuples(), jitter, dup_rank):
        if r.passed == 0 and r.total >= 7:
            offset = (-8, 7) if rank % 2 else (-8, -13)
            ax.annotate(r.sample_id.split("-")[1], (r.passed + jx, r.total), textcoords="offset points", xytext=offset, fontsize=7, color="#52514e")
    ax.set_xlabel("public tests passed (of 8)", color="#52514e")
    ax.set_ylabel("teacher total_score", color="#52514e")
    ax.set_title("EX02: public-test pass count vs teacher score", color="#0b0b0b", fontsize=10)
    ax.grid(color="#e6e5e0", linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------- main


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=EDA_DIR / "out" / "code")
    ap.add_argument("--cxx", default="clang++", help="local compiler (e.g. g++-13 inside a Linux container)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--timeout", type=float, default=2.0, help="per-test wall-clock limit (seconds, <=2)")
    ap.add_argument("--ngram", type=int, default=5)
    ap.add_argument("--sim-threshold", type=float, default=0.7)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.data_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, exams, _ = load_meta(root)
    df = load_samples(root)
    missing = df[~df.code_exists]
    if len(missing):
        print(f"> WARNING: {len(missing)} code_file missing: {list(missing.sample_id)}")
    df = df[df.code_exists].copy()
    df["abs_path"] = df.code_file.map(lambda p: str(root / p))

    print(f"# Code & auxiliary-signal EDA — {len(df)} samples, exams {dict(Counter(df.exam_id))}")
    print(f"\nEncodings: {dict(Counter(df.code_encoding))}")
    feats = pd.DataFrame([{"sample_id": r.sample_id, "exam_id": r.exam_id, "total_score": r.total_score, **static_features(r.code, exams.get(r.exam_id, {}))} for r in df.itertuples()])
    report_code_stats(df, feats, exams)
    proto = report_prototypes(df, exams)

    with tempfile.TemporaryDirectory(prefix="code_eda_") as tmp:
        workroot = Path(tmp)
        comp = report_compile(df, args.cxx, not args.no_compile, workroot)
        if not args.no_compile:
            report_harness(df, exams, args.cxx, workroot)
        ps = report_tests(df, comp, args.cxx, not (args.no_run or args.no_compile), min(args.timeout, 2.0), workroot)
    hc = report_hardcode(df, exams)
    report_similarity(df, exams, args.ngram, args.sim_threshold, args.out_dir)
    report_privacy(df)
    if not ps.empty:
        plot_pass_vs_score(ps, args.out_dir / "ex02_pass_vs_score.png")

    out = feats.merge(comp.drop(columns=[c for c in ["binary", "exam_id", "total"] if c in comp.columns]), on="sample_id", how="left")
    out = out.merge(hc.drop(columns=["total"]), on="sample_id", how="left")
    if not proto.empty:
        out = out.merge(proto.drop(columns=["total", "logic"]), on="sample_id", how="left")
    out.to_csv(args.out_dir / "code_features.csv", index=False)
    print(f"\n_Features → {args.out_dir / 'code_features.csv'}; figures → {args.out_dir}_")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
