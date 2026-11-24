"""Metadata / spec / cross-file consistency audit for the grading-challenge dataset.

Usage:
    python meta_audit.py --data-root /path/to/sample_dataset [--no-compile]

Prints markdown tables only. Never writes inside the data root. Student code is only
compiled (syntax-only for function-only exams, compile+link otherwise) inside a
temporary directory; binaries are never executed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from load_data import DIM_MAX, DIMS, load_meta, load_samples

CHARS_PER_TOKEN = 3.5
COMPILE_TIMEOUT_S = 30
DIAG_ERROR_RE = re.compile(r"^\S+:\d+:\d+: (fatal )?error:", re.M)
LINK_ERROR_RE = re.compile(r"(collect2|/ld|\bld): error|undefined reference to", re.M)
DIAG_REF_RE = re.compile(r"(student\.\w+|\S+\.cpp|\S+\.h):(\d+):(\d+): (?:fatal )?(error|warning)")
LINE_REF_RE = re.compile(r"(?:\bline|\bdòng(?: số)?)\s*(\d+)(?:\s*-\s*(\d+))?", re.I)
PROBLEM_MARK_RE = re.compile(r"\bP([1-9])\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
LONG_DIGITS_RE = re.compile(r"\b\d{7,}\b")
PII_KEYWORD_RE = re.compile(r"\b(MSSV|Họ\s*(và\s*)?tên|Ho\s*(va\s*)?ten|Lớp|Mã\s*SV|student\s*id)\b", re.I)
VS_TEMPLATE_RE = re.compile(r"This file contains the 'main' function")
WIN_PATH_RE = re.compile(r"[A-Z]:\\\\?[\w\- ()]")
# Order matters: the first matching rule wins for a problem's feedback segment.
EX01_STATUS_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("zero", re.compile(r"theo quy định là 0|lặp vô tận|vô hạn")),
    ("wrong", re.compile(r"sai|lỗi|crash|fail|rỗng|trống|hard ?code|gán cứng|không đúng")),
    ("missing", re.compile(r"không làm")),
    ("ok", re.compile(r"đúng")),
]
# Heuristic cues that a feedback reveals concrete code or a concrete fix (level-compliance proxy).
# "->" alone is not code: graders use it as an arrow in prose, so require identifiers on both sides.
CODE_FRAGMENT_RE = re.compile(r"\w->\w|==|!=|\b[A-Za-z_]\w*\s*=\s*[A-Za-z_]\w*|\b[A-Za-z_]\w*\(|:\d+:\d+: (?:fatal )?error")
FIX_CUE_RE = re.compile(r"chỉ cần|sửa|bỏ chữ|thêm |thay bằng|nên dùng|hãy dùng|phải dùng|chuyển thành", re.I)
# "hàm <ascii identifier>" names a concrete function; Vietnamese words after "hàm" are not locations.
LOCATION_CUE_RE = re.compile(r"\bP[1-9]\b|line \d|dòng|:\d+:\d+|hàm [A-Za-z_][A-Za-z0-9_]*(?!\w)|vòng lặp", re.I)


# ----------------------------------------------------------------------------- helpers


def md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Render a DataFrame as a GitHub markdown table without extra dependencies."""
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v).replace("|", "\\|").replace("\n", " ")

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def section(title: str) -> None:
    print(f"\n## {title}\n")


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({(100 * k / n if n else 0):.0f}%)"


def split_cpp(code: str) -> tuple[str, str, str]:
    """Return (code_without_comments_and_strings, comments_text, string_literals_text)."""
    out, comments, strings = [], [], []
    i, n = 0, len(code)
    while i < n:
        two = code[i:i + 2]
        if two == "//":
            j = code.find("\n", i)
            j = n if j < 0 else j
            comments.append(code[i + 2:j])
            out.append(" ")
            i = j
        elif two == "/*":
            j = code.find("*/", i + 2)
            j = n if j < 0 else j + 2
            comments.append(code[i + 2:j - 2])
            out.append(" " + "\n" * code[i:j].count("\n"))
            i = j
        elif code[i] in "\"'":
            quote, j = code[i], i + 1
            while j < n and code[j] != quote and code[j] != "\n":
                j += 2 if code[j] == "\\" else 1
            strings.append(code[i + 1:j])
            out.append(quote + quote)
            i = j + 1
        else:
            out.append(code[i])
            i += 1
    return "".join(out), "\n".join(comments), "\n".join(strings)


def norm_signature(sig: str) -> tuple[str, tuple[str, ...]]:
    """Normalise 'Ret name(T a, U& b)' into (Ret, (T, U&)) ignoring parameter names and std::."""
    sig = sig.replace("std::", "")
    head, _, params = sig.partition("(")
    params = params.rsplit(")", 1)[0]
    ret = re.sub(r"\s+", "", head.rsplit(None, 1)[0] if len(head.split()) > 1 else head)
    ret = re.sub(r"\w+$", "", ret) if not head.strip().endswith(("*", "&")) else ret
    types = []
    for p in filter(None, (x.strip() for x in params.split(","))):
        p = re.sub(r"\s*=.*$", "", p)
        m = re.match(r"^(.*?[\s\*&])(\w+)$", p)
        types.append(re.sub(r"\s+", "", m.group(1) if m else p))
    return ret, tuple(types)


def find_definitions(code_only: str, fname: str) -> list[str]:
    pattern = re.compile(r"([\w:<>]+[\s\*&]+(?:[\*&]\s*)*)" + re.escape(fname) + r"\s*\(([^)]*)\)\s*\{")
    return [f"{m.group(1).strip()} {fname}({m.group(2)})" for m in pattern.finditer(code_only)]


def tokens_est(text: str | None) -> float:
    return len(text or "") / CHARS_PER_TOKEN


def quantiles(s: pd.Series) -> dict[str, float]:
    s = s.dropna().astype(float)
    return {"p50": s.quantile(0.5), "p95": s.quantile(0.95), "max": s.max()}


# ----------------------------------------------------------------------------- 1. inventory


def audit_inventory(root: Path, tasks: dict[str, dict], exams: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_files = [p for p in root.rglob("*") if p.is_file()]
    rel = sorted(str(p.relative_to(root)) for p in all_files)
    hidden = [r for r in rel if Path(r).name.startswith(".")]
    rows.append({"check": "total files", "result": len(rel), "detail": dict(Counter(Path(r).suffix or Path(r).name for r in rel))})
    rows.append({"check": "hidden/system files", "result": len(hidden), "detail": hidden})
    cpp_on_disk = {r for r in rel if r.endswith(".cpp")}
    referenced: set[str] = set()
    for key, payload in tasks.items():
        ids = [s["sample_id"] for s in payload["samples"]]
        rows.append({"check": f"{key}: n_samples declared vs actual", "result": f"{payload.get('n_samples')} vs {len(ids)}",
                     "detail": "OK" if payload.get("n_samples") == len(ids) else "MISMATCH"})
        rows.append({"check": f"{key}: duplicate sample_id", "result": len(ids) - len(set(ids)), "detail": ""})
        referenced |= {s["input"]["code_file"] for s in payload["samples"]}
    id_lists = {k: [s["sample_id"] for s in p["samples"]] for k, p in tasks.items()}
    sets = {k: set(v) for k, v in id_lists.items()}
    rows.append({"check": "same sample_id set T1/T2/T3", "result": sets["t1"] == sets["t2"] == sets["t3"],
                 "detail": {f"only_{k}": sorted(v - set.intersection(*sets.values())) for k, v in sets.items() if v - set.intersection(*sets.values())}})
    rows.append({"check": "same sample order T1/T2/T3", "result": id_lists["t1"] == id_lists["t2"] == id_lists["t3"], "detail": ""})
    rows.append({"check": "cpp on disk not referenced", "result": len(cpp_on_disk - referenced), "detail": sorted(cpp_on_disk - referenced)})
    rows.append({"check": "referenced code_file missing", "result": len(referenced - cpp_on_disk), "detail": sorted(referenced - cpp_on_disk)})
    bad_path, bad_prefix, bad_exam_meta = [], [], []
    for s in tasks["t1"]["samples"]:
        inp = s["input"]
        if inp["code_file"] != f"submissions/{inp['exam_id']}/{s['sample_id']}.cpp":
            bad_path.append(s["sample_id"])
        if not s["sample_id"].startswith(inp["exam_id"] + "-"):
            bad_prefix.append(s["sample_id"])
        exam = exams.get(inp["exam_id"], {})
        if exam.get("exam_type") != inp.get("exam_type") or exam.get("language") != inp.get("language"):
            bad_exam_meta.append(s["sample_id"])
    rows.append({"check": "code_file != submissions/<exam>/<sample_id>.cpp", "result": len(bad_path), "detail": bad_path})
    rows.append({"check": "sample_id prefix != exam_id", "result": len(bad_prefix), "detail": bad_prefix})
    rows.append({"check": "exam_type/language != exams.json", "result": len(bad_exam_meta), "detail": bad_exam_meta})
    per_exam = Counter(s["input"]["exam_id"] for s in tasks["t1"]["samples"])
    rows.append({"check": "samples per exam", "result": dict(per_exam), "detail": ""})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2. cross-task inputs


def audit_inputs(tasks: dict[str, dict], label_space: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_id = {k: {s["sample_id"]: s for s in p["samples"]} for k, p in tasks.items()}
    ids = sorted(set.intersection(*(set(v) for v in by_id.values())))
    rows = []
    for a, b in [("t1", "t2"), ("t1", "t3"), ("t2", "t3")]:
        only_a, only_b, diff_vals, identical = Counter(), Counter(), Counter(), 0
        for sid in ids:
            ia, ib = by_id[a][sid]["input"], by_id[b][sid]["input"]
            only_a.update(set(ia) - set(ib))
            only_b.update(set(ib) - set(ia))
            diff_vals.update(k for k in set(ia) & set(ib) if ia[k] != ib[k])
            identical += ia == ib
        rows.append({"pair": f"{a} vs {b}", "identical_inputs": pct(identical, len(ids)),
                     f"keys_only_in_first": dict(only_a), "keys_only_in_second": dict(only_b),
                     "common_keys_with_diff_values": dict(diff_vals) or "none"})
    label_eq_order = sum(by_id["t3"][s]["input"].get("taxonomy_error") == by_id["t2"][s]["output"].get("taxonomy_error") for s in ids)
    label_eq_set = sum(set(by_id["t3"][s]["input"].get("taxonomy_error") or []) == set(by_id["t2"][s]["output"].get("taxonomy_error") or []) for s in ids)
    rows.append({"pair": "t3.input.taxonomy_error vs t2.output", "identical_inputs": f"list-equal {pct(label_eq_order, len(ids))}; set-equal {pct(label_eq_set, len(ids))}",
                 "keys_only_in_first": "", "keys_only_in_second": "", "common_keys_with_diff_values": ""})
    meta_rows = []
    header_keys = {k: [x for x in p if x != "samples"] for k, p in tasks.items()}
    for k, keys in header_keys.items():
        meta_rows.append({"task": k, "header_keys": keys, "input_fields": tasks[k].get("input_fields"),
                          "output_fields": tasks[k].get("output_fields"), "metric": tasks[k].get("metric")})
    t2_ls = tasks["t2"].get("label_space")
    meta_rows.append({"task": "t2.label_space == label_space.json.taxonomy_error",
                      "header_keys": t2_ls == label_space["taxonomy_error"],
                      "input_fields": f"same set: {set(t2_ls or []) == set(label_space['taxonomy_error'])}",
                      "output_fields": "", "metric": ""})
    return pd.DataFrame(rows), pd.DataFrame(meta_rows)


# ----------------------------------------------------------------------------- 3. label validity


def audit_labels(df: pd.DataFrame, label_space: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dims_spec = {d["name"]: tuple(d["range"]) for d in label_space["rubric"]["dimensions"]}
    rows = []
    for d in DIMS:
        lo, hi = dims_spec.get(d, (0, DIM_MAX[d]))
        vals = df[d]
        rows.append({"field": d, "spec_range": f"[{lo},{hi}]", "observed": f"[{vals.min()},{vals.max()}]",
                     "out_of_range": int(((vals < lo) | (vals > hi)).sum()),
                     "non_integer": int((vals % 1 != 0).sum()),
                     "value_counts": dict(sorted(Counter(vals.astype(float).tolist()).items()))})
    tlo, thi = label_space["rubric"]["total_range"]
    rows.append({"field": "total_score", "spec_range": f"[{tlo},{thi}]", "observed": f"[{df.total_score.min()},{df.total_score.max()}]",
                 "out_of_range": int(((df.total_score < tlo) | (df.total_score > thi)).sum()),
                 "non_integer": int((df.total_score % 1 != 0).sum()),
                 "value_counts": f"total==sum(6 dims): {pct(int((df.total_score == df.sum_dims).sum()), len(df))}"})
    dims_tbl = pd.DataFrame(rows)

    space = label_space["taxonomy_error"]
    lab_rows = []
    for lab in space:
        row = {"label": lab}
        for ex, g in df.groupby("exam_id"):
            row[ex] = int(g["labels"].map(lambda xs: lab in xs).sum())
        row["total"] = int(df["labels"].map(lambda xs: lab in xs).sum())
        lab_rows.append(row)
    unknown = Counter(l for xs in df["labels"] for l in xs if l not in space)
    lab_rows.append({"label": "[] (no error)", **{ex: int(g["labels"].map(len).eq(0).sum()) for ex, g in df.groupby("exam_id")},
                     "total": int(df["labels"].map(len).eq(0).sum())})
    lab_rows.append({"label": "UNKNOWN labels", "total": dict(unknown) or 0})
    labels_tbl = pd.DataFrame(lab_rows).fillna("")

    level_names = [lv["level"] for lv in label_space["feedback_level"]]
    lvl = df.groupby(["exam_id", "level"]).size().unstack(fill_value=0)
    lvl = lvl.reindex(columns=sorted(set(level_names) | set(lvl.columns)), fill_value=0)
    lvl["not_in_label_space"] = df.groupby("exam_id")["level"].apply(lambda s: int((~s.isin(level_names)).sum()))
    return dims_tbl, labels_tbl, lvl.reset_index()


# ----------------------------------------------------------------------------- 4. code facts


def code_facts(df: pd.DataFrame, exams: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        code = r.code or ""
        code_only, comments, strings = split_cpp(code)
        stmt = exams[r.exam_id]["statement"]
        allowed_m = re.search(r"Chỉ được dùng các thư viện:\s*`([^`]+)`", stmt)
        allowed = {x.strip() for x in allowed_m.group(1).split(",")} if allowed_m else None
        includes = re.findall(r'#\s*include\s*[<"]([^>"]+)[>"]', code_only + "\n" + "\n".join(
            l for l in code.splitlines() if l.strip().startswith("#include")))
        includes = sorted(set(includes))
        proto_status = {}
        for prob in exams[r.exam_id].get("problems", []):
            proto = prob.get("prototype")
            if not proto:
                continue
            fname = re.search(r"(\w+)\s*\(", proto).group(1)
            defs = find_definitions(code_only, fname)
            want = norm_signature(proto.rstrip(";"))
            if not defs:
                proto_status[prob["pid"]] = "absent"
            else:
                proto_status[prob["pid"]] = "match" if any(norm_signature(d) == want for d in defs) else "changed"
        rows.append({
            "sample_id": r.sample_id, "exam_id": r.exam_id,
            "has_main_code": bool(re.search(r"\b(int|void)\s+main\s*\(", code_only)),
            "has_main_comment_only": bool(re.search(r"\bmain\s*\(", comments)) and not re.search(r"\b(int|void)\s+main\s*\(", code_only),
            "void_main": bool(re.search(r"\bvoid\s+main\s*\(", code_only)),
            "pow_call": len(re.findall(r"(?<![\w.])pow\s*\(", code_only)),
            "own_pow_def": bool(re.search(r"\b\w+\s+pow\s*\([^)]*\)\s*\{", code_only)),
            "includes_cmath": any(h in ("cmath", "math.h") for h in includes),
            "includes": includes,
            "non_whitelisted_includes": sorted(set(includes) - allowed) if allowed else None,
            "proto_status": proto_status,
            "code_tokens": len(re.findall(r"\w+|[^\s\w]", code_only)),
            "skeleton_weights": re.findall(r"Problem\s+(\d)\s*\((\d+)\)", comments),
            "vs_template": bool(VS_TEMPLATE_RE.search(code)),
            "windows_path": bool(WIN_PATH_RE.search(strings)),
            "csv_literals": sorted(set(re.findall(r"[\w\- ().\\:]*\.csv", strings + "\n" + code_only))),
            "bom": code.startswith("\ufeff"), "crlf": "\r\n" in code,
            "comments": comments, "strings": strings, "code_only": code_only,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 5. exams.json internal


def statement_lines(stmt: str, pattern: str) -> list[str]:
    return [l.strip() for l in stmt.splitlines() if re.search(pattern, l, re.I)]


def audit_exams(exams: dict[str, dict], df: pd.DataFrame, facts: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for ex_id, ex in exams.items():
        stmt = ex["statement"]
        sub = df[df.exam_id == ex_id]
        fsub = facts[facts.exam_id == ex_id]
        n = len(sub)
        stmt_w = {f"P{a}": float(b.replace(",", ".")) for a, b in re.findall(r"Problem\s+(\d+)\s*\((\d+(?:[.,]\d+)?)\s*điểm\)", stmt)}
        json_w = {p["pid"]: p.get("max_score") for p in ex.get("problems", [])}
        if stmt_w:
            rows.append({"exam": ex_id, "check": "per-problem weight: statement vs problems[].max_score",
                         "evidence": f"statement {stmt_w} (sum {sum(stmt_w.values())}); json {json_w} (sum {sum(v or 0 for v in json_w.values())})",
                         "verdict": "MISMATCH" if any(stmt_w.get(k) != v for k, v in json_w.items()) else "OK"})
        skel = [w for ws in fsub.skeleton_weights for w in ws]
        if skel:
            rows.append({"exam": ex_id, "check": "weights seen in student-kept skeleton comments",
                         "evidence": f"{sorted(set(skel))} in {fsub[fsub.skeleton_weights.map(bool)].sample_id.tolist()}",
                         "verdict": "supports statement weights"})
        for pat, label in [(r"biên dịch bằng", "compile command"), (r"Nhận 0 điểm", "zero-score rule"),
                           (r"\bmain\b", "main rule"), (r"pow", "pow rule"), (r"\.csv", "data file"),
                           (r"Chỉ được dùng", "library whitelist"), (r"prototype", "prototype rule")]:
            for line in statement_lines(stmt, pat):
                rows.append({"exam": ex_id, "check": f"statement quote ({label})", "evidence": line[:220], "verdict": ""})
        rows.append({"exam": ex_id, "check": "note / has_hidden_testcases / grading_policy",
                     "evidence": f"note={ex.get('note')!r}; has_hidden_testcases={ex.get('has_hidden_testcases')}; grading_policy={ex.get('grading_policy')!r}",
                     "verdict": ""})
        rows.append({"exam": ex_id, "check": "submissions with a real main() (not commented)",
                     "evidence": f"{pct(int(fsub.has_main_code.sum()), n)}; void main: {int(fsub.void_main.sum())}",
                     "verdict": ""})
        main_scores = sub.merge(fsub[["sample_id", "has_main_code"]], on="sample_id")
        if ex["exam_type"] == "multi_problem":
            with_main = main_scores[main_scores.has_main_code]
            rows.append({"exam": ex_id, "check": "statement says delete main before submit -> violation should give 0",
                         "evidence": f"with main: n={len(with_main)}, total>0: {int((with_main.total_score > 0).sum())}, mean total {with_main.total_score.mean():.2f}; e.g. {with_main[with_main.total_score == with_main.total_score.max()].sample_id.tolist()[:3]}",
                         "verdict": "rule not applied in labels" if (with_main.total_score > 0).any() else "consistent"})
        protos = Counter()
        for st in fsub.proto_status:
            protos.update(f"{k}:{v}" for k, v in st.items())
        if protos:
            changed = fsub[fsub.proto_status.map(lambda d: "changed" in d.values())].sample_id.tolist()
            rows.append({"exam": ex_id, "check": "prototype match (types only, names ignored)",
                         "evidence": f"{dict(sorted(protos.items()))}; changed in {changed}", "verdict": ""})
            missing = fsub[fsub.proto_status.map(lambda d: "absent" in d.values())].merge(sub[["sample_id", "compilable", "total_score"]], on="sample_id")
            rows.append({"exam": ex_id, "check": "required function absent (statement: keep empty body) -> a driver calling it cannot link",
                         "evidence": f"{pct(len(missing), n)}; compilable=1 among them: {int(missing.compilable.sum())}; totals: {dict(zip(missing.sample_id, missing.total_score))}",
                         "verdict": "hidden by -fsyntax-only log"})
        bad_inc = fsub[fsub.non_whitelisted_includes.map(lambda x: bool(x))]
        if fsub.non_whitelisted_includes.notna().any():
            rows.append({"exam": ex_id, "check": "includes outside library whitelist",
                         "evidence": f"{pct(len(bad_inc), n)}: {dict(zip(bad_inc.sample_id, bad_inc.non_whitelisted_includes))}",
                         "verdict": ""})
        if re.search(r"pow", stmt):
            p = fsub[fsub.pow_call > 0]
            rows.append({"exam": ex_id, "check": "pow() calls in code (comments stripped)",
                         "evidence": f"{pct(len(p), n)}: {dict(zip(p.sample_id, zip(p.pow_call, p.own_pow_def)))} (calls, defines own pow)",
                         "verdict": ""})
        ce = sub[sub.compilable == 0]
        rows.append({"exam": ex_id, "check": "compilable=0 -> total_score",
                     "evidence": f"n={len(ce)}; totals={sorted(ce.total_score.tolist())}; non-zero dims: {sorted({d for d in DIMS if (ce[d] > 0).any()})}",
                     "verdict": ""})
        loops = sub[sub.feedback.fillna("").str.contains("lặp vô tận|vô hạn") |
                    sub.test_report.map(lambda r: isinstance(r, list) and any("TIMEOUT" in str(t.get("actual")) for t in r))]
        rows.append({"exam": ex_id, "check": "infinite loop / TIMEOUT evidence -> total",
                     "evidence": dict(zip(loops.sample_id, loops.total_score)), "verdict": ""})
        csv_mentions = sorted(set(re.findall(r"`?([\w\-]+\.csv)`?", stmt)))
        if csv_mentions:
            present = {c: any(p.name == c for p in root.rglob(c)) for c in csv_mentions}
            lits = Counter(l.strip() for ls in fsub.csv_literals for l in ls)
            rows.append({"exam": ex_id, "check": "data file referenced by statement present in dataset?",
                         "evidence": f"{present}; csv literals in student code: {dict(lits)}", "verdict": ""})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 6. EX01 policy bounds


def parse_problem_status(feedback: str, pids: list[str]) -> dict[str, str]:
    marks = list(PROBLEM_MARK_RE.finditer(feedback))
    status: dict[str, str] = {}
    pending: list[str] = []
    for i, m in enumerate(marks):
        seg = feedback[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(feedback)]
        pid = f"P{m.group(1)}"
        found = next((name for name, rx in EX01_STATUS_RULES if rx.search(seg)), None)
        if found is None and not seg.strip(" ,;:.-"):
            pending.append(pid)
            continue
        for p in pending + [pid]:
            status.setdefault(p, found or "unknown")
        pending = []
    for p in pending:
        status.setdefault(p, "unknown")
    return {p: status.get(p, "unmentioned") for p in pids}


def score_bounds(status: dict[str, str], weights: dict[str, float], prereq: str | None, compiled: bool) -> tuple[float, float]:
    if not compiled:
        return 0.0, 0.0
    prereq_failed = prereq is not None and status.get(prereq) in ("wrong", "missing", "zero")
    lo = hi = 0.0
    for pid, w in weights.items():
        st = status.get(pid, "unmentioned")
        if prereq_failed and pid != prereq:
            continue
        if st == "ok":
            lo, hi = lo + w, hi + w
        elif st in ("wrong", "unknown", "unmentioned"):
            hi += w
    return lo, hi


def audit_ex01_policy(df: pd.DataFrame, exams: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for ex_id, ex in exams.items():
        if ex["exam_type"] != "multi_problem":
            continue
        pids = [p["pid"] for p in ex["problems"]]
        stmt_w = {f"P{a}": float(b) for a, b in re.findall(r"Problem\s+(\d+)\s*\((\d+(?:\.\d+)?)\s*điểm\)", ex["statement"])}
        json_w = {p["pid"]: float(p["max_score"]) for p in ex["problems"]}
        prereq = next((p["pid"] for p in ex["problems"] if p.get("is_prerequisite_for")), None)
        for r in df[df.exam_id == ex_id].itertuples():
            st = parse_problem_status(r.feedback or "", pids)
            compiled = bool(r.compilable)
            b_stmt = score_bounds(st, stmt_w or json_w, prereq, compiled)
            b_json = score_bounds(st, json_w, prereq, compiled)
            rows.append({"sample_id": r.sample_id, "status": " ".join(f"{k}={v}" for k, v in st.items()),
                         "prereq_in_feedback": "tiên quyết" in (r.feedback or ""),
                         "bounds_stmt_1/4/2/3": f"[{b_stmt[0]:g},{b_stmt[1]:g}]", "bounds_json_2.5x4": f"[{b_json[0]:g},{b_json[1]:g}]",
                         "total_score": r.total_score,
                         "within_stmt": b_stmt[0] <= r.total_score <= b_stmt[1],
                         "within_json": b_json[0] <= r.total_score <= b_json[1],
                         "excess_vs_stmt": max(0.0, r.total_score - b_stmt[1])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 7. gold feedback vs level rules


def audit_gold_levels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for r in df.itertuples():
        fb = r.feedback or ""
        rows.append({"sample_id": r.sample_id, "level_num": r.level_num,
                     "code_fragment": sorted(set(m.group(0).strip() for m in CODE_FRAGMENT_RE.finditer(fb)))[:4],
                     "fix_cue": sorted(set(m.group(0).strip() for m in FIX_CUE_RE.finditer(fb))),
                     "location_cue": bool(LOCATION_CUE_RE.search(fb)),
                     "chars": len(fb)})
    flags = pd.DataFrame(rows)
    flags["has_code"] = flags.code_fragment.map(bool)
    flags["has_fix"] = flags.fix_cue.map(bool)
    agg = flags.groupby("level_num").agg(n=("sample_id", "size"), has_code=("has_code", "sum"), has_fix_cue=("has_fix", "sum"),
                                         has_location=("location_cue", "sum"), chars_p50=("chars", "median"),
                                         chars_max=("chars", "max")).reset_index()
    flagged = flags[(flags.level_num <= 2) & (flags.has_code | flags.has_fix | (flags.level_num.eq(1) & flags.location_cue))]
    vague_l3 = flags[(flags.level_num == 3) & ~flags.location_cue & ~flags.has_fix & ~flags.has_code]
    detail = pd.concat([flagged.assign(issue="L1/L2 gold reveals code/fix/location"),
                        vague_l3.assign(issue="L3 gold: no location/fix/code cue (check manually)")])
    return agg, detail[["sample_id", "level_num", "issue", "code_fragment", "fix_cue", "location_cue"]]


# ----------------------------------------------------------------------------- 8. token budget


def audit_lengths(df: pd.DataFrame, exams: dict[str, dict]) -> pd.DataFrame:
    d = pd.DataFrame({
        "exam_id": df.exam_id,
        "statement": df.exam_id.map(lambda e: tokens_est(exams[e]["statement"])),
        "code": df.code.map(tokens_est),
        "compile_log": df.compile_log.map(tokens_est),
        "test_report": df.test_report.map(lambda t: tokens_est(json.dumps(t, ensure_ascii=False)) if isinstance(t, list) else 0.0),
        "feedback": df.feedback.map(tokens_est),
    })
    d["prompt_t1_t2"] = d.statement + d.code + d.compile_log + d.test_report
    d["prompt_t3"] = d.statement + d.code + d.compile_log + df.t3_labels.map(lambda x: tokens_est(json.dumps(x, ensure_ascii=False))) + df.level.map(tokens_est)
    rows = []
    for ex, g in d.groupby("exam_id"):
        for col in ["statement", "code", "compile_log", "test_report", "feedback", "prompt_t1_t2", "prompt_t3"]:
            q = quantiles(g[col])
            rows.append({"exam_id": ex, "component": col, "n": len(g), **{k: round(v) for k, v in q.items()}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 9. toolchain + local compile


def audit_toolchain(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    logs = df[df.log_nonempty]
    feats = {
        "GNU libstdc++ include dir /usr/include/c++/<ver>": r"/usr/include/c\+\+/(\d+)",
        "gcc lib dir /usr/lib/gcc/<triplet>/<ver>": r"/usr/lib/gcc/([\w\-]+)/(\d+)",
        "python tempfile dir /tmp/tmpXXXX/<file>": r"/tmp/tmp\w+/([\w.]+)",
        "GCC-style 'In function' header": r"In function [‘']",
        "GCC-only '-fpermissive' tag": r"-fpermissive",
        "Unicode quotes ‘ ’ (UTF-8 locale)": r"[‘’]",
        "linker error (collect2/ld)": LINK_ERROR_RE.pattern,
        "Windows/MSVC markers (C:\\ , error C####)": r"[A-Z]:\\|error C\d{4}",
        "source-line gutter 'NN | code' (GCC>=9 or clang)": r"(?m)^\s+\d+ \| ",
    }
    for ex, g in logs.groupby("exam_id"):
        for name, pat in feats.items():
            hits = g.compile_log.map(lambda s: re.findall(pat, s))
            n_hit = int(hits.map(bool).sum())
            vals = Counter(str(v) for hs in hits for v in hs) if n_hit else {}
            rows.append({"exam_id": ex, "feature": name, "logs_with_feature": pct(n_hit, len(g)),
                         "values": dict(list(Counter(vals).most_common(3)))})
    fb_diag = df[df.feedback.fillna("").map(lambda s: bool(DIAG_REF_RE.search(s)))]
    for r in fb_diag.itertuples():
        m = DIAG_REF_RE.search(r.feedback)
        lm = next((x for x in DIAG_REF_RE.finditer(r.compile_log or "") if x.group(4) == "error"), None)
        rows.append({"exam_id": r.exam_id, "feature": f"gold feedback quotes compiler diag ({r.sample_id})",
                     "logs_with_feature": f"feedback {m.group(1)}:{m.group(2)} vs log {lm.group(1).split('/')[-1] + ':' + lm.group(2) if lm else '-'}",
                     "values": f"feedback quote style: {'unicode' if re.search('[‘’]', r.feedback) else 'ascii'}"})
    return pd.DataFrame(rows)


def compile_locally(df: pd.DataFrame, exams: dict[str, dict], root: Path) -> pd.DataFrame:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        return pd.DataFrame([{"note": "no C++ compiler found"}])
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for r in df.itertuples():
            src = tmpdir / "student.cpp"
            shutil.copyfile(root / r.code_file, src)
            syntax_only = "fsyntax-only" in (exams[r.exam_id].get("note") or "")
            cmd = [compiler, "-std=c++11", str(src)] + (["-fsyntax-only"] if syntax_only else ["-o", str(tmpdir / "a.out")])
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S, cwd=tmpdir)
                ok, err = res.returncode == 0, res.stderr
            except subprocess.TimeoutExpired:
                ok, err = False, "TIMEOUT"
            first_err = next((l.split("error:", 1)[1].strip() for l in err.splitlines() if "error:" in l), "")
            log = r.compile_log or ""
            rows.append({"sample_id": r.sample_id, "mode": "syntax-only" if syntax_only else "compile+link",
                         "gold_compilable": r.compilable,
                         "log_diag_error": bool(DIAG_ERROR_RE.search(log)),
                         "log_link_error": bool(LINK_ERROR_RE.search(log)),
                         "local_ok": ok, "local_first_error": first_err[:70]})
    out = pd.DataFrame(rows)
    out.attrs["compiler"] = version
    return out


# ----------------------------------------------------------------------------- 10. feedback line refs


def audit_line_refs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        fb, lines = r.feedback or "", (r.code or "").splitlines()
        refs = [(int(m.group(2)), "diag") for m in DIAG_REF_RE.finditer(fb)]
        refs += [(int(m.group(1)), "text") for m in LINE_REF_RE.finditer(fb)]
        log_lines = [int(m.group(2)) for m in DIAG_REF_RE.finditer(r.compile_log or "") if m.group(4) == "error"]
        for ln, kind in refs:
            code_at = lines[ln - 1].strip()[:60] if 0 < ln <= len(lines) else "<out of range>"
            rows.append({"sample_id": r.sample_id, "ref_kind": kind, "feedback_line": ln,
                         "first_log_error_line": log_lines[0] if log_lines else None,
                         "code_at_line": code_at,
                         "code_at_line_minus_1": lines[ln - 2].strip()[:60] if 1 < ln <= len(lines) + 1 else ""})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 11. anonymisation + duplicates


def audit_privacy(df: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    texts = {
        "code comments+strings": facts.set_index("sample_id").apply(lambda r: r.comments + "\n" + r.strings, axis=1),
        "feedback": df.set_index("sample_id").feedback.fillna(""),
        "compile_log": df.set_index("sample_id").compile_log.fillna(""),
    }
    for where, series in texts.items():
        for name, rx in [("[..._REDACTED] tokens", re.compile(r"\[\w+_REDACTED\]")), ("email-like", EMAIL_RE),
                         ("7+ digit numbers", LONG_DIGITS_RE), ("PII keywords (MSSV/Họ tên/Lớp...)", PII_KEYWORD_RE),
                         ("Visual Studio template header", VS_TEMPLATE_RE)]:
            hits = series.map(lambda s: len(rx.findall(s)))
            rows.append({"source": where, "pattern": name, "n_samples_with_hit": pct(int((hits > 0).sum()), len(series)),
                         "total_hits": int(hits.sum()), "sample_ids": hits[hits > 0].index.tolist()})
    return pd.DataFrame(rows)


def audit_duplicates(facts: pd.DataFrame, k: int = 5, top: int = 3) -> pd.DataFrame:
    def shingles(code: str) -> set[tuple[str, ...]]:
        toks = re.findall(r"\w+|[^\s\w]", code)
        return {tuple(toks[i:i + k]) for i in range(max(0, len(toks) - k + 1))}

    rows = []
    for ex, g in facts.groupby("exam_id"):
        sh = {r.sample_id: shingles(r.code_only) for r in g.itertuples()}
        pairs = []
        for a, b in itertools.combinations(sh, 2):
            if sh[a] and sh[b]:
                pairs.append((len(sh[a] & sh[b]) / len(sh[a] | sh[b]), a, b))
        for j, a, b in sorted(pairs, reverse=True)[:top]:
            rows.append({"exam_id": ex, "pair": f"{a} ~ {b}", "jaccard_5gram": round(j, 3), "n_pairs": len(pairs)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 12. README claims


def audit_readme(root: Path, df: pd.DataFrame, facts: pd.DataFrame, tasks: dict[str, dict]) -> pd.DataFrame:
    readme = (root / "README.md").read_text(encoding="utf-8")
    rows = []
    for k, n_all, ex in re.findall(r"(\d+)/(\d+) bài `(EX\d+)`", readme):
        sub = df[df.exam_id == ex]
        any_err = sub.compile_log.fillna("").map(lambda s: bool(DIAG_ERROR_RE.search(s) or LINK_ERROR_RE.search(s)))
        proxy = sub[any_err & (sub.compilable == 1)]
        f = facts.set_index("sample_id").loc[sub.sample_id]
        rows.append({"claim": f"{k}/{n_all} {ex} compiles_locally=false but compilable=1 (MSVC: void main / pow w/o cmath)",
                     "sample_evidence": f"field compiles_locally present: {any('compiles_locally' in s['input'] or 'compiles_locally' in s['output'] for s in tasks['t1']['samples'])}; "
                                        f"proxy (log error incl. linker & compilable=1): {pct(len(proxy), len(sub))} {proxy.sample_id.tolist()}; "
                                        f"void main: {int(f.void_main.sum())}; pow call w/o cmath: {int(((f.pow_call > 0) & ~f.includes_cmath & ~f.own_pow_def).sum())}",
                     "sample_size_vs_claim": f"sample {len(sub)} vs full {n_all}"})
    for ex, g in df.groupby("exam_id"):
        rows.append({"claim": f"level distribution {ex}", "sample_evidence": dict(Counter(g.level_num)),
                     "sample_size_vs_claim": len(g)})
    rows.append({"claim": "Level 4 present anywhere", "sample_evidence": int((df.level_num == 4).sum()), "sample_size_vs_claim": len(df)})
    for ex, g in df.groupby("exam_id"):
        tr = g.test_report
        rows.append({"claim": f"{ex} test_report shape", "sample_evidence": {
            "null": int(tr.isna().sum()), "n_tests": dict(Counter(g.n_tests.dropna().astype(int))),
            "keys": sorted({k for r in tr if isinstance(r, list) for t in r for k in t}),
            "empty_list_ids": g[g.n_tests == 0].sample_id.tolist()}, "sample_size_vs_claim": len(g)})
    tr_all = [t for r in df.test_report if isinstance(r, list) for t in r]
    if tr_all:
        exp_by_input: dict[str, set] = {}
        for t in tr_all:
            exp_by_input.setdefault(t["input"], set()).add(t["expected"])
        inputs_num = [int(x) for x in exp_by_input if x.strip().lstrip("-").isdigit()]
        rows.append({"claim": "public test inputs (same expected per input?)",
                     "sample_evidence": f"inputs={sorted(exp_by_input, key=lambda x: (len(x), x))}; conflicting expected: {sum(len(v) > 1 for v in exp_by_input.values())}; max input={max(inputs_num) if inputs_num else None}",
                     "sample_size_vs_claim": len(tr_all)})
        stripped = sum(1 for t in tr_all if t["actual"] != t["actual"].strip())
        rows.append({"claim": "test_report.actual keeps leading/trailing whitespace?",
                     "sample_evidence": f"actual values with outer whitespace: {stripped}/{len(tr_all)}",
                     "sample_size_vs_claim": ""})
    log_link = df[df.compile_log.fillna("").map(lambda s: bool(LINK_ERROR_RE.search(s)))]
    rows.append({"claim": "EX01 log via -fsyntax-only (no link step), EX02 full g++",
                 "sample_evidence": f"linker errors per exam: {dict(Counter(log_link.exam_id))}",
                 "sample_size_vs_claim": ""})
    test_score_keys = [k for s in tasks["t1"]["samples"] for k in list(s["input"]) + list(s["output"]) if "test" in k.lower() and k != "test_report"]
    rows.append({"claim": "'Điểm theo test' column not included", "sample_evidence": f"extra test-score keys: {sorted(set(test_score_keys)) or 'none'}",
                 "sample_size_vs_claim": ""})
    pred_hits = {f: len(re.findall(r"prediction|submission format|sample_submission|định dạng nộp", (root / f).read_text(encoding="utf-8"), re.I))
                 for f in ["README.md", "task1_grading.json", "task2_error_taxonomy.json", "task3_feedback.json"]}
    rows.append({"claim": "predictions file format specified?", "sample_evidence": pred_hits, "sample_size_vs_claim": ""})
    return pd.DataFrame(rows)


def audit_io_vs_tests(df: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    """Compare gold io_format with what test_report shows (prompt text, whitespace, pass count)."""
    rows = []
    strings = facts.set_index("sample_id").strings
    for r in df[df.n_tests.fillna(0) > 0].itertuples():
        actual = [t["actual"] for t in r.test_report]
        prefix = re.match(r"^(.*?)(True|False)$", actual[0]).group(1) if re.match(r"^(.*?)(True|False)$", actual[0]) else ""
        lits = strings.get(r.sample_id, "").split("\n")
        rows.append({"sample_id": r.sample_id, "io_format": r.io_format, "total_score": r.total_score,
                     "n_passed": f"{int(r.n_passed)}/{int(r.n_tests)}",
                     "actual_prefix_before_True/False": repr(prefix) if prefix else "",
                     "code_prints_padded_literal": any(l != l.strip() and l.strip() in ("True", "False") for l in lits),
                     "empty_actual": sum(a == "" for a in actual)})
    out = pd.DataFrame(rows)
    keep = (out.io_format == 0) | (out["actual_prefix_before_True/False"] != "") | out.code_prints_padded_literal
    return out[keep]


def audit_level_confound(df: pd.DataFrame) -> pd.DataFrame:
    """Is the target level independent of how good the submission is?"""
    t = pd.crosstab([df.exam_id, df.level_num], [df.total_score.eq(10).map({True: "total=10", False: "total<10"})])
    t["labels_empty"] = df.groupby(["exam_id", "level_num"])["labels"].apply(lambda s: int(s.map(len).eq(0).sum()))
    t["total_mean"] = df.groupby(["exam_id", "level_num"]).total_score.mean().round(2)
    return t.reset_index()


# ----------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--no-compile", action="store_true", help="skip local compilation")
    args = parser.parse_args()
    root: Path = args.data_root
    label_space, exams, tasks = load_meta(root)
    df = load_samples(root)
    facts = code_facts(df, exams)

    print(f"# Metadata & spec audit — {root} (n={len(df)})")
    section("1. File inventory and sample_id consistency")
    print(md(audit_inventory(root, tasks, exams)))

    section("2. Input fields across tasks")
    pairs, meta = audit_inputs(tasks, label_space)
    print(md(pairs))
    print()
    print(md(meta))

    section("3. Label validity vs label_space.json")
    dims_tbl, labels_tbl, lvl_tbl = audit_labels(df, label_space)
    print(md(dims_tbl))
    print()
    print(md(labels_tbl))
    print()
    print(md(lvl_tbl))

    section("4. README claims vs sample")
    print(md(audit_readme(root, df, facts, tasks)))
    print()
    print(md(audit_io_vs_tests(df, facts)))
    print()
    print(md(audit_level_confound(df)))

    section("5. exams.json internal consistency and statement rules vs labels")
    print(md(audit_exams(exams, df, facts, root)))

    section("6. multi_problem: score bounds implied by statement policy + gold feedback vs total_score")
    pol = audit_ex01_policy(df, exams)
    if not pol.empty:
        print(md(pol))
        parsed = pol[~pol.status.str.contains("unmentioned=|unknown") | pol.prereq_in_feedback]
        print(f"\noutside statement bounds: {pct(int((~pol.within_stmt).sum()), len(pol))}; "
              f"outside json bounds: {pct(int((~pol.within_json).sum()), len(pol))}; "
              f"fully-parsed or prerequisite rows: {len(parsed)}")

    section("7. Gold feedback vs level rules (heuristic flags)")
    agg, detail = audit_gold_levels(df)
    print(md(agg))
    print()
    print(md(detail))

    section(f"8. Estimated token budget (chars/{CHARS_PER_TOKEN})")
    print(md(audit_lengths(df, exams)))

    section("9. Toolchain fingerprint of compile_log")
    print(md(audit_toolchain(df)))
    if not args.no_compile:
        loc = compile_locally(df, exams, root)
        print(f"\nLocal compiler: {loc.attrs.get('compiler')} (NOT the GNU g++ used for compile_log)\n")
        print(md(loc))
        if "local_ok" in loc:
            log_err = loc.log_diag_error | loc.log_link_error
            print(f"\nagreement local_ok vs not(log error): {pct(int((loc.local_ok == ~log_err).sum()), len(loc))}; "
                  f"local_ok vs gold compilable: {pct(int((loc.local_ok == loc.gold_compilable.astype(bool)).sum()), len(loc))}; "
                  f"not(log error) vs gold compilable: {pct(int(((~log_err) == loc.gold_compilable.astype(bool)).sum()), len(loc))}")

    section("10. Line references in gold feedback vs current code file")
    print(md(audit_line_refs(df)))

    section("11. Anonymisation residue (counts only)")
    print(md(audit_privacy(df, facts)))

    section("12. Near-duplicate submissions (token 5-gram Jaccard, comments stripped)")
    print(md(audit_duplicates(facts)))

    section("13. Code facts per sample")
    cols = ["sample_id", "has_main_code", "void_main", "pow_call", "own_pow_def", "includes_cmath",
            "non_whitelisted_includes", "proto_status", "code_tokens", "vs_template", "windows_path", "bom", "crlf"]
    print(md(facts[cols]))


if __name__ == "__main__":
    main()
