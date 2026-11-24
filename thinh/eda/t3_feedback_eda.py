"""Task 3 EDA: level-controlled feedback.

Answers, with numbers, whether the gold feedback follows the level definitions in
label_space.json, how level relates to exam/score/labels, feedback length and style,
label coverage, feedback-vs-rubric conflicts and what Task 3 input lacks.

Regex detectors run on any dataset size. A manual annotation file (optional) is used
to measure detector precision/recall and the gold's self-compliance; on the full
dataset, without annotations, the detector-based numbers are still produced.

Run:
    python t3_feedback_eda.py --data-root /path/to/sample_dataset [--annotations t3_manual_annotations.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import DIM_MAX, DIMS, load_meta, load_samples  # noqa: E402

FULL_SCORE = sum(DIM_MAX.values())

# ---------------------------------------------------------------- regex detectors
# ' -> ' with spaces is used as "therefore" in grader notes, so only a spaceless a->b counts as code.
CODE_SNIPPET_RES: dict[str, str] = {
    "backtick": r"`[^`\n]+`",
    "member_arrow": r"\b[A-Za-z_][A-Za-z0-9_]*->[A-Za-z_][A-Za-z0-9_]*",
    "assign_compare": r"\b[A-Za-z_][A-Za-z0-9_]*\s*(?:==|!=|<=|>=|\+=|-=|\*=|/=|=)\s*[A-Za-z0-9_]+(?:->[A-Za-z_]\w*)?",
    "call": r"\b[A-Za-z_][A-Za-z0-9_]*\([^()\n]*\)",
    "statement": r"[)\]]\s*;|\breturn\s*;|\+\+|#include",
    "stream_io": r"\b(?:cout|cin)\s*(?:<<|>>)",
    "braces": r"[{}]",
}
CPP_KEYWORD_RE = re.compile(
    r"\b(?:int|double|float|bool|char|string|void|for|while|if|else|return|include|nullptr|NULL|new|delete|struct|cout|cin|endl)\b"
)
COMPILER_DUMP_RE = re.compile(r"(?m)^\S*:\d+:\d+:\s*(?:fatal\s+)?(?:error|warning)|^\s*\d+\s*\|\s")
FIX_RES = [
    r"\b(?:sửa|thay|đổi|thêm|bỏ|xoá|xóa|dùng|sử dụng|lấy)\b[^.;\n]{0,40}?\b(?:thành|bằng|là được|là xong)\b",
    r"\bchỉ cần\b",
    r"\b(?:nên|cần|phải)\s+(?:sửa|thay|đổi|thêm|bỏ|xoá|xóa|dùng|sử dụng|lấy|đếm|khởi tạo|kiểm tra|in|tính|khai báo)\b",
    r"\bchứ không phải\b",
    r"\bsửa\b",
]
GENERIC_ADVICE_RE = re.compile(r"(?i)\b(?:xem lại|coi (?:kỹ )?lại|học (?:kỹ|lại)|chú ý|ôn lại|làm bài tập)\b")
LOC_STATIC_RES: dict[str, str] = {
    "line_ref": r"(?i)\b(?:dòng|line)\s*(?:số\s*)?\d+|:\d+:\d+:",
    "problem_ref": r"\bP[1-4]\b|(?i:\bcâu\s*\d\b)",
    "named_function": r"(?i)\bhàm\s+(?:tính\s+\w+|[A-Za-z_][A-Za-z0-9_]*\b)",
    "construct": r"(?i)\bvòng lặp\b|\bdấu cách\b|\boutput\b",
}
CAUSE_RE = re.compile(
    r"(?i)\b(?:vì|do|bởi|dẫn đến|khiến|nên|chỉ phù hợp|không có|thiếu|thừa|lặp vô tận|vô hạn|crash|"
    r"chưa khởi tạo|assume|gán cứng|hard ?code|sai chỉ số|sai công thức|quá lớn|rất lớn|không set|mà)\b"
    r"|\s->\s|→|error:|No such file"
)
PRAISE_RE = re.compile(r"(?i)\b(?:tốt|đúng|giỏi|chính xác)\b")
NEGATION_RE = re.compile(r"(?i)\b(?:không|chưa|chẳng)\b")
SECOND_PERSON_RE = re.compile(r"(?i)\b(?:em|bạn)\b")
THIRD_PERSON_RE = re.compile(r"\bSV\b|(?i:\bsinh viên\b)")
ABBREV_RE = re.compile(r"\bSV\b|\bko\b|(?i:\bnhaa+\b)|\s->\s|\bP[1-4]\b")
ENGLISH_LOANS = ["assume", "crash", "fail", "list", "node", "hardcode", "convert", "include", "parameter", "set", "return", "data"]
ENGLISH_LOAN_RE = re.compile(r"(?i)\b(?:" + "|".join(ENGLISH_LOANS) + r")\b")
GRADING_TALK_RE = re.compile(r"(?i)theo quy định|tiên quyết|không (?:được )?chấm|không có điểm|là 0\b")
ENCOURAGE_RE = re.compile(r"(?i)\b(?:cố gắng|nhé|nha+|chú ý|xem lại|coi kỹ lại)\b")
RUNTIME_SYMPTOM_RE = re.compile(
    r"(?i)kết quả|output|in thừa|in ra|quá lớn|rất lớn|crash|lặp vô tận|bỏ qua chữ số|tạo ra trống|TRUE/FALSE"
)
LINE_REF_RE = re.compile(r"(?i)\b(?:dòng|line)\s*(?:số\s*)?(\d+)(?:\s*-\s*(\d+))?")
DIAG_REF_RE = re.compile(r"(\S+?):(\d+):(\d+):")
SNIPPET_LINE_RE = re.compile(r"(?m)^\s*(\d+)\s*\|\s?(.*\S)\s*$")

# Words that are common in Vietnamese notes or C++ boilerplate and must not count as a code location.
IDENT_STOPLIST = {
    "list", "node", "file", "data", "string", "int", "return", "include", "main", "std", "using", "namespace",
    "true", "false", "the", "and", "not", "error", "with", "value", "function", "token", "before", "such",
    "directory", "line", "for", "while", "else", "void", "bool", "char", "double", "float", "cout", "cin",
    "endl", "struct", "new", "delete", "nullptr", "null", "const", "auto", "iostream", "set", "fail", "crash",
}

LABEL_LEXICON: dict[str, str] = {
    "Lỗi biên dịch": r"biên dịch|compile|error:|fatal error|not declared",
    "Lỗi nhập/xuất": r"output|input|in thừa|in ra|dấu cách|định dạng|TRUE|đọc file|nhập|xuất",
    "Lỗi logic": r"logic|\bsai\b|không tính đúng|không đúng|crash|trống|rỗng|assume|khởi tạo",
    "Lỗi vòng lặp": r"vòng lặp|lặp vô tận|\bfor\b|\bwhile\b|->next|chỉ số",
    "Lỗi mảng/chuỗi": r"mảng|chuỗi|\bstring\b|chỉ số|index",
    "Lỗi hàm": r"\bhàm\b|stoi|stof|\breturn\b|tham số|parameter",
    "Lỗi edge case": r"edge|trường hợp|rỗng|\bBC\b",
    "Lỗi thuật toán": r"thuật toán|công thức|cách (?:tính|xóa|xoá)|phức tạp",
    "Lỗi hard-code": r"hard ?code|gán cứng|cố định|include file",
    "Lỗi style": r"style|phức tạp|comment|đặt tên|thụt",
}

ANNOT_FLAGS = ["loc", "cause", "fix_dir", "exact_edit", "code", "compiler_dump", "generic_advice", "praise"]


# ---------------------------------------------------------------- small helpers
def md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:.2f}") -> str:
    """Render a DataFrame as a GitHub markdown table without extra dependencies."""
    frame = df.reset_index() if index else df
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v).replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def section(title: str) -> None:
    print(f"\n## {title}\n")


def short(text: str, limit: int = 70) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def word_count(text: str) -> int:
    return len([t for t in text.split() if re.search(r"\w", t)])


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def cramers_v(table: pd.DataFrame) -> tuple[float, float, float]:
    """Return (Cramér's V, bias-corrected V (Bergsma 2013), chi2 p-value)."""
    obs = table.to_numpy(dtype=float)
    obs = obs[obs.sum(axis=1) > 0][:, obs.sum(axis=0) > 0]
    if min(obs.shape) < 2:
        return float("nan"), float("nan"), float("nan")
    chi2, p, _, _ = chi2_contingency(obs, correction=False)
    n = obs.sum()
    r, k = obs.shape
    v = np.sqrt(chi2 / n / (min(r, k) - 1))
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    v_corr = np.sqrt(phi2 / max(min(rc, kc) - 1, 1e-9))
    return float(v), float(v_corr), float(p)


def total_bucket(total: float) -> str:
    if total >= FULL_SCORE:
        return "10"
    if total >= 7:
        return "7-9"
    if total >= 1:
        return "1-6"
    return "0"


# ---------------------------------------------------------------- detectors
def code_snippets(text: str) -> list[str]:
    found: list[str] = []
    for pattern in CODE_SNIPPET_RES.values():
        found += [m.group(0) for m in re.finditer(pattern, text)]
    return found


def code_identifiers(code: str) -> set[str]:
    idents = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", code or ""))
    return {i for i in idents if i.lower() not in IDENT_STOPLIST}


def has_praise(text: str) -> bool:
    for m in PRAISE_RE.finditer(text):
        if not NEGATION_RE.search(text[max(0, m.start() - 15): m.start()]):
            return True
    return False


def detect(row: pd.Series) -> dict[str, object]:
    fb: str = row["feedback"] or ""
    code: str = row["code"] or ""
    log: str = row["compile_log"] or ""
    snippets = [s for s in code_snippets(fb) if s not in "{}"]
    novel = [s for s in snippets if squash(s) not in squash(code) and squash(s) not in squash(log)]
    idents_hit = sorted(i for i in code_identifiers(code) if re.search(rf"(?<![\w]){re.escape(i)}(?![\w])", fb))
    loc_hits = {k: bool(re.search(p, fb)) for k, p in LOC_STATIC_RES.items()}
    loc_hits["code_identifier"] = bool(idents_hit)
    return {
        "d_code": bool(snippets) or bool(re.search(CODE_SNIPPET_RES["braces"], fb)),
        "d_code_novel": bool(novel),
        "novel_snippets": novel,
        "d_cpp_keyword": bool(CPP_KEYWORD_RE.search(fb)),
        "d_compiler_dump": bool(COMPILER_DUMP_RE.search(fb)),
        "d_fix": any(re.search(p, fb, flags=re.I) for p in FIX_RES),
        "d_generic_advice": bool(GENERIC_ADVICE_RE.search(fb)),
        "d_loc": any(loc_hits.values()),
        "loc_kinds": ",".join(k for k, v in loc_hits.items() if v),
        "idents_hit": ",".join(idents_hit),
        "d_cause": bool(CAUSE_RE.search(fb)),
        "d_praise": has_praise(fb),
        "d_second_person": bool(SECOND_PERSON_RE.search(fb)),
        "d_third_person": bool(THIRD_PERSON_RE.search(fb)),
        "n_abbrev": len(ABBREV_RE.findall(fb)),
        "n_english_loans": len(ENGLISH_LOAN_RE.findall(fb)),
        "d_grading_talk": bool(GRADING_TALK_RE.search(fb)),
        "d_encourage": bool(ENCOURAGE_RE.search(fb)),
        "d_runtime_symptom": bool(RUNTIME_SYMPTOM_RE.search(fb)),
    }


def observed_level(loc: bool, cause: bool, fix_dir: bool, exact_edit: bool) -> int:
    """Lowest level whose definition covers what the text actually does."""
    if exact_edit:
        return 4
    if fix_dir:
        return 3
    if loc or cause:
        return 2
    return 1


def compliance(level: int, f: dict[str, bool]) -> dict[str, bool]:
    """Check one feedback against label_space prohibitions/requirements and the README metric rule.

    metric_violation: README rule (Level 1/2 giving fix code or a solution).
    def_violation: label_space 'cấm' column (L1: no location/cause/fix; L2: no fix/code; L3: no complete solution).
    under_spec: misses what the level definition requires (L2: location+cause; L3: +fix direction).
    """
    fix_code = f["novel_code"] or f["exact_edit"]
    metric_violation = level in (1, 2) and fix_code
    if level == 1:
        def_violation = f["loc"] or f["cause"] or f["fix_dir"] or fix_code
        under_spec = False
    elif level == 2:
        def_violation = f["fix_dir"] or fix_code
        under_spec = not (f["loc"] and f["cause"])
    elif level == 3:
        def_violation = f["exact_edit"]
        under_spec = not (f["loc"] and f["cause"] and f["fix_dir"])
    else:
        def_violation, under_spec = False, False
    return {"metric_violation": bool(metric_violation), "def_violation": bool(def_violation), "under_spec": bool(under_spec)}


# ---------------------------------------------------------------- checks against rubric / tests / code
def rubric_conflicts(row: pd.Series, det: dict[str, object]) -> list[str]:
    fb: str = row["feedback"] or ""
    flags: list[str] = []
    if re.search(r"(?i)tiên quyết", fb) and (row["logic"] > 0 or row["edge_case"] > 0):
        flags.append(f"prerequisite_claim_but_logic={row['logic']},edge={row['edge_case']}")
    if re.search(r"(?i)(?:là|=)\s*0\b", fb) and row["total_score"] > 0:
        flags.append(f"says_zero_but_total={row['total_score']}")
    if re.search(r"(?i)comment hết|comment toàn bộ", fb) and row["total_score"] >= FULL_SCORE:
        flags.append("commented_out_but_full_score")
    praise_only = det["d_praise"] and not (det["d_loc"] or det["d_cause"] or det["d_fix"] or det["d_generic_advice"])
    has_labels = isinstance(row["t3_labels"], list) and len(row["t3_labels"]) > 0
    if praise_only and (has_labels or row["total_score"] < FULL_SCORE):
        flags.append("praise_only_but_errors")
    if praise_only and pd.notna(row["pass_rate"]) and row["pass_rate"] < 1:
        flags.append(f"praise_only_but_tests_{int(row['n_passed'])}/{int(row['n_tests'])}")
    return flags


def verify_line_refs(row: pd.Series) -> list[dict[str, object]]:
    """Check that line numbers cited in the feedback point at the right place in the released code/log."""
    fb: str = row["feedback"] or ""
    code_lines = (row["code"] or "").split("\n")
    log: str = row["compile_log"] or ""
    out: list[dict[str, object]] = []
    for m in LINE_REF_RE.finditer(fb):
        a = int(m.group(1))
        b = int(m.group(2) or a)
        exists = b <= len(code_lines) and all(code_lines[i - 1].strip() for i in range(a, b + 1))
        out.append({"sample_id": row["sample_id"], "ref": m.group(0), "kind": "line_ref", "ok": exists,
                    "detail": f"code has {len(code_lines)} lines; cited lines non-blank={exists}"})
    for m in DIAG_REF_RE.finditer(fb):
        tag = f":{m.group(2)}:{m.group(3)}:"
        out.append({"sample_id": row["sample_id"], "ref": f"{m.group(1)}{tag}", "kind": "diag_ref",
                    "ok": tag in log, "detail": f"same line:col in released compile_log={tag in log}"})
    for m in SNIPPET_LINE_RE.finditer(fb):
        n, text = int(m.group(1)), m.group(2).strip()
        actual = code_lines[n - 1].strip() if n <= len(code_lines) else ""
        where = [i + 1 for i, line in enumerate(code_lines) if line.strip() == text]
        out.append({"sample_id": row["sample_id"], "ref": f"{n} | {short(text, 30)}", "kind": "snippet",
                    "ok": actual == text, "detail": f"snippet found in code at line(s) {where}"})
    return out


def privacy_counts(texts: Iterable[str]) -> dict[str, int]:
    patterns = {
        "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
        # Powers of ten (e.g. the 10^9 bound) are constants, not student IDs.
        "long_digit_run(>=7)": r"\b(?!10*\b)\d{7,}\b",
        "redacted_placeholder": r"\[(?:NAME|CLASS)_REDACTED\]",
        "id_keyword(MSSV/ho ten)": r"(?i)\bMSSV\b|họ (?:và )?tên",
    }
    counts = {k: 0 for k in patterns}
    for t in texts:
        for k, p in patterns.items():
            counts[k] += len(re.findall(p, t or ""))
    return counts


# ---------------------------------------------------------------- report sections
def report_schema(df: pd.DataFrame, root: Path) -> None:
    section("1. Schema Task 3 input (so với Task 1/2)")
    _, _, tasks = load_meta(root)
    t3_meta = {k: v for k, v in tasks["t3"].items() if k != "samples"}
    keys_t3 = df["t3_input"].map(lambda d: tuple(sorted(d))).value_counts()
    rows = [
        ("n samples", len(df)),
        ("t3_input key sets", "; ".join(f"{list(k)} x{v}" for k, v in keys_t3.items())),
        ("t3_input có test_report", int(df["t3_input"].map(lambda d: "test_report" in d).sum())),
        ("t1_input test_report khác null", int(df["t1_input"].map(lambda d: d.get("test_report") is not None).sum())),
        ("t3 compile_log == t1 compile_log", int(sum(a.get("compile_log") == b.get("compile_log") for a, b in zip(df["t3_input"], df["t1_input"])))),
        ("t3 taxonomy_error == t2 output (đúng thứ tự set)", int((df["t3_labels"].map(sorted) == df["labels"].map(sorted)).sum())),
        ("metric (task3 file)", t3_meta.get("metric", "")),
    ]
    print(md_table(pd.DataFrame(rows, columns=["check", "value"]), index=False))


def report_level_relations(df: pd.DataFrame) -> None:
    section("2. Level vs exam / điểm / số nhãn")
    print(md_table(pd.crosstab(df["level"], df["exam_id"], margins=True)))
    df = df.assign(total_bucket=df["total_score"].map(total_bucket), n_t3_labels=df["t3_labels"].map(len))
    print()
    print(md_table(pd.crosstab(df["level"], df["total_bucket"], margins=True)))
    print()
    print(md_table(pd.crosstab(df["level"], df["n_t3_labels"], margins=True)))
    rows = []
    for name, col, sub in [
        ("exam_id", "exam_id", df),
        ("total_bucket", "total_bucket", df),
        ("n_t3_labels", "n_t3_labels", df),
        ("total==10 (EX02 only)", "is_full", df[df["exam_id"] == "EX02"].assign(is_full=lambda d: d["total_score"] >= FULL_SCORE)),
        ("labels empty (EX02 only)", "no_label", df[df["exam_id"] == "EX02"].assign(no_label=lambda d: d["n_t3_labels"] == 0)),
    ]:
        v, vc, p = cramers_v(pd.crosstab(sub["level"], sub[col]))
        rows.append((name, len(sub), v, vc, p))
    print()
    print(md_table(pd.DataFrame(rows, columns=["vs level", "n", "Cramér V", "V bias-corr", "chi2 p"]), index=False))

    def rule(r: pd.Series) -> int:
        if r["exam_id"] == "EX01":
            return 2
        return 1 if r["total_score"] >= FULL_SCORE else 3
    pred = df.apply(rule, axis=1)
    acc = (pred == df["level_num"]).mean()
    print(f"\nRule 'EX01->L2; EX02 & total=10->L1; else L3' reproduces level: "
          f"{int((pred == df['level_num']).sum())}/{len(df)} = {acc:.1%}. "
          f"Levels present: {sorted(df['level_num'].dropna().unique().tolist())}; L4 count = {int((df['level_num'] == 4).sum())}.")


def report_length(df: pd.DataFrame) -> None:
    section("3. Độ dài feedback")
    d = df.assign(words=df["feedback"].map(word_count))
    agg = d.groupby(["exam_id", "level"]).agg(n=("sample_id", "size"), chars_mean=("feedback_chars", "mean"),
                                              chars_median=("feedback_chars", "median"), chars_min=("feedback_chars", "min"),
                                              chars_max=("feedback_chars", "max"), words_median=("words", "median"),
                                              words_min=("words", "min"), words_max=("words", "max"))
    print(md_table(agg))
    terse = d[d["words"] <= 5]
    print(f"\nEmpty feedback: {int((d['feedback'].fillna('').str.strip() == '').sum())}/{len(d)}; "
          f"exactly 'Tốt.': {int((d['feedback'].str.strip() == 'Tốt.').sum())}/{len(d)}; "
          f"<=5 words: {len(terse)}/{len(d)} ({len(terse) / len(d):.0%}) -> {', '.join(terse['sample_id'])}")
    print(f"Words overall: median {d['words'].median():.0f}, mean {d['words'].mean():.1f}, "
          f"EX01 median {d.loc[d.exam_id == 'EX01', 'words'].median():.0f}, EX02 median {d.loc[d.exam_id == 'EX02', 'words'].median():.0f}")


def report_detectors(df: pd.DataFrame, det: pd.DataFrame, ann: pd.DataFrame | None) -> None:
    section("4. Detector regex trên từng feedback")
    cols = ["d_code", "d_code_novel", "d_cpp_keyword", "d_compiler_dump", "d_fix", "d_generic_advice", "d_loc", "d_cause", "d_praise"]
    view = pd.concat([df[["sample_id", "level_num"]], det[cols].astype(int), det[["loc_kinds", "novel_snippets"]]], axis=1)
    view["novel_snippets"] = view["novel_snippets"].map(lambda xs: "; ".join(xs))
    print(md_table(view, index=False))
    print("\nDetector hit rate by target level:")
    print(md_table(pd.concat([df[["level_num"]], det[cols].astype(int)], axis=1).groupby("level_num").mean()))
    if ann is None:
        return
    print("\nDetector vs manual annotation (all n=%d feedbacks read by hand):" % len(ann))
    pairs = [("d_code", "code"), ("d_code_novel", "novel_code_manual"), ("d_compiler_dump", "compiler_dump"),
             ("d_fix", "fix_any"), ("d_generic_advice", "generic_advice"), ("d_loc", "loc"), ("d_cause", "cause"),
             ("d_praise", "praise"), ("d_second_person", "addr_em"), ("d_third_person", "addr_sv")]
    rows = []
    for d_col, m_col in pairs:
        pred, gold = det[d_col].astype(bool).to_numpy(), ann[m_col].astype(bool).to_numpy()
        tp, fp, fn = int((pred & gold).sum()), int((pred & ~gold).sum()), int((~pred & gold).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        fp_ids = ",".join(df.loc[pred & ~gold, "sample_id"])
        fn_ids = ",".join(df.loc[~pred & gold, "sample_id"])
        rows.append((d_col, m_col, int(gold.sum()), int(pred.sum()), tp, prec, rec, fp_ids, fn_ids))
    print(md_table(pd.DataFrame(rows, columns=["detector", "manual", "manual+", "pred+", "TP", "precision", "recall", "FP ids", "FN ids"]), index=False))


def report_compliance(df: pd.DataFrame, det: pd.DataFrame, ann: pd.DataFrame | None) -> pd.DataFrame:
    section("5. Gold có tự tuân thủ định nghĩa level không?")
    has_err = df["t3_labels"].map(len) > 0
    auto_flags = pd.DataFrame({
        "loc": det["d_loc"], "cause": det["d_cause"], "fix_dir": det["d_fix"],
        "exact_edit": False, "novel_code": det["d_code_novel"],
    })
    sources = {"auto(regex)": auto_flags}
    if ann is not None:
        sources["manual"] = pd.DataFrame({
            "loc": ann["loc"].astype(bool), "cause": ann["cause"].astype(bool), "fix_dir": ann["fix_dir"].astype(bool),
            "exact_edit": ann["exact_edit"].astype(bool), "novel_code": ann["novel_code_manual"].astype(bool),
        })
    results: dict[str, pd.DataFrame] = {}
    for name, flags in sources.items():
        rows = []
        for i, r in df.iterrows():
            f = flags.loc[i].to_dict()
            c = compliance(int(r["level_num"]), f)
            rows.append({"sample_id": r["sample_id"], "level": int(r["level_num"]), "has_err": bool(has_err.loc[i]),
                         "obs_level": observed_level(f["loc"], f["cause"], f["fix_dir"], f["exact_edit"] or f["novel_code"]), **c})
        res = pd.DataFrame(rows)
        results[name] = res
        sub = res[res["has_err"]]
        summary = sub.groupby("level").agg(n=("sample_id", "size"), metric_violation=("metric_violation", "sum"),
                                           def_violation=("def_violation", "sum"), under_spec=("under_spec", "sum"),
                                           obs_eq_target=("obs_level", lambda s: int((s == sub.loc[s.index, "level"]).sum())))
        print(f"\n**{name}** — chỉ tính {len(sub)} bài có >=1 nhãn lỗi (bài không lỗi: level không phân biệt được):")
        print(md_table(summary))
        tot = sub[["metric_violation", "def_violation", "under_spec"]].sum()
        ok = (~sub["def_violation"] & ~sub["under_spec"]).sum()
        print(f"Tổng: metric_violation {tot['metric_violation']}/{len(sub)}, def_violation {tot['def_violation']}/{len(sub)}, "
              f"under_spec {tot['under_spec']}/{len(sub)}, fully compliant {ok}/{len(sub)} ({ok / len(sub):.0%}).")
        print(f"Observed vs target level crosstab ({name}):")
        print(md_table(pd.crosstab(sub["level"].rename("target"), sub["obs_level"].rename("observed"))))
        bad = sub[sub["def_violation"] | sub["under_spec"]]
        listing = pd.DataFrame({
            "sample_id": bad["sample_id"], "target": bad["level"], "observed": bad["obs_level"],
            "issue": np.where(bad["def_violation"], "VƯỢT mức", "THIẾU yêu cầu"),
            "feedback": df.set_index("sample_id").loc[bad["sample_id"], "feedback"].map(lambda t: short(t, 80)).to_numpy(),
        })
        print(md_table(listing, index=False))
    best = "manual" if "manual" in sources else "auto(regex)"
    flags = sources[best][has_err.to_numpy()]
    diag = pd.Series(np.select(
        [flags["loc"] & flags["cause"], flags["loc"] & ~flags["cause"], ~flags["loc"] & flags["cause"]],
        ["loc+cause", "loc only", "cause only"], "no diagnosis"), index=flags.index)
    print(f"\nDiagnostic content of gold ({best}, n={len(flags)} with labels): "
          + ", ".join(f"{k} {v}" for k, v in diag.value_counts().items())
          + f" -> no diagnosis: {', '.join(df.loc[diag[diag == 'no diagnosis'].index, 'sample_id'])}")

    # A constant, content-free answer can never break the README compliance rule.
    print("\nConstant-output baselines under the README rule (auto detectors, all samples):")
    rows = []
    for text in ["Tốt.", "Chưa hoàn thiện.", "Em xem lại bài nhé."]:
        viol = 0
        for _, r in df.iterrows():
            d = detect(pd.Series({**r.to_dict(), "feedback": text}))
            f = {"loc": d["d_loc"], "cause": d["d_cause"], "fix_dir": d["d_fix"], "exact_edit": False, "novel_code": d["d_code_novel"]}
            viol += compliance(int(r["level_num"]), f)["metric_violation"]
        rows.append((text, len(df), viol, 1 - viol / len(df)))
    gold_viol = int(results[best]["metric_violation"].sum())
    rows.append((f"GOLD ({best})", len(df), gold_viol, 1 - gold_viol / len(df)))
    print(md_table(pd.DataFrame(rows, columns=["output", "n", "README violations", "compliance"]), index=False))
    return results[best]


def report_label_coverage(df: pd.DataFrame, ann: pd.DataFrame | None) -> None:
    section("6. Feedback có nhắc nhãn lỗi cho sẵn không?")
    rows = []
    per_sample = []
    for i, r in df.iterrows():
        fb = r["feedback"] or ""
        hits = {lab for lab, p in LABEL_LEXICON.items() if re.search(p, fb, flags=re.I)}
        labs = set(r["t3_labels"])
        manual = set(ann.loc[i, "covered"]) if ann is not None else set()
        per_sample.append({"sample_id": r["sample_id"], "n_labels": len(labs), "auto_covered": len(labs & hits),
                           "manual_covered": len(labs & manual), "auto_extra": sorted(hits - labs)})
        for lab in labs:
            rows.append({"label": lab, "auto": lab in hits, "manual": lab in manual})
    lab_df = pd.DataFrame(rows)
    by_label = lab_df.groupby("label").agg(n=("label", "size"), auto_cov=("auto", "mean"), manual_cov=("manual", "mean"))
    print(md_table(by_label.sort_values("n", ascending=False)))
    ps = pd.DataFrame(per_sample)
    with_err = ps[ps["n_labels"] > 0]
    print(f"\nLabel instances: {len(lab_df)}; auto-covered {int(lab_df['auto'].sum())} ({lab_df['auto'].mean():.0%}); "
          f"manual-covered {int(lab_df['manual'].sum())} ({lab_df['manual'].mean():.0%}).")
    if ann is not None:
        full = with_err[with_err["manual_covered"] == with_err["n_labels"]]
        none_ = with_err[with_err["manual_covered"] == 0]
        print(f"Samples with >=1 label: {len(with_err)}; all labels covered (manual) {len(full)} ({len(full) / len(with_err):.0%}); "
              f"no label covered {len(none_)} -> {', '.join(none_['sample_id'])}")
    extra = ps[ps["auto_extra"].map(len) > 0]
    print(f"Feedback mentioning a label NOT in t3_labels (lexicon, noisy): {len(extra)}/{len(ps)}")


def report_conflicts(df: pd.DataFrame, det_records: list[dict[str, object]], ann: pd.DataFrame | None) -> None:
    section("7. Mâu thuẫn feedback ↔ rubric T1 / test_report")
    rows = []
    for (_, r), d in zip(df.iterrows(), det_records):
        flags = rubric_conflicts(r, d)
        if flags:
            rows.append({"sample_id": r["sample_id"], "total": r["total_score"],
                         "rubric": "/".join(str(int(r[c])) for c in DIMS), "auto_flags": "; ".join(flags),
                         "feedback": short(r["feedback"], 70)})
    print(md_table(pd.DataFrame(rows), index=False) if rows else "No automatic conflict.")
    pre = df[df["feedback"].str.contains("tiên quyết", case=False)]
    bad_pre = pre[(pre["logic"] > 0) | (pre["edge_case"] > 0)]
    print(f"\nFeedback citing 'tiên quyết' (P1 failed -> P2-P4 not graded): {len(pre)}; with logic/edge_case > 0: "
          f"{len(bad_pre)}/{len(pre)} -> {', '.join(bad_pre['sample_id'])}")
    if ann is not None:
        rc = ann[ann["rubric_conflict"] != ""]
        tc = ann[ann["test_conflict"] != ""]
        print(f"Manual: rubric conflicts {len(rc)}/{len(ann)} ({', '.join(df.loc[rc.index, 'sample_id'])}); "
              f"test conflicts {len(tc)}/{len(ann)} ({', '.join(df.loc[tc.index, 'sample_id'])})")


def report_line_refs(df: pd.DataFrame) -> None:
    section("8. Kiểm tra số dòng mà feedback trích dẫn")
    refs = [x for _, r in df.iterrows() for x in verify_line_refs(r)]
    if not refs:
        print("No line references.")
        return
    tab = pd.DataFrame(refs)
    print(md_table(tab, index=False))
    print(f"\nRefs: {len(tab)} in {tab['sample_id'].nunique()} feedbacks; ok {int(tab['ok'].sum())}, mismatch {int((~tab['ok']).sum())}.")


def report_style(df: pd.DataFrame, det: pd.DataFrame, ann: pd.DataFrame | None) -> None:
    section("9. Văn phong: viết cho sinh viên hay ghi chú người chấm?")
    d = pd.concat([df[["sample_id", "exam_id", "level_num"]], det], axis=1)
    student = d["d_second_person"] | d["d_encourage"]
    grader = d["d_third_person"] | d["d_compiler_dump"] | d["d_grading_talk"] | (d["n_abbrev"] >= 2)
    d["register"] = np.select([student & ~grader, grader & ~student, student & grader], ["student-facing", "grader-note", "mixed"], "terse/neutral")
    agg = d.groupby("exam_id").agg(n=("sample_id", "size"), second_person=("d_second_person", "sum"),
                                   third_person_SV=("d_third_person", "sum"), compiler_dump=("d_compiler_dump", "sum"),
                                   grading_talk=("d_grading_talk", "sum"), encourage=("d_encourage", "sum"),
                                   abbrev_tokens=("n_abbrev", "sum"), english_loans=("n_english_loans", "sum"))
    print(md_table(agg))
    print()
    print(md_table(pd.crosstab(d["exam_id"], d["register"], margins=True)))
    if ann is not None:
        print()
        print(md_table(pd.crosstab(df["exam_id"], ann["addressee"].rename("manual addressee"), margins=True)))


def report_test_dependency(df: pd.DataFrame, det: pd.DataFrame) -> None:
    section("10. Task 3 không có test_report — feedback nào dựa vào hành vi lúc chạy?")
    d = pd.concat([df[["sample_id", "exam_id", "level_num", "n_passed", "n_tests", "total_score"]], det[["d_runtime_symptom"]]], axis=1)
    ex02 = d[d["exam_id"] == "EX02"]
    hits = ex02[ex02["d_runtime_symptom"]]
    print(f"EX02 feedback with runtime-symptom wording: {len(hits)}/{len(ex02)} -> {', '.join(hits['sample_id'])}")
    mism = ex02[(ex02["total_score"] >= FULL_SCORE) & (ex02["n_passed"].fillna(0) < ex02["n_tests"].fillna(0))]
    print(f"EX02 total=10 but not all public tests passed: {len(mism)} -> {', '.join(mism['sample_id'])}")
    io_hidden = df[(df["exam_id"] == "EX02") & (df["io_format"] == 0) & (df["pass_rate"] == 1)]
    print(f"EX02 io_format=0 while all public tests passed (test_report hides the I/O defect): {len(io_hidden)} -> "
          f"{', '.join(io_hidden['sample_id'])}")


def report_privacy(df: pd.DataFrame) -> None:
    section("11. Dấu vết ẩn danh sót (chỉ đếm)")
    rows = [("feedback", *privacy_counts(df["feedback"]).values()), ("code", *privacy_counts(df["code"]).values())]
    print(md_table(pd.DataFrame(rows, columns=["field", *privacy_counts([]).keys()]), index=False))


# ---------------------------------------------------------------- figures
def save_figures(df: pd.DataFrame, comp: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"

    sub = comp[comp["has_err"]]
    mat = pd.crosstab(sub["level"], sub["obs_level"]).reindex(index=[1, 2, 3, 4], columns=[1, 2, 3, 4], fill_value=0)
    fig, ax = plt.subplots(figsize=(4.8, 4.0), facecolor=surface)
    ax.imshow(mat.to_numpy(), cmap="Blues", vmin=0)
    for (i, j), v in np.ndenumerate(mat.to_numpy()):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > mat.to_numpy().max() / 2 else ink, fontsize=11)
    ax.set_xticks(range(4), [f"L{i}" for i in range(1, 5)])
    ax.set_yticks(range(4), [f"L{i}" for i in range(1, 5)])
    ax.set_xlabel("Observed level (what the gold text does)", color=muted)
    ax.set_ylabel("target_feedback_level", color=muted)
    ax.set_title(f"Gold feedback: target vs observed level\n(n={len(sub)} samples with ≥1 label)", fontsize=10, color=ink)
    fig.tight_layout()
    p = out_dir / "t3_target_vs_observed_level.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    d = df.assign(words=df["feedback"].map(word_count))
    fig, ax = plt.subplots(figsize=(6.0, 3.6), facecolor=surface)
    colors = {"EX01": "#2a78d6", "EX02": "#eb6834"}
    rng = np.random.default_rng(0)
    for k, (exam, g) in enumerate(d.groupby("exam_id")):
        x = g["level_num"].to_numpy() + (k - 0.5) * 0.25 + rng.uniform(-0.06, 0.06, len(g))
        ax.scatter(x, g["words"], s=36, color=colors.get(exam, muted), label=exam, edgecolor=surface, linewidth=1.5)
    ax.set_xticks([1, 2, 3, 4], ["L1", "L2", "L3", "L4"])
    ax.set_ylabel("words per feedback", color=muted)
    ax.set_title("Feedback length by target level (no L4 in sample)", fontsize=10, color=ink)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out_dir / "t3_feedback_words_by_level.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------- main
def load_annotations(path: Path | None, df: pd.DataFrame) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    missing = [s for s in df["sample_id"] if s not in raw]
    if missing:
        print(f"[annotations] {len(missing)} sample_ids not annotated; manual comparisons skipped.")
        return None
    ann = pd.DataFrame([raw[s] for s in df["sample_id"]], index=df.index)
    ann["fix_any"] = ann["fix_dir"].astype(bool) | ann["exact_edit"].astype(bool)
    ann["addr_em"] = ann["addressee"] == "em"
    ann["addr_sv"] = ann["addressee"] == "SV"
    return ann


def main(argv: list[str] | None = None) -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, default=here / "t3_manual_annotations.json")
    ap.add_argument("--out-dir", type=Path, default=here / "out" / "t3")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    df = load_samples(args.data_root).sort_values("sample_id").reset_index(drop=True)
    det_records = [detect(r) for _, r in df.iterrows()]
    det = pd.DataFrame(det_records, index=df.index)
    ann = load_annotations(args.annotations, df)
    if ann is not None:
        # Manual 'novel code' = manual code flag AND the regex snippet is not a quote of the student's code/log.
        ann["novel_code_manual"] = ann["code"].astype(bool) & (det["d_code_novel"] | ann["exact_edit"].astype(bool))

    print(f"# Task 3 EDA — data root: {args.data_root} — n={len(df)} — annotations: {'yes' if ann is not None else 'no'}")
    report_schema(df, args.data_root)
    report_level_relations(df)
    report_length(df)
    report_detectors(df, det, ann)
    comp = report_compliance(df, det, ann)
    report_label_coverage(df, ann)
    report_conflicts(df, det_records, ann)
    report_line_refs(df)
    report_style(df, det, ann)
    report_test_dependency(df, det)
    report_privacy(df)
    if not args.no_figures:
        paths = save_figures(df, comp, args.out_dir)
        section("Figures")
        for p in paths:
            print(f"- {p}")


if __name__ == "__main__":
    main()
