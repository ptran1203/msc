"""Independent re-check of the Task 3 (level-controlled feedback) EDA findings.

Recomputes every number from the raw dataset with fresh code (does not import the
analyst's t3 script). Subjective judgements (does a feedback give a location, a cause,
a fix?) come from a verifier annotation table below, with a STRICT and a LENIENT
reading for the grey cases, so every compliance number is reported as a range.

Run:
    python verify_t3_feedback.py --data-root /path/to/sample_dataset [--no-exec]

--no-exec skips compiling/running EX02 binaries (only EX02 is ever run; sources are
screened for dangerous calls first, run in a temp dir with a 2 s timeout).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import DIMS, load_samples  # noqa: E402

FULL_SCORE = 10
DANGEROUS_RE = re.compile(r"system\s*\(|popen|fork\s*\(|\bexec\w*\s*\(|remove\s*\(|rename\s*\(|ofstream|fopen|socket")
RUN_TIMEOUT_S = 2.0


# ---------------------------------------------------------------- verifier annotation
@dataclass(frozen=True)
class Judgement:
    """Verifier reading of one gold feedback. *_len = value under the lenient reading."""

    loc: bool
    cause: bool
    fix_dir: bool
    exact_fix: bool  # exact code edit / exact missing statement / formula given
    covered: tuple[str, ...]
    cause_len: bool | None = None
    fix_dir_len: bool | None = None
    exact_fix_len: bool | None = None
    covered_len: tuple[str, ...] | None = None
    l1_overreach_len: bool | None = None  # only used for the single L1-with-error case
    why: str = ""


L = "Lỗi logic"
J: dict[str, Judgement] = {
    "EX01-S101": Judgement(True, True, False, False, (L,), why="P4 + line 121-123 + only fits head node"),
    "EX01-S102": Judgement(True, False, False, False, (), cause_len=True,
                           why="'P2 list rỗng:' truncated; empty-list edge case, label is Lỗi thuật toán"),
    "EX01-S103": Judgement(True, True, False, False, ("Lỗi biên dịch", "Lỗi hard-code"), why="log + include csv"),
    "EX01-S104": Judgement(True, True, True, True, ("Lỗi thuật toán",), why="'chỉ cần lấy abs(...)' = formula fix"),
    "EX01-S105": Judgement(True, True, False, False, ("Lỗi mảng/chuỗi", L)),
    "EX01-S106": Judgement(True, True, False, True, (L, "Lỗi vòng lặp"), exact_fix_len=False,
                           why="'không có cur = cur->next': missing statement = cause (lenient) or fix code (strict)"),
    "EX01-S107": Judgement(True, True, False, True, ("Lỗi vòng lặp",), exact_fix_len=False,
                           covered_len=("Lỗi vòng lặp", L), why="same pattern as S106"),
    "EX01-S108": Judgement(True, True, False, False, ("Lỗi hard-code",)),
    "EX01-S109": Judgement(True, True, False, False, ("Lỗi biên dịch",), why="pasted log, quote not fix"),
    "EX01-S110": Judgement(True, True, True, False, (L,), fix_dir_len=False,
                           why="'phải đếm số dòng để có n': fix direction (strict) or explanation of requirement"),
    "EX01-S111": Judgement(True, True, False, False, ("Lỗi biên dịch",), why="pasted log"),
    "EX01-S112": Judgement(True, True, False, False, ("Lỗi hard-code",)),
    "EX01-S113": Judgement(True, True, False, False, ("Lỗi hàm",)),
    "EX01-S114": Judgement(True, True, False, False, ("Lỗi hàm",)),
    "EX01-S115": Judgement(False, False, False, False, ()),
    "EX02-S201": Judgement(False, False, False, False, ()),
    "EX02-S202": Judgement(False, False, False, False, ()),
    "EX02-S203": Judgement(False, False, False, False, ("Lỗi biên dịch",), why="category name + generic advice"),
    "EX02-S204": Judgement(True, False, False, False, (L, "Lỗi style"), why="'không tính đúng, em xem lại'"),
    "EX02-S205": Judgement(True, True, False, False, ("Lỗi nhập/xuất",), fix_dir_len=True,
                           why="'in thừa 1 dấu cách ở trước' makes the fix obvious (lenient)"),
    "EX02-S206": Judgement(True, True, True, False, ("Lỗi nhập/xuất", L), why="'X chứ không phải Y' twice"),
    "EX02-S207": Judgement(True, False, True, True, ("Lỗi thuật toán",), exact_fix_len=False,
                           why="'bỏ chữ int ở dòng số 9 là được': exact one-token edit; no mechanism given"),
    "EX02-S208": Judgement(False, False, False, False, ("Lỗi hàm",), why="'Gọi hàm chưa đúng' + generic advice"),
    "EX02-S209": Judgement(True, True, False, False, (L, "Lỗi vòng lặp"), fix_dir_len=True,
                           why="'chưa khởi tạo cẩn thận (temp, count)' implies initialise"),
    "EX02-S210": Judgement(False, False, False, False, ()),
    "EX02-S211": Judgement(False, False, False, False, ()),
    "EX02-S212": Judgement(True, True, False, False, ("Lỗi biên dịch", "Lỗi vòng lặp")),
    "EX02-S213": Judgement(True, False, False, False, ("Lỗi hàm",), covered_len=("Lỗi hàm", L)),
    "EX02-S214": Judgement(False, True, False, False, ("Lỗi style",), l1_overreach_len=False,
                           why="names the concrete problem (whole file commented out) at L1"),
    "EX02-S215": Judgement(True, True, False, False, ("Lỗi hàm",), covered_len=("Lỗi hàm", "Lỗi vòng lặp"),
                           why="'Sau vòng lặp' is a location, not a loop bug (strict)"),
    "EX02-S216": Judgement(True, True, False, False, (L,), covered_len=(L, "Lỗi vòng lặp"),
                           why="same phrase as S215 -> must be judged the same way"),
    "EX02-S217": Judgement(False, False, False, False, ("Lỗi biên dịch",), why="category name only"),
}


def pick(j: Judgement, field: str, lenient: bool) -> object:
    if lenient:
        alt = getattr(j, f"{field}_len", None)
        if alt is not None:
            return alt
    return getattr(j, field)


def level_check(level: int, j: Judgement, lenient: bool) -> dict[str, bool]:
    """label_space prohibitions (overreach) and definitions (under-spec) for one gold text."""
    loc, cause = bool(pick(j, "loc", lenient)), bool(pick(j, "cause", lenient))
    fix_dir, exact = bool(pick(j, "fix_dir", lenient)), bool(pick(j, "exact_fix", lenient))
    if level == 1:
        over = (loc or cause or fix_dir or exact) if not (lenient and j.l1_overreach_len is False) else False
        under = False
    elif level == 2:
        over, under = fix_dir or exact, not (loc and cause)
    elif level == 3:
        over, under = exact, not (loc and cause and (fix_dir or exact))
    else:
        over, under = False, False
    readme_violation = level in (1, 2) and exact  # README: L1/L2 giving fix code or a solution
    return {"over": over, "under": under, "readme_violation": readme_violation, "fix_any": fix_dir or exact}


# ---------------------------------------------------------------- helpers
def md(df: pd.DataFrame, index: bool = False) -> str:
    frame = df.reset_index() if index else df
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def section(title: str) -> None:
    print(f"\n## {title}\n")


def cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    """Cramér's V and Bergsma bias-corrected V for a contingency table (empty rows/cols dropped)."""
    obs = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0].to_numpy(float)
    chi2 = chi2_contingency(obs, correction=False)[0]
    n, (r, k) = obs.sum(), obs.shape
    v = float(np.sqrt(chi2 / n / (min(r, k) - 1)))
    phi2c = max(0.0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return v, float(np.sqrt(phi2c / (min(rc, kc) - 1)))


def words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def has_praise(text: str) -> bool:
    """'Tốt' or a non-negated 'đúng' (so 'không tính đúng' / 'không đúng quy định' do not count)."""
    if re.search(r"\bTốt\b", text):
        return True
    return any(not re.search(r"\b(?:không|chưa)\b", text[max(0, m.start() - 15): m.start()])
               for m in re.finditer(r"\bđúng\b", text))


def has_labels(row: pd.Series) -> bool:
    return isinstance(row["t3_labels"], list) and len(row["t3_labels"]) > 0


# ---------------------------------------------------------------- checks
def check_level_structure(df: pd.DataFrame) -> None:
    section("T3-01 level vs exam / score")
    ct = pd.crosstab(df["level_num"], df["exam_id"], margins=True)
    print(md(ct, index=True))
    v, vc = cramers_v(pd.crosstab(df["level_num"], df["exam_id"]))
    ex2 = df[df.exam_id == "EX02"]
    v2, vc2 = cramers_v(pd.crosstab(ex2["level_num"], ex2["total_score"] >= FULL_SCORE))
    print(f"\nCramér V level~exam = {v:.2f} (bias-corr {vc:.2f}); EX02 level~(total==10) = {v2:.2f} ({vc2:.2f})")
    l1 = ex2[ex2.level_num == 1]
    print(f"EX02 L1 with total=10: {(l1.total_score >= 10).sum()}/{len(l1)}; "
          f"EX02 non-L1 with total=10: {(ex2[ex2.level_num != 1].total_score >= 10).sum()}/{(ex2.level_num != 1).sum()}")
    rule = np.where(df.exam_id == "EX01", 2, np.where(df.total_score >= 10, 1, 3))
    exam_only = np.where(df.exam_id == "EX01", 2, 3)
    majority = df.level_num.mode()[0]
    print(f"Post-hoc rule exam+total: {(rule == df.level_num).sum()}/32; exam only: {(exam_only == df.level_num).sum()}/32; "
          f"majority (L{majority}): {(df.level_num == majority).sum()}/32")
    print(f"Misses of rule: {df.loc[rule != df.level_num, 'sample_id'].tolist()}")
    print(f"L4: {(df.level_num == 4).sum()}/32; L1 with >=1 label: "
          f"{df[(df.level_num == 1)].apply(has_labels, axis=1).sum()}/{(df.level_num == 1).sum()}; "
          f"no-label samples by level: {df[~df.apply(has_labels, axis=1)].groupby('level_num').sample_id.apply(list).to_dict()}")


def check_compliance(df: pd.DataFrame, metric_text: str) -> None:
    section("T3-02 / T3-03 gold self-compliance (verifier annotation, strict vs lenient)")
    lab = df[df.apply(has_labels, axis=1)]
    rows = []
    for lenient in (False, True):
        res = pd.DataFrame([{**level_check(int(r.level_num), J[r.sample_id], lenient), "sid": r.sample_id,
                             "level": int(r.level_num)} for r in lab.itertuples()])
        ok = ~(res.over | res.under)
        for lvl, g in res.groupby("level"):
            gok = ~(g.over | g.under)
            rows.append({"reading": "lenient" if lenient else "strict", "target": f"L{lvl}", "n": len(g),
                         "overreach": int(g.over.sum()), "under_spec": int(g.under.sum()),
                         "no_fix_dir": int((~g.fix_any).sum()), "compliant": int(gok.sum())})
        rows.append({"reading": "lenient" if lenient else "strict", "target": "ALL", "n": len(res),
                     "overreach": int(res.over.sum()), "under_spec": int(res.under.sum()),
                     "no_fix_dir": int((~res.fix_any).sum()), "compliant": int(ok.sum())})
        print(f"{'lenient' if lenient else 'strict'}: overreach={res[res.over].sid.tolist()} "
              f"under={res[res.under].sid.tolist()}")
    print(md(pd.DataFrame(rows)))

    print("\ntask3_feedback.json metric text:", metric_text)
    print("\nREADME metric (a) rule = 'L1/L2 output with fix code / solution is a violation' (all 32 samples):")
    out = []
    for lenient in (False, True):
        viol = [r.sample_id for r in df.itertuples()
                if level_check(int(r.level_num), J[r.sample_id], lenient)["readme_violation"]]
        out.append({"output": f"GOLD ({'lenient' if lenient else 'strict'})", "violations": len(viol),
                    "compliance": 1 - len(viol) / len(df), "ids": ",".join(viol)})
    for const in ("Tốt.", "Chưa hoàn thiện.", "Em xem lại bài nhé."):
        # A constant text has no code / fix, so the README rule can never fire.
        out.append({"output": const, "violations": 0, "compliance": 1.0, "ids": ""})
    print(md(pd.DataFrame(out)))
    full_viol_strict = [r.sample_id for r in df.itertuples() if has_labels(pd.Series(r._asdict()))
                        and level_check(int(r.level_num), J[r.sample_id], False)["over"]]
    print(f"\nIf (a) enforced every label_space prohibition (strict): gold violations {len(full_viol_strict)}/32 "
          f"-> compliance {1 - len(full_viol_strict) / 32:.2f} {full_viol_strict}")


def check_diagnosis(df: pd.DataFrame, root: Path) -> None:
    section("T3-04 diagnostic content, label coverage, cited lines")
    lab = df[df.apply(has_labels, axis=1)]
    for lenient in (False, True):
        kinds = {"loc+cause": 0, "loc only": 0, "cause only": 0, "none": 0}
        cov, tot = 0, 0
        for r in lab.itertuples():
            j = J[r.sample_id]
            lo, ca = bool(pick(j, "loc", lenient)), bool(pick(j, "cause", lenient))
            kinds["loc+cause" if lo and ca else "loc only" if lo else "cause only" if ca else "none"] += 1
            covered = set(pick(j, "covered", lenient) or ()) & set(r.t3_labels)
            cov += len(covered)
            tot += len(r.t3_labels)
        print(f"{'lenient' if lenient else 'strict'}: {kinds}; label coverage {cov}/{tot} = {cov / tot:.0%}")
    algo = lab[lab.t3_labels.map(lambda xs: "Lỗi thuật toán" in xs)]
    print(f"'Lỗi thuật toán' samples: {algo.sample_id.tolist()}; covered (strict): "
          f"{[s for s in algo.sample_id if 'Lỗi thuật toán' in J[s].covered]}")

    print("\nCompiler diagnostics quoted in gold vs released compile_log:")
    recs = []
    for r in df.itertuples():
        for m in re.finditer(r"([\w.]+):(\d+):(\d+): (?:fatal )?error", r.feedback):
            gfile, gline = m.group(1), int(m.group(2))
            log_lines = re.findall(r"([\w.]+):(\d+):\d+: (?:fatal )?error", r.compile_log or "")
            code_lines = (r.code or "").split("\n")
            quoted = re.search(rf"^\s*{gline}\s*\|\s*(.*\S)", r.feedback, re.M)
            quoted_txt = quoted.group(1).strip() if quoted else ""
            where = [i + 1 for i, c in enumerate(code_lines) if quoted_txt and c.strip() == quoted_txt]
            recs.append({"sample_id": r.sample_id, "gold_ref": f"{gfile}:{gline}",
                         "log_ref": ",".join(f"{f}:{n}" for f, n in log_lines[:1]),
                         "quoted_stmt_at_code_line": where, "gold_ascii_quotes": "'" in r.feedback,
                         "log_unicode_quotes": "‘" in (r.compile_log or "")})
    print(md(pd.DataFrame(recs)))
    s101 = df.set_index("sample_id").loc["EX01-S101", "code"].split("\n")
    print(f"\nS101 cites line 121-123 -> code: {[s101[i - 1].strip() for i in (121, 122, 123)]}")


def check_rubric_conflicts(df: pd.DataFrame) -> None:
    section("T3-05 feedback vs T1 rubric")
    pre = df[df.feedback.str.contains("tiên quyết", case=False)]
    print(md(pre[["sample_id", "total_score", "logic", "edge_case"]]))
    print(f"'tiên quyết' mentions: {len(pre)}; with logic>0 or edge>0: "
          f"{((pre.logic > 0) | (pre.edge_case > 0)).sum()} {pre[(pre.logic > 0) | (pre.edge_case > 0)].sample_id.tolist()}")
    zero = df[df.feedback.str.contains(r"là 0\b", regex=True)]
    print(md(zero[["sample_id", "total_score"] + DIMS]))
    s214 = df.set_index("sample_id").loc["EX02-S214"]
    code_wo_comments = re.sub(r"//[^\n]*|/\*.*?\*/", "", s214.code, flags=re.S)
    residual = len(re.sub(r"\s", "", code_wo_comments))
    print(f"\nS214: lines={s214.code_lines}, non-comment code chars={residual}, "
          f"log has 'undefined reference to `main'={'undefined reference to `main' in s214.compile_log}, "
          f"compilable={s214.compilable}, total={s214.total_score}, n_tests={s214.n_tests}, labels={s214.t3_labels}")
    s211 = df.set_index("sample_id").loc["EX02-S211"]
    print(f"S211: feedback={s211.feedback!r}, total={s211.total_score}, io_format={s211.io_format}, "
          f"passed {s211.n_passed}/{s211.n_tests}, actual[0]={s211.test_report[0]['actual']!r}")


def check_ex01_problem_bounds(df: pd.DataFrame, exams: dict) -> None:
    section("T3-05/T3-11 EX01 per-problem statements vs total, under both weightings")
    stmt = exams["EX01"]["statement"]
    stmt_w = [float(x) for x in re.findall(r"Problem \d \((\d+) điểm\)", stmt)]
    json_w = [p["max_score"] for p in exams["EX01"]["problems"]]
    print(f"statement weights={stmt_w} (sum {sum(stmt_w)}), exams.json max_score={json_w} (sum {sum(json_w)})")
    # status per problem read from gold: 1 = correct, 0 = zero (not done / not graded / infinite loop), None = partial.
    statuses = {
        "EX01-S101": [1, 1, 0, None], "EX01-S102": [1, None, 1, 1], "EX01-S104": [1, 1, None, 1],
        "EX01-S115": [1, 1, 1, 1], "EX01-S108": [0, 0, 0, 0], "EX01-S110": [0, 0, 0, 0], "EX01-S112": [0, 0, 0, 0],
    }
    rows = []
    for sid, st in statuses.items():
        tot = df.set_index("sample_id").loc[sid, "total_score"]
        for name, w in (("1/4/2/3", stmt_w), ("2.5x4", json_w)):
            ub = sum(wi for wi, s in zip(w, st) if s != 0)
            rows.append({"sample_id": sid, "weights": name, "per_problem_upper_bound": ub, "total": tot,
                         "total>bound": tot > ub})
    print(md(pd.DataFrame(rows)))


def check_leakage(df: pd.DataFrame) -> None:
    section("T3-06 / T3-07 inputs")
    print(f"t3 taxonomy_error == t2 output: {(df.t3_labels == df.labels).sum()}/32; "
          f"t3 compile_log == t1 compile_log: {(df.t3_input.map(lambda d: d.get('compile_log')) == df.compile_log).sum()}/32")
    print(f"t3_input keys: {sorted(set().union(*df.t3_input.map(dict.keys)))}")
    print(f"t3_input has test_report: {df.t3_input.map(lambda d: 'test_report' in d).sum()}/32")
    ex2 = df[df.exam_id == "EX02"]
    print(f"EX02 t1 test_report is list: {ex2.test_report.map(lambda x: isinstance(x, list)).sum()}/17, "
          f"non-empty: {ex2.test_report.map(lambda x: bool(x)).sum()}/17 "
          f"(empty: {ex2[ex2.test_report.map(lambda x: not x)].sample_id.tolist()})")
    stripped = sum(t["actual"] == t["actual"].strip() for r in ex2.test_report if r for t in r)
    total = sum(len(r) for r in ex2.test_report if r)
    print(f"test_report 'actual' already stripped of outer whitespace: {stripped}/{total}")
    s205 = ex2.set_index("sample_id").loc["EX02-S205"]
    lits = re.findall(r'cout\s*<<\s*"([^"]*)"', s205.code)
    print(f"S205 printed literals={lits!r}, io_format={s205.io_format}, passed {s205.n_passed}/{s205.n_tests}")
    # Verifier reading: which EX02 golds report a value that was OBSERVED at run time (vs. a static bug)?
    runtime_observed = {"EX02-S215": "sum quá lớn", "EX02-S216": "sum rất lớn"}
    static_derivable = {"EX02-S205": "cout << \" True\"", "EX02-S206": "TRUE/FALSE literal, *d vs ^d",
                        "EX02-S207": "shadowed int y", "EX02-S209": "for index"}
    print(f"EX02 golds stating run-time observed values: {len(runtime_observed)}/17 {list(runtime_observed)}; "
          f"symptom-like but statically visible: {list(static_derivable)}")


def check_style(df: pd.DataFrame) -> None:
    section("T3-08 style")
    log_re = re.compile(r"(?m)^\S*:\d+:\d+:|^\s*\d+\s*\|")
    rows = []
    for ex, g in df.groupby("exam_id"):
        fb = g.feedback
        prose = fb.map(lambda s: "\n".join(line for line in s.split("\n") if not log_re.search(line)))
        rows.append({
            "exam": ex, "n": len(g),
            "em": int(fb.str.contains(r"(?<!\w)em(?!\w)", case=False).sum()),
            "SV/sinh viên": int(fb.str.contains(r"\bSV\b|sinh viên", case=False).sum()),
            "grading_talk": int(fb.str.contains(r"tiên quyết|theo quy định|không chấm|không có điểm", case=False).sum()),
            "log_paste": int(fb.map(lambda s: bool(log_re.search(s))).sum()),
            "encourage_strict": int(fb.str.contains(r"cố gắng|cố lên|phát huy|chúc|giỏi|tốt lắm|nhé|nha+\b", case=False).sum()),
            "praise(Tốt/đúng)": int(fb.map(has_praise).sum()),
            "generic_advice": int(fb.str.contains(r"xem lại|coi kỹ lại|học kỹ|chú ý", case=False).sum()),
            "SV_tokens": int(fb.str.count(r"\bSV\b").sum()),
            "arrow_tokens": int(fb.str.count(r"\s->\s").sum()),
            "P1-P4_tokens": int(fb.str.count(r"\bP[1-4]\b").sum()),
            "eng_words_outside_logs": int(prose.str.count(
                r"(?i)\b(?:assume|crash|fail|list|node|hardcode|convert|parameter|set|data|include)\b").sum()),
        })
    print(md(pd.DataFrame(rows)))
    print("praise hits:", df[df.feedback.map(has_praise)].sample_id.tolist())
    print("encourage_strict hits:", df[df.feedback.str.contains(r"cố gắng|cố lên|phát huy|chúc|giỏi|tốt lắm|nhé|nha+\b",
                                                               case=False)].sample_id.tolist())


def check_substrings(df: pd.DataFrame) -> None:
    section("T3-09 snippet-is-substring check")
    for sid, stmt in (("EX01-S106", "cur = cur->next"), ("EX01-S107", "head = head->next")):
        code = df.set_index("sample_id").loc[sid, "code"].split("\n")
        hits = [i + 1 for i, c in enumerate(code) if re.sub(r"\s", "", stmt) in re.sub(r"\s", "", c)]
        print(f"{sid}: '{stmt}' found at code lines {hits}")
    code_like = {"EX01-S104": "abs(năm mất - năm sinh)", "EX01-S106": "cur = cur->next", "EX01-S107": "head = head->next",
                 "EX01-S109": "return;", "EX01-S111": "getline(sa,s,);", "EX02-S212": "i <= cnt"}
    ident_only = {"EX01-S112": "\"filename\"", "EX01-S113": "stof", "EX01-S114": "stoi", "EX02-S209": "isArm, temp, count"}
    print(f"feedback with C++ expression/statement: {len(code_like)}/32; with identifiers only: {len(ident_only)}/32")


def check_length(df: pd.DataFrame) -> None:
    section("T3-10 length")
    d = df.assign(words=df.feedback.map(words))
    g = d.groupby(["exam_id", "level_num"]).words.agg(["count", "median", "min", "max"]).reset_index()
    print(md(g))
    ex2 = d[d.exam_id == "EX02"]
    rho, p = spearmanr(ex2.level_num, ex2.words)
    print(f"\nEX02 Spearman(level, words) = {rho:.2f} (p={p:.4f}, n={len(ex2)})")
    print(f"'Tốt.' exact: {(d.feedback.str.strip() == 'Tốt.').sum()}/32; <=5 words: "
          f"{d[d.words <= 5].sample_id.tolist()}; empty: {(d.feedback.str.strip() == '').sum()}")


def screen_and_run(code: str, tests: list[dict], tmp: Path, name: str) -> tuple[int, list[str]] | None:
    """Compile an EX02 program and run the public tests; returns (passed, raw stdout of test 1)."""
    if DANGEROUS_RE.search(code):
        return None
    cxx = shutil.which("g++") or shutil.which("c++")
    if not cxx:
        return None
    src, exe = tmp / f"{name}.cpp", tmp / name
    src.write_text(code, encoding="utf-8")
    if subprocess.run([cxx, "-std=c++11", "-w", "-o", str(exe), str(src)], capture_output=True, timeout=60).returncode:
        return None
    passed, raws = 0, []
    for t in tests:
        try:
            out = subprocess.run([str(exe)], input=t["input"] + "\n", capture_output=True, text=True,
                                 timeout=RUN_TIMEOUT_S, cwd=tmp).stdout
        except subprocess.TimeoutExpired:
            out = "<timeout>"
        raws.append(out)
        passed += out.strip() == t["expected"].strip()
    return passed, raws[:1]


def check_execution(df: pd.DataFrame) -> None:
    section("Execution checks (EX02 only; clang on macOS, differs from the GNU g++ that produced compile_log)")
    idx = df.set_index("sample_id")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for sid, patch in (("EX02-S205", None), ("EX02-S207", None), ("EX02-S207", "drop 'int' on line 9"),
                           ("EX02-S211", None)):
            row = idx.loc[sid]
            code = row.code
            if patch:
                lines = code.split("\n")
                lines[8] = lines[8].replace("int y", "y", 1)  # the gold's suggested one-token edit
                code = "\n".join(lines)
            res = screen_and_run(code, row.test_report, tmp, sid.replace("-", "_") + ("_p" if patch else ""))
            if res is None:
                print(f"{sid} {patch or 'original'}: skipped (screen/compile)")
                continue
            print(f"{sid} {patch or 'original'}: passes {res[0]}/{len(row.test_report)} public tests "
                  f"(test_report says {row.n_passed}); raw stdout test1={res[1][0]!r}")


def check_edge_case_gap(df: pd.DataFrame) -> None:
    section("Extra: edge_case deductions vs labels / gold")
    comp = df[df.compilable == 1]
    ded = comp[comp.edge_case < 2]
    mention = ded[ded.feedback.str.contains(r"rỗng|\bBC\b|trường hợp|biên|edge|số 0|n = 0", case=False)]
    print(f"compilable=1: {len(comp)}; edge_case<2 among them: {len(ded)}; 'Lỗi edge case' label in t3 input: "
          f"{df.t3_labels.map(lambda xs: 'Lỗi edge case' in xs).sum()}/32; deducted samples whose gold mentions an "
          f"edge-case concept: {len(mention)}/{len(ded)} {mention.sample_id.tolist()}")
    ex2 = df[df.exam_id == "EX02"]
    vec = ex2[DIMS].astype(int).astype(str).agg(",".join, axis=1)
    top = vec.value_counts().head(1)
    print(f"most common EX02 rubric vector: [{top.index[0]}] x{top.iloc[0]}/17 -> "
          f"{ex2[vec == top.index[0]].sample_id.tolist()}; their gold texts all differ: "
          f"{ex2[vec == top.index[0]].feedback.nunique()} distinct")


def check_io_standard(df: pd.DataFrame) -> None:
    section("Extra: io_format standard - extra text printed around True/False (EX02, comments stripped)")
    rows = []
    for r in df[df.exam_id == "EX02"].itertuples():
        code = re.sub(r"//[^\n]*|/\*.*?\*/", "", r.code, flags=re.S)
        lits = re.findall(r'"([^"]*)"', code)
        prompt = [x for x in lits if re.search(r"[A-Za-z]", x) and x.strip().rstrip("\\n") not in ("True", "False")]
        padded = [x for x in lits if x.strip().rstrip("\\n") in ("True", "False") and x != x.lstrip()]
        if prompt or padded:
            rows.append({"sample_id": r.sample_id, "extra_text": prompt + padded, "io_format": r.io_format,
                         "io_label": "Lỗi nhập/xuất" in r.t3_labels, "passed": f"{int(r.n_passed)}/{int(r.n_tests)}",
                         "total": r.total_score, "level": f"L{r.level_num}", "feedback": r.feedback[:45]})
    print(md(pd.DataFrame(rows)))


def normalise_log_line(line: str) -> str:
    """Drop temp-dir prefixes and unify quote styles so gold and released logs compare fairly."""
    line = re.sub(r"/tmp/\w+/", "", line)
    return re.sub(r"[‘’`]", "'", line).strip()


def check_gold_log_vs_released(df: pd.DataFrame) -> None:
    section("Extra: gold-pasted logs vs released compile_log (copy baseline)")
    log_line_re = re.compile(r"^\S*:\d+:\d+:|^\s*\d+\s*\|")
    rows = []
    for r in df.itertuples():
        gold_log = [ln for ln in r.feedback.split("\n") if log_line_re.search(ln)]
        if not gold_log:
            continue
        released = {normalise_log_line(ln) for ln in (r.compile_log or "").split("\n")}
        raw_released = set((r.compile_log or "").split("\n"))
        rows.append({"sample_id": r.sample_id, "gold_log_lines": len(gold_log),
                     "found_verbatim": sum(ln in raw_released for ln in gold_log),
                     "found_after_norm": sum(normalise_log_line(ln) in released for ln in gold_log),
                     "gold_files": ",".join(sorted(set(re.findall(r"(student\.\w+):\d", r.feedback)))),
                     "gold_quotes": "ascii" if "'" in r.feedback else "none",
                     "log_quotes": "unicode" if "‘" in (r.compile_log or "") else "none"})
    print(md(pd.DataFrame(rows)))
    best = df[df.feedback.map(lambda s: bool(log_line_re.search(s.split("\n")[0])))]
    print(f"gold feedback that STARTS with a compiler diagnostic: {best.sample_id.tolist()}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--no-exec", action="store_true", help="skip compiling/running EX02 programs")
    args = ap.parse_args(argv)
    df = load_samples(args.data_root)
    exams = {e["exam_id"]: e for e in json.loads((args.data_root / "exams.json").read_text(encoding="utf-8"))["exams"]}
    missing = set(df.sample_id) - set(J)
    if missing:
        print(f"> NOTE: {len(missing)} samples have no verifier annotation; annotation-based checks cover the sample set only.")
        df = df[df.sample_id.isin(J)]
    print(f"# Task 3 verification (n={len(df)})")
    readme = (args.data_root / "README.md").read_text(encoding="utf-8")
    for needle in ("bias của bộ mẫu nhỏ", "để Task 3 độc lập với Task 2", "dùng **chung một tập sample_id**"):
        print(f"README contains '{needle}': {needle in readme}")
    check_level_structure(df)
    t3_meta = json.loads((args.data_root / "task3_feedback.json").read_text(encoding="utf-8"))
    check_compliance(df, str(t3_meta.get("metric", "")))
    check_diagnosis(df, args.data_root)
    check_rubric_conflicts(df)
    check_ex01_problem_bounds(df, exams)
    check_leakage(df)
    check_style(df)
    check_substrings(df)
    check_length(df)
    check_edge_case_gap(df)
    check_io_standard(df)
    check_gold_log_vs_released(df)
    if not args.no_exec:
        check_execution(df)


if __name__ == "__main__":
    main()
