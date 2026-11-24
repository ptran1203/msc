"""Task 1 (rubric grading) EDA, reusable on the full dataset.

Sections (all printed as markdown):
  1. Per-dimension / total distributions by exam, integrity checks, degenerate dims.
  2. Gate structure (compilable=0, io_format vs logic) and deterministic couplings.
  3. compile_log signal vs gold `compilable` (+ optional local clang check).
  4. EX02: test_report vs rubric (Spearman, crosstab, anomalies) + optional local run.
  5. EX01: statement weights vs max_score, prototypes, per-problem status, P1-gate audit.
  6. QWK: label-free baselines, labels=None vs labels=range(11), bootstrap CI.

Usage:
  python t1_rubric_eda.py --data-root /path/to/sample_dataset [--no-compile] [--no-run]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from load_data import COMPILE_ERROR_RE, DIM_MAX, DIMS, load_meta, load_samples

TOTAL_MAX = 10
SCORE_LABELS = list(range(TOTAL_MAX + 1))
RUN_TIMEOUT_S = 2.0
TIMEOUT_MARK = "<TIMEOUT>"
NEAR_CONSTANT_SHARE = 0.9
LINK_ERROR_RE = re.compile(r"undefined reference|ld returned \d+ exit status|collect2: error|ld: error", re.I)
WARNING_RE = re.compile(r"^\S+:\d+:\d+: warning:", re.M)
# Calls we refuse to execute in untrusted student code (checked on raw source, comments included).
UNSAFE_RE = re.compile(
    r"\bsystem\s*\(|\bpopen\b|\bfork\s*\(|\bexec\w*\s*\(|\bremove\s*\(|\brename\s*\(|\bofstream\b|\bfopen\b|\bsocket\b"
)
STMT_WEIGHT_RE = re.compile(r"Problem\s+(\d)\s*\((\d+(?:[.,]\d+)?)\s*điểm\)")
# Extra in-range probes for EX02 (not in the public tests): 3/7/9-digit Armstrong numbers and upper bound.
EX02_EXTRA_PROBES = ["2", "407", "4210818", "9926315", "912985153", "999999999", "1000000000"]

# Manual per-problem reading of the 15 sample EX01 submissions (code + feedback, EDA only).
# status: correct | wrong | not_done | ungraded (attempted but behind the P1 gate) | unknown
MANUAL_EX01: dict[str, dict] = {
    "EX01-S101": {"P1": "correct", "P2": "correct", "P3": "not_done", "P4": "wrong", "inf_loop": False,
                  "note": "prob1 allocates new General[n] from caller n, never sets n (same defect as S110)"},
    "EX01-S102": {"P1": "correct", "P2": "wrong", "P3": "correct", "P4": "correct", "inf_loop": False,
                  "note": "addTail never assigns head -> prob2 returns empty list"},
    "EX01-S103": {"P1": "wrong", "P2": "not_done", "P3": "not_done", "P4": "not_done", "inf_loop": False,
                  "note": "#include of the csv file -> compile error"},
    "EX01-S104": {"P1": "correct", "P2": "correct", "P3": "wrong", "P4": "correct", "inf_loop": False,
                  "note": "BC age formula 100-born+death"},
    "EX01-S105": {"P1": "wrong", "P2": "wrong", "P3": "not_done", "P4": "not_done", "inf_loop": False,
                  "note": "returns address of local VLA; prob3/prob4 declared only"},
    "EX01-S106": {"P1": "unknown", "P2": "wrong", "P3": "wrong", "P4": "unknown", "inf_loop": True,
                  "note": "prob3 loop never advances cur"},
    "EX01-S107": {"P1": "unknown", "P2": "unknown", "P3": "wrong", "P4": "unknown", "inf_loop": True,
                  "note": "prob3 loop never advances head"},
    "EX01-S108": {"P1": "wrong", "P2": "ungraded", "P3": "not_done", "P4": "not_done", "inf_loop": False,
                  "note": "prob1 reads cin, fixed size 21"},
    "EX01-S109": {"P1": "unknown", "P2": "unknown", "P3": "not_done", "P4": "not_done", "inf_loop": False,
                  "note": "compile error in prob2"},
    "EX01-S110": {"P1": "wrong", "P2": "ungraded", "P3": "not_done", "P4": "ungraded", "inf_loop": False,
                  "note": "prob1 allocates new General[n] from caller n, never sets n"},
    "EX01-S111": {"P1": "wrong", "P2": "not_done", "P3": "not_done", "P4": "not_done", "inf_loop": False,
                  "note": "compile error in prob1"},
    "EX01-S112": {"P1": "wrong", "P2": "wrong", "P3": "ungraded", "P4": "wrong", "inf_loop": False,
                  "note": "opens literal \"filename\"; prob2 loop never runs; prob4 overwrites instead of deleting"},
    "EX01-S113": {"P1": "wrong", "P2": "not_done", "P3": "not_done", "P4": "not_done", "inf_loop": False,
                  "note": "only prob1, runtime stof failure"},
    "EX01-S114": {"P1": "wrong", "P2": "not_done", "P3": "not_done", "P4": "not_done", "inf_loop": False,
                  "note": "only prob1, runtime stoi failure"},
    "EX01-S115": {"P1": "correct", "P2": "correct", "P3": "correct", "P4": "correct", "inf_loop": False,
                  "note": "all correct per feedback"},
}
PIDS = ["P1", "P2", "P3", "P4"]
FB_WRONG_RE = re.compile(r"\bsai\b|crash|lỗi|lặp vô tận|vô hạn|gán cứng|hardcode|rỗng|trống|fail", re.I)
FB_NOT_DONE_RE = re.compile(r"không làm", re.I)
FB_CORRECT_RE = re.compile(r"đúng", re.I)
FB_GATE_RE = re.compile(r"tiên quyết|không (được )?chấm", re.I)
FB_INF_LOOP_RE = re.compile(r"lặp vô tận|vòng lặp vô hạn|lặp vô hạn", re.I)


# ----------------------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub markdown table without the optional `tabulate` dependency."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for value in row:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                cells.append("")
            elif isinstance(value, (float, np.floating)):
                cells.append(str(int(value)) if float(value).is_integer() else floatfmt.format(value))
            else:
                cells.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def section(title: str) -> None:
    print(f"\n## {title}\n")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return re.sub(r"//[^\n]*", "", code)


def function_body(code: str, name: str) -> str | None:
    """Body of the first definition `name(...) {` using brace matching, or None."""
    for match in re.finditer(rf"\b{name}\s*\(([^)]*)\)\s*\{{", code):
        start, depth = match.end() - 1, 0
        for i in range(start, len(code)):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    return code[start + 1 : i]
    return None


def log_kind(log: str | None) -> str:
    log = log or ""
    if COMPILE_ERROR_RE.search(log):
        return "compile_error"
    if LINK_ERROR_RE.search(log):
        return "link_error"
    if WARNING_RE.search(log):
        return "warning_only"
    return "empty" if not log.strip() else "other"


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan"), float("nan"), int(mask.sum())
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p), int(mask.sum())


def qwk(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            value = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)
        except ValueError:
            return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def rubric_vector(row: pd.Series) -> str:
    return "".join(str(int(row[d])) for d in DIMS)


def normalize_lenient(text: str) -> str:
    """Drop surrounding whitespace and an interactive prompt prefix like 'Input n: '."""
    return (text or "").strip().rsplit(":", 1)[-1].strip()


def armstrong_reference(n: int) -> str:
    digits = str(n)
    return "True" if sum(int(c) ** len(digits) for c in digits) == n else "False"


# ----------------------------------------------------------------------------- section 1
def report_distributions(df: pd.DataFrame) -> None:
    section("1. Phân bố từng chiều và total theo exam + kiểm tra toàn vẹn")
    cols = DIMS + ["total_score"]
    rows = []
    for exam, sub in [("ALL", df)] + list(df.groupby("exam_id")):
        for col in cols:
            counts = sub[col].value_counts().sort_index()
            share = counts.max() / len(sub)
            rows.append({
                "exam": exam, "dim": col, "n": len(sub), "max": DIM_MAX.get(col, TOTAL_MAX),
                "value:count": " ".join(f"{int(k)}:{v}" for k, v in counts.items()),
                "mean": sub[col].mean(), "std": sub[col].std(ddof=0), "n_unique": counts.size,
                "majority_share": share,
                "flag": "CONSTANT" if counts.size == 1 else ("near-constant" if share >= NEAR_CONSTANT_SHARE else ""),
            })
    print(md_table(pd.DataFrame(rows)))

    checks = []
    for col in cols:
        hi = DIM_MAX.get(col, TOTAL_MAX)
        is_int = df[col].map(lambda v: isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer()))
        checks.append({"field": col, "n_missing": int(df[col].isna().sum()), "n_non_integer": int((~is_int).sum()),
                       "n_out_of_range": int(((df[col] < 0) | (df[col] > hi)).sum())})
    mismatch = df[df["total_score"] != df["sum_dims"]]
    checks.append({"field": "total_score == sum(6 dims)", "n_missing": 0, "n_non_integer": 0,
                   "n_out_of_range": len(mismatch)})
    print("\n**Integrity** (n_out_of_range ở dòng cuối = số mẫu total ≠ Σ6 chiều):\n")
    print(md_table(pd.DataFrame(checks)))
    if len(mismatch):
        print("\nMismatched total:", ", ".join(mismatch["sample_id"]))

    print("\n**Rubric vectors (compilable,io,logic,edge,complexity,quality) theo exam:**\n")
    vec = df.assign(vector=df.apply(rubric_vector, axis=1))
    table = (vec.groupby(["exam_id", "vector", "total_score"]).agg(n=("sample_id", "size"),
             sample_ids=("sample_id", lambda s: ", ".join(x.split("-")[1] for x in s))).reset_index())
    print(md_table(table))
    for exam, sub in vec.groupby("exam_id"):
        print(f"\n- {exam}: {sub['vector'].nunique()} vector khác nhau / n={len(sub)}; "
              f"{sub['total_score'].nunique()} mức total khác nhau: {sorted(sub['total_score'].unique().tolist())}")


# ----------------------------------------------------------------------------- section 2
def report_gate(df: pd.DataFrame) -> None:
    section("2. Cấu trúc 'cổng' và các ràng buộc tất định giữa các chiều")
    vec = df.assign(vector=df.apply(rubric_vector, axis=1))
    for title, mask in [("compilable=0", vec["compilable"] == 0),
                        ("compilable=1 & logic=0", (vec["compilable"] == 1) & (vec["logic"] == 0))]:
        sub = vec[mask]
        table = sub.groupby(["exam_id", "vector"]).agg(n=("sample_id", "size"), total=("total_score", "first"),
                                                       ids=("sample_id", ", ".join)).reset_index()
        print(f"**{title}** (n={len(sub)}):\n")
        print(md_table(table))
        print()

    print("**Crosstab io_format × logic theo exam:**\n")
    ct = pd.crosstab([df["exam_id"], df["io_format"]], df["logic"]).reset_index()
    ct.columns = [str(c) if not isinstance(c, (int, np.integer)) else f"logic={c}" for c in ct.columns]
    print(md_table(ct))
    io0 = df[(df["io_format"] == 0) & (df["logic"] > 0)]["sample_id"].tolist()
    io1 = df[(df["io_format"] == 1) & (df["logic"] == 0)]["sample_id"].tolist()
    print(f"\n- io_format=0 & logic>0: {len(io0)} → {io0}")
    print(f"- io_format=1 & logic=0: {len(io1)} → {io1}")

    rules = {
        "edge_case == logic/2": lambda s: s["edge_case"] * 2 == s["logic"],
        "complexity == code_quality": lambda s: s["complexity"] == s["code_quality"],
        "code_quality == 1": lambda s: s["code_quality"] == 1,
        "code_quality == [logic>0]": lambda s: s["code_quality"] == (s["logic"] > 0).astype(int),
        "complexity == [logic>0]": lambda s: s["complexity"] == (s["logic"] > 0).astype(int),
        "compilable=0 ⇒ io=logic=edge=complexity=0": lambda s: (s["compilable"] == 1)
        | (s[["io_format", "logic", "edge_case", "complexity"]].sum(axis=1) == 0),
        "compilable=0 ⇒ code_quality=0": lambda s: (s["compilable"] == 1) | (s["code_quality"] == 0),
        "logic=0 ⇒ edge_case=0": lambda s: (s["logic"] > 0) | (s["edge_case"] == 0),
    }
    rows = []
    for name, rule in rules.items():
        row = {"rule": name}
        for exam, sub in df.groupby("exam_id"):
            held = rule(sub)
            row[exam] = f"{int(held.sum())}/{len(sub)}"
            row[f"{exam} violators"] = ", ".join(sub.loc[~held, "sample_id"].str.split("-").str[1])
        rows.append(row)
    print("\n**Ràng buộc tất định (holds n/N):**\n")
    print(md_table(pd.DataFrame(rows)))


# ----------------------------------------------------------------------------- section 3
def compile_locally(src: Path, workdir: Path, link: bool) -> tuple[bool, str, Path | None]:
    """Compile with the local `g++` (Apple clang on macOS). Returns (ok, first error message, binary)."""
    binary = workdir / (src.stem + ".bin")
    cmd = ["g++", "-std=c++11", "-w", str(src)] + (["-o", str(binary)] if link else ["-fsyntax-only"])
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60, cwd=workdir)
    match = re.search(r"error: (.*)", proc.stderr)
    return proc.returncode == 0, (match.group(1)[:70] if match else ""), (binary if link and proc.returncode == 0 else None)


def report_compile(df: pd.DataFrame, root: Path, do_compile: bool) -> pd.DataFrame:
    section("3. compile_log so với nhãn compilable")
    df = df.assign(log_kind=df["compile_log"].map(log_kind))
    df["log_fail_any"] = df["log_kind"].isin(["compile_error", "link_error"])
    ct = pd.crosstab([df["exam_id"], df["log_kind"]], df["compilable"]).reset_index()
    ct.columns = [str(c) if not isinstance(c, (int, np.integer)) else f"compilable={c}" for c in ct.columns]
    print(md_table(ct))
    miss = df[df["log_fail_any"] & ~df["log_has_error"]]["sample_id"].tolist()
    print(f"\n- Log có lỗi link nhưng `log_has_error` (regex file:line:col) của loader bỏ sót: {miss}")
    disagree = df[df["log_fail_any"] == (df["compilable"] == 1)]
    print(f"- Bất đồng log_fail_any vs compilable: {len(disagree)}/{len(df)} → {disagree['sample_id'].tolist()}")

    if do_compile:
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            for r in df.itertuples():
                ok, err, _ = compile_locally(root / r.code_file, Path(tmp), link=(r.exam_type == "single_problem"))
                rows.append({"sample_id": r.sample_id, "clang_ok": ok, "clang_first_error": err})
        df = df.merge(pd.DataFrame(rows), on="sample_id")
        print("\n**Biên dịch lại bằng Apple clang** (EX01: -fsyntax-only; EX02: compile+link). "
              "Lưu ý: clang ≠ GNU g++ 13 dùng để sinh compile_log.\n")
        tab = pd.crosstab([df["exam_id"], df["clang_ok"]], [df["log_fail_any"], df["compilable"]])
        tab.columns = [f"log_fail={a},compilable={b}" for a, b in tab.columns]
        print(md_table(tab.reset_index()))
        diff = df[df["clang_ok"] == df["log_fail_any"]]
        if len(diff):
            print("\nMẫu clang và compile_log (GNU) cho kết quả khác nhau:\n")
            print(md_table(diff[["sample_id", "log_kind", "clang_ok", "clang_first_error", "compilable"]]))
    return df


# ----------------------------------------------------------------------------- section 4
def run_binary(binary: Path, stdin: str, cwd: Path) -> tuple[str, int | None]:
    try:
        proc = subprocess.run([str(binary)], input=stdin + "\n", capture_output=True, text=True,
                              errors="replace", timeout=RUN_TIMEOUT_S, cwd=cwd)
        return proc.stdout, proc.returncode
    except subprocess.TimeoutExpired:
        return TIMEOUT_MARK, None


def is_fully_commented(code: str) -> bool:
    return bool(code.strip()) and not strip_comments(code).strip()


def local_run_ex02(ex2: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Compile+run each safe EX02 submission on the public inputs and extra probes (clang, 2s timeout)."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for r in ex2.itertuples():
            code = r.code or ""
            row: dict = {"sample_id": r.sample_id}
            if UNSAFE_RE.search(code):
                rows.append({**row, "local_status": "skipped_unsafe"})
                continue
            src = root / r.code_file
            counterfactual = False
            if is_fully_commented(code):
                counterfactual = True
                src = work / f"{r.sample_id}_uncommented.cpp"
                src.write_text(re.sub(r"^(\s*)//", r"\1", code, flags=re.M), encoding="utf-8")
            ok, err, binary = compile_locally(src, work, link=True)
            row.update({"local_status": "built" if ok else f"build_fail: {err}", "uncommented_run": counterfactual})
            if binary is None:
                rows.append(row)
                continue
            tests = r.test_report if isinstance(r.test_report, list) and r.test_report else [
                {"input": i, "expected": e} for i, e in [("1634", "True"), ("123", "False"), ("153", "True"),
                                                          ("9474", "True"), ("9", "True"), ("10", "False"),
                                                          ("1", "True"), ("370", "True")]]
            strict = lenient = same_as_report = crashes = timeouts = lead_ws = 0
            for t in tests:
                # After two timeouts assume the program hangs on every input (keeps full-dataset runs short).
                out, code_rc = (TIMEOUT_MARK, None) if timeouts >= 2 else run_binary(binary, t["input"], work)
                timeouts += out == TIMEOUT_MARK
                crashes += code_rc is not None and code_rc < 0
                lead_ws += out[:1].isspace() if out else 0
                strict += out.strip() == t["expected"]
                lenient += normalize_lenient(out) == t["expected"]
                if "actual" in t:
                    same_as_report += (out.strip() if out != TIMEOUT_MARK else out) == t["actual"]
            probe_pass = 0 if timeouts >= 2 else sum(
                normalize_lenient(run_binary(binary, p, work)[0]) == armstrong_reference(int(p))
                for p in EX02_EXTRA_PROBES)
            row.update({"local_strict": strict, "local_lenient": lenient, "local_eq_report": same_as_report,
                        "local_crash": crashes, "local_timeout": timeouts, "stdout_leading_ws": lead_ws,
                        "probe_pass": f"{probe_pass}/{len(EX02_EXTRA_PROBES)}"})
            rows.append(row)
    return pd.DataFrame(rows)


def report_ex02_tests(df: pd.DataFrame, root: Path, do_run: bool) -> pd.DataFrame:
    section("4. EX02: test_report so với rubric")
    ex2 = df[df["exam_type"] == "single_problem"].copy()
    if ex2.empty:
        print("Không có bài single_problem.")
        return ex2

    def report_counts(report: object) -> pd.Series:
        tests = report if isinstance(report, list) else []
        actual = [t.get("actual", "") for t in tests]
        return pd.Series({
            "n_lenient": sum(normalize_lenient(a) == t.get("expected") for a, t in zip(actual, tests)),
            "n_timeout": sum(a == TIMEOUT_MARK for a in actual),
            "n_empty": sum(a == "" for a in actual),
            "has_prompt": any(":" in a for a in actual),
            "const_output": len(set(actual)) == 1 and len(actual) > 1,
        })

    ex2 = pd.concat([ex2, ex2["test_report"].apply(report_counts)], axis=1)
    ex2["n_passed0"] = ex2["n_passed"].fillna(0)  # compile failures have an empty report -> 0 passed
    inputs = sorted({t["input"] for r in ex2["test_report"] if isinstance(r, list) for t in r}, key=lambda s: int(s))
    print(f"Public test inputs (union): {inputs}; max digits = {max(len(i) for i in inputs)} "
          f"(ràng buộc đề: n ≤ 10^9). n_tests theo mẫu: {ex2['n_tests'].value_counts().to_dict()}")

    rows = []
    for x in ["n_passed0", "n_lenient"]:
        for y in ["logic", "edge_case", "io_format", "total_score"]:
            for scope, sub in [("all EX02", ex2), ("built (n_tests>0)", ex2[ex2["n_tests"] > 0])]:
                rho, p, n = safe_spearman(sub[x].astype(float), sub[y].astype(float))
                rows.append({"x": x, "y": y, "scope": scope, "n": n, "spearman_rho": rho, "p_value": p})
    print("\n**Spearman** (n_passed0 = số test pass, compile fail tính 0; n_lenient = pass sau khi bỏ prompt 'xxx:'):\n")
    print(md_table(pd.DataFrame(rows)))

    print("\n**Crosstab n_passed (report gốc) × total_score:**\n")
    ct = pd.crosstab(ex2["n_passed0"].astype(int), ex2["total_score"])
    ct.columns = [f"total={c}" for c in ct.columns]
    print(md_table(ct.reset_index().rename(columns={"n_passed0": "n_passed"})))

    if do_run:
        ex2 = ex2.merge(local_run_ex02(ex2, root), on="sample_id", how="left")
    cols = ["sample_id", "compilable", "io_format", "logic", "edge_case", "complexity", "total_score", "log_kind",
            "n_tests", "n_passed0", "n_lenient", "n_timeout", "n_empty", "has_prompt", "const_output"]
    cols += [c for c in ["local_status", "uncommented_run", "local_strict", "local_lenient", "local_eq_report",
                         "local_crash", "local_timeout", "stdout_leading_ws", "probe_pass"] if c in ex2.columns]
    print("\n**Per-sample EX02** (local_* = chạy lại bằng clang trên input công khai; probe_pass = "
          f"{len(EX02_EXTRA_PROBES)} input thêm trong miền {EX02_EXTRA_PROBES}):\n")
    print(md_table(ex2[cols]))

    anomalies = {
        "total=10 nhưng n_passed<8 (gồm n_tests=0)": ex2[(ex2["total_score"] == TOTAL_MAX) & (ex2["n_passed0"] < 8)],
        "pass 8/8 nhưng total<10": ex2[(ex2["n_passed0"] == 8) & (ex2["total_score"] < TOTAL_MAX)],
        "total≥7 nhưng n_passed=0": ex2[(ex2["total_score"] >= 7) & (ex2["n_passed0"] == 0)],
        "complexity=1 nhưng có TIMEOUT": ex2[(ex2["complexity"] == 1) & (ex2["n_timeout"] > 0)],
        "io_format=1 nhưng output có prompt": ex2[(ex2["io_format"] == 1) & ex2["has_prompt"]],
        "compilable=1 nhưng log lỗi compile/link": ex2[(ex2["compilable"] == 1)
                                                      & ex2["log_kind"].isin(["compile_error", "link_error"])],
    }
    print("\n**Bất thường:**\n")
    for name, sub in anomalies.items():
        print(f"- {name}: {len(sub)}/{len(ex2)} → {sub['sample_id'].tolist()}")
    logic2 = ex2[ex2["logic"] == 2]
    print(f"- logic=2 (total=7) trải trên n_passed = {sorted(logic2['n_passed0'].astype(int).tolist())} "
          f"(n={len(logic2)})")
    return ex2


# ----------------------------------------------------------------------------- section 5
def parse_feedback_status(feedback: str) -> dict:
    """Heuristic per-problem status from teacher feedback (EDA only; feedback is a label, never a model input)."""
    text = nfc(feedback)
    status = {p: "unmentioned" for p in PIDS}
    marks = list(re.finditer(r"\bP([1-4])\b", text))
    pending: list[str] = []
    for i, m in enumerate(marks):
        seg = text[m.end() : marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        pid = f"P{m.group(1)}"
        if not re.sub(r"[\s,;:và]", "", seg):  # "P3, P4 không làm" -> P3 inherits P4's status
            pending.append(pid)
            continue
        # An error mention wins over "không làm" ("P1 lỗi stof ... SV không làm các câu còn lại").
        if FB_WRONG_RE.search(seg):
            value = "wrong"
        elif FB_NOT_DONE_RE.search(seg):
            value = "not_done"
        elif FB_CORRECT_RE.search(seg):
            value = "correct"
        else:
            value = "unknown"
        for p in pending + [pid]:
            status[p] = value
        pending = []
    status["gate_mentioned"] = bool(FB_GATE_RE.search(text))
    status["inf_loop_mentioned"] = bool(FB_INF_LOOP_RE.search(text))
    return status


def statement_weights(exam: dict) -> dict[str, float]:
    found = STMT_WEIGHT_RE.findall(nfc(exam.get("statement", "")))
    return {f"P{k}": float(v.replace(",", ".")) for k, v in found}


def policy_score(status: dict, weights: dict[str, float], compile_fail: bool, inf_loop: bool) -> float:
    """Score implied by the written rules: compile fail / infinite loop -> 0; P1 not correct -> 0 (gate)."""
    if compile_fail or inf_loop or status.get("P1") != "correct":
        return 0.0
    return float(sum(w for p, w in weights.items() if status.get(p) == "correct"))


def prototype_status(code: str, name: str) -> str:
    body = function_body(code, name)
    if body is not None:
        return "defined" if body.strip() else "empty_body"
    if re.search(rf"^\s*[\w:<>]+[\s\*&]+{name}\s*\([^)]*\)\s*;", code, flags=re.M):
        return "declared_only"
    return "absent"


def report_ex01(df: pd.DataFrame, exams: dict) -> pd.DataFrame:
    section("5. EX01 (multi_problem): trọng số, prototype, trạng thái từng câu và chính sách cổng P1")
    ex1 = df[df["exam_type"] == "multi_problem"].copy()
    for exam_id in ex1["exam_id"].unique():
        exam = exams[exam_id]
        stmt = statement_weights(exam)
        meta = {p["pid"]: p.get("max_score") for p in exam.get("problems", [])}
        table = pd.DataFrame([{"pid": p, "statement_weight": stmt.get(p), "problems[].max_score": meta.get(p)}
                              for p in sorted(set(stmt) | set(meta))])
        print(f"**{exam_id}: trọng số trong statement vs problems[].max_score**\n")
        print(md_table(table))
        print(f"\nΣ statement = {sum(stmt.values())}, Σ max_score = {sum(v or 0 for v in meta.values())}; "
              f"grading_policy: {exam.get('grading_policy')!r}")

    exam = exams[ex1["exam_id"].iloc[0]]
    weights_stmt = statement_weights(exam)
    weights_meta = {p["pid"]: float(p["max_score"]) for p in exam["problems"]}
    names = {p["pid"]: re.search(r"(\w+)\s*\(", p["prototype"]).group(1) for p in exam["problems"]}

    rows = []
    for r in ex1.itertuples():
        code = strip_comments(r.code or "")
        parsed = parse_feedback_status(r.feedback or "")
        protos = {p: prototype_status(code, names[p]) for p in PIDS}
        for p in PIDS:  # fill silent problems from code structure so the parser alone works on the full set
            if parsed[p] != "unmentioned":
                continue
            if protos[p] in ("absent", "declared_only", "empty_body"):
                parsed[p] = "not_done"
            elif p != "P1" and parsed["P1"] == "wrong" and parsed["gate_mentioned"]:
                parsed[p] = "ungraded"
        manual = MANUAL_EX01.get(r.sample_id)
        final = {p: (manual or parsed)[p] for p in PIDS}
        inf_loop = manual["inf_loop"] if manual else parsed["inf_loop_mentioned"]
        p1_body = function_body(code, names["P1"]) or ""
        row = {
            "sample_id": r.sample_id,
            "protos": "".join({"defined": "D", "empty_body": "E", "declared_only": "d", "absent": "-"}[protos[p]]
                              for p in PIDS),
            "has_main": bool(re.search(r"\bint\s+main\s*\(", code)),
            "p1_uses_caller_n": bool(re.search(r"(new\s+General\s*\[\s*n\s*\]|General\s+\w+\s*\[\s*n\s*\])", p1_body))
            and not re.search(r"(?<![\w.>])n\s*(=(?!=)|\+\+|\+=)|\+\+\s*n\b", p1_body),
            "p1_literal_path": bool(re.search(r"(open\s*\(\s*\"|stream\s+\w+\s*\(\s*\")", p1_body)),
            **{f"fb_{p}": parsed[p] for p in PIDS},
            "fb_gate": parsed["gate_mentioned"],
            **{p: final[p] for p in PIDS},
            "inf_loop": inf_loop,
            "compilable": r.compilable, "io": r.io_format, "logic": r.logic, "edge": r.edge_case,
            "rubric_total": r.total_score,
            "policy_stmt_1/4/2/3": policy_score(final, weights_stmt, r.compilable == 0, inf_loop),
            "policy_meta_2.5": policy_score(final, weights_meta, r.compilable == 0, inf_loop),
            "raw_stmt_no_gate": float(sum(w for p, w in weights_stmt.items() if final[p] == "correct")),
        }
        row["violates_P1_gate"] = final["P1"] == "wrong" and r.logic > 0
        rows.append(row)
    audit = pd.DataFrame(rows)
    print("\n**Per-sample EX01** (protos: D=định nghĩa, d=chỉ khai báo, E=thân rỗng, -=không có, theo P1..P4; "
          "fb_* = parse tự động từ feedback; P1..P4 = trạng thái cuối (đọc tay code+feedback); "
          "policy_* = điểm theo quy định đề):\n")
    print(md_table(audit))

    n_p1_wrong = int((audit["P1"] == "wrong").sum())
    viol = audit[audit["violates_P1_gate"]]
    print(f"\n- P1 sai (đọc tay): {n_p1_wrong}/{len(audit)}; trong đó logic>0 (vi phạm chính sách cổng): "
          f"{len(viol)}/{n_p1_wrong} → {viol['sample_id'].tolist()}")
    print(f"- Có `int main` (ngoài comment): {int(audit['has_main'].sum())}/{len(audit)}")
    print(f"- prob1 dùng n do caller truyền mà không gán n: "
          f"{audit.loc[audit['p1_uses_caller_n'], ['sample_id', 'fb_P1', 'rubric_total']].values.tolist()}")
    pairs = pd.concat([audit[[f"fb_{p}", p]].set_axis(["parsed", "manual"], axis=1) for p in PIDS])
    known = pairs[~pairs["parsed"].isin(["unmentioned", "unknown"]) & (pairs["manual"] != "unknown")]
    print(f"- Parser feedback(+cấu trúc code) khớp đọc tay: {int((known['parsed'] == known['manual']).sum())}/"
          f"{len(known)} ô xác định; ô parser không quyết được: {len(pairs) - len(known)}/{len(pairs)}")
    for col in ["policy_stmt_1/4/2/3", "policy_meta_2.5", "raw_stmt_no_gate"]:
        rho, p, n = safe_spearman(audit[col], audit["rubric_total"].astype(float))
        mae = mean_absolute_error(audit["rubric_total"], audit[col])
        print(f"- rubric_total vs {col}: Spearman ρ={rho:.3f} (p={p:.3g}, n={n}); MAE={mae:.2f}")
    built = audit[audit["compilable"] == 1]
    rho, p, n = safe_spearman(built["raw_stmt_no_gate"], built["logic"].astype(float))
    print(f"- (compilable=1) logic vs raw_stmt_no_gate: ρ={rho:.3f} (p={p:.3g}, n={n})")
    return audit


# ----------------------------------------------------------------------------- section 6
def baseline_predictions(df: pd.DataFrame, ex2: pd.DataFrame) -> dict[str, np.ndarray]:
    lenient = df["sample_id"].map(ex2.set_index("sample_id")["n_lenient"]) if "n_lenient" in ex2 else None
    fail_any = df["compile_log"].map(log_kind).isin(["compile_error", "link_error"])
    is_ex2 = df["exam_type"] == "single_problem"
    rate = (df["n_passed"] / df["n_tests"]).where(df["n_tests"] > 0, 0.0).fillna(0.0)
    rate_len = ((lenient / df["n_tests"]).where(df["n_tests"] > 0, 0.0).fillna(0.0)) if lenient is not None else rate
    compile_rule = np.where(fail_any, 0, TOTAL_MAX)
    preds = {
        "const_5 (a priori)": np.full(len(df), 5),
        "const_median(gold)": np.full(len(df), int(np.median(df["total_score"]))),
        "exam_median(gold)": df.groupby("exam_id")["total_score"].transform("median").round().astype(int).to_numpy(),
        "compile_rule_loader (0/10)": np.where(df["log_has_error"], 0, TOTAL_MAX),
        "compile_rule_strict (0/10)": compile_rule,
        "pass_rate_strict": np.where(is_ex2, np.where(fail_any, 0, np.rint(TOTAL_MAX * rate)), compile_rule),
        "pass_rate_lenient": np.where(is_ex2, np.where(fail_any, 0, np.rint(TOTAL_MAX * rate_len)), compile_rule),
        "rubric_prior (2+8·lenient)": np.where(is_ex2, np.where(fail_any, 0, 2 + np.rint(8 * rate_len)), compile_rule),
    }
    return {k: np.asarray(v, dtype=int) for k, v in preds.items()}


def report_qwk(df: pd.DataFrame, ex2: pd.DataFrame, n_boot: int, suspect_ids: list[str]) -> None:
    section("6. Metric: QWK của baseline không dùng nhãn gold làm input")
    y = df["total_score"].to_numpy(dtype=int)
    preds = baseline_predictions(df, ex2)
    rows = []
    for name, p in preds.items():
        row = {"baseline": name, "QWK_pooled labels=None": qwk(y, p, None),
               "QWK_pooled labels=0..10": qwk(y, p, SCORE_LABELS)}
        for exam in sorted(df["exam_id"].unique()):
            m = (df["exam_id"] == exam).to_numpy()
            row[f"{exam} None"] = qwk(y[m], p[m], None)
            row[f"{exam} 0..10"] = qwk(y[m], p[m], SCORE_LABELS)
        row.update({"MAE": mean_absolute_error(y, p), "exact": float((y == p).mean())})
        rows.append(row)
    print(md_table(pd.DataFrame(rows)))
    print("\n(QWK=NaN khi 1 phía hằng số trên tập con → mẫu số 0.)")

    keep = ~df["sample_id"].isin(suspect_ids).to_numpy()
    print(f"\n**Độ nhạy với nhãn nghi sai** (bỏ {len(suspect_ids)} mẫu: {suspect_ids}; labels=0..10):\n")
    rows = [{"baseline": name, "QWK all": qwk(y, p, SCORE_LABELS), "QWK w/o suspects": qwk(y[keep], p[keep], SCORE_LABELS),
             "MAE all": mean_absolute_error(y, p), "MAE w/o suspects": mean_absolute_error(y[keep], p[keep])}
            for name, p in preds.items() if not name.startswith("const")]
    print(md_table(pd.DataFrame(rows)))

    best = max(preds, key=lambda k: np.nan_to_num(qwk(y, preds[k], SCORE_LABELS), nan=-9))
    rng = np.random.default_rng(0)
    for name in dict.fromkeys([best, "compile_rule_strict (0/10)"]):
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            v = qwk(y[idx], preds[name][idx], SCORE_LABELS)
            if np.isfinite(v):
                vals.append(v)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"- Bootstrap {n_boot}× (labels=0..10) `{name}`: QWK={qwk(y, preds[name], SCORE_LABELS):.3f}, "
              f"95% CI [{lo:.3f}, {hi:.3f}]")

    print("\n**Độ nhạy với `labels` trên dữ liệu thật (cùng một dự đoán):**\n")
    sens = []
    for exam in sorted(df["exam_id"].unique()):
        m = (df["exam_id"] == exam).to_numpy()
        yt = y[m]
        present = sorted(set(yt.tolist()))
        if len(present) < 3:
            continue
        # Move every sample at the upper end of the widest gap between observed levels down to the lower end.
        _, lo_level, hi_level = max((b - a, a, b) for a, b in zip(present, present[1:]))
        pert = np.where(yt == hi_level, lo_level, yt)
        sens.append({"exam": exam, "observed_levels": str(present), "perturbation": f"{hi_level}→{lo_level}",
                     "n_changed": int((yt == hi_level).sum()), "point_error": hi_level - lo_level,
                     "QWK labels=None": qwk(yt, pert, None), "QWK labels=0..10": qwk(yt, pert, SCORE_LABELS)})
    print(md_table(pd.DataFrame(sens)))
    print("\n(labels=None → sklearn dùng hợp các giá trị quan sát được làm trục; trọng số bậc hai tính theo "
          "*chỉ số* trong danh sách, nên 2 mức kề nhau trong danh sách luôn cách nhau 1 bước dù chênh nhiều điểm.)")

    section("6b. Exact-match từng chiều của dự đoán hằng số (mode theo exam)")
    rows = []
    for exam, sub in df.groupby("exam_id"):
        row = {"exam": exam, "n": len(sub)}
        for d in DIMS:
            row[d] = float((sub[d] == sub[d].mode().iloc[0]).mean())
        rows.append(row)
    row = {"exam": "ALL (mode pooled)", "n": len(df)}
    for d in DIMS:
        row[d] = float((df[d] == df[d].mode().iloc[0]).mean())
    rows.append(row)
    print(md_table(pd.DataFrame(rows)))


# ----------------------------------------------------------------------------- figures
def save_figures(ex2: pd.DataFrame, audit: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ink, muted, accent, accent2 = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
    paths = []

    if not ex2.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        rng = np.random.default_rng(1)
        jitter = rng.uniform(-0.12, 0.12, len(ex2))
        flag = (ex2["total_score"] >= 7) & (ex2["n_passed0"] <= 2) | ((ex2["n_passed0"] == 8) & (ex2["total_score"] < 10))
        ax.scatter(ex2["n_passed0"] + jitter, ex2["total_score"], s=46, c=np.where(flag, accent2, accent),
                   edgecolors="white", linewidths=1.5, zorder=3)
        # One label per (passed, total) cell so coincident points do not overprint each other.
        for (x_val, y_val), grp in ex2[flag].groupby(["n_passed0", "total_score"]):
            ax.annotate(", ".join(grp["sample_id"].str.split("-").str[1]), (x_val, y_val), xytext=(10, -3),
                        textcoords="offset points", fontsize=8, color=ink)
        ax.set_xlabel("public tests passed (report, compile fail = 0)", color=muted)
        ax.set_ylabel("gold total_score", color=muted)
        ax.set_title("EX02: tests passed vs gold total (orange = contradiction)", fontsize=10, color=ink, loc="left")
        ax.set_xticks(range(0, 9))
        ax.set_yticks(range(0, 11))
        ax.grid(alpha=0.25)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        path = out_dir / "t1_ex02_passed_vs_total.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    if not audit.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        a = audit.sort_values("rubric_total").reset_index(drop=True)
        x = np.arange(len(a))
        for i, r in a.iterrows():
            ax.plot([i, i], [r["policy_stmt_1/4/2/3"], r["rubric_total"]], color="#c9c8c2", lw=2, zorder=1)
        ax.scatter(x, a["rubric_total"], s=46, c=accent, label="gold rubric total", zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.scatter(x, a["policy_stmt_1/4/2/3"], s=80, facecolors="none", edgecolors=accent2, linewidths=2,
                   label="policy score (weights 1/4/2/3, P1 gate)", zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(a["sample_id"].str.split("-").str[1], rotation=60, fontsize=8)
        ax.set_ylabel("score (0-10)", color=muted)
        ax.set_yticks(range(0, 11))
        ax.set_title("EX01: gold rubric total vs score implied by the written policy", fontsize=10, color=ink,
                     loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.25)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        path = out_dir / "t1_ex01_rubric_vs_policy.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "out" / "t1_")
    parser.add_argument("--no-compile", action="store_true", help="skip local clang -fsyntax-only/link check")
    parser.add_argument("--no-run", action="store_true", help="skip running EX02 binaries")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    root = args.data_root.resolve()
    df = load_samples(root)
    _, exams, _ = load_meta(root)
    print(f"# Task 1 EDA — n={len(df)} mẫu; theo exam: {df['exam_id'].value_counts().sort_index().to_dict()}")

    report_distributions(df)
    report_gate(df)
    df = report_compile(df, root, do_compile=not args.no_compile)
    ex2 = report_ex02_tests(df, root, do_run=not args.no_run)
    audit = report_ex01(df, exams) if (df["exam_type"] == "multi_problem").any() else pd.DataFrame()
    suspects = []
    if not ex2.empty:
        suspects += ex2.loc[(ex2["compilable"] == 1) & ex2["log_kind"].isin(["compile_error", "link_error"]),
                            "sample_id"].tolist()
    if not audit.empty:
        suspects += audit.loc[audit["violates_P1_gate"], "sample_id"].tolist()
    report_qwk(df, ex2, args.bootstrap, suspects)
    if not args.no_figures:
        paths = save_figures(ex2, audit, args.out_dir)
        print("\nFigures:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
