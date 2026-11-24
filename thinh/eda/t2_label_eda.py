"""EDA for Task 2 (multi-label error taxonomy).

Covers: label-string integrity, prevalence / cardinality / combinations, co-occurrence,
consistency of labels with the Task 1 rubric and with input signals (compile_log,
test_report), cross-task leakage (Task 2 output vs Task 3 input), and macro-F1
sensitivity (zero_division conventions, label-set choice, small-test-set instability).

Run:
    .venv/bin/python t2_label_eda.py --data-root /path/to/sample_dataset [--compile] [--exec]

--compile  re-checks every submission with the local compiler (Apple clang on macOS,
           NOT GNU g++ used for compile_log): EX01 with -fsyntax-only, EX02 compile+link
           into a temp dir. Nothing is executed.
--exec     for EX02 submissions whose source is entirely commented out, uncomments it,
           and (only if a static grep finds no dangerous calls) runs the public tests in
           a temp dir with a 2 s timeout per test.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binom
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import DIMS, load_meta, load_samples  # noqa: E402

LBL_COMPILE = "Lỗi biên dịch"
LBL_IO = "Lỗi nhập/xuất"
LBL_LOGIC = "Lỗi logic"
LBL_LOOP = "Lỗi vòng lặp"
LBL_EDGE = "Lỗi edge case"
LBL_ALGO = "Lỗi thuật toán"
LBL_STYLE = "Lỗi style"
ZERO_DIVISIONS: list[tuple[str, float]] = [("0", 0.0), ("1", 1.0), ("nan", np.nan)]
# Linker failures ("collect2: error: ld returned 1 exit status") are invisible to the
# shared loader regex, which only matches "file:line:col: error:" diagnostics.
LINK_ERROR_RE = re.compile(r"(collect2|ld): error|undefined reference to|ld returned \d+ exit status")
EDGE_WORDS_RE = re.compile(r"rỗng|\bBC\b|trường hợp|đặc biệt|edge|tràn|số âm", re.I)
DANGEROUS_RE = re.compile(r"system\s*\(|popen|fork\s*\(|\bexec\w*\s*\(|remove\s*\(|rename\s*\(|ofstream|fopen|socket")
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


# ---------------------------------------------------------------- formatting helpers
def md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:.3f}") -> str:
    frame = df.reset_index() if index else df
    def cell(v: object) -> str:
        if isinstance(v, float):
            return "nan" if np.isnan(v) else floatfmt.format(v)
        return str(v).replace("|", "\\|")
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def section(title: str) -> None:
    print(f"\n## {title}\n")


def short(label: str) -> str:
    return label.replace("Lỗi ", "")


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "0/0"


# ---------------------------------------------------------------- data preparation
def is_fully_commented(code: str | None) -> bool:
    if not isinstance(code, str):
        return False
    stripped = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return all(not ln.strip() or ln.strip().startswith("//") for ln in stripped.splitlines())


def extra_output_text(report: object) -> bool:
    """A failing test whose output still contains the expected token (prompt / decoration)."""
    if not isinstance(report, list):
        return False
    return any(
        not t.get("passed") and t.get("expected", "").strip()
        and t["expected"].strip() in t.get("actual", "") and t["actual"].strip() != t["expected"].strip()
        for t in report
    )


def any_timeout(report: object) -> bool:
    return isinstance(report, list) and any("TIMEOUT" in str(t.get("actual", "")) for t in report)


def prepare(root: Path) -> tuple[pd.DataFrame, list[str]]:
    label_space, _, _ = load_meta(root)
    labels = list(label_space["taxonomy_error"])
    df = load_samples(root)
    df["label_set"] = df["labels"].map(lambda xs: frozenset(unicodedata.normalize("NFC", x.strip()) for x in xs))
    for lab in labels:
        df[lab] = df["label_set"].map(lambda s, lab=lab: lab in s).astype(int)
    df["is_empty"] = df["label_set"].map(len).eq(0)
    log = df["compile_log"].fillna("")
    df["log_link_error"] = log.map(lambda s: bool(LINK_ERROR_RE.search(s)))
    df["log_any_error"] = df["log_has_error"] | df["log_link_error"]
    df["fully_commented"] = df["code"].map(is_fully_commented)
    df["extra_output_text"] = df["test_report"].map(extra_output_text)
    df["any_timeout"] = df["test_report"].map(any_timeout)
    df["edge_words_in_feedback"] = df["feedback"].fillna("").map(lambda s: bool(EDGE_WORDS_RE.search(s)))
    return df, labels


# ---------------------------------------------------------------- 1. label integrity
def label_integrity(root: Path, df: pd.DataFrame, labels: list[str]) -> None:
    section("1. Label-string integrity (exact / NFC / NFD / whitespace / order)")
    _, _, tasks = load_meta(root)
    header_space = tasks["t2"].get("label_space")
    print(f"- label_space.json taxonomy == task2 header label_space (ordered): {header_space == labels}")
    print(f"- label_space strings all NFC: {all(unicodedata.is_normalized('NFC', x) for x in labels)}; "
          f"NFD differs from NFC for {sum(unicodedata.normalize('NFD', x) != x for x in labels)}/{len(labels)} labels")
    order = {lab: i for i, lab in enumerate(labels)}
    rows = []
    for col in ("labels", "t3_labels"):
        occ = [x for xs in df[col] if isinstance(xs, list) for x in xs]
        lists = [xs for xs in df[col] if isinstance(xs, list)]
        rows.append({
            "field": col,
            "occurrences": len(occ),
            "exact_in_label_space": sum(x in order for x in occ),
            "is_NFC": sum(unicodedata.is_normalized("NFC", x) for x in occ),
            "NFD_only": sum(not unicodedata.is_normalized("NFC", x) for x in occ),
            "whitespace_issue": sum(x != x.strip() or "  " in x for x in occ),
            "casefold_only_match": sum(x not in order and x.casefold() in {k.casefold() for k in order} for x in occ),
            "lists_with_duplicates": sum(len(xs) != len(set(xs)) for xs in lists),
            "lists_not_canonical_order": sum(
                [order.get(x, 99) for x in xs] != sorted(order.get(x, 99) for x in xs) for xs in lists),
        })
    print(md_table(pd.DataFrame(rows), index=False))
    unordered = df.loc[df["labels"].map(lambda xs: [order.get(x, 99) for x in xs] != sorted(order.get(x, 99) for x in xs)),
                       "sample_id"].tolist()
    print(f"\nSample_ids whose label list is not in label_space order: {unordered}")


# ---------------------------------------------------------------- 2. prevalence
def prevalence(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    section("2. Label prevalence (overall and per exam)")
    exams = sorted(df["exam_id"].unique())
    rows = []
    for lab in labels:
        row = {"label": lab, "n": int(df[lab].sum()), "pct": 100 * df[lab].mean()}
        for ex in exams:
            sub = df[df["exam_id"] == ex]
            row[f"{ex} n/{len(sub)}"] = int(sub[lab].sum())
        rows.append(row)
    prev = pd.DataFrame(rows).set_index("label")
    print(md_table(prev, floatfmt="{:.1f}"))
    zero = prev.index[prev["n"] == 0].tolist()
    print(f"\n- Labels with 0 positives: {zero}")
    for ex in exams:
        sub = df[df["exam_id"] == ex]
        absent = [short(lab) for lab in labels if sub[lab].sum() == 0]
        print(f"- {ex}: {len(absent)}/{len(labels)} labels absent -> {absent}")

    section("2b. Labels per sample, empty-set rate, distinct combinations")
    card = df.groupby(["exam_id", "n_labels"]).size().unstack(0, fill_value=0)
    card["all"] = card.sum(axis=1)
    print(md_table(card))
    print(f"\n- mean labels/sample: {df['n_labels'].mean():.3f} "
          f"(EX01 {df.loc[df.exam_id == 'EX01', 'n_labels'].mean():.3f}, EX02 {df.loc[df.exam_id == 'EX02', 'n_labels'].mean():.3f})")
    print(f"- empty set: {pct(int(df['is_empty'].sum()), len(df))} -> {df.loc[df.is_empty, 'sample_id'].tolist()}")
    combos = df["label_set"].map(lambda s: " + ".join(short(x) for x in labels if x in s) or "[]")
    vc = combos.value_counts()
    print(f"- distinct label combinations: {len(vc)} over {len(df)} samples; singletons (seen once): {int((vc == 1).sum())}")
    print(md_table(vc.rename("count").to_frame().rename_axis("combination")))
    return prev


# ---------------------------------------------------------------- 3. co-occurrence
def cooccurrence(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    section("3. Co-occurrence P(B | A) (row = A, col = B; diagonal = count of A)")
    present = [lab for lab in labels if df[lab].sum() > 0]
    y = df[present].to_numpy()
    counts = y.T @ y
    cond = counts / np.diag(counts)[:, None]
    tab = pd.DataFrame(cond, index=[short(x) for x in present], columns=[short(x) for x in present])
    for i, lab in enumerate(present):
        tab.iloc[i, i] = float(counts[i, i])
    print(md_table(tab, floatfmt="{:.2f}"))
    pairs = [(short(a), short(b), int(counts[i, j])) for (i, a), (j, b) in combinations(enumerate(present), 2) if counts[i, j]]
    print("\nNon-zero pairs (count): " + ", ".join(f"{a}+{b}={c}" for a, b, c in sorted(pairs, key=lambda t: -t[2])))
    return tab


# ---------------------------------------------------------------- 4. consistency
def crosstab_row(name: str, scope: pd.DataFrame, label: pd.Series, signal: pd.Series) -> dict:
    lab, sig = label.astype(bool), signal.astype(bool)
    both, lab_only, sig_only, neither = (lab & sig).sum(), (lab & ~sig).sum(), (~lab & sig).sum(), (~lab & ~sig).sum()
    kappa = cohen_kappa_score(lab, sig) if lab.nunique() > 1 or sig.nunique() > 1 else np.nan
    return {
        "pair (label ~ signal)": name, "n": len(scope), "L1S1": int(both), "L1S0": int(lab_only),
        "L0S1": int(sig_only), "L0S0": int(neither), "agree%": 100 * (both + neither) / len(scope),
        "kappa": float(kappa),
        "label_without_signal": ",".join(scope.loc[lab & ~sig, "sample_id"]) or "-",
        "signal_without_label": ",".join(scope.loc[~lab & sig, "sample_id"]) or "-",
    }


def consistency(df: pd.DataFrame) -> pd.DataFrame:
    section("4. Label vs rubric / input-signal consistency (L=label present, S=signal true)")
    ok = df[df["compilable"] == 1]
    ex2 = df[df["exam_id"] == "EX02"]
    ex2_tests = ex2[ex2["n_tests"].fillna(0) > 0]
    specs = [
        ("Lỗi biên dịch ~ compilable=0", df, df[LBL_COMPILE], df["compilable"] == 0),
        ("Lỗi biên dịch ~ log error (loader regex)", df, df[LBL_COMPILE], df["log_has_error"]),
        ("Lỗi biên dịch ~ log error incl. linker", df, df[LBL_COMPILE], df["log_any_error"]),
        ("compilable=0 ~ log error incl. linker", df, df["compilable"] == 0, df["log_any_error"]),
        ("[] ~ total=10", df, df["is_empty"], df["total_score"] == 10),
        ("[] ~ all public tests pass (EX02, tests>0)", ex2_tests, ex2_tests["is_empty"], ex2_tests["pass_rate"] == 1),
        ("Lỗi nhập/xuất ~ io_format=0", df, df[LBL_IO], df["io_format"] == 0),
        ("Lỗi nhập/xuất ~ io_format=0 | compilable=1", ok, ok[LBL_IO], ok["io_format"] == 0),
        ("Lỗi nhập/xuất ~ extra text in output (EX02)", ex2, ex2[LBL_IO], ex2["extra_output_text"]),
        ("io_format=0 ~ extra text in output (EX02)", ex2, ex2["io_format"] == 0, ex2["extra_output_text"]),
        ("Lỗi edge case ~ edge_case<2", df, df[LBL_EDGE], df["edge_case"] < 2),
        ("Lỗi edge case ~ edge_case<2 | compilable=1", ok, ok[LBL_EDGE], ok["edge_case"] < 2),
        ("Lỗi edge case ~ edge words in feedback", df, df[LBL_EDGE], df["edge_words_in_feedback"]),
        ("Lỗi style ~ code_quality=0", df, df[LBL_STYLE], df["code_quality"] == 0),
        ("Lỗi thuật toán ~ complexity=0", df, df[LBL_ALGO], df["complexity"] == 0),
        ("Lỗi thuật toán ~ complexity=0 | compilable=1", ok, ok[LBL_ALGO], ok["complexity"] == 0),
        ("Lỗi logic ~ logic<4 | compilable=1", ok, ok[LBL_LOGIC], ok["logic"] < 4),
        ("Lỗi vòng lặp ~ any TIMEOUT (EX02)", ex2, ex2[LBL_LOOP], ex2["any_timeout"]),
        ("[] ~ fully commented source", df, df["is_empty"], df["fully_commented"]),
    ]
    res = pd.DataFrame([crosstab_row(*s) for s in specs])
    print(md_table(res, index=False, floatfmt="{:.2f}"))
    print(f"\n- fully commented submissions: {df.loc[df.fully_commented, 'sample_id'].tolist()} "
          f"labels={df.loc[df.fully_commented, 'labels'].tolist()} total={df.loc[df.fully_commented, 'total_score'].tolist()}")
    print(f"- compile_log with linker error only: {df.loc[df.log_link_error & ~df.log_has_error, 'sample_id'].tolist()}")
    print(f"- edge_case<2: {pct(int((df.edge_case < 2).sum()), len(df))}; among compilable=1: "
          f"{pct(int((ok.edge_case < 2).sum()), len(ok))}; label '{LBL_EDGE}': {int(df[LBL_EDGE].sum())}")
    print(f"- feedback with edge-related words: {df.loc[df.edge_words_in_feedback, ['sample_id', 'labels']].values.tolist()}")

    section("4b. Is edge_case an independent signal? (crosstab logic x edge_case, compilable=1)")
    print(md_table(pd.crosstab([ok["exam_id"], ok["logic"]], ok["edge_case"])))

    section("4c. Mean rubric profile per label (all samples carrying the label)")
    prof = []
    for lab in [c for c in df.columns if c.startswith("Lỗi ")] + ["is_empty"]:
        sub = df[df[lab].astype(bool)]
        if len(sub):
            prof.append({"label": short(lab), "n": len(sub), **sub[DIMS + ["total_score"]].mean().round(2).to_dict(),
                         "pass_rate(EX02)": sub["pass_rate"].mean()})
    print(md_table(pd.DataFrame(prof), index=False, floatfmt="{:.2f}"))
    return res


# ---------------------------------------------------------------- 5. cross-task leakage
def leakage(df: pd.DataFrame) -> None:
    section("5. Cross-task leakage: Task 2 output vs Task 3 input, shared inputs")
    n = len(df)
    t3_sets = df["t3_labels"].map(lambda xs: frozenset(xs) if isinstance(xs, list) else None)
    same_ordered = (df["labels"] == df["t3_labels"]).sum()
    same_set = (df["label_set"] == t3_sets.map(lambda s: frozenset(unicodedata.normalize("NFC", x.strip()) for x in s))).sum()
    jacc = [len(a & b) / len(a | b) if (a | b) else 1.0 for a, b in zip(df["label_set"], t3_sets)]
    keys = ["exam_id", "exam_type", "language", "code_file", "compile_log"]
    t1_t3 = sum(all(a.get(k) == b.get(k) for k in keys) for a, b in zip(df["t1_input"], df["t3_input"]))
    rows = [
        {"comparison": "T2 output == T3 input taxonomy_error (ordered list)", "match": pct(int(same_ordered), n)},
        {"comparison": "T2 output == T3 input taxonomy_error (as set, NFC)", "match": pct(int(same_set), n)},
        {"comparison": "mean Jaccard(T2 output, T3 input)", "match": f"{np.mean(jacc):.3f}"},
        {"comparison": "T1 input == T2 input (whole dict)", "match": pct(int((df["t1_input"] == df["t2_input"]).sum()), n)},
        {"comparison": "T1 input == T3 input on shared keys", "match": pct(t1_t3, n)},
    ]
    print(md_table(pd.DataFrame(rows), index=False))


# ---------------------------------------------------------------- 6. metric simulation
def f1_per_label(y: np.ndarray, p: np.ndarray, zero_division: float) -> np.ndarray:
    tp = (y & p).sum(0).astype(float)
    denom = 2 * tp + (~y & p).sum(0) + (y & ~p).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), zero_division)


def macro_f1(y: np.ndarray, p: np.ndarray, zero_division: float) -> float:
    return float(np.nanmean(f1_per_label(y, p, zero_division)))


def to_matrix(sets: list[frozenset], labels: list[str]) -> np.ndarray:
    return np.array([[lab in s for lab in labels] for s in sets], dtype=bool)


def baselines(df: pd.DataFrame, labels: list[str]) -> dict[str, list[frozenset]]:
    top = max(labels, key=lambda lab: df[lab].sum())
    n = len(df)
    comp = df["log_has_error"].tolist()
    comp_ext = df["log_any_error"].tolist()
    return {
        "empty []": [frozenset()] * n,
        f"always {{{short(top)}}}": [frozenset({top})] * n,
        "all 10 labels": [frozenset(labels)] * n,
        "compile rule (loader regex)": [frozenset({LBL_COMPILE}) if c else frozenset() for c in comp],
        "compile rule (+linker)": [frozenset({LBL_COMPILE}) if c else frozenset() for c in comp_ext],
        f"compile rule else {{{short(top)}}}": [frozenset({LBL_COMPILE}) if c else frozenset({top}) for c in comp],
        "lookup T3 input taxonomy_error": df["t3_labels"].map(frozenset).tolist(),
        "oracle (gold)": df["label_set"].tolist(),
    }


def metric_simulation(df: pd.DataFrame, labels: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    section("6. Macro-F1 sensitivity: baselines x zero_division x label set")
    y = to_matrix(df["label_set"].tolist(), labels)
    observed = [lab for lab in labels if df[lab].sum() > 0]
    obs_idx = [labels.index(lab) for lab in observed]
    rows = []
    for name, preds in baselines(df, labels).items():
        p = to_matrix(preds, labels)
        row: dict[str, object] = {"baseline": name}
        for tag, zd in ZERO_DIVISIONS:
            row[f"macro10 zd={tag}"] = macro_f1(y, p, zd)
            ref = f1_score(y, p, average="macro", zero_division=zd)
            assert np.isclose(row[f"macro10 zd={tag}"], ref), (name, tag)
        row[f"macro{len(observed)} (observed labels) zd=0"] = macro_f1(y[:, obs_idx], p[:, obs_idx], 0.0)
        row["micro zd=0"] = f1_score(y, p, average="micro", zero_division=0)
        row["samples zd=0"] = f1_score(y, p, average="samples", zero_division=0)
        row["samples zd=1"] = f1_score(y, p, average="samples", zero_division=1)
        rows.append(row)
    res = pd.DataFrame(rows)
    print(md_table(res, index=False))
    print(f"\nObserved labels used for macro{len(observed)}: {[short(x) for x in observed]}. "
          "macro10 zd=nan silently drops labels absent from both gold and prediction.")

    section("6b. Oracle macro-F1 per exam (labels absent in that exam drive the score)")
    per_exam = []
    for ex, sub in df.groupby("exam_id"):
        ys = to_matrix(sub["label_set"].tolist(), labels)
        per_exam.append({"exam": ex, "n": len(sub), "labels_absent": int((ys.sum(0) == 0).sum()),
                         **{f"oracle macro10 zd={t}": macro_f1(ys, ys, zd) for t, zd in ZERO_DIVISIONS}})
    print(md_table(pd.DataFrame(per_exam), index=False))

    section("6c. P(label has zero positives in a test set of size N), binomial")
    n = len(df)
    prob_rows = []
    for lab in labels:
        k = int(df[lab].sum())
        p_hat, p_jef = k / n, (k + 0.5) / (n + 1)
        upper = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
        row = {"label": short(lab), "k/n": f"{k}/{n}", "p_hat": p_hat, "p_jeffreys": p_jef, "CP95_upper": upper}
        for size in (100, 300):
            row[f"E[pos] N={size}"] = size * p_hat
            row[f"P0 p_hat N={size}"] = (1 - p_hat) ** size
            row[f"P0 jeff N={size}"] = (1 - p_jef) ** size
            row[f"P(<=2) p_hat N={size}"] = float(binom.cdf(2, size, p_hat))
        prob_rows.append(row)
    print(md_table(pd.DataFrame(prob_rows), index=False, floatfmt="{:.4f}"))

    section(f"6d. Bootstrap ({n_boot} resamples of the sample rows) - macro-F1 spread for a fixed predictor")
    rng = np.random.default_rng(seed)
    top = max(labels, key=lambda lab: df[lab].sum())
    rule = to_matrix([frozenset({LBL_COMPILE}) if c else frozenset({top}) for c in df["log_has_error"]], labels)
    boot_rows = []
    for size in (len(df), 100, 300):
        scores = {f"rule zd={t}": [] for t, _ in ZERO_DIVISIONS}
        absent = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(df), size)
            yb, pb = y[idx], rule[idx]
            absent.append(int((yb.sum(0) == 0).sum()))
            for t, zd in ZERO_DIVISIONS:
                scores[f"rule zd={t}"].append(macro_f1(yb, pb, zd))
        for key, vals in scores.items():
            arr = np.array(vals)
            boot_rows.append({"N": size, "metric": key, "mean": arr.mean(), "std": arr.std(),
                              "p5": np.percentile(arr, 5), "p95": np.percentile(arr, 95),
                              "mean_labels_without_positive": float(np.mean(absent))})
    print(md_table(pd.DataFrame(boot_rows), index=False))
    print("\nNote: resampling 32 rows can never create a positive for a label absent from the sample "
          "(edge case), so this is a lower bound on instability for the full set.")
    return res


# ---------------------------------------------------------------- optional compiler / exec checks
def local_compile_check(df: pd.DataFrame, root: Path) -> None:
    section("7. Local recompile (Apple clang via g++ alias; compile_log was produced by GNU g++ 13 on Linux)")
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        print("No compiler found; skipped.")
        return
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(f"compiler: `{version}`\n")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for _, r in df.iterrows():
            src = Path(tmp) / "student.cpp"
            shutil.copyfile(root / r["code_file"], src)
            args = [compiler, "-std=c++11", str(src)]
            args += ["-fsyntax-only"] if r["exam_id"] == "EX01" else ["-o", str(Path(tmp) / "a.out")]
            proc = subprocess.run(args, capture_output=True, text=True, timeout=60)
            results.append(proc.returncode != 0)
    df = df.assign(clang_error=results)
    rows = [crosstab_row("Lỗi biên dịch ~ local clang error", df, df[LBL_COMPILE], df["clang_error"]),
            crosstab_row("compilable=0 ~ local clang error", df, df["compilable"] == 0, df["clang_error"]),
            crosstab_row("log error incl. linker ~ local clang error", df, df["log_any_error"], df["clang_error"])]
    print(md_table(pd.DataFrame(rows), index=False, floatfmt="{:.2f}"))


def exec_commented_submissions(df: pd.DataFrame, root: Path) -> None:
    section("8. Fully-commented EX02 submissions: uncomment and run public tests (sandboxed, 2 s timeout)")
    compiler = shutil.which("g++") or shutil.which("clang++")
    targets = df[(df["exam_id"] == "EX02") & df["fully_commented"]]
    if targets.empty or not compiler:
        print("Nothing to run.")
        return
    rows = []
    for _, r in targets.iterrows():
        code = re.sub(r"^\s*//", "", r["code"], flags=re.M)
        if DANGEROUS_RE.search(code):
            rows.append({"sample_id": r["sample_id"], "status": "skipped: dangerous call pattern"})
            continue
        # Expected values come from any EX02 sample's public test report (same 8 tests for the exam).
        tests = next(t for t in df.loc[df["exam_id"] == "EX02", "test_report"] if isinstance(t, list) and t)
        with tempfile.TemporaryDirectory() as tmp:
            src, exe = Path(tmp) / "student.cpp", Path(tmp) / "a.out"
            src.write_text(code, encoding="utf-8")
            build = subprocess.run([compiler, "-std=c++11", str(src), "-o", str(exe)], capture_output=True, text=True, timeout=60)
            if build.returncode != 0:
                rows.append({"sample_id": r["sample_id"], "status": "uncommented code does not compile"})
                continue
            passed, failed = 0, []
            for t in tests:
                try:
                    out = subprocess.run([str(exe)], input=t["input"] + "\n", capture_output=True, text=True,
                                         timeout=2, cwd=tmp).stdout.strip()
                except subprocess.TimeoutExpired:
                    out = "<TIMEOUT>"
                if out == t["expected"].strip():
                    passed += 1
                else:
                    failed.append(f"{t['input']}->{out} (exp {t['expected']})")
            rows.append({"sample_id": r["sample_id"], "status": f"passed {passed}/{len(tests)}",
                         "failed": "; ".join(failed) or "-", "gold labels": r["labels"],
                         "logic": r["logic"], "total": r["total_score"]})
    print(md_table(pd.DataFrame(rows), index=False))


# ---------------------------------------------------------------- figures
def save_figures(prev: pd.DataFrame, cooc: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
                         "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 9})
    paths = []

    exam_cols = [c for c in prev.columns if c.startswith("EX")]
    fig, ax = plt.subplots(figsize=(7, 4))
    ypos = np.arange(len(prev))
    h = 0.38
    for i, col in enumerate(exam_cols):
        bars = ax.barh(ypos + (i - 0.5) * h, prev[col], height=h - 0.04, color=SERIES[i], label=col)
        ax.bar_label(bars, labels=[str(v) if v else "" for v in prev[col]], padding=2, fontsize=8, color=INK_2)
    for y_i, total in zip(ypos, prev["n"]):
        if total == 0:
            ax.text(0.05, y_i, "0 - absent in every exam", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(ypos, [short(x) for x in prev.index])
    ax.invert_yaxis()
    ax.set_xlabel("number of samples with label")
    ax.set_title("Task 2 label prevalence by exam (sample n=32)", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e6e3", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    paths.append(out_dir / "t2_prevalence_by_exam.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    mat = cooc.to_numpy(copy=True)
    diag = np.diag(mat).copy()
    np.fill_diagonal(mat, np.nan)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = f"n={int(diag[i])}" if i == j else (f"{mat[i, j]:.2f}" if mat[i, j] > 0 else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=INK if i == j or mat[i, j] < 0.6 else "white")
    ax.set_xticks(range(len(cooc.columns)), cooc.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cooc.index)), cooc.index)
    ax.set_xlabel("B")
    ax.set_ylabel("A (conditioning label)")
    ax.set_title("P(B | A) between Task 2 labels", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    paths.append(out_dir / "t2_cooccurrence.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    cols = [c for c in metrics.columns if c.startswith("macro10")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ypos = np.arange(len(metrics))
    h = 0.26
    for i, col in enumerate(cols):
        vals = metrics[col].astype(float).to_numpy()
        bars = ax.barh(ypos + (i - 1) * h, vals, height=h - 0.03, color=SERIES[i], label=col.replace("macro10 ", ""))
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=2, fontsize=7, color=INK_2)
    ax.set_yticks(ypos, metrics["baseline"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("macro-F1 over the 10 labels")
    ax.set_title("Same predictions, different macro-F1: zero_division convention", loc="left")
    ax.legend(frameon=False, loc="center right", title="zero_division")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    paths.append(out_dir / "t2_macro_f1_zero_division.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- entry point
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "out" / "t2")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="recompile submissions with the local compiler")
    parser.add_argument("--exec", action="store_true", help="run public tests on uncommented fully-commented EX02 files")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    df, labels = prepare(args.data_root)
    print(f"# Task 2 EDA - {len(df)} samples, {df['exam_id'].value_counts().to_dict()}")
    label_integrity(args.data_root, df, labels)
    prev = prevalence(df, labels)
    cooc = cooccurrence(df, labels)
    consistency(df)
    leakage(df)
    metrics = metric_simulation(df, labels, args.n_boot, args.seed)
    if args.compile:
        local_compile_check(df, args.data_root)
    if args.exec:
        exec_commented_submissions(df, args.data_root)
    if not args.no_figures:
        paths = save_figures(prev, cooc, metrics, args.out_dir)
        print("\nFigures: " + ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
