"""Shared loader for the grading-challenge dataset.

Merges the three task files (same sample_id set) with the source code and a few
derived columns, so every EDA script works on one flat DataFrame. The data root is
always passed in explicitly, so the same code runs on the sample set now and on the
full dataset later.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

DIMS = ["compilable", "io_format", "logic", "edge_case", "complexity", "code_quality"]
DIM_MAX = {"compilable": 1, "io_format": 1, "logic": 4, "edge_case": 2, "complexity": 1, "code_quality": 1}
TASK_FILES = {
    "t1": "task1_grading.json",
    "t2": "task2_error_taxonomy.json",
    "t3": "task3_feedback.json",
}
# gcc/clang diagnostic line, e.g. "a.cpp:12:5: error: ..."; avoids matching "-Werror" or "0 errors".
COMPILE_ERROR_RE = re.compile(r"^\S+:\d+:\d+: (fatal )?error:", re.M)
# Link failures carry no file:line:col (e.g. a fully commented-out file with no main).
LINK_ERROR_RE = re.compile(r"undefined reference|collect2: error|ld returned|linker command failed")
# Label noise confirmed by the course staff. Released labels stay the default; callers opt in.
LABEL_FIXES = {
    "EX02-S214": {"compilable": 0},  # whole file commented out, cannot build
    "EX02-S211": {"io_format": 0},   # extra input prompt fails 8/8 public tests
    "EX02-S216": {"io_format": 0},   # same extra-prompt pattern as S211
}
# Rubric on a failed build: EX01 per its grading policy, EX02 as every released EX02 build failure shows.
COMPILE_FAIL_RUBRIC = {"EX01": [0, 0, 0, 0, 0, 0], "EX02": [0, 0, 0, 0, 0, 1]}
# Taxonomy label implied when a dimension is corrected to 0.
FIX_IMPLIED_LABEL = {"compilable": "Lỗi biên dịch", "io_format": "Lỗi nhập/xuất"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_code(path: Path) -> tuple[str, str]:
    """Return (text, encoding_used); falls back so a bad byte never crashes the EDA."""
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1258", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def level_number(level: str | None) -> int | None:
    match = re.search(r"(\d)", level or "")
    return int(match.group(1)) if match else None


def load_meta(root: Path) -> tuple[dict, dict, dict[str, dict]]:
    """Return (label_space, exams_by_id, raw task files)."""
    label_space = read_json(root / "label_space.json")
    exams = {e["exam_id"]: e for e in read_json(root / "exams.json")["exams"]}
    tasks = {key: read_json(root / name) for key, name in TASK_FILES.items()}
    return label_space, exams, tasks


def load_samples(root: Path) -> pd.DataFrame:
    """One row per sample_id with inputs, all three outputs, code and derived columns."""
    _, _, tasks = load_meta(root)
    rows: dict[str, dict] = {}
    for key, payload in tasks.items():
        for sample in payload["samples"]:
            row = rows.setdefault(sample["sample_id"], {"sample_id": sample["sample_id"]})
            row[f"{key}_input"] = sample["input"]
            out = sample["output"]
            if key == "t1":
                row.update({d: out["rubric"].get(d) for d in DIMS})
                row["total_score"] = out.get("total_score")
            elif key == "t2":
                row["labels"] = out.get("taxonomy_error")
            else:
                row["feedback"] = out.get("feedback")
                row["level"] = sample["input"].get("target_feedback_level")
                row["t3_labels"] = sample["input"].get("taxonomy_error")

    df = pd.DataFrame(list(rows.values()))
    base = df["t1_input"].apply(pd.Series)
    df = pd.concat([df, base[["exam_id", "exam_type", "language", "code_file", "compile_log", "test_report"]]], axis=1)

    codes = df["code_file"].map(lambda p: read_code(root / p) if (root / p).exists() else (None, None))
    df["code"] = codes.map(lambda t: t[0])
    df["code_encoding"] = codes.map(lambda t: t[1])
    df["code_exists"] = df["code"].notna()

    df["sum_dims"] = df[DIMS].sum(axis=1)
    df["level_num"] = df["level"].map(level_number)
    df["n_labels"] = df["labels"].map(lambda xs: len(xs) if isinstance(xs, list) else None)
    df["log_has_error"] = df["compile_log"].fillna("").map(lambda s: bool(COMPILE_ERROR_RE.search(s)))
    df["log_has_link_error"] = df["compile_log"].fillna("").map(lambda s: bool(LINK_ERROR_RE.search(s)))
    df["log_build_failed"] = df["log_has_error"] | df["log_has_link_error"]
    df["log_nonempty"] = df["compile_log"].fillna("").str.strip().ne("")
    df["n_tests"] = df["test_report"].map(lambda r: len(r) if isinstance(r, list) else None)
    df["n_passed"] = df["test_report"].map(lambda r: sum(bool(t.get("passed")) for t in r) if isinstance(r, list) else None)
    df["pass_rate"] = df["n_passed"] / df["n_tests"]
    df["code_lines"] = df["code"].map(lambda c: c.count("\n") + 1 if isinstance(c, str) else None)
    df["code_chars"] = df["code"].map(lambda c: len(c) if isinstance(c, str) else None)
    df["feedback_chars"] = df["feedback"].map(lambda s: len(s) if isinstance(s, str) else None)
    return df


def is_nfc(text: str) -> bool:
    return unicodedata.is_normalized("NFC", text)
