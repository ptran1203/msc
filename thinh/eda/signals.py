"""Deterministic, LLM-free signals read from a submission: build status and test verdicts.

Shared by the EDA page and the POC pipeline so both classify a sample the same way.
"""

from __future__ import annotations

import re

PROMPT_RE = re.compile(r"^(?P<prompt>.*?[:?]\s*)(?P<rest>\S+)$")


def strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return re.sub(r"//[^\n]*", "", code)


def compile_status(log: str) -> str:
    if not log.strip():
        return "ok"
    if re.search(r"undefined reference|collect2: error|ld returned", log):
        return "link"
    if re.search(r"^\S+:\d+:\d+: (fatal )?error:", log, re.M):
        return "error"
    return "warning"


def classify_test(t: dict) -> str:
    """pass | prompt (format only) | prompt_wrong | wrong | crash (empty output) | timeout."""
    if t.get("passed"):
        return "pass"
    actual, expected = (t.get("actual") or "").strip(), (t.get("expected") or "").strip()
    if actual == "<TIMEOUT>":
        return "timeout"
    if actual == "":
        return "crash"
    m = PROMPT_RE.match(actual)
    if m:
        return "prompt" if m.group("rest").lower() == expected.lower() else "prompt_wrong"
    return "wrong"


def test_summary(tests: list[dict] | None) -> dict | None:
    """Counts per verdict plus strict/lenient pass; None when the sample has no public tests."""
    if not tests:
        return None
    verdicts = [classify_test(t) for t in tests]
    return {
        "n": len(verdicts),
        "strict_pass": sum(v == "pass" for v in verdicts),
        "lenient_pass": sum(v in ("pass", "prompt") for v in verdicts),
        "verdicts": {v: verdicts.count(v) for v in sorted(set(verdicts))},
    }


def ex01_functions(code: str) -> dict[str, str]:
    body = strip_comments(code)
    out = {}
    for k in range(1, 5):
        if re.search(rf"\bprob{k}\s*\([^;{{}}]*\)\s*\{{", body):
            out[f"prob{k}"] = "defined"
        elif re.search(rf"\bprob{k}\s*\([^;{{}}]*\)\s*;", body):
            out[f"prob{k}"] = "declared"
        else:
            out[f"prob{k}"] = "missing"
    out["main"] = "defined" if re.search(r"\b(int|void)\s+main\s*\(", body) else "missing"
    return out


STATEMENT_WEIGHT_RE = re.compile(r"\*\*Problem\s+(\d+)\s*\((\d+(?:[.,]\d+)?)\s*điểm\)", re.I)


def statement_weights(statement: str) -> dict[str, float]:
    """Point weight per problem as written in the statement, e.g. '**Problem 2 (4 điểm).**' -> {'prob2': 4.0}."""
    return {f"prob{k}": float(w.replace(",", ".")) for k, w in STATEMENT_WEIGHT_RE.findall(statement)}
