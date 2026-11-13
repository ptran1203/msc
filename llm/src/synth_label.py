"""Derive the official rubric/taxonomy schema from objective oracle signals + known injected bug.

Taxonomy label comes from injection INTENT (we chose the bug), not from re-diagnosing the
oracle's symptoms -- except compile failure, which always overrides intent (see below).
Rubric scores come entirely from objective oracle signals, never from LLM self-report.
"""
from synth_bugs import BUG_TAXONOMY

RUBRIC_KEYS = ["compilable", "io_format", "logic", "edge_case", "complexity", "code_quality"]


def derive_label(grade: dict, bug_id: str) -> dict | None:
    """Returns the {rubric, total_score, taxonomy_error} output dict, or None if the
    sample fails quality control (LLM didn't follow the injection instruction cleanly)."""
    if not grade["compilable"]:
        if bug_id != "compile_error":
            return None  # QC: unintended compile break, discard -- can't trust a clean single-bug injection
        rubric = {k: 0 for k in RUBRIC_KEYS}
        return {"rubric": rubric, "total_score": 0, "taxonomy_error": ["Lỗi biên dịch"]}

    if bug_id == "compile_error":
        return None  # QC: intended compile break but it actually compiled -- injection failed

    gen_raw, gen_norm = grade["gen_pass_raw"], grade["gen_pass_norm"]
    edge_raw, edge_norm = grade["edge_pass_raw"], grade["edge_pass_norm"]

    if gen_raw >= 0.9:
        io_format, logic_basis = 1, gen_raw
    elif gen_norm >= 0.9:
        io_format, logic_basis = 0, gen_norm
    else:
        io_format, logic_basis = 1, gen_norm  # ambiguous: assume format fine, blame logic

    rubric = {
        "compilable": 1,
        "io_format": io_format,
        "logic": round(4 * logic_basis),
        "edge_case": round(2 * (edge_norm if io_format == 0 else edge_raw)),
        "complexity": 1 if grade["complexity_ok"] else 0,
        "code_quality": 0 if bug_id == "style_error" else 1,
    }

    # QC: the intended bug should actually degrade exactly the dimension(s) it's supposed to.
    fully_clean = rubric["logic"] == 4 and rubric["edge_case"] == 2 and rubric["io_format"] == 1
    if bug_id == "style_error":
        if not fully_clean:
            return None  # QC: style must be cosmetic-only -- behavior changed, discard
    elif bug_id == "io_format_error":
        if io_format == 1:
            return None  # QC: format bug should show up as io_format=0
    else:
        if fully_clean:
            return None  # QC: injected bug had no measurable effect -- injection likely failed

    total_score = sum(rubric.values())
    return {"rubric": rubric, "total_score": total_score, "taxonomy_error": [BUG_TAXONOMY[bug_id]["label"]]}
