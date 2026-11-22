"""Objective compile-log enrichment: the released data always leaves compile_log empty,
so the model has to guess compilability purely by reading code -- measured (via our own
g++ compile) to be wrong ~20-25% of the time even on code that actually compiles. Fill in
the real signal instead, same oracle principle as the synthetic-data pipeline: this only
uses the input code, never the hidden gold labels.
"""
import json
import re
import subprocess
from pathlib import Path

from data import REPO_ROOT, read_code

WORK_DIR = REPO_ROOT / "synth/work/compile_check"
WORK_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = REPO_ROOT / "outputs"


def _compile(code: str, tag: str) -> tuple[bool, str]:
    cpp_path = WORK_DIR / f"{tag}.cpp"
    obj_path = WORK_DIR / f"{tag}.o"
    cpp_path.write_text(code, encoding="utf-8")
    try:
        # -c: compile only, no link. Multi-problem exams require submitting ONLY the asked
        # functions (graders link their own main), so linking failed ~18 public files
        # that compile fine. -w: warnings came first in stderr and pushed the real
        # "error:" line past the 500-char cut.
        result = subprocess.run(
            ["g++", "-std=c++11", "-c", "-w", str(cpp_path), "-o", str(obj_path)],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    return result.returncode == 0, result.stderr


def first_compile_error(log: str) -> str:
    """The first g++ error line (+ the source line g++ prints under it) -- how teachers
    quote compile errors in Level 2 feedback ("student.cpp:51:26: error: ...")."""
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if "error:" in line:
            if i + 1 < len(lines) and re.match(r"\s*\d+ \|", lines[i + 1]):
                return f"{line}\n{lines[i + 1]}"
            return line
    return "Lỗi biên dịch."


def attach_real_compile_log(ds, task_num: int) -> None:
    """Mutates ds.tasks[task_num] samples in place, setting input.compile_log from a real
    g++ compile. Cached to disk per dataset root so repeat runs don't recompile."""
    # v2: compile-only + no warnings (see _compile); v1 entries were link-based
    cache_path = CACHE_DIR / f"compile_log_cache_v2_{ds.root.name}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    changed = False
    for sample in ds.tasks[task_num]:
        sid = sample["sample_id"]
        if sid not in cache:
            ok, stderr = _compile(read_code(ds, sample), sid)
            cache[sid] = "Biên dịch thành công." if ok else f"Lỗi biên dịch:\n{stderr[:500]}"
            changed = True
        # g++ prints our local work-dir path; teachers quote it as "student.cpp:51:26: error: ..."
        sample["input"]["compile_log"] = re.sub(r"(?:[A-Za-z]:)?[^\s:]*[\\/][^\s\\/:]+\.cpp", "student.cpp", cache[sid])

    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
