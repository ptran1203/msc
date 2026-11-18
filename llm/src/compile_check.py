"""Objective compile-log enrichment: the released data always leaves compile_log empty,
so the model has to guess compilability purely by reading code -- measured (via our own
g++ compile) to be wrong ~20-25% of the time even on code that actually compiles. Fill in
the real signal instead, same oracle principle as the synthetic-data pipeline: this only
uses the input code, never the hidden gold labels.
"""
import json
import subprocess
from pathlib import Path

from data import REPO_ROOT, read_code

WORK_DIR = REPO_ROOT / "synth/work/compile_check"
WORK_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = REPO_ROOT / "outputs"


def _compile(code: str, tag: str) -> tuple[bool, str]:
    cpp_path = WORK_DIR / f"{tag}.cpp"
    exe_path = WORK_DIR / f"{tag}.exe"
    cpp_path.write_text(code, encoding="utf-8")
    try:
        result = subprocess.run(
            ["g++", "-std=c++11", str(cpp_path), "-o", str(exe_path)],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    return result.returncode == 0, result.stderr


def attach_real_compile_log(ds, task_num: int) -> None:
    """Mutates ds.tasks[task_num] samples in place, setting input.compile_log from a real
    g++ compile. Cached to disk per dataset root so repeat runs don't recompile."""
    cache_path = CACHE_DIR / f"compile_log_cache_{ds.root.name}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    changed = False
    for sample in ds.tasks[task_num]:
        sid = sample["sample_id"]
        if sid not in cache:
            ok, stderr = _compile(read_code(ds, sample), sid)
            cache[sid] = "Biên dịch thành công." if ok else f"Lỗi biên dịch:\n{stderr[:500]}"
            changed = True
        sample["input"]["compile_log"] = cache[sid]

    if changed:
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
