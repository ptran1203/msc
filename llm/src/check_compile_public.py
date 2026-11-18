"""Independently compile every public_test submission with g++ -- an objective ground-truth
signal we're allowed to use (it's derived from the input code only, never from hidden gold
labels), to measure how often the model's own "compilable" judgment is simply wrong.

Must run via PowerShell, not Bash (g++'s cc1plus.exe subprocess is blocked under Bash's sandbox).
"""
import json
import subprocess
from pathlib import Path

from data import REPO_ROOT, load_public_test, read_code

WORK_DIR = REPO_ROOT / "synth/work/compile_check"
WORK_DIR.mkdir(parents=True, exist_ok=True)


def compiles(code: str, tag: str) -> tuple[bool, str]:
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


def main():
    ds = load_public_test()
    samples = ds.tasks[1]
    results = []
    for i, sample in enumerate(samples):
        code = read_code(ds, sample)
        ok, err = compiles(code, sample["sample_id"])
        results.append({"sample_id": sample["sample_id"], "compiles": ok, "stderr": err[:300] if not ok else ""})
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(samples)}...")

    n_ok = sum(r["compiles"] for r in results)
    print(f"\n{n_ok}/{len(results)} ({n_ok/len(results):.1%}) actually compile with g++ -std=c++11")

    out_path = REPO_ROOT / "outputs" / "public_test_compile_check.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
