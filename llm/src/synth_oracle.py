"""Objective oracle: compile + run candidate C++ against reference-generated test cases.

Must be invoked via the PowerShell tool, not Bash -- the sandboxed Bash tool blocks
g++'s cc1plus.exe subprocess from spawning (see llm/CLAUDE.md "Tooling note").
Run as: python synth_oracle.py (smoke test) from a PowerShell call.
"""
import random
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = REPO_ROOT / "synth/references"
WORK_DIR = REPO_ROOT / "synth/work"
SEED = 42

TIMEOUT_S = 5
STRESS_TIMEOUT_S = 2  # complexity check: must finish large input within this


def compile_cpp(cpp_path: Path, exe_path: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["g++", "-std=c++11", str(cpp_path), "-o", str(exe_path)],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0, result.stderr


def run_exe(exe_path: Path, input_text: str, timeout: float = TIMEOUT_S) -> tuple[str | None, bool]:
    """Returns (stdout, timed_out). stdout is None on crash/nonzero exit (but not timeout)."""
    try:
        result = subprocess.run(
            [str(exe_path)], input=input_text, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, True
    if result.returncode != 0:
        return None, False
    return result.stdout, False


def normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


# ---- per-problem test case generators (seeded for reproducibility) ----

def gen_armstrong_cases(rng: random.Random):
    general = [rng.randint(1, 10**9) for _ in range(8)] + [1634, 9474, 123, 9926315]
    edge = [1, 9, 10**9, 10, 999999999]
    return [(str(n) + "\n", False) for n in general] + [(str(n) + "\n", True) for n in edge]


def gen_remove_duplicates_cases(rng: random.Random):
    def fmt(arr):
        return f"{len(arr)}\n{' '.join(map(str, arr))}\n"

    general = []
    for _ in range(6):
        n = rng.randint(3, 20)
        arr = [rng.randint(-50, 50) for _ in range(n)]
        general.append(fmt(arr))
    edge = [
        fmt([5]),  # single element
        fmt([7, 7, 7, 7, 7]),  # all identical
        fmt([1, 2, 3, 4, 5]),  # no duplicates at all
        fmt([-1, -1, -2, -2, -3]),  # negatives
    ]
    return [(c, False) for c in general] + [(c, True) for c in edge]


def gen_earliest_login_cases(rng: random.Random):
    import string

    def rand_name(n):
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))

    general = []
    for _ in range(6):
        f, l = rand_name(rng.randint(2, 10)), rand_name(rng.randint(2, 10))
        general.append(f"{f} {l}\n")
    general += ["harry potter\n", "tom riddle\n"]
    edge = [
        f"{rand_name(1)} {rand_name(1)}\n",  # both single-char
        "aa aa\n",  # tie-prone pattern
        "aaaa a\n",
    ]
    return [(c, False) for c in general] + [(c, True) for c in edge]


PROBLEM_GENERATORS = {
    "armstrong": gen_armstrong_cases,
    "remove_duplicates": gen_remove_duplicates_cases,
    "earliest_login": gen_earliest_login_cases,
}


def build_oracle(problem_name: str):
    """Compile the reference solution and precompute expected outputs for all test cases."""
    rng = random.Random(SEED)
    cases = PROBLEM_GENERATORS[problem_name](rng)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ref_cpp = REF_DIR / f"{problem_name}.cpp"
    ref_exe = WORK_DIR / f"{problem_name}_ref.exe"
    ok, err = compile_cpp(ref_cpp, ref_exe)
    assert ok, f"reference solution {problem_name} failed to compile: {err}"

    oracle_cases = []
    for input_text, is_edge in cases:
        expected, timed_out = run_exe(ref_exe, input_text)
        assert expected is not None and not timed_out, f"reference solution crashed/timed out on {input_text!r}"
        oracle_cases.append({"input": input_text, "expected": expected, "is_edge": is_edge})
    return oracle_cases


def grade_candidate(problem_name: str, candidate_cpp: Path, oracle_cases: list, tag: str) -> dict:
    """Objective grading signals for a candidate .cpp against precomputed oracle cases."""
    exe_path = WORK_DIR / f"{tag}.exe"
    ok, compile_err = compile_cpp(candidate_cpp, exe_path)
    if not ok:
        return {"compilable": False, "compile_err": compile_err}

    def pass_rates(cases):
        if not cases:
            return 1.0, 1.0
        raw_ok = norm_ok = 0
        for c in cases:
            actual, timed_out = run_exe(exe_path, c["input"])
            if timed_out or actual is None:
                continue
            if actual.strip("\n") == c["expected"].strip("\n"):
                raw_ok += 1
                norm_ok += 1
            elif normalize(actual) == normalize(c["expected"]):
                norm_ok += 1
        return raw_ok / len(cases), norm_ok / len(cases)

    general_cases = [c for c in oracle_cases if not c["is_edge"]]
    edge_cases = [c for c in oracle_cases if c["is_edge"]]
    gen_raw, gen_norm = pass_rates(general_cases)
    edge_raw, edge_norm = pass_rates(edge_cases)

    # complexity: stress-test with a large input, must finish within STRESS_TIMEOUT_S
    complexity_ok = True
    if problem_name == "remove_duplicates":
        big = f"10000\n{' '.join(str(i % 500) for i in range(10000))}\n"
        _, timed_out = run_exe(exe_path, big, timeout=STRESS_TIMEOUT_S)
        complexity_ok = not timed_out

    return {
        "compilable": True,
        "gen_pass_raw": gen_raw, "gen_pass_norm": gen_norm,
        "edge_pass_raw": edge_raw, "edge_pass_norm": edge_norm,
        "complexity_ok": complexity_ok,
    }


if __name__ == "__main__":
    for name in PROBLEM_GENERATORS:
        oracle = build_oracle(name)
        grade = grade_candidate(name, REF_DIR / f"{name}.cpp", oracle, tag=f"selfcheck_{name}")
        print(name, "n_cases=", len(oracle), "self-grade (should be perfect):", grade)
        assert grade["compilable"]
        assert grade["gen_pass_raw"] == 1.0 and grade["edge_pass_raw"] == 1.0, grade
    print("\nORACLE SELF-CHECK PASSED")
