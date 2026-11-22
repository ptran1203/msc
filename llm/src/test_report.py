"""Fill the (always empty) test_report input field by RUNNING each submission on
LLM-generated test cases -- same principle as compile_log: a fact about the student's own
code, never a gold label. Instructor QA: hand-written test cases are not allowed,
LLM-generated ones are.

Expected outputs are never written by the LLM directly. Per single-problem exam, the LLM
writes 3 reference solutions + test inputs; a reference is trusted only if it reproduces
every example in the statement, and a test is kept only if all trusted references agree.

    python test_report.py gen [--split public|sample]   # LLM step (GPU), writes outputs/llm_tests/
    python test_report.py run [--split public|sample]   # runs submissions (CPU), prints a summary
"""
import argparse
import hashlib
import json
import re
import subprocess

from data import REPO_ROOT, load_public_test, load_sample_dataset, read_code
from llm_client import SEED, LocalLLM, extract_code_block, extract_json
from synth_generate import strip_cpp_comments

TESTS_DIR = REPO_ROOT / "outputs/llm_tests"
WORK_DIR = REPO_ROOT / "synth/work/test_report"
CACHE_DIR = REPO_ROOT / "outputs"
N_REFS = 3  # trusted references wanted per exam
MAX_REF_TRIES = 12  # the 7B gets even easy problems wrong often (cubes every digit, sorts instead of keeping order)
RUN_TIMEOUT_S = 2
SYSTEM = "Bạn là lập trình viên C++ cẩn thận. Làm đúng yêu cầu, không giải thích dài."


# ---------------------------------------------------------------- generation (LLM)

def table_cells(statement: str):
    """(input cells, output cells) of the statement's example table, in order."""
    ins, outs, in_table = [], [], False
    for line in statement.splitlines():
        if re.match(r"\|\s*Input\s*\|\s*Output\s*\|", line):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                in_table = False
                continue
            cells = [c.strip().strip("`").strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 2 and not set(cells[0]) <= set("-: "):
                ins.append(cells[0])
                outs.append(cells[1])
    return ins, outs


def _tokens(s: str):
    return s.split()


def extract_examples(llm, statement: str):
    """LLM groups table rows into examples; accepted only if the grouping reproduces the
    table's cells exactly (so a mis-grouped or invented example is rejected)."""
    ins, outs = table_cells(statement)
    prompt = f"""{statement}

Trích các ví dụ (Example) trong đề thành JSON. Một ví dụ có thể trải trên nhiều dòng của bảng
(ví dụ dòng đầu là n, dòng sau là mảng) -- gộp đúng thành input/output hoàn chỉnh, dùng "\\n" giữa các dòng.
Trả lời CHỈ JSON: {{"examples": [{{"input": "...", "output": "..."}}]}}"""
    for attempt in range(4):  # greedy first, then seeded samples
        temperature = None if attempt == 0 else 0.7
        parsed = extract_json(llm.generate(SYSTEM, prompt, max_new_tokens=600, temperature=temperature, seed=SEED + attempt)) or {}
        examples = [e for e in parsed.get("examples", []) if isinstance(e, dict) and isinstance(e.get("input"), str) and isinstance(e.get("output"), str)]
        if (_tokens(" ".join(e["input"] for e in examples)) == _tokens(" ".join(ins))
                and _tokens(" ".join(e["output"] for e in examples)) == _tokens(" ".join(outs))):
            return [{"input": e["input"].rstrip("\n") + "\n", "output": e["output"]} for e in examples]
        print(f"  example extraction attempt {attempt}: doesn't match the statement table, retrying")
    raise RuntimeError(f"example extraction never matched the statement table {ins} / {outs}")


def gen_reference(llm, statement: str, seed: int):
    prompt = f"""{statement}

Viết lời giải C++11 HOÀN CHỈNH, ĐÚNG cho đề trên: có hàm main, đọc từ stdin và in ra stdout
ĐÚNG định dạng output của đề (không in thêm chữ nào khác). Trả lời CHỈ bằng một code block cpp."""
    temperature = None if seed == 0 else 0.7
    return extract_code_block(llm.generate(SYSTEM, prompt, max_new_tokens=900, temperature=temperature, seed=SEED + seed))


def gen_inputs(llm, statement: str, seed: int):
    prompt = f"""{statement}

Sinh 12 input kiểm thử cho đề trên, đúng định dạng input và trong giới hạn (constraints) của đề:
khoảng một nửa là trường hợp thông thường, một nửa là trường hợp biên (giá trị nhỏ nhất/lớn nhất,
phần tử trùng nhau, một phần tử, ...). Chỉ cần input, KHÔNG cần output.
Trả lời CHỈ JSON: {{"inputs": ["<input 1, dùng \\n giữa các dòng>", "..."]}}"""
    temperature = None if seed == 0 else 0.7
    parsed = extract_json(llm.generate(SYSTEM, prompt, max_new_tokens=900, temperature=temperature, seed=SEED + seed)) or {}
    return [s.rstrip("\n") + "\n" for s in parsed.get("inputs", []) if isinstance(s, str) and s.strip()]


# ---------------------------------------------------------------- compile / run

def compile_exe(code: str, tag: str):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cpp, exe = WORK_DIR / f"{tag}.cpp", WORK_DIR / f"{tag}.exe"
    cpp.write_text(code, encoding="utf-8")
    try:
        r = subprocess.run(["g++", "-std=c++11", "-w", "-O1", str(cpp), "-o", str(exe)], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    return exe if r.returncode == 0 else None


def run_exe(exe, stdin: str):
    """-> (stdout, status) with status in ok / timeout / crash."""
    try:
        r = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return "", "timeout"
    return r.stdout, "ok" if r.returncode == 0 else "crash"


def exact(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip("\n").split("\n"))


def loose(s: str) -> str:
    return " ".join(s.lower().split())


def build_tests(llm, exam):
    st = exam["statement"]
    examples = extract_examples(llm, st)
    # Trust = right CONTENT on every statement example (whitespace-insensitive); expected
    # outputs come from a reference that also matches the statement's EXACT format, so
    # format-only mistakes in student code stay detectable.
    refs, exact_refs = [], []
    for i in range(MAX_REF_TRIES):
        if len(refs) >= N_REFS and exact_refs:
            break
        code = gen_reference(llm, st, i)
        exe = compile_exe(code, f"ref_{exam['exam_id']}_{i}") if code else None
        if exe is None:
            print(f"  ref {i}: no code / doesn't compile")
            continue
        outs = [run_exe(exe, e["input"]) for e in examples]
        content_ok = all(s == "ok" and loose(o) == loose(e["output"]) for (o, s), e in zip(outs, examples))
        format_ok = content_ok and all(exact(o) == exact(e["output"]) for (o, _), e in zip(outs, examples))
        print(f"  ref {i}: {'exact' if format_ok else 'content-only' if content_ok else 'WRONG'} on {len(examples)} statement examples")
        if content_ok:
            refs.append(exe)
        if format_ok:
            exact_refs.append(exe)
    if len(refs) < 2 or not exact_refs:
        raise RuntimeError(f"{exam['exam_id']}: {len(refs)} content-correct / {len(exact_refs)} exact references, need >= 2 / >= 1")

    seen = {loose(e["input"]) for e in examples}
    inputs = [x for x in dict.fromkeys(x for s in range(2) for x in gen_inputs(llm, st, s)) if loose(x) not in seen]
    tests, n_disagree = [{"input": e["input"], "expected": e["output"], "is_example": True} for e in examples], 0
    for x in inputs:
        outs = [run_exe(exe, x) for exe in refs]
        if all(s == "ok" for _, s in outs) and len({loose(o) for o, _ in outs}) == 1:
            tests.append({"input": x, "expected": exact(run_exe(exact_refs[0], x)[0]), "is_example": False})
        else:
            n_disagree += 1
    print(f"  {len(tests)} tests kept ({len(examples)} from the statement), {n_disagree} generated inputs dropped (refs disagree/crash)")
    return {"exam_id": exam["exam_id"], "n_refs": len(refs), "tests": tests}


def gen(split: str):
    ds = load_public_test() if split == "public" else load_sample_dataset()
    llm = LocalLLM()
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    for exam in ds.exams.values():
        if exam["exam_type"] != "single_problem" or (TESTS_DIR / f"{exam['exam_id']}.json").exists():
            continue
        print(f"== {exam['exam_id']}")
        try:
            out = build_tests(llm, exam)
        except RuntimeError as e:
            print(f"  SKIPPED: {e}")  # no test_report for this exam beats one built on a wrong reference
            continue
        (TESTS_DIR / f"{exam['exam_id']}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- per-submission report

def _short(s: str, n: int = 60) -> str:
    s = s.strip().replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "..."


def _dedupe(tests: list) -> list:
    seen, out = set(), []
    for t in tests:
        if loose(t["input"]) not in seen:
            seen.add(loose(t["input"]))
            out.append(t)
    return out


def classify(out: str, status: str, expected: str) -> str:
    if status != "ok":
        return status  # timeout / crash
    if exact(out) == exact(expected):
        return "ok"
    # same content, different whitespace/case -- or an input prompt printed first
    # ("Input number: True"): a format problem, not wrong content
    if loose(out) == loose(expected) or loose(out).endswith(" " + loose(expected)):
        return "format"
    return "wrong"


def run_submission(code: str, tests: list, tag: str):
    """Per-test (classification, actual output); 'no_code' / 'no_compile' for the whole run."""
    if not strip_cpp_comments(code).strip():
        return "no_code"
    exe = compile_exe(code, tag)
    if exe is None:
        return "no_compile"
    results = []
    for t in tests:
        out, status = run_exe(exe, t["input"])
        results.append([classify(out, status, t["expected"]), out[:200]])
    return results


def out_of_spec_tests(all_results: list, n_tests: int) -> set:
    """Generated tests that most otherwise-strong submissions fail are likely outside the
    statement's constraints (e.g. n=0 when 1 <= n): a correct student shouldn't be told
    they're wrong. Uses only student-code outcomes, never labels."""
    runs = [r for r in all_results if isinstance(r, list)]
    strong = [r for r in runs if sum(c in ("ok", "format") for c, _ in r) >= 0.75 * n_tests]
    if len(strong) < 3:
        return set()
    return {j for j in range(n_tests) if sum(r[j][0] not in ("ok", "format") for r in strong) > 0.5 * len(strong)}


def make_report(result, tests: list, drop: set) -> str:
    if result == "no_code":
        return "Bài nộp không có code chạy được (toàn bộ code bị comment hoặc trống)."
    if result == "no_compile":
        return "Không chạy được test vì chương trình không biên dịch được."
    kept = [(t, r) for j, (t, r) in enumerate(zip(tests, result)) if j not in drop]
    count = lambda c: sum(r[0] == c for _, r in kept)
    fails = [(t, r) for t, r in kept if r[0] == "wrong"]
    lines = [f"Chạy {len(kept)} test sinh tự động: đúng {count('ok')}/{len(kept)}."]
    if count("format"):
        lines.append(f"- {count('format')} test: nội dung đúng nhưng SAI ĐỊNH DẠNG output (khoảng trắng/xuống dòng/chữ hoa-thường, hoặc in thêm chữ thừa).")
    if count("timeout"):
        lines.append(f"- {count('timeout')} test quá thời gian {RUN_TIMEOUT_S}s (có thể lặp vô tận).")
    if count("crash"):
        lines.append(f"- {count('crash')} test chương trình bị lỗi khi chạy (crash).")
    for t, r in fails[:3]:
        lines.append(f"- Sai: input `{_short(t['input'])}` -> mong đợi `{_short(t['expected'])}`, chương trình in `{_short(r[1])}`")
    if len(fails) > 3:
        lines.append(f"- ... và {len(fails) - 3} test sai khác.")
    return "\n".join(lines)


def attach_test_report(ds, task_num: int) -> None:
    """Sets input.test_report for samples of exams that have generated tests (single-
    problem exams after `gen`); others are left untouched. Raw per-test results are
    cached per dataset root, keyed by a hash of the exam's test suite."""
    cache_path = CACHE_DIR / f"test_results_cache_{ds.root.name}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    suites = {p.stem: _dedupe(json.loads(p.read_text(encoding="utf-8"))["tests"]) for p in TESTS_DIR.glob("*.json")} if TESTS_DIR.exists() else {}
    changed = False
    by_exam = {}
    for s in ds.tasks[task_num]:
        tests = suites.get(s["input"]["exam_id"])
        if tests is None:
            continue
        key = f"{s['sample_id']}:{hashlib.md5(json.dumps(tests, sort_keys=True).encode()).hexdigest()[:10]}"
        if key not in cache:
            cache[key] = run_submission(read_code(ds, s), tests, s["sample_id"])
            changed = True
        by_exam.setdefault(s["input"]["exam_id"], []).append((s, cache[key]))
    for exam_id, items in by_exam.items():
        tests = suites[exam_id]
        drop = out_of_spec_tests([r for _, r in items], len(tests))
        if drop:
            print(f"  {exam_id}: dropping {len(drop)} likely out-of-spec test(s): {[_short(tests[j]['input'], 30) for j in sorted(drop)]}")
        for s, r in items:
            s["input"]["test_report"] = make_report(r, tests, drop)
    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def run(split: str):
    ds = load_public_test() if split == "public" else load_sample_dataset()
    attach_test_report(ds, 3)
    with_report = [s for s in ds.tasks[3] if s["input"].get("test_report")]
    print(f"{len(with_report)}/{len(ds.tasks[3])} Task 3 samples got a test_report")
    for s in with_report[:5]:
        print(f"\n[{s['sample_id']}] level={s['input']['target_feedback_level'][:7]}\n{s['input']['test_report']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gen", "run"])
    ap.add_argument("--split", choices=["public", "sample"], default="public")
    args = ap.parse_args()
    gen(args.split) if args.step == "gen" else run(args.split)
