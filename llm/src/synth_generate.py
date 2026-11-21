"""Generate synthetic Task 1 + Task 2 training data via controlled bug injection.

Scope: Task 1 (rubric) + Task 2 (taxonomy) only. Task 3 is out of scope here -- we have
no objective oracle for feedback-TEXT quality, so synthetically generating it would just
reintroduce the self-labeling problem this whole pipeline exists to avoid (see METHOD.md).

The LLM is used ONLY to write buggy code from a precise instruction (something it's good
at); it never grades its own output. Labels come from synth_label.derive_label, which is
driven entirely by the objective oracle in synth_oracle.py. Samples that fail quality
control (LLM didn't follow the injection instruction cleanly) are discarded, not mislabeled.

Needs the GPU (LLM calls) -- run via PowerShell, not Bash (g++ subprocess spawn is blocked
under Bash's sandbox). Fit a single sample first (see smoke-test block) before the full loop.
"""
import argparse
import json
import re
import time
from pathlib import Path

from data import REPO_ROOT, load_public_test, load_sample_dataset
from llm_client import LocalLLM, SEED, extract_code_block
from synth_bugs import BUG_TAXONOMY
from synth_label import derive_label
from synth_oracle import REF_DIR, build_oracle, grade_candidate

SYSTEM_MUTATOR = (
    "Bạn là một công cụ chỉnh sửa code C++ theo yêu cầu CHÍNH XÁC. "
    "Chỉ thực hiện ĐÚNG MỘT thay đổi được yêu cầu, không thay đổi gì khác. "
    "Trả lời CHỈ bằng code C++ hoàn chỉnh trong một code block, không giải thích."
)

GEN_DIR = REPO_ROOT / "synth/generated"
VARIANTS_PER_BUG = 3

# problem -> which real exam's statement/metadata to reuse (same problem, same rubric/policy)
PROBLEM_TO_REAL_EXAM = {
    "armstrong": ("EX02", "sample"),
    "remove_duplicates": ("Class03-Midterm-2", "public"),
    "earliest_login": ("Class03-Final-1", "public"),
}
# More problems beyond these 3 real ones aren't free: the only other single_problem real
# exam (Class03-Midterm-1) turns out to be a duplicate Armstrong instance, and the
# remaining real exams (EX01, Class01-Test1/2, Class02-Test1) are a different struct +
# file-I/O + linked-list family with no stdin/stdout main() -- would need a whole new
# oracle harness, too big a lift for a diversity pass. These two instead synthesize their
# own exam metadata (no real exam backing) but reuse the exact same stdin/stdout harness.
PROBLEM_SYNTHETIC_EXAM = {
    "binary_search": {
        "exam_type": "single_problem",
        "language": "cpp",
        "statement": (
            "## Tìm kiếm nhị phân\n\n"
            "Cho một mảng gồm n số nguyên ĐÃ SẮP XẾP TĂNG DẦN và một giá trị target. "
            "Viết chương trình tìm vị trí (chỉ số, bắt đầu từ 0) của target trong mảng. "
            "Nếu không tìm thấy, in ra -1. Yêu cầu dùng thuật toán tìm kiếm nhị phân "
            "(binary search), không duyệt tuyến tính toàn bộ mảng.\n\n"
            "**Input:** dòng 1: n; dòng 2: n số nguyên đã sắp xếp tăng; dòng 3: target.\n"
            "**Output:** chỉ số của target, hoặc -1 nếu không có."
        ),
        "grading_policy": "Chấm trên một bài duy nhất; điểm rubric là điểm của cả bài.",
    },
    "palindrome_check": {
        "exam_type": "single_problem",
        "language": "cpp",
        "statement": (
            "## Kiểm tra palindrome\n\n"
            "Viết chương trình đọc một dòng chuỗi (có thể chứa khoảng trắng, dấu câu, "
            "chữ hoa/thường) và kiểm tra chuỗi đó có là palindrome hay không SAU KHI bỏ "
            "qua mọi ký tự không phải chữ/số và không phân biệt chữ hoa/thường.\n\n"
            "**Input:** một dòng chuỗi.\n"
            "**Output:** in `True` nếu là palindrome, `False` nếu không."
        ),
        "grading_policy": "Chấm trên một bài duy nhất; điểm rubric là điểm của cả bài.",
    },
}
SYN_EXAM_ID = {
    "armstrong": "SYN-ARMSTRONG", "remove_duplicates": "SYN-DEDUP", "earliest_login": "SYN-LOGIN",
    "binary_search": "SYN-BINSEARCH", "palindrome_check": "SYN-PALINDROME",
}
ALL_PROBLEMS = list(PROBLEM_TO_REAL_EXAM) + list(PROBLEM_SYNTHETIC_EXAM)


def mutation_prompt(problem_name: str, ref_code: str, bug_id: str, variant_idx: int) -> str:
    instruction = BUG_TAXONOMY[bug_id]["instruction"]
    variation_hint = "" if variant_idx == 0 else f"\n(Đây là biến thể #{variant_idx + 1} — hãy chọn một cách khác để tạo lỗi này so với các biến thể trước, đa dạng hơn.)"
    return f"""Đây là một lời giải ĐÚNG cho bài toán:
```cpp
{ref_code}
```

Yêu cầu: {instruction}{variation_hint}

Trả lời CHỈ bằng code C++ đã sửa, trong một code block duy nhất.
"""


_CPP_TOKENS_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/', re.S)


def strip_cpp_comments(code: str) -> str:
    """The mutator LLM annotates its own edit in ~56% of samples ("// Sai == thành !=",
    "// Xóa dấu chấm phẩy ở đây") -- the answer written into the input. References have
    no comments, so drop them all. String literals are left alone and line numbers are
    preserved (feedback targets cite them)."""
    def repl(m):
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")
    return "\n".join(line.rstrip() for line in _CPP_TOKENS_RE.sub(repl, code).split("\n"))


def build_synthetic_exams():
    exams = {}
    for problem, (real_exam_id, split) in PROBLEM_TO_REAL_EXAM.items():
        ds = load_sample_dataset() if split == "sample" else load_public_test()
        real_exam = ds.exams[real_exam_id]
        syn_id = SYN_EXAM_ID[problem]
        exams[syn_id] = {**real_exam, "exam_id": syn_id, "note": f"Synthetic bug-injected variants of {real_exam_id}'s problem."}
    for problem, defn in PROBLEM_SYNTHETIC_EXAM.items():
        syn_id = SYN_EXAM_ID[problem]
        exams[syn_id] = {"exam_id": syn_id, "note": f"Synthetic bug-injected variants of a standalone {problem} problem (no real exam backing).", **defn}
    return exams


def generate(limit_bugs: int | None = None, variants: int = VARIANTS_PER_BUG):
    llm = LocalLLM()
    exams = build_synthetic_exams()
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    task1_samples, task2_samples = [], []
    n_discarded = 0
    bug_ids = list(BUG_TAXONOMY)[:limit_bugs] if limit_bugs else list(BUG_TAXONOMY)

    for problem in ALL_PROBLEMS:
        syn_exam_id = SYN_EXAM_ID[problem]
        ref_code = (REF_DIR / f"{problem}.cpp").read_text(encoding="utf-8")
        oracle_cases = build_oracle(problem)
        submissions_dir = GEN_DIR / "submissions" / syn_exam_id
        submissions_dir.mkdir(parents=True, exist_ok=True)

        for bug_id in bug_ids:
            for v in range(variants):
                sample_id = f"{syn_exam_id}-{bug_id}-{v}"
                t0 = time.time()
                prompt = mutation_prompt(problem, ref_code, bug_id, v)
                # v==0 stays deterministic (greedy); later variants are seeded samples so
                # "a different way to do this" actually diverges instead of repeating the
                # same greedy completion with a different hint sentence attached.
                temperature = None if v == 0 else 0.8
                raw = llm.generate(SYSTEM_MUTATOR, prompt, max_new_tokens=800, temperature=temperature, seed=v)
                code = extract_code_block(raw)
                if code is not None:
                    code = strip_cpp_comments(code)
                if code is None:
                    print(f"  SKIP {sample_id}: no code block extracted ({time.time()-t0:.1f}s)")
                    n_discarded += 1
                    continue

                cpp_path = submissions_dir / f"{sample_id}.cpp"
                cpp_path.write_text(code, encoding="utf-8")
                grade = grade_candidate(problem, cpp_path, oracle_cases, tag=sample_id)
                labeled = derive_label(grade, bug_id)
                if labeled is None:
                    grade_short = {k: (v[:150] + "..." if isinstance(v, str) and len(v) > 150 else v) for k, v in grade.items()}
                    print(f"  DISCARD {sample_id}: failed QC ({time.time()-t0:.1f}s) grade={grade_short}")
                    cpp_path.unlink()
                    n_discarded += 1
                    continue

                rel_code_path = f"submissions/{syn_exam_id}/{sample_id}.cpp"
                base_input = {
                    "exam_id": syn_exam_id,
                    "exam_type": exams[syn_exam_id]["exam_type"],
                    "language": exams[syn_exam_id]["language"],
                    "code_file": rel_code_path,
                    "compile_log": "",
                    "test_report": None,
                }
                task1_samples.append({
                    "sample_id": sample_id,
                    "input": base_input,
                    "output": {"rubric": labeled["rubric"], "total_score": labeled["total_score"]},
                })
                task2_samples.append({
                    "sample_id": sample_id,
                    "input": base_input,
                    "output": {"taxonomy_error": labeled["taxonomy_error"]},
                })
                print(f"  OK {sample_id} ({time.time()-t0:.1f}s): total={labeled['total_score']} labels={labeled['taxonomy_error']}")

    (GEN_DIR / "exams.json").write_text(json.dumps({"exams": list(exams.values())}, ensure_ascii=False, indent=2), encoding="utf-8")
    label_space_src = (load_sample_dataset().root / "label_space.json").read_text(encoding="utf-8")
    (GEN_DIR / "label_space.json").write_text(label_space_src, encoding="utf-8")
    (GEN_DIR / "task1_grading.json").write_text(
        json.dumps({"task": "task1_rubric_grading", "n_samples": len(task1_samples), "samples": task1_samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (GEN_DIR / "task2_error_taxonomy.json").write_text(
        json.dumps({"task": "task2_error_taxonomy", "n_samples": len(task2_samples), "samples": task2_samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nGenerated {len(task1_samples)} samples, discarded {n_discarded} (failed QC or extraction) -> {GEN_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true", help="1 bug type x 1 variant x 1 problem, before the full loop")
    ap.add_argument("--variants", type=int, default=VARIANTS_PER_BUG)
    args = ap.parse_args()
    if args.smoke_test:
        generate(limit_bugs=1, variants=1)
    else:
        generate(variants=args.variants)
