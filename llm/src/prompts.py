"""Prompt templates for Task 1/2/3, zero-shot and few-shot."""
import json

from data import Dataset, read_code

SYSTEM_GRADER = (
    "Bạn là trợ giảng chấm bài lập trình C++ cho sinh viên. "
    "Chấm nghiêm túc, khách quan, chỉ dựa trên đề bài và code được cung cấp. "
    "Trả lời ĐÚNG định dạng JSON được yêu cầu, không thêm giải thích ngoài JSON."
)

# Hand-picked from sample_dataset for diversity, not random. Kept deliberately SHORT
# (2-3 exemplars, not more): a 4-exemplar set pushed total input tokens from ~4241 to
# ~4799 on real samples, crossing the RTX 3060's 12GB VRAM ceiling and silently spilling
# into ~15-20x slower Windows "shared GPU memory" -- see llm/CLAUDE.md Tooling note.
# Task 1 -- spans low/mid/high of the total_score range (0, 6, 10) across both exam types,
#   since calibration (not comprehension) was the identified zero-shot weakness.
# Task 2 -- empty-label + one rare multi-label case (Lỗi mảng/chuỗi, only 1 occurrence
#   in the whole 32-sample set). Deliberately EXCLUDES any single-label compile-error
#   exemplar: an earlier version including EX01-S109 ("Lỗi biên dịch") caused the model to
#   anchor on that exact label for 23/32 samples regardless of actual code shown, and this
#   held regardless of the exemplar's position in the list (not just recency bias). See METHOD.md.
# Task 3 -- one exemplar per level actually present in the data (1, 2, 3); already short
#   enough (~2K tokens) that no reduction was needed.
FEWSHOT_IDS = {
    1: ["EX01-S109", "EX01-S108", "EX02-S211"],
    2: ["EX01-S115", "EX01-S105"],
    3: ["EX02-S201", "EX01-S101", "EX02-S207"],
}


def _problem_context(ds: Dataset, sample: dict) -> str:
    exam = ds.exams[sample["input"]["exam_id"]]
    parts = [f"## Đề bài ({exam['exam_id']}, {exam['exam_type']})", exam["statement"]]
    if exam.get("grading_policy"):
        parts.append(f"\n## Chính sách chấm\n{exam['grading_policy']}")
    return "\n".join(parts)


def _find_sample(fewshot_ds: Dataset, task_num: int, sample_id: str) -> dict:
    return next(s for s in fewshot_ds.tasks[task_num] if s["sample_id"] == sample_id)


def _fewshot_block_from_items(items: list, output_fn) -> str:
    """items: list of (ds, sample) tuples, already selected (static IDs or RAG retrieval).
    output_fn(sample) -> the JSON dict an exemplar should demonstrate as output."""
    if not items:
        return ""
    parts = [
        "# Ví dụ minh hoạ (few-shot) — mỗi ví dụ ĐỘC LẬP, chỉ để tham khảo cách trả lời đúng định dạng.\n"
        "QUAN TRỌNG: KHÔNG copy đáp án của bất kỳ ví dụ nào. Đáp án thật phải dựa HOÀN TOÀN vào code "
        "thực tế được đưa ra sau phần ví dụ, không dựa vào việc code đó 'giống' ví dụ nào.\n"
    ]
    for ex_ds, ex in items:
        code = read_code(ex_ds, ex)
        answer = json.dumps(output_fn(ex), ensure_ascii=False)
        parts.append(f"### Ví dụ: {ex['sample_id']}\n```cpp\n{code}\n```\nĐáp án: {answer}\n")
    parts.append("# Hết ví dụ. Bây giờ hãy làm với bài nộp thật dưới đây.\n")
    return "\n".join(parts)


def _fewshot_block(fewshot_ds: Dataset, task_num: int, exclude_id: str | None, output_fn) -> str:
    """Static exemplar selection from FEWSHOT_IDS, resolved within a single Dataset."""
    ids = [sid for sid in FEWSHOT_IDS[task_num] if sid != exclude_id]
    items = [(fewshot_ds, _find_sample(fewshot_ds, task_num, sid)) for sid in ids]
    return _fewshot_block_from_items(items, output_fn)


def task1_prompt(ds: Dataset, sample: dict, fewshot_ds: Dataset | None = None, fewshot_items: list | None = None) -> str:
    code = read_code(ds, sample)
    rubric_dims = ds.label_space["rubric"]["dimensions"]
    rubric_desc = "\n".join(f"- {d['name']} ({d['range'][0]}-{d['range'][1]}): {d['vi']}" for d in rubric_dims)
    log = sample["input"].get("compile_log") or "(không có)"
    report = sample["input"].get("test_report") or "(không có)"
    output_fn = lambda ex: {"rubric": ex["output"]["rubric"], "total_score": ex["output"]["total_score"]}
    if fewshot_items is not None:
        fewshot = _fewshot_block_from_items(fewshot_items, output_fn)
    elif fewshot_ds is not None:
        fewshot = _fewshot_block(fewshot_ds, 1, sample["sample_id"], output_fn)
    else:
        fewshot = ""
    return f"""{fewshot}{_problem_context(ds, sample)}

## Code sinh viên
```cpp
{code}
```

## compile_log
{log}

## test_report
{report}

## Rubric 6 chiều
{rubric_desc}

Hãy chấm bài trên theo rubric. Trả lời CHỈ một JSON object đúng định dạng:
{{"rubric": {{"compilable": <0-1>, "io_format": <0-1>, "logic": <0-4>, "edge_case": <0-2>, "complexity": <0-1>, "code_quality": <0-1>}}, "total_score": <0-10>}}
"""


def task2_prompt(ds: Dataset, sample: dict, fewshot_ds: Dataset | None = None, fewshot_items: list | None = None) -> str:
    code = read_code(ds, sample)
    labels = ds.label_space["taxonomy_error"]
    labels_str = ", ".join(f'"{l}"' for l in labels)
    log = sample["input"].get("compile_log") or "(không có)"
    report = sample["input"].get("test_report") or "(không có)"
    output_fn = lambda ex: {"taxonomy_error": ex["output"]["taxonomy_error"]}
    if fewshot_items is not None:
        fewshot = _fewshot_block_from_items(fewshot_items, output_fn)
    elif fewshot_ds is not None:
        fewshot = _fewshot_block(fewshot_ds, 2, sample["sample_id"], output_fn)
    else:
        fewshot = ""
    return f"""{fewshot}{_problem_context(ds, sample)}

## Code sinh viên
```cpp
{code}
```

## compile_log
{log}

## test_report
{report}

## Danh sách nhãn lỗi (taxonomy)
[{labels_str}]

Hãy xác định các lỗi mà bài nộp trên mắc phải, chỉ chọn trong danh sách nhãn trên.
Nếu bài không có lỗi nào, trả về danh sách rỗng ([]), KHÔNG được trả về object rỗng ({{}}).
Trả lời CHỈ một JSON object đúng định dạng, luôn có key "taxonomy_error" dù danh sách rỗng hay không, ví dụ:
{{"taxonomy_error": []}}
hoặc
{{"taxonomy_error": ["Lỗi logic", "Lỗi vòng lặp"]}}
"""


def task3_prompt(ds: Dataset, sample: dict, fewshot_ds: Dataset | None = None, fewshot_items: list | None = None) -> str:
    """Schema forks depending on the split:
    - sample_dataset: taxonomy_error given in input -> output is {feedback} only.
    - public_test_release: taxonomy_error NOT given -> model must self-diagnose,
      output is {taxonomy_error, feedback} (matches README's original Task 3 spec).
    """
    code = read_code(ds, sample)
    level = sample["input"]["target_feedback_level"]
    level_defs = ds.label_space["feedback_level"]
    level_def = next((d for d in level_defs if d["level"] == level), None)
    level_desc = (
        f"{level}\nĐịnh nghĩa: {level_def['định nghĩa']}\nCấm: {level_def['cấm']}"
        if level_def
        else level
    )
    log = sample["input"].get("compile_log") or "(không có)"
    given_errors = sample["input"].get("taxonomy_error")

    if given_errors is not None:
        diagnosis_block = f"## Nhãn lỗi đã xác định (cho sẵn, không cần tự chẩn đoán lại)\n{given_errors}"
        output_instruction = 'Trả lời CHỈ một JSON object đúng định dạng:\n{"feedback": "<đoạn phản hồi tiếng Việt>"}'
    else:
        labels = ds.label_space["taxonomy_error"]
        labels_str = ", ".join(f'"{l}"' for l in labels)
        diagnosis_block = f"## Danh sách nhãn lỗi (taxonomy)\n[{labels_str}]\n\nNhãn lỗi KHÔNG được cho sẵn — hãy tự chẩn đoán trước khi viết phản hồi."
        output_instruction = (
            'Trả lời CHỈ một JSON object đúng định dạng:\n'
            '{"taxonomy_error": ["<nhãn1>", ...], "feedback": "<đoạn phản hồi tiếng Việt>"}'
        )

    if fewshot_ds is not None:
        if given_errors is not None:
            output_fn = lambda ex: {"feedback": ex["output"]["feedback"]}
        else:
            output_fn = lambda ex: {"taxonomy_error": ex["input"]["taxonomy_error"], "feedback": ex["output"]["feedback"]}
        fewshot = _fewshot_block(fewshot_ds, 3, sample["sample_id"], output_fn)
    else:
        fewshot = ""

    return f"""{fewshot}{_problem_context(ds, sample)}

## Code sinh viên
```cpp
{code}
```

## compile_log
{log}

{diagnosis_block}

## Mức phản hồi yêu cầu
{level_desc}

Hãy viết phản hồi tiếng Việt cho sinh viên, ĐÚNG mức độ yêu cầu ở trên (không vượt mức cho phép).
{output_instruction}
"""
