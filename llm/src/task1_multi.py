"""Task 1 v3: per-problem scoring for multi-problem exams + compile rule for every item.

Why: every earlier Task 1 file scored public multi-problem submissions ~0.7-1.2/10 (single:
~5.6) -> leaderboard QWK -0.21. Causes found in the model's reasoning:
  - the old link-based compile log called function-only files (no main, as the exam
    requires) uncompilable -> 20/61 compiling files got compilable=0, total 0;
  - the statement's "If you fail to read the file, the remaining problems will not be
    graded" + the P1 prerequisite policy -> the model can't run the file read, assumes P1
    fails, zeroes P2-P4. Instructor QA: ignore that sentence; problems weigh EQUALLY.
Fix:
  - compile rule (teacher pattern, 6/6 on sample): real g++ failure -> multi all 0,
    single code_quality=1 only; clean compile -> compilable=1. No LLM call needed.
  - multi + compiles: the model scores EACH problem on the 6 dims (prerequisite sentence and
    policy removed, point weights replaced by "equal"), dims averaged over problems, rounded.
  - single-problem items: taken from --base (another run's Task 1 file) + the compile rule.

Usage:
  python -u task1_multi.py --split sample                         # 15 EX01 items vs gold
  python -u task1_multi.py --split public --base <task1 public json> --out-tag <tag>
"""
import argparse
import json
import re

from compile_check import attach_real_compile_log
from data import REPO_ROOT, load_public_test, load_sample_dataset, read_code
from eval import task1_metrics
from llm_client import LocalLLM, extract_json
from prompts import SYSTEM_GRADER

OUT = REPO_ROOT / "outputs"


def clean_statement(statement: str) -> str:
    lines = [l for l in statement.splitlines() if "fail to read the file" not in l]
    text = "\n".join(lines)
    return re.sub(r"(\*\*Problem \d+) \(\d+ điểm\)", r"\1 (trọng số bằng các câu khác)", text)


def problem_ids(statement: str) -> list:
    return [f"P{n}" for n in re.findall(r"\*\*Problem (\d+)", statement)]


def multi_prompt(ds, sample) -> str:
    exam = ds.exams[sample["input"]["exam_id"]]
    probs = problem_ids(exam["statement"])
    dims = ds.label_space["rubric"]["dimensions"]
    rubric_desc = "\n".join(f"- {d['name']} ({d['range'][0]}-{d['range'][1]}): {d['vi']}" for d in dims)
    one = ", ".join(f'"{d["name"]}": <{d["range"][0]}-{d["range"][1]}>' for d in dims)
    schema = ", ".join(f'"{p}": {{{one}}}' for p in probs)
    return f"""## Đề bài ({exam['exam_id']}, nhiều câu: {', '.join(probs)})
{clean_statement(exam['statement'])}

## Quy tắc chấm (thay cho mọi quy định trong đề)
- Chấm TỪNG câu {', '.join(probs)} ĐỘC LẬP theo rubric 6 chiều; mỗi câu có trọng số NGANG NHAU.
- KHÔNG áp dụng ràng buộc tiên quyết: câu P1 (đọc file) sai hay thiếu thì các câu sau VẪN được chấm bình thường.
- File dữ liệu không có sẵn: đánh giá phần đọc file bằng cách đọc code, không cần chạy.
- Bài nộp chỉ gồm các hàm (không có main) là ĐÚNG quy định, không phải lỗi.
- Câu bỏ trống / không làm -> mọi chiều của câu đó = 0.

## Code sinh viên
```cpp
{read_code(ds, sample)}
```

## compile_log (g++ -c, kiểm tra biên dịch thật)
{sample['input'].get('compile_log') or '(không có)'}

## Rubric 6 chiều (áp dụng cho TỪNG câu)
{rubric_desc}

Trước khi chấm, phân tích ngắn gọn từng câu vào trường "reasoning" (đúng/sai ở đâu).
Trả lời CHỈ một JSON object, "reasoning" đứng TRƯỚC:
{{"reasoning": "<phân tích ngắn từng câu>", {schema}}}
"""


def clamp_rubric(rubric: dict, dims: list) -> dict:
    out = {}
    for d in dims:
        lo, hi = d["range"]
        v = rubric.get(d["name"], 0)
        out[d["name"]] = min(max(int(v) if isinstance(v, (int, float)) else 0, lo), hi)
    return out


def aggregate(parsed: dict, probs: list, dims: list) -> dict:
    """Equal-weight mean of each dim over problems, rounded half up."""
    per = [clamp_rubric(parsed.get(p) or {}, dims) for p in probs]
    return {d["name"]: int(sum(r[d["name"]] for r in per) / len(per) + 0.5) for d in dims}


def compile_rule(inp: dict, rubric: dict | None) -> dict | None:
    """Teacher pattern for real compile failures; None = not decided by the rule."""
    if not (inp.get("compile_log") or "").startswith("Lỗi biên dịch"):
        if rubric is not None:
            rubric = dict(rubric, compilable=1)
        return rubric
    zero = {k: 0 for k in ("compilable", "io_format", "logic", "edge_case", "complexity", "code_quality")}
    if inp["exam_type"] == "single_problem":
        zero["code_quality"] = 1
    return zero


def score_multi(llm, ds, items, ckpt_path):
    dims = ds.label_space["rubric"]["dimensions"]
    done = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["sample_id"]] = r["rubric"]
    with open(ckpt_path, "a", encoding="utf-8") as ck:
        for i, s in enumerate(items):
            sid = s["sample_id"]
            if sid in done:
                continue
            ruled = compile_rule(s["input"], None)
            if ruled is not None:
                rubric, raw = ruled, "(compile rule)"
            else:
                probs = problem_ids(ds.exams[s["input"]["exam_id"]]["statement"])
                raw = llm.generate(SYSTEM_GRADER, multi_prompt(ds, s), max_new_tokens=800)
                rubric = compile_rule(s["input"], aggregate(extract_json(raw) or {}, probs, dims))
            done[sid] = rubric
            ck.write(json.dumps({"sample_id": sid, "rubric": rubric, "raw": raw}, ensure_ascii=False) + "\n")
            ck.flush()
            print(f"  [{i + 1}/{len(items)}] {sid} total={sum(rubric.values())}", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["sample", "public"], required=True)
    ap.add_argument("--base", help="public: Task 1 file whose single-problem predictions are kept")
    ap.add_argument("--out-tag", default="task1-multi-v3")
    args = ap.parse_args()

    ds = load_sample_dataset() if args.split == "sample" else load_public_test()
    attach_real_compile_log(ds, 1)
    multi = [s for s in ds.tasks[1] if s["input"]["exam_type"] == "multi_problem"]
    run_dir = OUT / args.out_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    n_llm = sum(1 for s in multi if compile_rule(s["input"], None) is None)
    print(f"{args.split}: {len(multi)} multi-problem items, {n_llm} need the LLM (rest = compile rule)")
    llm = LocalLLM() if n_llm else None
    multi_pred = score_multi(llm, ds, multi, run_dir / f"multi_{args.split}.ckpt.jsonl")

    if args.split == "sample":
        pred = {sid: {"rubric": r, "total_score": sum(r.values())} for sid, r in multi_pred.items()}
        m = task1_metrics(multi, pred)
        print(f"sample multi (n={len(multi)}): QWK {m['qwk']:.3f} MAE {m['mae']:.2f} exact {m['exact_match']}")
        print("pred :", [pred[s['sample_id']]['total_score'] for s in multi])
        print("gold :", [s['output']['total_score'] for s in multi])
        return

    base = {x["sample_id"]: x["output"] for x in json.loads(open(args.base, encoding="utf-8").read())}
    out = []
    for s in ds.tasks[1]:
        sid = s["sample_id"]
        rubric = multi_pred[sid] if sid in multi_pred else compile_rule(s["input"], base[sid]["rubric"])
        out.append({"sample_id": sid, "output": {"rubric": rubric, "total_score": sum(rubric.values())}})
    path = run_dir / "predictions_task1_public.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
