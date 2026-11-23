"""Task 1 v3: per-problem scoring for multi-problem exams + compile rule for every item.

Why: every earlier Task 1 file scored public multi-problem submissions ~0.7-1.2/10 (single:
~5.6). Causes found in the model's reasoning:
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

Options (v5):
  --model/--thinking  e.g. Qwen/Qwen3-8B with thinking (budget-forced, see LocalLLM)
  --anchor            one teacher-graded EX01 submission (same exam style as Class01/02:
                      General struct, CSV, linked list) as a SCALE example -- rubric numbers
                      only, never the feedback field. Most similar compiling EX01 item by code
                      embedding; on --split sample it is leave-one-out (never the item itself).

Usage:
  python -u task1_multi.py --split sample [--model Qwen/Qwen3-8B --thinking] [--anchor] --out-tag <tag>
  python -u task1_multi.py --split public --base <task1 public json> --out-tag <tag> [same options]
"""
import argparse
import json
import re

from compile_check import attach_real_compile_log
from test_report import attach_test_report
from data import REPO_ROOT, load_public_test, load_sample_dataset, read_code
from eval import task1_metrics
from llm_client import MODEL_NAME, LocalLLM, extract_json
from prompts import SYSTEM_GRADER

OUT = REPO_ROOT / "outputs"
ANCHOR_MAX_CHARS = 4000  # anchor code cap; keeps prompt + thinking well under the 12GB VRAM cliff


def clean_statement(statement: str) -> str:
    lines = [l for l in statement.splitlines() if "fail to read the file" not in l]
    text = "\n".join(lines)
    return re.sub(r"(\*\*Problem \d+) \(\d+ điểm\)", r"\1 (trọng số bằng các câu khác)", text)


def problem_ids(statement: str) -> list:
    return [f"P{n}" for n in re.findall(r"\*\*Problem (\d+)", statement)]


def anchor_block(anchor) -> str:
    if anchor is None:
        return ""
    a_ds, a = anchor
    out = a["output"]
    return f"""## Ví dụ tham khảo: một bài CÙNG DẠNG ĐỀ ({a['input']['exam_id']}) đã được giảng viên chấm
```cpp
{read_code(a_ds, a)}
```
Điểm giảng viên cho cả bài:{json.dumps(out['rubric'])}, tổng {out['total_score']}/10.
Dùng ví dụ này để hiểu THANG ĐIỂM / mức khắt khe của giảng viên; đề khác nên KHÔNG sao chép điểm.

"""


def pick_anchors(items, items_ds, pool_ds):
    """{sample_id: (pool_ds, anchor)}: most similar compiling multi-problem pool item by code
    embedding, never the item itself (leave-one-out when items and pool are the same split)."""
    from rag import _get_model
    pool = [s for s in pool_ds.tasks[1] if s["input"]["exam_type"] == "multi_problem"
            and not (s["input"].get("compile_log") or "").startswith("Lỗi biên dịch")
            and len(read_code(pool_ds, s)) <= ANCHOR_MAX_CHARS]
    enc = _get_model()
    P = enc.encode([read_code(pool_ds, s) for s in pool], normalize_embeddings=True)
    T = enc.encode([read_code(items_ds, s) for s in items], normalize_embeddings=True)
    out = {}
    for s, t in zip(items, T):
        order = sorted(range(len(pool)), key=lambda j: -float(P[j] @ t))
        out[s["sample_id"]] = next((pool_ds, pool[j]) for j in order if pool[j]["sample_id"] != s["sample_id"])
    return out


def multi_prompt(ds, sample, anchor=None) -> str:
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

{anchor_block(anchor)}Trước khi chấm, phân tích ngắn gọn từng câu vào trường "reasoning" (đúng/sai ở đâu).
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


def test_rule(inp: dict) -> dict | None:
    """Single-problem rubric from verified test outcomes, mirroring the teacher's pattern on
    the sample (EX02): all pass -> 10; only format failures -> 9 (io_format 0); some pass +
    some wrong -> 7 (logic 2, edge 1); nothing passes -> 2 (logic/edge/io 0). None = no
    report or no-code / no-compile (handled elsewhere)."""
    rep = inp.get("test_report") or ""
    m = re.search(r"đúng (\d+)/(\d+)\.", rep)
    if inp["exam_type"] != "single_problem" or not m:
        return None
    ok, n = int(m.group(1)), int(m.group(2))
    fmt = sum(int(x) for x in re.findall(r"- (\d+) test: nội dung đúng nhưng SAI ĐỊNH DẠNG", rep))
    full = {"compilable": 1, "io_format": 1, "logic": 4, "edge_case": 2, "complexity": 1, "code_quality": 1}
    if ok == n:
        return full
    if ok + fmt == n:
        return dict(full, io_format=0)
    if ok > 0:
        return dict(full, logic=2, edge_case=1)
    return dict(full, io_format=0, logic=0, edge_case=0)


def score_multi(ds, items, ckpt_path, model=MODEL_NAME, thinking=False, anchors=None):
    llm = None  # loaded on first item that needs it (fully checkpointed runs never load it)
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
                llm = llm or LocalLLM(model_name=model, thinking=thinking)
                anchor = (anchors or {}).get(sid)
                raw = llm.generate(SYSTEM_GRADER, multi_prompt(ds, s, anchor), max_new_tokens=800)
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
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--thinking", action="store_true", help="Qwen3-style thinking (budget-forced)")
    ap.add_argument("--anchor", action="store_true", help="one teacher-graded EX01 example as a scale anchor (rubric only)")
    ap.add_argument("--single-test-rule", action="store_true", help="single-problem items with a test_report: rubric from test outcomes (test_rule)")
    args = ap.parse_args()

    ds = load_sample_dataset() if args.split == "sample" else load_public_test()
    attach_real_compile_log(ds, 1)
    attach_test_report(ds, 1)
    multi = [s for s in ds.tasks[1] if s["input"]["exam_type"] == "multi_problem"]
    run_dir = OUT / args.out_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    n_llm = sum(1 for s in multi if compile_rule(s["input"], None) is None)
    print(f"{args.split}: {len(multi)} multi-problem items, {n_llm} need the LLM (rest = compile rule)")
    anchors = None
    if args.anchor:
        pool_ds = ds if args.split == "sample" else load_sample_dataset()
        if pool_ds is not ds:
            attach_real_compile_log(pool_ds, 1)
        anchors = pick_anchors(multi, ds, pool_ds)
    multi_pred = score_multi(ds, multi, run_dir / f"multi_{args.split}.ckpt.jsonl", args.model, args.thinking, anchors)

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
        if sid in multi_pred:
            rubric = multi_pred[sid]
        else:
            ruled = test_rule(s["input"]) if args.single_test_rule else None
            rubric = compile_rule(s["input"], ruled or base[sid]["rubric"])
        out.append({"sample_id": sid, "output": {"rubric": rubric, "total_score": sum(rubric.values())}})
    path = run_dir / "predictions_task1_public.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
