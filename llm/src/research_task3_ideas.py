"""Research spike: try 5 ideas to improve Task 3 self-diagnosis + feedback quality.
Does NOT touch prompts.py/run_prompting.py -- builds a local self-diagnose prompt
variant (sample_dataset normally gives taxonomy_error, so self-diagnosis is forced
here by simply never looking at that field) and reuses prompts.py's private helpers
(_problem_context, _fewshot_block_from_items) + the production given_errors branch
(task3_prompt itself) for idea 5's second pass.

Usage: python research_task3_ideas.py [--limit N]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import prompts as P
from compile_check import attach_real_compile_log
from data import REPO_ROOT, load_sample_dataset, read_code
from eval import task2_metrics, task3_compliance, task3_overlap_proxy
from llm_client import LocalLLM, SEED, extract_json
from rag import _get_model

OUT_DIR = REPO_ROOT / "outputs" / "research_task3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_CODE_CHARS_FOR_EXEMPLARS = 3000  # same VRAM-cliff guard as run_prompting.py


def build_self_diagnose_prompt(ds, sample, fewshot_items=None, diagnosis_hint=None, compile_guard=False):
    """Local copy of task3_prompt's self-diagnose (given_errors is None) branch,
    parametrized with an optional hint line / compile-log guard line."""
    code = read_code(ds, sample)
    level = sample["input"]["target_feedback_level"]
    level_defs = ds.label_space["feedback_level"]
    level_def = next((d for d in level_defs if d["level"] == level), None)
    level_desc = (
        f"{level}\nĐịnh nghĩa: {level_def['định nghĩa']}\nCấm: {level_def['cấm']}"
        if level_def else level
    )
    log = sample["input"].get("compile_log") or "(không có)"
    labels = ds.label_space["taxonomy_error"]
    labels_str = ", ".join(f'"{l}"' for l in labels)

    hint_line = ""
    if diagnosis_hint is not None:
        hint_line = (
            f"\nGợi ý tham khảo từ một hệ thống chẩn đoán khác (CÓ THỂ SAI, phải tự xác minh "
            f"lại qua code thực tế, không copy mù): {diagnosis_hint}"
        )
    guard_line = ""
    if compile_guard:
        guard_line = (
            "\nQUAN TRỌNG: nếu compile_log ở trên cho biết code BIÊN DỊCH THÀNH CÔNG, "
            "TUYỆT ĐỐI KHÔNG chọn nhãn \"Lỗi biên dịch\"."
        )
    diagnosis_block = (
        f"## Danh sách nhãn lỗi (taxonomy)\n[{labels_str}]\n\n"
        f"Nhãn lỗi KHÔNG được cho sẵn — hãy tự chẩn đoán trước khi viết phản hồi.{hint_line}{guard_line}"
    )
    reasoning_instr = (
        "Trước khi viết phản hồi, hãy tự phân tích ngắn gọn vào trường \"reasoning\": code đúng/sai "
        "ở đâu so với đề bài. \"reasoning\" không bị giới hạn độ dài và không tính vào mức độ phản hồi "
        "(Level) -- CHỈ \"feedback\" mới phải tuân luật ngắn gọn/không tiết lộ đáp án ở trên."
    )
    output_instruction = (
        f'{reasoning_instr}\nTrả lời CHỈ một JSON object đúng định dạng, "reasoning" đứng TRƯỚC "taxonomy_error":\n'
        '{"reasoning": "<phân tích ngắn gọn>", "taxonomy_error": ["<nhãn1>", ...], "feedback": "<đoạn phản hồi tiếng Việt>"}'
    )

    fewshot = ""
    if fewshot_items and len(code) <= MAX_CODE_CHARS_FOR_EXEMPLARS:
        output_fn = lambda ex: {"taxonomy_error": ex["input"]["taxonomy_error"], "feedback": ex["output"]["feedback"]}
        fewshot = P._fewshot_block_from_items(fewshot_items, output_fn)

    return f"""{fewshot}{P._problem_context(ds, sample)}

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
QUAN TRỌNG: phản hồi phải NGẮN GỌN đúng phong cách giáo viên thực tế — thường chỉ 1 câu, hoặc vài
vế ngắn nối bằng dấu ';' (ví dụ: "Tốt.", "P1 đúng; P2 sai công thức, em xem lại."). KHÔNG viết
đoạn văn dài, KHÔNG giải thích nhiều bước, KHÔNG lặp lại đề bài.
BẮT BUỘC viết TOÀN BỘ bằng tiếng Việt — KHÔNG chèn tiếng Trung, tiếng Anh hay ngôn ngữ khác vào phản hồi.
{output_instruction}
"""


def static_fewshot_items(ds, ids):
    return [(ds, next(s for s in ds.tasks[3] if s["sample_id"] == sid)) for sid in ids]


def sanitize(parsed, valid_labels):
    if parsed is None:
        return {"taxonomy_error": [], "feedback": ""}
    labels = parsed.get("taxonomy_error", [])
    if isinstance(labels, str):
        labels = [labels]
    elif not isinstance(labels, list):
        labels = []
    parsed["taxonomy_error"] = [l for l in labels if isinstance(l, str) and l in valid_labels]
    fb = parsed.get("feedback", "")
    if not isinstance(fb, str):
        fb = str(fb)
    parsed["feedback"] = fb
    return parsed


def run_variant(name, llm, ds, valid_labels, prompt_fn):
    """prompt_fn(sample) -> prompt string. Runs greedy over all task3 samples."""
    records = []
    for sample in ds.tasks[3]:
        prompt = prompt_fn(sample)
        raw = llm.generate(P.SYSTEM_GRADER, prompt, max_new_tokens=700)
        parsed = sanitize(extract_json(raw), valid_labels)
        records.append({"sample_id": sample["sample_id"], "pred_labels": parsed["taxonomy_error"], "pred_feedback": parsed["feedback"]})
    (OUT_DIR / f"{name}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def run_variant_selfconsistency(name, llm, ds, valid_labels, fewshot_items_fn, fewshot_ds_full, n_votes=3):
    records = []
    for sample in ds.tasks[3]:
        items = fewshot_items_fn(sample)
        prompt = build_self_diagnose_prompt(ds, sample, fewshot_items=items)
        label_sets = []
        for v in range(n_votes):
            raw = llm.generate(P.SYSTEM_GRADER, prompt, max_new_tokens=700, temperature=0.7, seed=SEED + v)
            parsed = sanitize(extract_json(raw), valid_labels)
            label_sets.append(set(parsed["taxonomy_error"]))
        counts = {}
        for s in label_sets:
            for l in s:
                counts[l] = counts.get(l, 0) + 1
        voted = [l for l, c in counts.items() if c / n_votes >= 0.5]
        sample2 = dict(sample)
        sample2["input"] = dict(sample["input"])
        sample2["input"]["taxonomy_error"] = voted
        # fewshot_ds_full is a separately-loaded, never-truncated Dataset so the static
        # FEWSHOT_IDS lookup inside task3_prompt's given_errors branch always resolves,
        # independent of any --limit applied to the evaluation set `ds`.
        final_prompt = P.task3_prompt(ds, sample2, fewshot_ds=fewshot_ds_full)
        raw2 = llm.generate(P.SYSTEM_GRADER, final_prompt, max_new_tokens=700)
        parsed2 = sanitize(extract_json(raw2), valid_labels)
        records.append({"sample_id": sample["sample_id"], "pred_labels": voted, "pred_feedback": parsed2["feedback"]})
    (OUT_DIR / f"{name}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def score(name, ds, records):
    gold_by_id = {s["sample_id"]: s for s in ds.tasks[3]}
    label_space = ds.label_space["taxonomy_error"]
    gold_samples = [{"sample_id": sid, "output": {"taxonomy_error": g["input"]["taxonomy_error"]}} for sid, g in gold_by_id.items()]
    pred_by_id = {r["sample_id"]: r["pred_labels"] for r in records}
    m2 = task2_metrics(gold_samples, pred_by_id, label_space)

    jaccards = []
    for r in records:
        gold_set = set(gold_by_id[r["sample_id"]]["input"]["taxonomy_error"])
        pred_set = set(r["pred_labels"])
        union = gold_set | pred_set
        jaccards.append(len(gold_set & pred_set) / len(union) if union else 1.0)
    mean_jaccard = sum(jaccards) / len(jaccards)

    level_by_id = {sid: g["input"]["target_feedback_level"] for sid, g in gold_by_id.items()}
    triples = [(r["sample_id"], level_by_id[r["sample_id"]], r["pred_feedback"]) for r in records]
    comp = task3_compliance(triples)

    gold_fb_by_id = {sid: g["output"]["feedback"] for sid, g in gold_by_id.items()}
    overlap_triples = [(r["sample_id"], r["pred_feedback"], gold_fb_by_id[r["sample_id"]]) for r in records]
    overlap = task3_overlap_proxy(overlap_triples)

    return {
        "name": name,
        "macro_f1": round(m2["macro_f1"], 4),
        "micro_f1": round(m2["micro_f1"], 4),
        "mean_jaccard": round(mean_jaccard, 4),
        "compliance_rate": round(comp["compliance_rate"], 4),
        "overlap_f1": round(overlap["mean_overlap_f1"], 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ds = load_sample_dataset()
    attach_real_compile_log(ds, 3)
    valid_labels = set(ds.label_space["taxonomy_error"])

    static_ids = P.FEWSHOT_IDS[3]  # ["EX02-S201", "EX01-S101", "EX02-S207"]
    static_items_all = static_fewshot_items(ds, static_ids)  # resolved BEFORE any --limit truncation
    fewshot_ds_full = load_sample_dataset()  # separate, never-truncated instance for idea 5's final call

    if args.limit:
        ds.tasks[3] = ds.tasks[3][: args.limit]

    print(f"Loading model on {len(ds.tasks[3])} Task 3 samples...")
    llm = LocalLLM()

    def static_fewshot_fn(sample):
        return [(d, s) for d, s in static_items_all if s["sample_id"] != sample["sample_id"]]

    results = []

    # --- baseline: self-diagnose, static 3-exemplar fewshot, no hint ---
    recs = run_variant("baseline", llm, ds, valid_labels,
                        lambda s: build_self_diagnose_prompt(ds, s, fewshot_items=static_fewshot_fn(s)))
    results.append(score("baseline", ds, recs))

    # --- idea 1: diagnosis hint from Task 2's own (zero-shot, no fewshot) predictions ---
    print("idea1: generating Task 2 zero-shot hints on the same 32 samples...")
    hint_by_id = {}
    for sample in ds.tasks[3]:
        t2_prompt = P.task2_prompt(ds, sample)  # zero-shot, no fewshot_ds
        raw = llm.generate(P.SYSTEM_GRADER, t2_prompt, max_new_tokens=700)
        parsed = extract_json(raw) or {}
        labels = parsed.get("taxonomy_error", [])
        hint_by_id[sample["sample_id"]] = [l for l in labels if isinstance(l, str) and l in valid_labels] if isinstance(labels, list) else []
    recs = run_variant("idea1_task2_hint", llm, ds, valid_labels,
                        lambda s: build_self_diagnose_prompt(ds, s, fewshot_items=static_fewshot_fn(s), diagnosis_hint=hint_by_id[s["sample_id"]]))
    results.append(score("idea1_task2_hint", ds, recs))

    # --- idea 2: RAG-style same-level nearest-neighbor single exemplar ---
    print("idea2: embedding Task 3 pool for same-level nearest-neighbor exemplar...")
    model = _get_model()
    codes = [read_code(ds, s) for s in ds.tasks[3]]
    embeddings = model.encode(codes, normalize_embeddings=True)

    def rag_fewshot_fn(sample):
        idx = ds.tasks[3].index(sample)
        level = sample["input"]["target_feedback_level"]
        sims = embeddings @ embeddings[idx]
        order = np.argsort(-sims)
        for i in order:
            cand = ds.tasks[3][i]
            if cand["sample_id"] == sample["sample_id"]:
                continue
            if cand["input"]["target_feedback_level"] == level:
                return [(ds, cand)]
        return []

    recs = run_variant("idea2_rag_exemplar", llm, ds, valid_labels,
                        lambda s: build_self_diagnose_prompt(ds, s, fewshot_items=rag_fewshot_fn(s)))
    results.append(score("idea2_rag_exemplar", ds, recs))

    # --- idea 3: compile_log logic guard ---
    recs = run_variant("idea3_compile_guard", llm, ds, valid_labels,
                        lambda s: build_self_diagnose_prompt(ds, s, fewshot_items=static_fewshot_fn(s), compile_guard=True))
    results.append(score("idea3_compile_guard", ds, recs))

    # --- idea 4: 2 short exemplars per level (<=1200 chars), 6 total ---
    MAX_EXEMPLAR_CHARS = 1200
    by_level = {}
    for s in ds.tasks[3]:
        if len(read_code(ds, s)) <= MAX_EXEMPLAR_CHARS:
            by_level.setdefault(s["input"]["target_feedback_level"], []).append(s)
    richer_items_all = []
    for level, items in by_level.items():
        for s in items[:2]:
            richer_items_all.append((ds, s))
    print(f"idea4: richer fewshot pool = {len(richer_items_all)} items across {len(by_level)} levels")

    def richer_fewshot_fn(sample):
        return [(d, s) for d, s in richer_items_all if s["sample_id"] != sample["sample_id"]]

    recs = run_variant("idea4_more_exemplars", llm, ds, valid_labels,
                        lambda s: build_self_diagnose_prompt(ds, s, fewshot_items=richer_fewshot_fn(s)))
    results.append(score("idea4_more_exemplars", ds, recs))

    # --- idea 5: self-consistency voting (3x sampled diagnosis -> 1 final feedback call) ---
    print("idea5: self-consistency voting (3 samples per item)...")
    recs = run_variant_selfconsistency("idea5_self_consistency", llm, ds, valid_labels, static_fewshot_fn, fewshot_ds_full, n_votes=3)
    results.append(score("idea5_self_consistency", ds, recs))

    print("\n=== RESULTS (sample_dataset, n=%d) ===" % len(ds.tasks[3]))
    header = f"{'variant':<24} {'macro_f1':>9} {'micro_f1':>9} {'jaccard':>9} {'compliance':>11} {'overlap_f1':>11}"
    print(header)
    for r in results:
        print(f"{r['name']:<24} {r['macro_f1']:>9} {r['micro_f1']:>9} {r['mean_jaccard']:>9} {r['compliance_rate']:>11} {r['overlap_f1']:>11}")
    (OUT_DIR / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
