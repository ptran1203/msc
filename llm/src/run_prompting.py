"""Run the zero-shot prompting baseline over a dataset split and write predictions.json per task.

Usage:
    python run_prompting.py --split sample --tasks 1,2,3     # validates against gold labels
    python run_prompting.py --split public --tasks 1,2,3     # produces submission files
"""
import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from data import REPO_ROOT, load_public_test, load_sample_dataset, read_code
from eval import task1_metrics, task2_metrics, task3_compliance, task3_overlap_proxy
from compile_check import attach_real_compile_log
from llm_client import LocalLLM, MODEL_NAME, SEED, extract_json
from prompts import SYSTEM_GRADER, task1_prompt, task2_prompt, task3_prompt
from rag import RagIndex

OUTPUT_DIR = REPO_ROOT / "outputs"
_CJK_RE = re.compile(r"[一-鿿　-〿＀-￯]")  # Han ideographs + CJK/fullwidth punctuation
# A few public-test submissions (the multi-problem, file-I/O/struct-based exams) run
# 5000+ chars -- combined with few-shot/RAG exemplars this crosses the RTX 3060's VRAM
# cliff (~4241-4799 input tokens, see llm/CLAUDE.md) and stalls for tens of minutes per
# sample instead of ~10s. Zero-shot alone handles these fine (confirmed: this exact
# sample graded normally under zero-shot), so just drop exemplars rather than truncate
# the student's code -- the code itself must stay intact for accurate grading.
MAX_CODE_CHARS_FOR_EXEMPLARS = 3000


RUBRIC_DEFAULT = {d: 0 for d in ["compilable", "io_format", "logic", "edge_case", "complexity", "code_quality"]}

DEFAULTS = {
    1: {"rubric": dict(RUBRIC_DEFAULT), "total_score": 0},
    2: {"taxonomy_error": []},
    3: {"feedback": ""},  # "taxonomy_error" default added dynamically if the split requires it
}


def required_fields(task_num, ds):
    if task_num != 3:
        return {1: {"rubric", "total_score"}, 2: {"taxonomy_error"}}[task_num]
    task3_self_diagnoses = "taxonomy_error" not in ds.tasks[3][0]["input"]
    return {"taxonomy_error", "feedback"} if task3_self_diagnoses else {"feedback"}


def run_task(llm, ds, task_num, log_path, fewshot_ds=None, rag_index=None, rag_k=2):
    builder = {1: task1_prompt, 2: task2_prompt, 3: task3_prompt}[task_num]
    samples = ds.tasks[task_num]
    required = required_fields(task_num, ds)
    defaults = dict(DEFAULTS[task_num])
    if task_num == 3 and "taxonomy_error" in required:
        defaults["taxonomy_error"] = []

    predictions = []
    n_parse_fail = 0
    n_patched = 0
    n_cjk_retried = 0
    n_exemplars_dropped = 0
    with open(log_path, "w", encoding="utf-8") as logf:
        bar = tqdm(samples, desc=f"task{task_num}", unit="sample")
        for sample in bar:
            query_code = read_code(ds, sample)
            code_too_long = len(query_code) > MAX_CODE_CHARS_FOR_EXEMPLARS
            if code_too_long:
                n_exemplars_dropped += 1
                prompt = builder(ds, sample)
            elif rag_index is not None:
                fewshot_items = rag_index.retrieve(query_code, k=rag_k, exclude_id=sample["sample_id"])
                prompt = builder(ds, sample, fewshot_items=fewshot_items)
            else:
                prompt = builder(ds, sample, fewshot_ds=fewshot_ds)
            raw = llm.generate(SYSTEM_GRADER, prompt, max_new_tokens=700)
            parsed = extract_json(raw)
            if parsed is None:
                n_parse_fail += 1
                parsed = dict(defaults)
            else:
                missing = required - set(parsed.keys())
                if missing:
                    n_patched += 1
                    for key in missing:
                        parsed[key] = defaults[key]
            # Task 3 occasionally mixes in Chinese characters despite the Vietnamese-only
            # instruction (observed ~6-9% of samples) -- retry once with an emphatic reminder.
            if task_num == 3 and parsed.get("feedback") and _CJK_RE.search(parsed["feedback"]):
                # Greedy decoding deterministically repeats this failure mode on the same
                # prompt even with an added reminder -- sample instead (seeded, reproducible)
                # to actually escape it.
                retry_prompt = prompt + "\n\n[Lưu ý: lần trước bạn lẫn ký tự KHÔNG phải tiếng Việt vào phản hồi. Viết lại HOÀN TOÀN bằng tiếng Việt, không chèn tiếng Trung/Anh.]"
                retry_raw = llm.generate(SYSTEM_GRADER, retry_prompt, max_new_tokens=700, temperature=0.7)
                retry_parsed = extract_json(retry_raw)
                if retry_parsed and retry_parsed.get("feedback") and not _CJK_RE.search(retry_parsed["feedback"]):
                    parsed = retry_parsed
                    raw = retry_raw
                    n_cjk_retried += 1
                else:
                    # Last resort: the sampled retry hit the same failure mode (observed on
                    # a couple of stubborn samples) -- strip CJK chars rather than ship them.
                    parsed["feedback"] = _CJK_RE.sub(" ", parsed["feedback"])
                    parsed["feedback"] = re.sub(r"\s{2,}", " ", parsed["feedback"]).strip()
            if task_num == 1 and "rubric" in parsed:
                # total_score is NOT an independent judgment -- it must equal the rubric sum,
                # confirmed as a hard invariant in gold labels and enforced by the grader.
                parsed["total_score"] = sum(parsed["rubric"].values())
            # "reasoning" is only there to make the model think before answering (it's listed
            # first in the schema so it's generated first) -- strip it before submission since
            # it's not part of the official output schema; the raw log below keeps it for inspection.
            parsed.pop("reasoning", None)
            predictions.append({"sample_id": sample["sample_id"], "output": parsed})
            logf.write(json.dumps({"sample_id": sample["sample_id"], "raw": raw}, ensure_ascii=False) + "\n")
            logf.flush()
            bar.set_postfix(parse_fail=n_parse_fail, patched=n_patched, cjk_retried=n_cjk_retried, exemplars_dropped=n_exemplars_dropped)
    print(f"  (dropped exemplars for {n_exemplars_dropped}/{len(samples)} samples with code >{MAX_CODE_CHARS_FOR_EXEMPLARS} chars, to avoid the VRAM cliff)")
    return predictions, n_parse_fail


def validate_format(task_num, predictions, ds):
    required = required_fields(task_num, ds)
    errors = []
    for p in predictions:
        if "sample_id" not in p or "output" not in p:
            errors.append(f"{p}: missing sample_id/output")
            continue
        missing = required - set(p["output"].keys())
        if missing:
            errors.append(f"{p['sample_id']}: missing output fields {missing}")
    return errors


def short_model_name(model_name: str) -> str:
    """HF model id -> filesystem-safe short tag, e.g. 'Qwen/Qwen2.5-Coder-7B-Instruct' -> 'qwen2.5-coder-7b'."""
    name = model_name.split("/")[-1].lower()
    return name.replace("-instruct", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["sample", "public"], required=True)
    ap.add_argument("--tasks", default="1,2,3")
    ap.add_argument("--limit", type=int, default=None, help="only run first N samples (smoke-testing)")
    ap.add_argument("--fewshot", action="store_true", help="prepend hand-picked exemplars from sample_dataset")
    ap.add_argument("--rag", action="store_true", help="retrieve top-k nearest exemplars by code similarity (sample_dataset + synthetic pool) instead of a fixed list; task3 has no synthetic data so falls back to --fewshot behavior")
    ap.add_argument("--rag-k", type=int, default=2, help="exemplars to retrieve per sample (keep small -- see VRAM cliff note in llm/CLAUDE.md)")
    ap.add_argument("--model", default=MODEL_NAME, help="HF model id or local path -- try a different model/checkpoint")
    ap.add_argument("--run-tag", default=None, help="override the auto-derived tag in output filenames (default: <model>[_fewshot|_rag])")
    ap.add_argument("--num-shards", type=int, default=1, help="split each task's samples into this many disjoint shards -- for running one shard per GPU in parallel (see merge_shards.py)")
    ap.add_argument("--shard-id", type=int, default=0, help="which shard (0-indexed) this process handles; pair with CUDA_VISIBLE_DEVICES to pin it to one GPU")
    args = ap.parse_args()
    task_nums = [int(t) for t in args.tasks.split(",")]

    ds = load_sample_dataset() if args.split == "sample" else load_public_test()
    if args.limit:
        for t in ds.tasks:
            ds.tasks[t] = ds.tasks[t][: args.limit]
    if args.num_shards > 1:
        for t in ds.tasks:
            ds.tasks[t] = ds.tasks[t][args.shard_id :: args.num_shards]
    fewshot_ds = load_sample_dataset() if (args.fewshot or args.rag) else None

    # The released data always leaves compile_log empty, forcing the model to guess
    # compilability purely by reading code -- measured to be wrong ~20-25% of the time
    # even on code that compiles fine (see METHOD.md). Fill in the real signal via our
    # own g++ compile instead, same oracle principle as the synthetic-data pipeline.
    print("computing real compile_log (g++, cached)...")
    for t in task_nums:
        attach_real_compile_log(ds, t)
        if fewshot_ds is not None:
            attach_real_compile_log(fewshot_ds, t)

    method_suffix = "_rag" if args.rag else ("_fewshot" if args.fewshot else "")
    method_tag = args.run_tag or short_model_name(args.model) + method_suffix
    shard_suffix = f"_shard{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else ""
    tag = f"{args.split}_{method_tag}{shard_suffix}"

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Loading {args.model} (seed={SEED}, fewshot={args.fewshot}, run_tag={method_tag})...")
    llm = LocalLLM(model_name=args.model)

    for task_num in task_nums:
        print(f"\n=== Task {task_num} ({tag}, n={len(ds.tasks[task_num])}) ===")
        log_path = OUTPUT_DIR / f"raw_task{task_num}_{tag}.jsonl"
        rag_index = None
        if args.rag and task_num in (1, 2):
            print(f"  building RAG index for task{task_num} (embedding pool, CPU)...")
            rag_index = RagIndex(task_num)
            print(f"  pool size: {len(rag_index.pool)}")
        predictions, n_fail = run_task(llm, ds, task_num, log_path, fewshot_ds=fewshot_ds, rag_index=rag_index, rag_k=args.rag_k)

        errors = validate_format(task_num, predictions, ds)
        if errors:
            print(f"  FORMAT ERRORS ({len(errors)}):", errors[:5])
        else:
            print("  format OK")

        pred_path = OUTPUT_DIR / f"predictions_task{task_num}_{tag}.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"  wrote {pred_path} (parse failures: {n_fail}/{len(predictions)})")

        if args.split == "sample":
            gold = ds.tasks[task_num]
            pred_by_id = {p["sample_id"]: p["output"] for p in predictions}
            if task_num == 1:
                m = task1_metrics(gold, pred_by_id)
                print("  QWK:", m["qwk"], "MAE:", m["mae"], "exact_match:", m["exact_match"])
            elif task_num == 2:
                m = task2_metrics(gold, {k: v["taxonomy_error"] for k, v in pred_by_id.items()}, ds.label_space["taxonomy_error"])
                print("  macro_f1:", m["macro_f1"], "micro_f1:", m["micro_f1"])
            elif task_num == 3:
                level_by_id = {s["sample_id"]: s["input"]["target_feedback_level"] for s in gold}
                triples = [(p["sample_id"], level_by_id[p["sample_id"]], p["output"]["feedback"]) for p in predictions]
                m = task3_compliance(triples)
                print("  level compliance:", m["compliance_rate"], "violations:", m["violations"])
                gold_fb_by_id = {s["sample_id"]: s["output"]["feedback"] for s in gold}
                overlap_triples = [(p["sample_id"], p["output"]["feedback"], gold_fb_by_id[p["sample_id"]]) for p in predictions]
                mo = task3_overlap_proxy(overlap_triples)
                print("  feedback overlap proxy (not the official metric):", mo["mean_overlap_f1"])


if __name__ == "__main__":
    main()
