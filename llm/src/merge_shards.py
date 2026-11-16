"""Merge per-GPU shard outputs from run_prompting.py --num-shards back into one
predictions_task{N}_{split}_{method_tag}.json per task, matching the filename a
single non-sharded run would have produced (so validate_submission.py needs no changes).

Usage: python merge_shards.py --split public --tasks 1,2,3 --method-tag qwen2.5-coder-7b_rag --num-shards 2
"""
import argparse
import json

from data import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "outputs"

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["sample", "public"], required=True)
ap.add_argument("--tasks", default="1,2,3")
ap.add_argument("--method-tag", required=True, help="the <model>[_fewshot|_rag] tag used for the shard runs")
ap.add_argument("--num-shards", type=int, required=True)
args = ap.parse_args()

for task_num in (int(t) for t in args.tasks.split(",")):
    merged = []
    for shard_id in range(args.num_shards):
        path = OUTPUT_DIR / f"predictions_task{task_num}_{args.split}_{args.method_tag}_shard{shard_id}of{args.num_shards}.json"
        merged += json.load(open(path, encoding="utf-8"))
    out_path = OUTPUT_DIR / f"predictions_task{task_num}_{args.split}_{args.method_tag}.json"
    json.dump(merged, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"task{task_num}: merged {len(merged)} samples from {args.num_shards} shards -> {out_path}")
