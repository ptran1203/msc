"""Regenerate a single failed sample's prediction and patch it into an existing predictions.json."""
import argparse
import json

from data import REPO_ROOT, load_public_test, load_sample_dataset
from llm_client import LocalLLM, extract_json
from prompts import SYSTEM_GRADER, task1_prompt, task2_prompt, task3_prompt

ap = argparse.ArgumentParser()
ap.add_argument("task_num", type=int)
ap.add_argument("split", choices=["sample", "public"])
ap.add_argument("sample_id")
args = ap.parse_args()

ds = load_sample_dataset() if args.split == "sample" else load_public_test()
builder = {1: task1_prompt, 2: task2_prompt, 3: task3_prompt}[args.task_num]
sample = next(s for s in ds.tasks[args.task_num] if s["sample_id"] == args.sample_id)

llm = LocalLLM()
prompt = builder(ds, sample)
raw = llm.generate(SYSTEM_GRADER, prompt, max_new_tokens=500)
parsed = extract_json(raw)
print("raw:", raw)
print("parsed:", parsed)
assert parsed is not None and parsed.get("feedback", "").strip(), "still failed to parse"

pred_path = REPO_ROOT / f"outputs/predictions_task{args.task_num}_{args.split}.json"
preds = json.loads(pred_path.read_text(encoding="utf-8"))
for p in preds:
    if p["sample_id"] == args.sample_id:
        p["output"] = parsed
        break
pred_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"patched {args.sample_id} in {pred_path}")
