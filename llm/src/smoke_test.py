"""Single-sample check: model loads, fits in VRAM, output parses as valid JSON. Run before any full loop."""
import time

import torch

from data import load_sample_dataset
from llm_client import LocalLLM, extract_json
from prompts import SYSTEM_GRADER, task1_prompt, task2_prompt, task3_prompt

ds = load_sample_dataset()
sample1 = ds.tasks[1][0]
sample2 = ds.tasks[2][0]
sample3 = ds.tasks[3][0]

print("Loading model (first run downloads weights, may take a while)...")
t0 = time.time()
llm = LocalLLM()
print(f"Model loaded in {time.time() - t0:.1f}s")
if torch.cuda.is_available():
    print(f"GPU mem allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

for name, sample, builder in [
    ("task1", sample1, task1_prompt),
    ("task2", sample2, task2_prompt),
    ("task3", sample3, task3_prompt),
]:
    prompt = builder(ds, sample)
    t0 = time.time()
    raw = llm.generate(SYSTEM_GRADER, prompt, max_new_tokens=400)
    dt = time.time() - t0
    parsed = extract_json(raw)
    print(f"\n--- {name} ({sample['sample_id']}, {dt:.1f}s) ---")
    print("raw:", raw[:300])
    print("parsed:", parsed)
    assert parsed is not None, f"{name}: failed to parse JSON from model output"

print("\nSMOKE TEST PASSED — safe to run the full loop.")
