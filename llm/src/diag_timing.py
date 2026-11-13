"""Isolate why task2 few-shot generation is ~5x slower than task1 few-shot despite similar prompt size."""
import time

import torch

import prompts
from data import load_sample_dataset
from llm_client import LocalLLM
from prompts import SYSTEM_GRADER, task1_prompt, task2_prompt

ds = load_sample_dataset()
s1 = ds.tasks[1][5]
s2 = ds.tasks[2][5]
p1 = task1_prompt(ds, s1, fewshot_ds=ds)
p2 = task2_prompt(ds, s2, fewshot_ds=ds)
prompts.FEWSHOT_IDS[2] = prompts.FEWSHOT_IDS[2][:2]  # halve task2's exemplar count
p2_short = task2_prompt(ds, s2, fewshot_ds=ds)
print(f"task1 prompt: {len(p1)} chars, task2 prompt: {len(p2)} chars, task2_short prompt: {len(p2_short)} chars")

llm = LocalLLM()
tok1 = llm.tokenizer(llm.tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_GRADER}, {"role": "user", "content": p1}], tokenize=False, add_generation_prompt=True), return_tensors="pt")
tok2 = llm.tokenizer(llm.tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_GRADER}, {"role": "user", "content": p2}], tokenize=False, add_generation_prompt=True), return_tensors="pt")
tok2s = llm.tokenizer(llm.tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_GRADER}, {"role": "user", "content": p2_short}], tokenize=False, add_generation_prompt=True), return_tensors="pt")
print(f"task1 input tokens: {tok1['input_ids'].shape[1]}, task2 input tokens: {tok2['input_ids'].shape[1]}, task2_short input tokens: {tok2s['input_ids'].shape[1]}")

for name, prompt, mnt in [("task1", p1, 500), ("task2-full", p2, 100), ("task2-short", p2_short, 100), ("task2-short-again", p2_short, 100)]:
    t0 = time.time()
    out = llm.generate(SYSTEM_GRADER, prompt, max_new_tokens=mnt)
    dt = time.time() - t0
    print(f"{name}: {dt:.1f}s, output={out[:60]!r}")
    if torch.cuda.is_available():
        print(f"  GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
