"""Build the Kaggle Task 3 fine-tuning notebook: generate grounded feedback targets for
the synthetic data (14B teacher, both GPUs), QLoRA-train the 7B on them, then evaluate
base vs fine-tuned on the 32 gold sample_dataset items -- see finetune_task3.py."""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/ptran1203/msc.git"
KAGGLE_USERNAME = "ptran1203"
DATASET_MOUNT = f"/kaggle/input/datasets/{KAGGLE_USERNAME}/llm-autograding-course-data"
SYNTH_MOUNT = f"/kaggle/input/notebooks/{KAGGLE_USERNAME}/llm-synth-data-generator/synth_generated"

cells = []

cells.append(new_markdown_cell(
"""# Task 3 fine-tune -- Kaggle runner

1. **Targets**: synth data has no Task 3 labels, so Qwen2.5-Coder-14B writes Level 2/3
   feedback grounded on facts known by construction (injected bug, diff vs reference,
   oracle rubric, real g++ log). Level 1 and compile-error Level 2 are deterministic.
2. **Train**: QLoRA on Qwen2.5-Coder-7B (the model the pipeline actually uses), same
   self-diagnose prompt as inference, loss on the answer only.
3. **Eval**: base vs fine-tuned on the 32 gold `sample_dataset` items, same prompt, so
   the delta is the fine-tune alone.

**Needs accelerator "GPU T4 x2"** (Settings -> Accelerator) -- not settable via the push API."""
))

cells.append(new_code_cell(
"""import torch
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.device_count() == 2, "set the accelerator to GPU T4 x2"
"""
))

cells.append(new_code_cell(
f"""%cd /kaggle/working
!rm -rf msc
!git clone --depth 1 {REPO_URL}
%cd /kaggle/working/msc/llm
!pip install -q -r requirements.txt
!which g++ || (apt-get update -qq && apt-get install -y -qq g++ > /dev/null)
"""
))

cells.append(new_code_cell(
f"""import shutil, pathlib, json
repo_llm = pathlib.Path("/kaggle/working/msc/llm")
shutil.copytree("{DATASET_MOUNT}/data", repo_llm / "data")
(repo_llm / "synth").mkdir(exist_ok=True)
shutil.copytree("{SYNTH_MOUNT}", repo_llm / "synth" / "generated")
n = json.loads((repo_llm / "synth/generated/task2_error_taxonomy.json").read_text())["n_samples"]
print("synth samples:", n)
"""
))

cells.append(new_markdown_cell(
"""## Smoke test first -- 2 targets + 2 optimizer steps

Per the project rule: fit a tiny batch end-to-end before the real run. Smoke outputs are
deleted afterwards so they don't end up in the real training set."""
))

cells.append(new_code_cell(
"""%cd /kaggle/working/msc/llm/src
import os, subprocess, glob
!CUDA_VISIBLE_DEVICES=0 python -u finetune_task3.py gen-targets --limit 2
!cat ../synth/generated/task3_ft_targets_shard0.jsonl
!CUDA_VISIBLE_DEVICES=0 python -u finetune_task3.py train --max-steps 2
for p in glob.glob("../synth/generated/task3_ft_targets_shard*.jsonl"):
    os.remove(p)
shutil.rmtree("../outputs/ft_task3_lora", ignore_errors=True)
"""
))

cells.append(new_markdown_cell("## 1. Generate targets -- one 14B teacher per GPU"))

cells.append(new_code_cell(
"""from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-Coder-14B-Instruct")  # once, so the 2 shards don't race on the cache
cmd = lambda i: ["python", "-u", "finetune_task3.py", "gen-targets", "--num-shards", "2", "--shard-id", str(i)]
procs = [subprocess.Popen(cmd(i), env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}) for i in range(2)]
for p in procs:
    p.wait()
assert all(p.returncode == 0 for p in procs), [p.returncode for p in procs]
!wc -l ../synth/generated/task3_ft_targets_shard*.jsonl
# 14B weights (~30GB) aren't needed anymore -- free the disk before the 7B download
shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-14B-Instruct"), ignore_errors=True)
"""
))

cells.append(new_markdown_cell("## 2. QLoRA train the 7B (single GPU)"))

cells.append(new_code_cell(
"""!CUDA_VISIBLE_DEVICES=0 python -u finetune_task3.py train
"""
))

cells.append(new_markdown_cell("## 3. Eval on the 32 gold samples -- base (GPU 0) vs fine-tuned (GPU 1) in parallel"))

cells.append(new_code_cell(
"""procs = [
    subprocess.Popen(["python", "-u", "finetune_task3.py", "eval", "--base"], env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}),
    subprocess.Popen(["python", "-u", "finetune_task3.py", "eval"], env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"}),
]
for p in procs:
    p.wait()
assert all(p.returncode == 0 for p in procs), [p.returncode for p in procs]
for name in ("base_zeroshot", "finetuned"):
    print(json.loads(pathlib.Path(f"../outputs/ft_task3_eval/{name}.json").read_text())["metrics"])
"""
))

cells.append(new_markdown_cell(
"""## Collect + clean up

Keeps only `/kaggle/working/ft_task3/` (targets, LoRA adapter, eval results) and deletes
the cloned repo so the saved output stays small."""
))

cells.append(new_code_cell(
"""out = pathlib.Path("/kaggle/working/ft_task3")
out.mkdir(exist_ok=True)
for p in glob.glob("../synth/generated/task3_ft_targets_shard*.jsonl"):
    shutil.copy(p, out)
shutil.copytree("../outputs/ft_task3_lora", out / "lora", ignore=shutil.ignore_patterns("checkpoints"), dirs_exist_ok=True)
shutil.copytree("../outputs/ft_task3_eval", out / "eval", dirs_exist_ok=True)
%cd /kaggle/working
shutil.rmtree("/kaggle/working/msc", ignore_errors=True)
print(sorted(str(p.relative_to(out)) for p in out.rglob("*")))
"""
))

# Kaggle's papermill engine needs kernelspec metadata or it fails before any cell runs.
KERNELSPEC_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nb = new_notebook(cells=cells, metadata=KERNELSPEC_METADATA)
with open("../kaggle_kernel_finetune/finetune_task3.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote kaggle_kernel_finetune/finetune_task3.ipynb")
