"""Build the Kaggle runner notebook: clones the public code repo, attaches the private
data via the Kaggle dataset mount, and runs the grading pipeline on Kaggle's free GPU."""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/ptran1203/msc.git"
KAGGLE_USERNAME = "ptran1203"
DATASET_SLUG = "llm-autograding-course-data"
DATASET_MOUNT = f"/kaggle/input/datasets/{KAGGLE_USERNAME}/{DATASET_SLUG}"
SYNTH_KERNEL_SLUG = "llm-synth-data-generator"
SYNTH_MOUNT = f"/kaggle/input/notebooks/{KAGGLE_USERNAME}/{SYNTH_KERNEL_SLUG}/synth_generated"

cells = []

cells.append(new_markdown_cell(
f"""# LLM auto-grading -- Kaggle runner

Clones the public code repo ([{REPO_URL}]({REPO_URL})) and attaches the private data
via the Kaggle dataset `ptran1203/{DATASET_SLUG}`, plus the synthetic bug-injection data
via the `ptran1203/{SYNTH_KERNEL_SLUG}` notebook's output (both added automatically when
pushed via `kernel_sources`/`dataset_sources` in kernel-metadata.json). Course rules forbid
distributing the real student data outside the class, so it lives only in that private
dataset, never in the repo itself.

Edit the **config cell** below, then Run All."""
))

cells.append(new_code_cell(
"""import torch
print("GPU available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
"""
))

cells.append(new_code_cell(
f"""%cd /kaggle/working
!rm -rf msc
!git clone --depth 1 {REPO_URL}
%cd /kaggle/working/msc/llm
!pip install -q -r requirements.txt
"""
))

cells.append(new_code_cell(
f"""import shutil, pathlib
dataset_root = pathlib.Path("{DATASET_MOUNT}")
synth_root = pathlib.Path("{SYNTH_MOUNT}")
repo_llm = pathlib.Path("/kaggle/working/msc/llm")
shutil.copytree(dataset_root / "data", repo_llm / "data")
(repo_llm / "synth").mkdir(exist_ok=True)
shutil.copytree(synth_root, repo_llm / "synth" / "generated")
print("data staged OK")
"""
))

cells.append(new_markdown_cell(
"""## Run -- generates the public submission (328 samples, all 3 tasks), TWO configs

Runs **both** `--fewshot` (static hand-picked real exemplars) and `--rag` (retrieval from
the real + synthetic pool) back to back, so the two can be compared directly on the
leaderboard instead of guessing from one run. Both now also get the **real compile_log**
(see `compile_check.py`): the released data leaves it empty, so the model previously had
to guess compilability purely by reading code and was wrong ~20-25% of the time even on
code that compiles -- confirmed the actual cause of Task 1's negative QWK, not RAG vs.
few-shot (which scored almost identically, -0.2171 vs -0.2108, before this fix).

`--rag` had anchored Task 2 hard toward single-label outputs last time (every synthetic
exemplar has exactly one injected bug), so it's still a real risk for that task
specifically -- that's exactly what running both configs side by side is for.

Each config runs across **both GPUs in parallel** (`--num-shards 2`, one process per GPU
via `CUDA_VISIBLE_DEVICES`, merged by `merge_shards.py`). **Needs the kernel's accelerator
set to "GPU T4 x2"** in the Kaggle notebook editor (Settings -> Accelerator) -- not
exposed through the push API, so it has to be set once by hand and then persists across
future `kaggle kernels push` updates."""
))

cells.append(new_code_cell(
"""%cd /kaggle/working/msc/llm/src
import os, subprocess

RUN_TAGS = {"fewshot": "qwen-fewshot", "rag": "qwen-rag"}
METHOD_FLAGS = {"fewshot": "--fewshot", "rag": "--rag"}

for method, run_tag in RUN_TAGS.items():
    cmd = lambda shard: [
        "python", "-u", "run_prompting.py", "--split", "public", "--tasks", "1,2,3",
        METHOD_FLAGS[method], "--run-tag", run_tag, "--num-shards", "2", "--shard-id", str(shard),
    ]
    envs = [{**os.environ, "CUDA_VISIBLE_DEVICES": str(i)} for i in range(2)]
    procs = [subprocess.Popen(cmd(i), env=envs[i]) for i in range(2)]
    for p in procs:
        p.wait()
    assert all(p.returncode == 0 for p in procs), (method, [p.returncode for p in procs])
    subprocess.run(
        ["python", "merge_shards.py", "--split", "public", "--tasks", "1,2,3",
         "--method-tag", run_tag, "--num-shards", "2"],
        check=True,
    )
    print(f"=== {method} done ===")
"""
))

cells.append(new_code_cell(
"""import glob, re, subprocess, pathlib
valid_paths = set()
for path in sorted(glob.glob("../outputs/*/predictions_task*_public.json")):
    path = str(pathlib.Path(path).resolve())
    task_num = re.search(r"predictions_task(\\d)_", path).group(1)
    r = subprocess.run(["python", "validate_submission.py", task_num, path, "--split", "public"])
    print(path, "-> OK" if r.returncode == 0 else "-> FAIL (will NOT be collected)")
    if r.returncode == 0:
        valid_paths.add(path)
"""
))

cells.append(new_markdown_cell(
"""## Collect submissions + clean up

Each run-tag now writes to its own `outputs/{run_tag}/` folder (one folder per run id,
instead of flat files disambiguated only by a filename suffix). Copies only the predictions
files that PASSED validation above into `/kaggle/working/submission/`, under plain
`predictions_task{N}_{fewshot,rag}.json` names, then deletes the cloned repo -- so the
notebook's saved output is only that one folder instead of the whole source tree + per-shard
intermediates. Earlier this silently shipped invalid files regardless of validation result
(`!python validate_submission.py` doesn't stop the notebook on a non-zero exit code) --
raises instead now, so a schema violation surfaces as a failed kernel, not a rejected
Kaggle submission."""
))

cells.append(new_code_cell(
"""import shutil, pathlib
sub_dir = pathlib.Path("/kaggle/working/submission")
sub_dir.mkdir(exist_ok=True)
missing_or_invalid = []
for method, run_tag in RUN_TAGS.items():
    for t in (1, 2, 3):
        src = pathlib.Path(f"/kaggle/working/msc/llm/outputs/{run_tag}/predictions_task{t}_public.json")
        if str(src.resolve()) in valid_paths:
            shutil.copy(src, sub_dir / f"predictions_task{t}_{method}.json")
        else:
            missing_or_invalid.append(str(src))
%cd /kaggle/working
shutil.rmtree("/kaggle/working/msc", ignore_errors=True)
print("submission files:", sorted(p.name for p in sub_dir.iterdir()))
assert not missing_or_invalid, f"did NOT collect (missing or failed validation): {missing_or_invalid}"
"""
))

cells.append(new_markdown_cell(
"""## Output

`/kaggle/working/submission/predictions_task{1,2,3}_{fewshot,rag}.json` -- pick whichever
config scores better per task directly, already validated against the schema above."""
))

# Kaggle's papermill execution engine requires kernelspec metadata to even start --
# without it, the kernel fails immediately with "No kernel name found in notebook"
# before any cell runs (found this the hard way on the sibling synth-data notebook).
KERNELSPEC_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nb = new_notebook(cells=cells, metadata=KERNELSPEC_METADATA)
with open("../kaggle_kernel/kaggle_runner.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote kaggle_kernel/kaggle_runner.ipynb")
