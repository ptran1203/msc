"""Build the Kaggle runner notebook: clones the public code repo, attaches the private
data via the Kaggle dataset mount, and runs the grading pipeline on Kaggle's free GPU."""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/ptran1203/msc.git"
DATASET_SLUG = "llm-autograding-course-data"

cells = []

cells.append(new_markdown_cell(
f"""# LLM auto-grading -- Kaggle runner

Clones the public code repo ([{REPO_URL}]({REPO_URL})) and attaches the private data
via the Kaggle dataset `ptran1203/{DATASET_SLUG}` (must be added under Notebook ->
Add Input before running). Course rules forbid distributing the real student data
outside the class, so it lives only in that private dataset, never in the repo itself.

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
dataset_root = pathlib.Path("/kaggle/input/{DATASET_SLUG}")
repo_llm = pathlib.Path("/kaggle/working/msc/llm")
shutil.copytree(dataset_root / "data", repo_llm / "data")
(repo_llm / "synth").mkdir(exist_ok=True)
shutil.copytree(dataset_root / "synth_generated", repo_llm / "synth" / "generated")
print("data staged OK")
"""
))

cells.append(new_markdown_cell(
"""## Config -- edit this, then run the cell below

- `SPLIT`: `"sample"` (32 labeled, validates against gold) or `"public"` (328 unlabeled, the real submission target)
- `TASKS`: comma-separated task numbers, e.g. `"1,2"`
- `MODE`: `"zeroshot"`, `"fewshot"` (static hand-picked exemplars), or `"rag"` (retrieval from the 89-sample pool)"""
))

cells.append(new_code_cell(
"""SPLIT = "sample"
TASKS = "1,2"
MODE = "rag"
"""
))

cells.append(new_code_cell(
"""%cd /kaggle/working/msc/llm/src
mode_flag = {"zeroshot": "", "fewshot": "--fewshot", "rag": "--rag"}[MODE]
!python -u run_prompting.py --split {SPLIT} --tasks {TASKS} {mode_flag}
"""
))

cells.append(new_markdown_cell(
"""## Output

Predictions and logs are written to `/kaggle/working/msc/llm/outputs/` -- anything under
`/kaggle/working` is saved as this notebook's output automatically on commit."""
))

nb = new_notebook(cells=cells)
with open("../kaggle_kernel/kaggle_runner.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote kaggle_kernel/kaggle_runner.ipynb")
