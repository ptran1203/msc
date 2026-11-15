"""Build the Kaggle runner notebook: clones the public code repo, attaches the private
data via the Kaggle dataset mount, and runs the grading pipeline on Kaggle's free GPU."""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/ptran1203/msc.git"
KAGGLE_USERNAME = "ptran1203"
DATASET_SLUG = "llm-autograding-course-data"
DATASET_MOUNT = f"/kaggle/input/datasets/{KAGGLE_USERNAME}/{DATASET_SLUG}"

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
dataset_root = pathlib.Path("{DATASET_MOUNT}")
repo_llm = pathlib.Path("/kaggle/working/msc/llm")
shutil.copytree(dataset_root / "data", repo_llm / "data")
(repo_llm / "synth").mkdir(exist_ok=True)
shutil.copytree(dataset_root / "synth_generated", repo_llm / "synth" / "generated")
print("data staged OK")
"""
))

cells.append(new_markdown_cell(
"""## Run -- generates the public submission (328 samples, all 3 tasks, static few-shot)

To try a different config, just edit the command line directly:
`--split sample|public`, `--tasks 1,2,3`, `--fewshot` / `--rag` / neither (zero-shot)."""
))

cells.append(new_code_cell(
"""%cd /kaggle/working/msc/llm/src
!python -u run_prompting.py --split public --tasks 1,2,3 --fewshot
"""
))

cells.append(new_code_cell(
"""import glob, re
for path in sorted(glob.glob("../outputs/predictions_task*_public_*.json")):
    task_num = re.search(r"predictions_task(\\d)_", path).group(1)
    get_ipython().system(f"python validate_submission.py {task_num} {path} --split public")
"""
))

cells.append(new_markdown_cell(
"""## Output

Predictions and logs are written to `/kaggle/working/msc/llm/outputs/` -- anything under
`/kaggle/working` is saved as this notebook's output automatically on commit. The cell
above validates each generated `predictions_task*.json` against the schema before you
trust it as submission-ready."""
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
