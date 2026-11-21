"""Build the Kaggle synthetic-data-generation notebook: clones the public code repo,
attaches the private data (for exam statements), ensures g++ is available, and runs the
bug-injection pipeline (synth_bugs/synth_oracle/synth_label/synth_generate) on Kaggle's GPU."""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/ptran1203/msc.git"
KAGGLE_USERNAME = "ptran1203"
DATASET_SLUG = "llm-autograding-course-data"
DATASET_MOUNT = f"/kaggle/input/datasets/{KAGGLE_USERNAME}/{DATASET_SLUG}"

cells = []

cells.append(new_markdown_cell(
f"""# Synthetic data generation -- Kaggle runner

Generates new Task 1 + Task 2 training data via controlled bug injection (see
[METHOD.md](https://github.com/ptran1203/msc/blob/main/llm/METHOD.md) section 4 for the
design: the LLM only writes buggy code from a precise instruction, labels come from an
objective oracle -- never from the LLM self-grading its own output).

Attach the private dataset `ptran1203/{DATASET_SLUG}` under Notebook -> Add Input before running."""
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
!apt-get update -qq && apt-get install -y -qq g++ > /dev/null
!g++ --version
"""
))

cells.append(new_code_cell(
f"""import shutil, pathlib
dataset_root = pathlib.Path("{DATASET_MOUNT}")
repo_llm = pathlib.Path("/kaggle/working/msc/llm")
shutil.copytree(dataset_root / "data", repo_llm / "data")
print("data staged OK")
"""
))

cells.append(new_markdown_cell(
"""## Smoke test -- oracle self-check, then 1 bug type x 1 variant x 5 problems

Per the no-heavy-compute-without-checking-first rule: verify the oracle (compile/run/compare,
no model needed) and the full LLM pipeline both work before the real run."""
))

cells.append(new_code_cell(
"""%cd /kaggle/working/msc/llm/src
!python synth_oracle.py
"""
))

cells.append(new_code_cell(
"""!python synth_generate.py --smoke-test
"""
))

cells.append(new_markdown_cell(
"""## Full run -- 5 problems x 10 bug types x 20 variants (up to 1000 candidates before QC discards)

Two new problems added (binary_search, palindrome_check) beyond the original 3
(armstrong, remove_duplicates, earliest_login) -- the only other real single-problem
exam (Class03-Midterm-1) turned out to be a duplicate Armstrong instance, and the
remaining real exams are a different struct + file-I/O + linked-list family with no
stdin/stdout main(), too big a lift to retrofit here -- so these two synthesize their
own exam metadata instead of reusing a real exam's.

Variants >=1 use seeded sampling (not greedy), so they actually diversify instead of
repeating near-identical edits -- locally this roughly doubled yield at variants=6 (57->109
kept) and even got `compile_error` producing usable samples at all (0/9 under pure greedy).
Pushing variants higher for maximum yield; some bug types (e.g. `style_error`) will likely
saturate toward 100% long before 20, others (e.g. `edge_case_error`) have more headroom.
Edit `--variants` to generate more/fewer per bug type."""
))

cells.append(new_code_cell(
"""!python -u synth_generate.py --variants 20
"""
))

cells.append(new_markdown_cell(
"""## Collect output + clean up

Copies just the generated data out of the repo clone into `/kaggle/working/synth_generated/`,
then deletes the repo -- so the notebook's saved output is only the synthetic data, not the
whole source tree + per-candidate submission files."""
))

cells.append(new_code_cell(
"""import shutil
shutil.rmtree("/kaggle/working/synth_generated", ignore_errors=True)
shutil.copytree("/kaggle/working/msc/llm/synth/generated", "/kaggle/working/synth_generated")
%cd /kaggle/working
shutil.rmtree("/kaggle/working/msc", ignore_errors=True)
print("kept files:", len(list(__import__("pathlib").Path("/kaggle/working/synth_generated").rglob("*"))))
"""
))

cells.append(new_markdown_cell(
"""## Output

`/kaggle/working/synth_generated/` (schema-compatible with `sample_dataset`, loadable via
`data.load_dataset()`).

**Before using this data elsewhere**: it embeds real exam statements (course content), so per
the course rules it should go back into the private Kaggle dataset
(`ptran1203/{slug}`), never into the public code repo or anywhere public -- re-upload it there,
don't just leave it only as a kernel output.""".format(slug=DATASET_SLUG)
))

# Kaggle's papermill execution engine requires kernelspec metadata to even start --
# without it the kernel fails immediately with "No kernel name found in notebook"
# before any cell runs (this is exactly what happened on the first push of this notebook).
KERNELSPEC_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nb = new_notebook(cells=cells, metadata=KERNELSPEC_METADATA)
with open("../kaggle_kernel_synth/synth_data_runner.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote kaggle_kernel_synth/synth_data_runner.ipynb")
