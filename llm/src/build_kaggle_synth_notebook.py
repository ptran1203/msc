"""Build the Kaggle synthetic-data-generation notebook: clones the public code repo,
attaches the private data (for exam statements), ensures g++ is available, and runs the
bug-injection pipeline (synth_bugs/synth_oracle/synth_label/synth_generate) on Kaggle's GPU."""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/ptran1203/msc.git"
DATASET_SLUG = "llm-autograding-course-data"

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
dataset_root = pathlib.Path("/kaggle/input/{DATASET_SLUG}")
repo_llm = pathlib.Path("/kaggle/working/msc/llm")
shutil.copytree(dataset_root / "data", repo_llm / "data")
print("data staged OK")
"""
))

cells.append(new_markdown_cell(
"""## Smoke test -- oracle self-check, then 1 bug type x 1 variant x 3 problems

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
"""## Full run -- 3 problems x 10 bug types x 3 variants (up to 90 candidates before QC discards)

Edit `--variants` to generate more/fewer per bug type."""
))

cells.append(new_code_cell(
"""!python -u synth_generate.py --variants 3
"""
))

cells.append(new_markdown_cell(
"""## Output

Written to `/kaggle/working/msc/llm/synth/generated/` (schema-compatible with `sample_dataset`,
loadable via `data.load_dataset()`) -- saved as this notebook's output automatically on commit.

**Before using this data elsewhere**: it embeds real exam statements (course content), so per
the course rules it should go back into the private Kaggle dataset
(`ptran1203/{slug}`), never into the public code repo or anywhere public -- re-upload it there,
don't just leave it only as a kernel output.""".format(slug=DATASET_SLUG)
))

nb = new_notebook(cells=cells)
with open("../kaggle_kernel_synth/synth_data_runner.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote kaggle_kernel_synth/synth_data_runner.ipynb")
