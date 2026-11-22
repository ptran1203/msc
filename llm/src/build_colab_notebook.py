"""Build a Colab notebook (run from VS Code's Google Colab extension, GPU T4) that runs the
Task 3 test_report ablation (eval_testreport_sample.py) with a thinking model on the 32
gold items.

The notebook embeds the user's Kaggle API token (to pull the two private datasets), so it
is written to llm/colab/, which is gitignored -- never commit it. This builder holds no secret.

Usage: python build_colab_notebook.py [--model Qwen/Qwen3-8B] [--no-thinking]
"""
import argparse
import json
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/ptran1203/msc.git"
COURSE_DATASET = "ptran1203/llm-autograding-course-data"
TESTS_DATASET = "ptran1203/llm-autograding-llm-tests"
LLM = Path(__file__).resolve().parent.parent
OUT_DIR = LLM / "colab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()

    token = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text(encoding="utf-8"))
    flags = "" if args.no_thinking else " --thinking"
    run_args = f"--model {args.model}{flags}"

    cells = [
        new_markdown_cell(
            f"# Task 3 ablation on Colab -- `{args.model}`{' (thinking)' if not args.no_thinking else ''}\n\n"
            "Select kernel -> Colab -> New Colab Server -> **GPU (T4)**, then Run All.\n\n"
            "**Contains a Kaggle API token -- private file, never commit or share.**"
        ),
        new_code_cell(
            "import subprocess, torch\n"
            "assert torch.cuda.is_available(), 'no GPU: pick a GPU (T4) Colab server'\n"
            "print(torch.cuda.get_device_name(0))\n"
            "print(subprocess.run(['g++', '--version'], capture_output=True, text=True).stdout.splitlines()[0])"
        ),
        new_code_cell(
            "import json, os, pathlib\n"
            "p = pathlib.Path.home() / '.kaggle' / 'kaggle.json'\n"
            "p.parent.mkdir(exist_ok=True)\n"
            f"p.write_text(json.dumps({{'username': {token['username']!r}, 'key': {token['key']!r}}}))\n"
            "os.chmod(p, 0o600)"
        ),
        new_code_cell(
            "%cd /content\n"
            "!rm -rf msc && git clone -q --depth 1 " + REPO_URL + "\n"
            "%cd /content/msc/llm\n"
            "!pip install -q -r requirements.txt kaggle"
        ),
        new_code_cell(
            "import shutil, glob\n"
            f"!kaggle datasets download -q {COURSE_DATASET} -p /content/ds_course --unzip\n"
            f"!kaggle datasets download -q {TESTS_DATASET} -p /content/ds_tests --unzip\n"
            "shutil.copytree('/content/ds_course/data', 'data', dirs_exist_ok=True)\n"
            "os.makedirs('outputs/llm_tests', exist_ok=True)\n"
            "for f in glob.glob('/content/ds_tests/*.json'):\n"
            "    if not f.endswith('dataset-metadata.json'):\n"
            "        shutil.copy(f, 'outputs/llm_tests')\n"
            "print(sorted(os.listdir('outputs/llm_tests')))\n"
            "assert len(os.listdir('outputs/llm_tests')) >= 3"
        ),
        new_markdown_cell("Smoke test on 2 items first (model download + thinking parse), then the full 32."),
        new_code_cell(
            "%cd /content/msc/llm/src\n"
            f"!python -u eval_testreport_sample.py {run_args} --limit 2"
        ),
        new_code_cell(f"!python -u eval_testreport_sample.py {run_args}"),
        new_code_cell(
            "import glob\n"
            "for f in sorted(glob.glob('../outputs/research_task3/testreport_ablation/*/summary.json')):\n"
            "    print(f); print(open(f).read())"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
    })
    OUT_DIR.mkdir(exist_ok=True)
    name = args.model.split("/")[-1].lower() + ("" if args.no_thinking else "-thinking")
    path = OUT_DIR / f"colab_task3_{name}.ipynb"
    nbf.write(nb, str(path))
    print(f"wrote {path}  (contains Kaggle token -- gitignored, do not commit)")


if __name__ == "__main__":
    main()
