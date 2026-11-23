"""Build a Colab notebook that QLoRA fine-tunes the Task 3 student model (finetune_task3.py)
with everything that matters saved to Google Drive: llm/outputs is a symlink to
MyDrive/llm_ft/<model>/outputs, so checkpoints (every 10 steps), the adapter, eval results and
public predictions survive a disconnect. Re-running all cells resumes: training restarts
from the last checkpoint, finished steps are skipped.

Data comes from private Kaggle datasets (course data, synth data + 14B targets, test suites),
so the notebook embeds the user's Kaggle token -> written to llm/colab/ (gitignored, never commit).

Usage: python build_colab_finetune_notebook.py [--model Qwen/Qwen2.5-Coder-7B-Instruct]
"""
import argparse
import json
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/ptran1203/msc.git"
DATASETS = {"course": "ptran1203/llm-autograding-course-data", "synth": "ptran1203/llm-synth-generated",
            "tests": "ptran1203/llm-autograding-llm-tests"}
LLM = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args = ap.parse_args()
    token = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text(encoding="utf-8"))
    short = args.model.split("/")[-1].lower()
    run_tag = f"{short}-ft-task3"

    cells = [
        new_markdown_cell(
            f"# Task 3 fine-tune on Colab -- `{args.model}` (QLoRA)\n\n"
            "Runtime -> Change runtime type -> **T4 GPU**, then Runtime -> Run all. Allow the Drive mount.\n\n"
            f"Everything is saved to **MyDrive/llm_ft/{short}/outputs**. If Colab disconnects, just Run all "
            "again: training resumes from the last checkpoint (every 10 steps), finished steps are skipped.\n\n"
            "**Contains a Kaggle API token -- private file, never commit or share.**"
        ),
        new_code_cell(
            "import subprocess, torch, os, json, glob, shutil, pathlib, zipfile\n"
            "assert torch.cuda.is_available(), 'no GPU: Runtime -> Change runtime type -> T4 GPU'\n"
            "print(torch.cuda.get_device_name(0))\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            f"MODEL = {args.model!r}\n"
            f"DRIVE = pathlib.Path('/content/drive/MyDrive/llm_ft/{short}')\n"
            "(DRIVE / 'outputs').mkdir(parents=True, exist_ok=True)\n"
            "print('saving to', DRIVE, '->', os.listdir(DRIVE / 'outputs'))"
        ),
        new_code_cell(
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
            "".join(f"!kaggle datasets download -q {slug} -p /content/ds_{k} --unzip\n" for k, slug in DATASETS.items())
            + "shutil.copytree('/content/ds_course/data', 'data', dirs_exist_ok=True)\n"
            "gen = pathlib.Path('synth/generated'); gen.mkdir(parents=True, exist_ok=True)\n"
            "for f in glob.glob('/content/ds_synth/*'):\n"
            "    if f.endswith('submissions.zip'):  # the CLI zipped the submissions/ folder\n"
            "        zipfile.ZipFile(f).extractall(gen / 'submissions')\n"
            "    elif os.path.isdir(f):\n"
            "        shutil.copytree(f, gen / os.path.basename(f), dirs_exist_ok=True)\n"
            "    elif not f.endswith('dataset-metadata.json'):\n"
            "        shutil.copy(f, gen)\n"
            "n_targets = sum(1 for p in gen.glob('task3_ft_targets_shard*.jsonl') for _ in open(p, encoding='utf-8'))\n"
            "n_cpp = len(list((gen / 'submissions').rglob('*.cpp')))\n"
            "print('targets:', n_targets, '| synth .cpp:', n_cpp)\n"
            "assert n_targets >= 580 and n_cpp > 500"
        ),
        new_markdown_cell(
            "## Smoke test -- 1 optimizer step (8 examples), local disk only\n\n"
            "Project rule: fit one batch before the real run. Skipped when Drive already has a "
            "checkpoint or adapter (a resumed session)."
        ),
        new_code_cell(
            "%cd /content/msc/llm/src\n"
            "def step(*a):\n"
            "    # check=True: a failing `!python` does not stop a notebook\n"
            "    subprocess.run(['python', '-u', *a], check=True)\n"
            "\n"
            "resumed = any((DRIVE / 'outputs').rglob('checkpoint-*')) or any((DRIVE / 'outputs').rglob('adapter_model.safetensors'))\n"
            "if not resumed:\n"
            "    step('finetune_task3.py', 'train', '--model', MODEL, '--levels-per-sample', '1', '--max-steps', '1')\n"
            "shutil.rmtree('../outputs', ignore_errors=True)\n"
            "os.symlink(DRIVE / 'outputs', '../outputs')  # from here on, every output lands on Drive\n"
            "os.makedirs('../outputs/llm_tests', exist_ok=True)\n"
            "for f in glob.glob('/content/ds_tests/*.json'):\n"
            "    if not f.endswith('dataset-metadata.json'):\n"
            "        shutil.copy(f, '../outputs/llm_tests')\n"
            "print('smoke ok' if not resumed else 'resumed session, smoke skipped')"
        ),
        new_markdown_cell(
            "## 1. Train -- one feedback level per synth sample (588 examples, ~74 steps)\n\n"
            "7B on a T4 ~ 196 s/step ~ 4h; 3B roughly half. Resumes from the last checkpoint on Drive."
        ),
        new_code_cell(
            "import sys; sys.path.insert(0, '.')\n"
            "from finetune_task3 import adapter_dir\n"
            "ADAPTER = adapter_dir(MODEL)\n"
            "if (ADAPTER / 'adapter_model.safetensors').exists():\n"
            "    print('adapter already trained:', ADAPTER)\n"
            "else:\n"
            "    step('finetune_task3.py', 'train', '--model', MODEL, '--levels-per-sample', '1')\n"
            "print(os.listdir(ADAPTER))"
        ),
        new_markdown_cell("## 2. Eval on the 32 gold samples -- fine-tuned vs base, same prompt"),
        new_code_cell(
            "suffix = '' if MODEL == 'Qwen/Qwen2.5-Coder-7B-Instruct' else '_' + MODEL.split('/')[-1].lower()\n"
            "for name, extra in (('finetuned', []), ('base_zeroshot', ['--base'])):\n"
            "    path = pathlib.Path(f'../outputs/ft_task3_eval/{name}{suffix}.json')\n"
            "    if not path.exists():\n"
            "        step('finetune_task3.py', 'eval', '--model', MODEL, *extra)\n"
            "    print(name, json.loads(path.read_text(encoding='utf-8'))['metrics'])"
        ),
        new_markdown_cell(
            "## 3. Task 3 public predictions (328) with the adapter\n\n"
            "`--no-test-report`: the adapter was trained without a test report in the prompt. "
            "Resumable: rerunning continues from the last finished item."
        ),
        new_code_cell(
            f"RUN_TAG = {run_tag!r}\n"
            "step('run_prompting.py', '--split', 'public', '--tasks', '3', '--model', MODEL, '--adapter', str(ADAPTER),\n"
            "     '--no-test-report', '--run-tag', RUN_TAG)\n"
            "step('validate_submission.py', '3', f'../outputs/{RUN_TAG}/predictions_task3_public.json', '--split', 'public')\n"
            "print('saved on Drive:', DRIVE / 'outputs' / RUN_TAG)"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
    })
    out_dir = LLM / "colab"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"colab_finetune_{short}.ipynb"
    nbf.write(nb, str(path))
    print(f"wrote {path}  (contains Kaggle token -- gitignored, do not commit)")


if __name__ == "__main__":
    main()
