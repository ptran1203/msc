"""Build the Kaggle runner notebook: clones the public code repo, attaches the private
data via Kaggle dataset mounts, and runs the grading pipeline on Kaggle's free GPU."""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/ptran1203/msc.git"
KAGGLE_USERNAME = "ptran1203"
DATASET_SLUG = "llm-autograding-course-data"
DATASET_MOUNT = f"/kaggle/input/datasets/{KAGGLE_USERNAME}/{DATASET_SLUG}"
TESTS_SLUG = "llm-autograding-llm-tests"
TESTS_MOUNT = f"/kaggle/input/datasets/{KAGGLE_USERNAME}/{TESTS_SLUG}"
RUN_TAG = "qwen7b-fewshot-testreport"

cells = []

cells.append(new_markdown_cell(
f"""# LLM auto-grading -- Kaggle runner

Clones the public code repo ([{REPO_URL}]({REPO_URL})) and attaches two private datasets:
`ptran1203/{DATASET_SLUG}` (the real student data -- course rules forbid distributing it,
so it never goes in the repo) and `ptran1203/{TESTS_SLUG}` (the verified LLM-generated test
suites from `test_report.py`, uploaded as-is so the tests here are exactly the ones checked
locally against the 32 gold samples).

Every submission gets two input signals the released data leaves empty, both obtained by
running the student's own code (never gold labels): the real g++ `compile_log`
(`compile_check.py`, compile-only) and `test_report` (`test_report.py`, single-problem exams).

**Needs accelerator "GPU T4 x2"** (Settings -> Accelerator), set once by hand in the editor."""
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
f"""import shutil, pathlib, glob
repo_llm = pathlib.Path("/kaggle/working/msc/llm")
shutil.copytree("{DATASET_MOUNT}/data", repo_llm / "data")
(repo_llm / "outputs/llm_tests").mkdir(parents=True, exist_ok=True)
for p in glob.glob("{TESTS_MOUNT}/*.json"):
    shutil.copy(p, repo_llm / "outputs/llm_tests")
print("test suites:", sorted(p.name for p in (repo_llm / "outputs/llm_tests").iterdir()))
assert len(list((repo_llm / "outputs/llm_tests").iterdir())) >= 3, "test-suite dataset not attached"
"""
))

cells.append(new_markdown_cell(
f"""## Run -- Tasks 1, 2, 3 on the public test (328 samples), few-shot + compile_log + test_report

One process per GPU (`--num-shards 2`), merged by `merge_shards.py`. Each step uses
`subprocess.run(check=True)` -- a failing `!python ...` does NOT stop a notebook."""
))

cells.append(new_code_cell(
f"""%cd /kaggle/working/msc/llm/src
import os, subprocess
RUN_TAG = "{RUN_TAG}"
cmd = lambda i: ["python", "-u", "run_prompting.py", "--split", "public", "--tasks", "1,2,3", "--fewshot",
                 "--run-tag", RUN_TAG, "--num-shards", "2", "--shard-id", str(i)]
procs = [subprocess.Popen(cmd(i), env={{**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}}) for i in range(2)]
for p in procs:
    p.wait()
assert all(p.returncode == 0 for p in procs), [p.returncode for p in procs]
subprocess.run(["python", "merge_shards.py", "--split", "public", "--tasks", "1,2,3", "--method-tag", RUN_TAG, "--num-shards", "2"], check=True)
"""
))

cells.append(new_markdown_cell(
"""## Task 3 rules + validation

`postprocess_task3.py`: Level 2 + real compile error -> the g++ error line; Level 1 +
passes every verified test -> "Tốt.". Then every file must pass `validate_submission.py`,
or the kernel fails here instead of shipping an invalid file."""
))

cells.append(new_code_cell(
"""subprocess.run(["python", "postprocess_task3.py", f"../outputs/{RUN_TAG}/predictions_task3_public.json",
                "--out-tag", f"{RUN_TAG}-rules"], check=True)
FINAL = {
    1: f"../outputs/{RUN_TAG}/predictions_task1_public.json",
    2: f"../outputs/{RUN_TAG}/predictions_task2_public.json",
    3: f"../outputs/{RUN_TAG}-rules/predictions_task3_public.json",
}
for t, path in FINAL.items():
    subprocess.run(["python", "validate_submission.py", str(t), path, "--split", "public"], check=True)
"""
))

cells.append(new_markdown_cell(
"""## Collect + clean up

Keeps only `/kaggle/working/submission/` and deletes the cloned repo."""
))

cells.append(new_code_cell(
"""sub_dir = pathlib.Path("/kaggle/working/submission")
sub_dir.mkdir(exist_ok=True)
for t, path in FINAL.items():
    shutil.copy(path, sub_dir / f"predictions_task{t}.json")
# Task 3 before the rules too, so the rules' effect can be measured on the leaderboard
shutil.copy(f"../outputs/{RUN_TAG}/predictions_task3_public.json", sub_dir / "predictions_task3_norules.json")
%cd /kaggle/working
shutil.rmtree("/kaggle/working/msc", ignore_errors=True)
print("submission files:", sorted(p.name for p in sub_dir.iterdir()))
"""
))

# Kaggle's papermill engine needs kernelspec metadata or it fails before any cell runs.
KERNELSPEC_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nb = new_notebook(cells=cells, metadata=KERNELSPEC_METADATA)
with open("../kaggle_kernel/kaggle_runner.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote kaggle_kernel/kaggle_runner.ipynb")
