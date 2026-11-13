"""Load exams, label space, and per-task samples for the auto-grading dataset."""
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATASET = REPO_ROOT / "data/public-test-cham-diem-phan-hoi-files/sample_dataset"
PUBLIC_TEST = REPO_ROOT / "data/public-test-cham-diem-phan-hoi-files/public_test_release"

TASK_FILES = {
    1: "task1_grading.json",
    2: "task2_error_taxonomy.json",
    3: "task3_feedback.json",
}


@dataclass
class Dataset:
    root: Path
    exams: dict  # exam_id -> exam dict
    label_space: dict
    tasks: dict  # task_num -> list of sample dicts


def _index_exams(exams_json: dict) -> dict:
    return {e["exam_id"]: e for e in exams_json["exams"]}


def load_dataset(root: Path) -> Dataset:
    exams = _index_exams(json.loads((root / "exams.json").read_text(encoding="utf-8")))
    label_space = json.loads((root / "label_space.json").read_text(encoding="utf-8"))
    tasks = {}
    for task_num, fname in TASK_FILES.items():
        path = root / fname
        if path.exists():
            tasks[task_num] = json.loads(path.read_text(encoding="utf-8"))["samples"]
    return Dataset(root=root, exams=exams, label_space=label_space, tasks=tasks)


def load_sample_dataset() -> Dataset:
    return load_dataset(SAMPLE_DATASET)


def load_public_test() -> Dataset:
    return load_dataset(PUBLIC_TEST)


def read_code(ds: Dataset, sample: dict) -> str:
    code_path = ds.root / sample["input"]["code_file"]
    return code_path.read_text(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    for name, loader in [("sample_dataset", load_sample_dataset), ("public_test", load_public_test)]:
        ds = loader()
        print(f"{name}: exams={list(ds.exams)} tasks={{{', '.join(f'{k}:{len(v)}' for k, v in ds.tasks.items())}}}")
        s = ds.tasks[1][0]
        code = read_code(ds, s)
        print(f"  first sample {s['sample_id']}: code_file ok, {len(code)} chars")
