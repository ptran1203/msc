"""Write gold-free grading inputs so an LLM grader can be evaluated without label leakage.

Two variants per exam: code-only (statement + rubric + code) and with-signals (adds compile_log,
test_report and the deterministic signal summary). No rubric score, taxonomy or feedback is written.
With --per-sample, also writes one with-signals file per submission (<out-dir>/per-sample/<sample_id>.json) so
a grader can be called once per submission, as in production, without seeing other students' code.
Usage: python poc/blind_inputs.py --data-root <dataset> --out-dir <dir outside the dataset> [--per-sample]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_data import load_meta, load_samples  # noqa: E402
from signals import compile_status, test_summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-sample", action="store_true")
    args = ap.parse_args()
    label_space, exams, _ = load_meta(args.data_root)
    df = load_samples(args.data_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for exam_id, exam in exams.items():
        sub = df[df.exam_id == exam_id].sort_values("sample_id")
        for variant in ("code-only", "with-signals"):
            samples = []
            for r in sub.itertuples():
                s = {"sample_id": r.sample_id, "code": r.code or ""}
                if variant == "with-signals":
                    tests = r.test_report if isinstance(r.test_report, list) else None
                    s.update(compile_log=r.compile_log or "", build_status=compile_status(r.compile_log or ""),
                             test_report=tests, test_summary=test_summary(tests))
                samples.append(s)
            payload = {
                "exam": {k: exam.get(k) for k in ("exam_id", "exam_type", "language", "statement", "problems",
                                                   "grading_policy", "has_hidden_testcases", "note")},
                "rubric": label_space["rubric"],
                "taxonomy_error": label_space["taxonomy_error"],
                "samples": samples,
            }
            out = args.out_dir / f"{exam_id.lower()}-{variant}.json"
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{out} ({len(samples)} samples)")
            if args.per_sample and variant == "with-signals":
                single_dir = args.out_dir / "per-sample"
                single_dir.mkdir(exist_ok=True)
                for s in samples:
                    single = {**payload, "samples": [s]}
                    (single_dir / f"{s['sample_id']}.json").write_text(json.dumps(single, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
