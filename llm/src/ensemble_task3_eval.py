"""Pick the Task 3 two-method ensemble (prompting + fine-tuned) on the 32 gold samples
before applying anything to the public test. Both inputs are per-sample records
{sample_id, pred_labels, pred_feedback} from the same forced-self-diagnosis setup.

Feedback text can't be averaged, so strategies differ in WHICH method's feedback each
sample gets; labels can be voted (union/intersection). Exam-type routing is the
principled split: the fine-tune only saw single-problem synth code, while the prompting
exemplar can show the multi-problem "P1 đúng; P2 ..." checklist style.

Usage: python ensemble_task3_eval.py <prompting_records.json> <finetuned_records.json>
"""
import argparse
import json

from data import load_sample_dataset
from research_task3_ideas import score


def load_records(path):
    data = json.load(open(path, encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) else data
    return {r["sample_id"]: r for r in records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompting")
    ap.add_argument("finetuned")
    args = ap.parse_args()

    ds = load_sample_dataset()
    pr, ft = load_records(args.prompting), load_records(args.finetuned)
    exam_type = {s["sample_id"]: ds.exams[s["input"]["exam_id"]]["exam_type"] for s in ds.tasks[3]}
    ids = [s["sample_id"] for s in ds.tasks[3]]

    def combine(name, labels_fn, feedback_fn):
        recs = [{"sample_id": i, "pred_labels": labels_fn(i), "pred_feedback": feedback_fn(i)} for i in ids]
        return score(name, ds, recs)

    union = lambda i: sorted(set(pr[i]["pred_labels"]) | set(ft[i]["pred_labels"]))
    inter = lambda i: sorted(set(pr[i]["pred_labels"]) & set(ft[i]["pred_labels"]))
    fb_pr = lambda i: pr[i]["pred_feedback"]
    fb_ft = lambda i: ft[i]["pred_feedback"]
    fb_route = lambda i: ft[i]["pred_feedback"] if exam_type[i] == "single_problem" else pr[i]["pred_feedback"]

    rows = [
        combine("prompting only", lambda i: pr[i]["pred_labels"], fb_pr),
        combine("finetuned only", lambda i: ft[i]["pred_labels"], fb_ft),
        combine("labels union / fb prompting", union, fb_pr),
        combine("labels union / fb finetuned", union, fb_ft),
        combine("labels inter / fb prompting", inter, fb_pr),
        combine("labels union / fb by exam_type", union, fb_route),
        combine("labels prompting / fb by exam_type", lambda i: pr[i]["pred_labels"], fb_route),
    ]
    keys = ["macro_f1", "micro_f1", "mean_jaccard", "compliance_rate", "overlap_f1"]
    print(f"{'strategy':38s} " + " ".join(f"{k:>15s}" for k in keys))
    for r in rows:
        print(f"{r['name']:38s} " + " ".join(f"{r[k]:>15.4f}" for k in keys))


if __name__ == "__main__":
    main()
