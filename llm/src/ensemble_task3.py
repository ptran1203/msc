"""Task 3 ensemble of prompting + fine-tuned outputs, weighted by each source's leaderboard score.

The leaderboard scores whole submissions, so weights are per source (not per item/level):
  taxonomy_error  weighted vote: keep a label if sum(w of sources predicting it) / sum(w) >= --threshold
  feedback        text can't be averaged -> taken from the highest-weight source, or per level
                  with --level-from "Level 1=ft" (e.g. routing chosen on the 32 sample items)
Then run postprocess_task3.py on the result (grounded compile / all-tests-pass rules).

  python ensemble_task3.py --source 7b=<json>:0.3011 --source think=<json>:<lb> --source ft=<json>:<lb> \
      [--level-from "Level 1=ft"] --out-tag task3-ens [--split public]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from data import REPO_ROOT, load_public_test, load_sample_dataset


def parse_source(text):
    name, rest = text.split("=", 1)
    path, weight = rest.rsplit(":", 1)
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return name, float(weight), {r["sample_id"]: r["output"] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="name=<predictions_task3 json>:<leaderboard score>")
    ap.add_argument("--level-from", action="append", default=[], help='"Level N=<source name>": feedback source for that level')
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--split", choices=["sample", "public"], default="public")
    args = ap.parse_args()

    sources = [parse_source(s) for s in args.source]
    weights = {n: w for n, w, _ in sources}
    preds = {n: p for n, _, p in sources}
    route = dict(x.split("=", 1) for x in args.level_from)
    assert set(route.values()) <= set(preds), f"unknown source in --level-from: {route}"
    default_fb = max(weights, key=weights.get)
    total_w = sum(weights.values())
    print("weights:", {n: round(w / total_w, 3) for n, w in weights.items()}, "| default feedback:", default_fb)

    ds = load_public_test() if args.split == "public" else load_sample_dataset()
    out, fb_used = [], Counter()
    for s in ds.tasks[3]:
        sid, level = s["sample_id"], s["input"]["target_feedback_level"]
        votes = Counter()
        for n, p in preds.items():
            for label in dict.fromkeys(p[sid].get("taxonomy_error") or []):
                votes[label] += weights[n]
        labels = [l for l, v in votes.items() if v / total_w >= args.threshold]
        fb_src = next((src for lv, src in route.items() if level.startswith(lv)), default_fb)
        fb_used[fb_src] += 1
        out.append({"sample_id": sid, "output": {"taxonomy_error": labels, "feedback": preds[fb_src][sid]["feedback"]}})

    path = REPO_ROOT / "outputs" / args.out_tag / f"predictions_task3_{args.split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    n_lab = sum(len(o["output"]["taxonomy_error"]) for o in out) / len(out)
    print(f"feedback from: {dict(fb_used)} | labels/item {n_lab:.2f} -> {path}")


if __name__ == "__main__":
    main()
