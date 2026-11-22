"""One-off cleanup of llm/: moves Kaggle downloads, synth backups and legacy flat outputs
into gitignored folders. Dry run by default -- prints every move; --apply performs them.

  llm/kaggle_check_*                 -> llm/kaggle_outputs/<kernel>_<date>/   (Kaggle downloads)
  llm/synth/generated_*backup*       -> llm/synth/backups/<name>/
  llm/outputs/ loose logs            -> llm/outputs/_logs/
  llm/outputs/ flat predictions/raw/
    old v1 caches (pre run-folder)   -> llm/outputs/_legacy/
  llm/outputs/submission_task2_ensemble.json
                                     -> llm/outputs/submission_task2_ensemble/predictions_task2_public.json

Left in place (code reads/writes them): outputs/<run_tag>/ folders, compile_log_cache_v2_*,
test_results_cache_*, llm_tests/, research_task3/, ft_task3_*, public_test_compile_check.json,
synth/generated/. Never overwrites: a move whose target exists is reported and skipped.

Usage: python restructure_workspace.py [--apply]
"""
import argparse
import datetime
import re
import shutil
from pathlib import Path

LLM = Path(__file__).resolve().parent.parent
OUT = LLM / "outputs"
GITIGNORE_ADD = ["kaggle_outputs/", "synth/backups/"]

KAGGLE_KERNEL = {"runner": "runner", "synth": "synth", "finetune": "finetune"}
KEEP_IN_OUTPUTS = re.compile(r"^(compile_log_cache_v2_.*|test_results_cache_.*|public_test_compile_check\.json)$")
LEGACY_FILE = re.compile(r"^(predictions_task\d_.*\.json|raw_task\d_.*\.jsonl|compile_log_cache_(?!v2_).*\.json|test_report_cache_.*\.json)$")
LOG_FILE = re.compile(r".*\.(log|txt)$")


def mdate(p: Path) -> str:
    return datetime.date.fromtimestamp(p.stat().st_mtime).isoformat()


def plan():
    moves = []
    for d in sorted(LLM.glob("kaggle_check_*")):
        kind = re.match(r"kaggle_check_([a-z]+)", d.name).group(1)
        moves.append((d, LLM / "kaggle_outputs" / f"{KAGGLE_KERNEL.get(kind, kind)}_{mdate(d)}_{d.name}"))
    for d in sorted((LLM / "synth").glob("generated_*backup*")):
        moves.append((d, LLM / "synth" / "backups" / d.name))
    for f in sorted(OUT.iterdir()):
        if not f.is_file() or KEEP_IN_OUTPUTS.match(f.name):
            continue
        if f.name == "submission_task2_ensemble.json":
            moves.append((f, OUT / "submission_task2_ensemble" / "predictions_task2_public.json"))
        elif LEGACY_FILE.match(f.name):
            moves.append((f, OUT / "_legacy" / f.name))
        elif LOG_FILE.match(f.name):
            moves.append((f, OUT / "_logs" / f.name))
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    moves = plan()
    for src, dst in moves:
        clash = "  !! target exists, SKIP" if dst.exists() else ""
        print(f"{src.relative_to(LLM)}  ->  {dst.relative_to(LLM)}{clash}")
    gi = LLM / ".gitignore"
    missing = [e for e in GITIGNORE_ADD if e not in gi.read_text(encoding="utf-8").splitlines()]
    print(f"\n{len(moves)} moves; .gitignore additions: {missing or 'none'}")

    if not args.apply:
        print("dry run -- rerun with --apply")
        return
    done = 0
    for src, dst in moves:
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        done += 1
    if missing:
        with open(gi, "a", encoding="utf-8") as f:
            f.write("\n# Kaggle downloads + synth backups: outputs on real student data, never commit\n")
            f.write("\n".join(missing) + "\n")
    print(f"applied {done}/{len(moves)} moves")


if __name__ == "__main__":
    main()
