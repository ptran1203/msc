"""Recompile every submission with GNU g++ inside Docker and cache the result as JSON.

Compile only, never run (student code is untrusted): EX01 uses -fsyntax-only (no main, as the
release log does); single_problem builds and links. Container has no network.
Usage: python viz/gcc_compile_check.py --data-root <dataset> [--image IMG] [--out viz/gcc-results.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_data import load_samples  # noqa: E402

DEFAULT_IMAGE = "ghcr.io/open-webui/open-webui:main"  # local image that ships Debian g++ 12.2


def compile_one(image: str, root: Path, code_file: str, multi: bool) -> dict:
    src = f"/data/{code_file}"
    cmd = f"g++ -std=c++11 -fsyntax-only {src}" if multi else f"g++ -std=c++11 {src} -o /tmp/a.out"
    proc = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--memory", "512m", "-v", f"{root}:/data:ro",
         image, "sh", "-c", cmd],
        capture_output=True, text=True, timeout=120,
    )
    return {"ok": proc.returncode == 0, "stderr_head": proc.stderr[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("gcc-results.json"))
    args = ap.parse_args()
    root = args.data_root.resolve()
    version = subprocess.run(["docker", "run", "--rm", "--network", "none", args.image, "g++", "--version"],
                             capture_output=True, text=True).stdout.splitlines()[0]
    df = load_samples(root)
    results = {r.sample_id: compile_one(args.image, root, r.code_file, r.exam_type == "multi_problem")
               for r in df.itertuples()}
    args.out.write_text(json.dumps({"compiler": version, "image": args.image, "results": results},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{version}: {sum(not v['ok'] for v in results.values())}/{len(results)} fail -> {args.out}")


if __name__ == "__main__":
    main()
