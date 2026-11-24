#!/usr/bin/env bash
# Run every EDA script on a dataset root and collect the markdown output per area.
# Usage: ./run-eda.sh <data-root> [--with-exec]
#   --with-exec  also recompile submissions and run EX02 binaries on public tests
#                (untrusted code: only do this on a machine/container you can throw away)
set -euo pipefail

DATA_ROOT="${1:?usage: ./run-eda.sh <data-root> [--with-exec]}"
WITH_EXEC="${2:-}"
EDA_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$EDA_DIR/.venv/bin/python"
OUT="$EDA_DIR/out/report"
mkdir -p "$OUT"

if [[ "$WITH_EXEC" == "--with-exec" ]]; then
  T1_FLAGS=(); T2_FLAGS=(--compile --exec); CODE_FLAGS=(); META_FLAGS=()
else
  T1_FLAGS=(--no-compile --no-run); T2_FLAGS=(); CODE_FLAGS=(--no-compile --no-run); META_FLAGS=(--no-compile)
fi

run() {  # run <name> <script> [flags...]
  local name="$1" script="$2"; shift 2
  echo ">> $name"
  "$PY" "$EDA_DIR/$script" --data-root "$DATA_ROOT" "$@" > "$OUT/$name.md"
}

run t1-rubric      t1_rubric_eda.py   ${T1_FLAGS[@]+"${T1_FLAGS[@]}"}
run t2-taxonomy    t2_label_eda.py    ${T2_FLAGS[@]+"${T2_FLAGS[@]}"}
run t3-feedback    t3_feedback_eda.py
run code-signals   code_eda.py        ${CODE_FLAGS[@]+"${CODE_FLAGS[@]}"}
run meta-audit     meta_audit.py      ${META_FLAGS[@]+"${META_FLAGS[@]}"}
[[ "$WITH_EXEC" == "--with-exec" ]] && "$PY" "$EDA_DIR/viz/gcc_compile_check.py" --data-root "$DATA_ROOT"
echo ">> viz (input + rubric page)"
"$PY" "$EDA_DIR/viz/build_input_rubric_viz.py" --data-root "$DATA_ROOT" --out "$EDA_DIR/viz/input-rubric-eda.html"
echo "done -> $OUT"
