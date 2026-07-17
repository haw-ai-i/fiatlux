#!/usr/bin/env bash
# Start NVIDIA's Isaac-GR00T PolicyServer for the `groot` baseline policy spec.
#
# The 3B VLA runs in the Isaac-GR00T repo's own uv venv (CUDA 12.8 / py3.12), separate
# from the fiatlux Isaac Lab venv; the benchmark side talks to it over ZMQ
# (fiatlux_task/groot.py). Setup: journal/specs/groot-sonic-baseline.md.
#
#   GROOT_REPO=~/tools/Isaac-GR00T GROOT_PORT=5555 scripts/groot/serve.sh
set -euo pipefail

GROOT_REPO="${GROOT_REPO:-$HOME/tools/Isaac-GR00T}"
GROOT_MODEL="${GROOT_MODEL:-nvidia/GR00T-N1.7-3B}"
GROOT_PORT="${GROOT_PORT:-5555}"
# REAL_G1 is the base checkpoint's G1 head (decoupled whole-body: upper-body targets +
# navigate/base-height commands). UNITREE_G1_SONIC needs a finetuned checkpoint.
GROOT_EMBODIMENT="${GROOT_EMBODIMENT:-REAL_G1}"

exec uv run --project "$GROOT_REPO" python "$GROOT_REPO/gr00t/eval/run_gr00t_server.py" \
    --model-path "$GROOT_MODEL" \
    --embodiment-tag "$GROOT_EMBODIMENT" \
    --port "$GROOT_PORT" \
    "$@"
