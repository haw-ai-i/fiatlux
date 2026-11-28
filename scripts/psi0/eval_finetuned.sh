#!/usr/bin/env bash
# Serve a fine-tuned Psi-0 run dir on a second port and evaluate it on S06 with the same adapter,
# server settings, instruction, and sweep as the zero-shot baseline (docs/psi0_finetune.md,
# "Evaluation"). The zero-shot :8014 server must NOT be running: one 24 GB GPU fits one server
# plus one Isaac instance.
#
#   RUN_DIR=~/psi0_ft/runs/finetune/<run> STEP=2000 scripts/psi0/eval_finetuned.sh
#   VIDEO_SEEDS="0 1 2 3" PSI0_OUT=logs/runs/psi0_ft_s06_video RUN_DIR=... scripts/psi0/eval_finetuned.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${RUN_DIR:?the fine-tuned run dir (argv.txt, run_config.json, checkpoints/ckpt_N)}"
STEP="${STEP:-2000}"
PORT="${PORT:-8015}"
export PSI0_OUT="${PSI0_OUT:-logs/runs/psi0_ft_s06}" SEEDS="${SEEDS:-0 1 2 3}" VIDEO_SEEDS="${VIDEO_SEEDS:-0}"

mkdir -p "$PSI0_OUT"
PSI0_RUN_DIR="$RUN_DIR" PSI0_CKPT_STEP="$STEP" PSI0_PORT="$PORT" scripts/psi0/serve.sh > "$PSI0_OUT/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
    curl -s -m 5 "localhost:$PORT/health" | grep -q ok && break
    kill -0 $server 2>/dev/null || { echo "server exited; see $PSI0_OUT/server.log" >&2; exit 1; }
    sleep 5
done
PSI0_SPEC="psi0:localhost:$PORT" scripts/psi0/sweep.sh S06
"${PY:-.venv/bin/python}" scripts/psi0/summarize.py "$PSI0_OUT" || true
