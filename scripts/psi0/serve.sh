#!/usr/bin/env bash
# Start the Psi-0 (SONIC) HTTP policy server for the `psi0` baseline policy spec.
#
# The 2.8B VLA runs in the Psi0 repo's own uv venv (torch 2.7 / py3.11), separate from the
# fiatlux Isaac Lab venv; the benchmark side talks to it over HTTP POST /act
# (fiatlux_task/psi0.py). Setup (see docs/psi0_baseline.md for the full notes):
#
#   git clone https://github.com/physical-superintelligence-lab/Psi0 ~/tools/Psi0
#   cd ~/tools/Psi0 && uv venv .venv-psi --python 3.11 && source .venv-psi/bin/activate
#   rm uv.lock   # upstream lock at 4f3720d has a duplicated TOML table and does not parse
#   GIT_LFS_SKIP_SMUDGE=1 uv sync --group serve --group viz --group psi \
#       --index-strategy unsafe-best-match --active
#   hf download USC-PSI-Lab/psi-model \
#       --include "psi0/sonic-checkpoints/multi-task.psi-dream.2609092156/*" \
#       --local-dir ~/tools/psi0_checkpoints
#
#   PSI0_PORT=8014 scripts/psi0/serve.sh
set -euo pipefail

PSI0_REPO="${PSI0_REPO:-$HOME/tools/Psi0}"
PSI0_RUN_DIR="${PSI0_RUN_DIR:-$HOME/tools/psi0_checkpoints/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156}"
PSI0_CKPT_STEP="${PSI0_CKPT_STEP:-40000}"
PSI0_PORT="${PSI0_PORT:-8014}"
# Rows of each 30-row (1 s at 30 Hz) chunk that /act returns. 15 with test-time RTC is the
# synchronous analog of Psi0's own deployment server (serve_psi0_sonic.py: H=30, s_min=15,
# test-time guidance for this --model.no-rtc checkpoint); the sim clock is paused while a
# request is in flight, so the inference delay is exactly 0 and nothing else changes.
PSI0_EXEC_HORIZON="${PSI0_EXEC_HORIZON:-15}"
PSI0_RTC="${PSI0_RTC:-1}"

rtc_args=()
if [ "$PSI0_RTC" = "1" ]; then
    rtc_args=(--rtc --rtc-mode test_time)
fi

# serve_http.py is upstream's serve_psi0_sonic_http plus the CLIP pooled-instruction input the
# SONIC checkpoints need (its docstring says why); it runs in the Psi0 venv.
SERVE_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serve_http.py"
cd "$PSI0_REPO"
exec "$PSI0_REPO/.venv-psi/bin/python" "$SERVE_PY" \
    --policy psi0 \
    --host 127.0.0.1 \
    --port "$PSI0_PORT" \
    --run-dir "$PSI0_RUN_DIR" \
    --ckpt-step "$PSI0_CKPT_STEP" \
    --action-exec-horizon "$PSI0_EXEC_HORIZON" \
    "${rtc_args[@]}" \
    "$@"
