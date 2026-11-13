#!/usr/bin/env bash
# Stages a, b, d and e of docs/psi0_reproduce.md section 9, on a machine with no Isaac Lab:
# teleop bags -> SONIC tokens -> LeRobot pack, released checkpoint -> warm start, then the two
# CPU checks. No GPU. Idempotent: each stage is skipped when its output already exists.
#
# Two things differ from the runbook, neither of which needs the stock scripts changed:
#   - stage a is run through run_encode.py, because encode_sonic_tokens.py imports two leaf
#     modules of a package whose __init__ pulls in Isaac Lab (see that file);
#   - build_psi0_pack.py shells out to ffmpeg and ffprobe, which are not on Koa's default PATH.
#     The FFmpeg module is loaded in a subshell so its GCCcore paths never reach the trainer.
#
#   PSI0_KOA_REPO=/path/to/fiatlux scripts/psi0/koa/prep_koa.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

FFMPEG_MODULE="${FFMPEG_MODULE:-vis/FFmpeg/7.1.2-GCCcore-14.3.0}"
PACK_ID="${PACK_ID:-s06_dispose_teleop}"
PACK="$FT/packs/$PACK_ID"
INIT="$FT/init/psi_dream_40k"
RUNDIR="$CKPTS/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156"
TASK="${TASK:-put the light bulb into the yellow crate and let go of it}"
PSI="$PSI0_REPO/.venv-psi/bin/python"
say() { echo "[prep] $*"; }
say "node=$(hostname) started=$(date)"

# a. SONIC tokens. The encode step gets a small CPU venv of its own: the Psi0 venv is the
#    training environment, and onnxruntime's pins have no business perturbing it.
if [ ! -x "$ROOT/venv-encode/bin/python" ]; then
    UV_HTTP_TIMEOUT=600 UV_TORCH_BACKEND=cpu "$UV" venv "$ROOT/venv-encode" --python 3.11
    UV_HTTP_TIMEOUT=600 UV_TORCH_BACKEND=cpu "$UV" pip install \
        --python "$ROOT/venv-encode/bin/python" numpy torch onnxruntime h5py
fi
ENC="$ROOT/venv-encode/bin/python"

# encode_sonic_tokens.py defaults --encoder to ~/tools/...; on Koa everything is in scratch.
ENCODER="$SONIC_DIR/policy/release/model_encoder.onnx"
[ -s "$ENCODER" ] || { echo "FATAL: no $ENCODER (setup step 3)" >&2; exit 1; }

say "encoder layout check (standing pose)"
"$ENC" "$HERE/run_encode.py" "$REPO" --check-stand --encoder "$ENCODER"
if [ ! -d "$FT/encoded/success" ]; then
    "$ENC" "$HERE/run_encode.py" "$REPO" \
        --takes "$FT/raw/success" --out "$FT/encoded/success" --encoder "$ENCODER"
fi
say "encoded: $(find "$FT/encoded/success" -name '*.npz' | wc -l) takes"

# b. LeRobot pack.
if [ ! -s "$PACK/meta/stats_psi0.json" ]; then
    (
        module load "$FFMPEG_MODULE"
        exec "$PSI" "$REPO/scripts/psi0/build_psi0_pack.py" \
            --encoded "$FT/encoded/success" --takes "$FT/raw/success" --out "$PACK" \
            --ckpt-run-dir "$RUNDIR" --task "$TASK"
    )
fi
say "pack: $PACK ($(du -sh "$PACK" | cut -f1))"

# d. Warm start.
if [ ! -s "$INIT/model.safetensors" ]; then
    PSI0_REPO="$PSI0_REPO" RUN="$RUNDIR" HF_FROM="$FT/hf_from" OUT="$INIT" \
        bash "$REPO/scripts/psi0/export_warmstart.sh"
fi
say "warm start: $(find "$INIT" -maxdepth 1 -name '*.safetensors' -printf '%f ' )"

# e. CPU checks: the data path, then the warm-start key match.
export PSI0_REPO PACK_ID
export PACK_ROOT="$FT/packs" INIT_DIR="$INIT" OUT_DIR="$FT/runs"
DRY_RUN=1    bash "$REPO/scripts/psi0/finetune_s06.sh"
CHECK_INIT=1 bash "$REPO/scripts/psi0/finetune_s06.sh"
say "finished=$(date)"
