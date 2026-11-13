#!/usr/bin/env bash
# One-time setup for the Psi-0 baseline and the S06 fine-tune: the Psi0 repo + its own venv, and
# every artifact the experiments read. Idempotent: each step is skipped when its output exists.
# Needs git, uv, and the `hf` CLI (Hugging Face); HF_TOKEN must be set for the private teleop
# dataset (step 6) only. See docs/psi0_reproduce.md for what each step is for.
#
#   scripts/psi0/setup.sh            # everything
#   STEPS="1 2 3" scripts/psi0/setup.sh
set -euo pipefail

TOOLS="${TOOLS:-$HOME/tools}"
PSI0_REPO="${PSI0_REPO:-$TOOLS/Psi0}"
PSI0_COMMIT="${PSI0_COMMIT:-4f3720d}"
CKPTS="${CKPTS:-$TOOLS/psi0_checkpoints}"
SONIC_DIR="${SONIC_DIR:-$TOOLS/sonic_models}"
FT="${FT:-$HOME/psi0_ft}"              # fine-tune workspace (data, packs, runs)
STEPS="${STEPS:-1 2 3 4 5 6}"
UV="${UV:-uv}"

want() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "[setup] $*"; }

# 1. Psi0 repo at the tested commit, with its own py3.11 venv. Upstream's uv.lock at 4f3720d has a
#    duplicated TOML table and does not parse (docs/psi0_baseline.md #2), so it is re-resolved from
#    pyproject.toml's pins. No flash-attn: inference and our training both use SDPA (#3, finetune #10).
if want 1; then
    if [ ! -d "$PSI0_REPO/.git" ]; then
        GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/physical-superintelligence-lab/Psi0 "$PSI0_REPO"
    fi
    git -C "$PSI0_REPO" checkout -q "$PSI0_COMMIT"
    if [ ! -x "$PSI0_REPO/.venv-psi/bin/python" ]; then
        (
            cd "$PSI0_REPO"
            rm -f uv.lock
            UV_HTTP_TIMEOUT=600 "$UV" venv .venv-psi --python 3.11
            # shellcheck disable=SC1091
            source .venv-psi/bin/activate
            GIT_LFS_SKIP_SMUDGE=1 UV_HTTP_TIMEOUT=600 "$UV" sync --group serve --group viz --group psi \
                --index-strategy unsafe-best-match --active
        )
    fi
    # Psi0's scripts/train.py asserts a .env exists (docs/psi0_finetune.md #11). Nothing here is
    # read by the server.
    [ -f "$PSI0_REPO/.env" ] || printf 'PSI_HOME=%s\nTOKENIZERS_PARALLELISM=false\nAV_LOG_LEVEL=quiet\nPROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python\nNO_ALBUMENTATIONS_UPDATE=1\n' "$FT" > "$PSI0_REPO/.env"
    "$PSI0_REPO/.venv-psi/bin/python" -c "import psi; print('[setup] psi importable')"
fi

# 2. The released Psi-0 SONIC checkpoint (deployable run dir, ~11 GB), and the HF config/processor
#    files of its post-trained VLM (no weights), which the fine-tune's warm start needs.
if want 2; then
    run="$CKPTS/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156"
    [ -s "$run/checkpoints/ckpt_40000/model.safetensors" ] || hf download USC-PSI-Lab/psi-model \
        --include "psi0/sonic-checkpoints/multi-task.psi-dream.2609092156/*" --local-dir "$CKPTS"
    hf_from="$FT/hf_from"
    if [ ! -s "$hf_from/config.json" ]; then
        hf download USC-PSI-Lab/psi-model --include "psi0/postpre.sonic1.0.unifolm.2609092156.40k/*" \
            --exclude "*.safetensors" --local-dir "$FT/_hf_postpre"
        mkdir -p "$hf_from"
        cp "$FT"/_hf_postpre/psi0/postpre.sonic1.0.unifolm.2609092156.40k/* "$hf_from"/
    fi
    # Pre-fetch the two public HF models the server and trainer load by name (the trainer runs
    # with HF_HUB_OFFLINE=1): the Qwen3-VL config/processor and the frozen CLIP-L text encoder.
    hf download Qwen/Qwen3-VL-2B-Instruct --exclude "*.safetensors" > /dev/null
    hf download openai/clip-vit-large-patch14 --include "*.json" "*.txt" "model.safetensors" > /dev/null
    say "checkpoint: $run"
fi

# 3. GEAR-SONIC release (v1.0) decoder + encoder, where fiatlux_task/groot.py's DEFAULT_SONIC_ONNX
#    and encode_sonic_tokens.py look for them.
if want 3; then
    if [ ! -s "$SONIC_DIR/policy/release/model_decoder.onnx" ]; then
        hf download nvidia/GEAR-SONIC model_decoder.onnx model_encoder.onnx observation_config.yaml config.json \
            --local-dir "$SONIC_DIR/_hf"
        mkdir -p "$SONIC_DIR/policy/release"
        for f in model_decoder.onnx model_encoder.onnx observation_config.yaml; do
            ln -sfn "$SONIC_DIR/_hf/$f" "$SONIC_DIR/policy/release/$f"
        done
    fi
    say "SONIC release models: $SONIC_DIR/policy/release"
fi

# 4. Psi-0's public UnifoLM SONIC validation pack (33 MB): ground truth for
#    verify_sonic_tracking.py and unifolm_reencode.py.
if want 4; then
    uni="$FT/unifolm_sonic_lerobot_val"
    if [ ! -d "$uni/meta" ]; then
        hf download USC-PSI-Lab/psi-data sonic/unifolm_sonic_lerobot_val.zip --repo-type dataset --local-dir "$FT/_hf_data"
        (cd "$FT" && unzip -q -o _hf_data/sonic/unifolm_sonic_lerobot_val.zip)
    fi
    "$PSI0_REPO/.venv-psi/bin/python" "$(dirname "$0")/unifolm_to_npz.py" "$uni" "$FT/unifolm_npz"
fi

# 5. (nothing to fetch for the zero-shot sweep itself beyond steps 1-3 and the fiatlux assets.)

# 6. The S06 dispose-bulb VR teleop takes (private: haw-ai-i/fiatlux-teleoperation, needs HF_TOKEN),
#    laid out as $FT/raw/{success,fail}/<take>/.
if want 6; then
    : "${HF_TOKEN:?set HF_TOKEN for the private teleop dataset}"
    sub="2026-09-13-dex3-teleop-takes/FIATLUX-S06-DisposeBulb-Teleop-v0/dex3/hdf5/vr"
    hf download haw-ai-i/fiatlux-teleoperation --repo-type dataset --include "$sub/*" --local-dir "$FT/_hf_teleop"
    for kind in success fail; do
        mkdir -p "$FT/raw/$kind"
        for take in "$FT/_hf_teleop/$sub/$kind"/*/; do
            ln -sfn "${take%/}" "$FT/raw/$kind/$(basename "$take")"
        done
    done
    say "teleop takes: $(ls "$FT/raw/success" | wc -l) success, $(ls "$FT/raw/fail" | wc -l) fail"
fi
say "done"
