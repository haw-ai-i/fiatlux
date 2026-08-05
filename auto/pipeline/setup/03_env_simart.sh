#!/usr/bin/env bash
# Stage-3 environment: SimArt (monolithic mesh -> articulated parts + URDF).
#
# CPython 3.10 per SimArt's README.  torch 2.4.0+cu121 per its requirements.txt
# -- cu121 wheels run fine against the box's 580.x driver.
#
# flash_attn is installed BEST-EFFORT only.  `infer.py:291` hardcodes
# attn_implementation="flash_attention_2", and building flash_attn from source
# takes ~30 min and needs an exact torch/python/CUDA match; patch 01 makes that
# a runtime choice defaulting to `sdpa` instead, so a failure here is harmless.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTO="$(cd "$HERE/.." && pwd)"
SIMART="$AUTO/SimArt"
VENV="$HERE/.venvs/simart"
TOOLS="$HERE/.tools"
UV="${UV:-$HOME/.local/bin/uv}"
PY="$VENV/bin/python"

# --no-config is load-bearing: these venvs live inside the fiatlux tree, and uv
# walks up to /home/shadeform/fiatlux/pyproject.toml, whose
# `override-dependencies = ["torch==2.7.0", ...]` and `requires-python ==3.11.*`
# would otherwise be forced onto this (deliberately different) environment.
echo "==> creating $VENV (CPython 3.10)"
"$UV" venv --no-config --allow-existing --python 3.10 "$VENV"

pipi() { VIRTUAL_ENV="$VENV" "$UV" pip install --no-config --python "$PY" "$@"; }

echo "==> torch 2.4.0+cu121"
pipi torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "==> SimArt requirements"
pipi accelerate==1.7.0 transformers==4.57.0 trimesh==4.5.3 \
     qwen-vl-utils==0.0.14 scipy==1.14.1 open3d==0.19.0 numpy==1.26.4 tqdm \
     "huggingface_hub[cli]" hf_xet

echo "==> flash-attn (best effort; sdpa fallback is wired in via patch 01)"
pipi flash_attn==2.7.4.post1 --no-build-isolation 2>/dev/null \
    && echo "   flash_attn OK" \
    || echo "   flash_attn unavailable -> stage 3 will run with SIMART_ATTN_IMPL=sdpa"

# ---------------------------------------------------------------- checkpoints
# infer.py's defaults are --model_path ./checkpoints/simart_mllm and
# --vqvae_ckpt_dir ./checkpoints/simart_vqvae, and the HF repo's top-level
# layout is exactly those two directories, so a plain repo download lands right.
CKPT="$SIMART/checkpoints"
if [ -f "$CKPT/simart_vqvae/vq.pt" ] && [ -f "$CKPT/simart_mllm/config.json" ]; then
    echo "==> checkpoints already present ($(du -sh "$CKPT" | cut -f1))"
else
    echo "==> downloading ByteDance-Seed/SimArt (~19 GB) -> $CKPT"
    "$VENV/bin/hf" download ByteDance-Seed/SimArt \
        --local-dir "$CKPT"
fi

# ------------------------------------------------------------------- blender
# utils/render_utils.py auto-downloads Blender 3.0.1 into /tmp mid-run.  Do it
# here instead, into a stable location we pass via --blender_path, so a cleared
# /tmp can't cost us a 200 MB re-download in the middle of a GPU run.
BLENDER="$TOOLS/blender-3.0.1-linux-x64/blender"
if [ -x "$BLENDER" ]; then
    echo "==> blender already present: $BLENDER"
else
    echo "==> downloading Blender 3.0.1 -> $TOOLS"
    mkdir -p "$TOOLS"
    wget -q --show-progress -O "$TOOLS/blender.tar.xz" \
        https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz
    tar -xf "$TOOLS/blender.tar.xz" -C "$TOOLS"
    rm -f "$TOOLS/blender.tar.xz"
fi
"$BLENDER" --version | head -1

"$PY" - <<'PY'
import torch, transformers, open3d, trimesh, numpy
print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  transformers {transformers.__version__}")
print(f"  open3d       {open3d.__version__}")
print("  simart env OK")
PY
