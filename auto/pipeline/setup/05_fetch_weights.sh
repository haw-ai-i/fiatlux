#!/usr/bin/env bash
# Prefetch every model the pipeline downloads lazily, so the first real run
# starts GPU work immediately instead of stalling ~10 min on ~35 GB of
# mid-stage downloads.  Idempotent: hf skips files that are already complete.
#
# (03_env_simart.sh already prefetches SimArt's 19 GB and Blender on the same
# rationale; this script covers the Hunyuan3D side, which was the gap.)
#
# Layout matters for the shape model: hy3dshape's loader
# (hy3dshape/utils/utils.py:96-110) does NOT use the normal HF hub cache -- it
# calls snapshot_download with local_dir=$HY3DGEN_MODELS/tencent/Hunyuan3D-2.1
# (default ~/.cache/hy3dgen).  Prefetching into the hub cache would be invisible
# to it and the 4 GB would download a second time.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYH="$HERE/.venvs/hunyuan/bin/python"
# The hunyuan venv pins huggingface-hub 0.30.2, whose CLI is still named
# `huggingface-cli` (the `hf` alias arrived later); accept either.
HF="$HERE/.venvs/hunyuan/bin/hf"
[ -x "$HF" ] || HF="$HERE/.venvs/hunyuan/bin/huggingface-cli"
[ -x "$HF" ] || { echo "ERROR: run setup/02_env_hunyuan.sh first (need $HF)"; exit 1; }

HY3DGEN="${HY3DGEN_MODELS:-$HOME/.cache/hy3dgen}"

echo "==> Hunyuan3D shape DiT (~4 GB) -> $HY3DGEN/tencent/Hunyuan3D-2.1"
"$HF" download tencent/Hunyuan3D-2.1 \
    --include "hunyuan3d-dit-v2-1/*" \
    --local-dir "$HY3DGEN/tencent/Hunyuan3D-2.1"

echo "==> Hunyuan3D paint PBR UNet (~8 GB, normal hub cache)"
"$HF" download tencent/Hunyuan3D-2.1 --include "hunyuan3d-paintpbr-v2-1/*"

echo "==> DINOv2-giant (~4.5 GB, normal hub cache)"
"$HF" download facebook/dinov2-giant

echo "==> rembg u2net (~170 MB) -> ~/.u2net"
"$PYH" - <<'PY'
from rembg import new_session
new_session()          # downloads u2net.onnx on first call, no-op after
print("  u2net OK")
PY

echo "==> all weights present"
