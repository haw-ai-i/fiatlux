#!/usr/bin/env bash
# Stage-1 environment: Hunyuan3D-2.1 (image -> textured mesh).
#
# CPython 3.11 is forced, and the reason is worth recording because the repo's
# own README says 3.10.
#
# bpy is not optional: `hy3dpaint/DifferentiableRenderer/mesh_utils.py:17`
# imports it at module level, MeshRender.save_mesh comes from that module, and
# MeshRender.py swallows the ImportError -- so a missing bpy surfaces as a
# NameError deep inside the paint stage.  But `bpy==4.0` (the pin) no longer
# exists on PyPI at all, and every bpy release still published ships **cp311**
# wheels only (4.2.x-5.0.x) or cp313 (5.1+).  There is no cp310 bpy to install.
#
# Meanwhile the pins that actually blocked 3.12 -- numpy==1.24.4,
# pymeshlab==2022.2.post3, open3d==0.18.0, onnxruntime==1.16.3 -- all publish
# cp311 wheels.  So 3.11 satisfies everything, and as a bonus matches fiatlux's
# own `requires-python = "==3.11.*"`.
#
# Run 01_system_deps.sh first -- custom_rasterizer is a CUDAExtension and needs nvcc.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTO="$(cd "$HERE/.." && pwd)"
HY="$AUTO/Hunyuan3D-2.1"
VENV="$HERE/.venvs/hunyuan"
UV="${UV:-$HOME/.local/bin/uv}"
PY="$VENV/bin/python"

# A100 is sm_80.  Building only that arch cuts the custom_rasterizer compile
# from ~15 min to ~2 and produces exactly the code this box can run.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export CUDA_HOME="${CUDA_HOME:-$(ls -d /usr/local/cuda-12.* /usr/local/cuda 2>/dev/null | head -1)}"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_NVCC_FLAGS="-allow-unsupported-compiler"
# Pin a matched host-compiler pair.  Left to itself nvcc picks up /usr/bin/gcc
# (12) while /usr/bin/g++ is 11 on this image; 01_system_deps.sh installs g++-12
# so the pair exists, and these make the choice explicit rather than incidental.
export CC="${CC:-gcc-12}" CXX="${CXX:-g++-12}"

command -v nvcc >/dev/null || { echo "ERROR: nvcc not found. Run setup/01_system_deps.sh first."; exit 1; }
echo "==> CUDA_HOME=$CUDA_HOME  arch=$TORCH_CUDA_ARCH_LIST"

# --no-config: see the note in 03_env_simart.sh -- fiatlux's pyproject.toml
# would otherwise override torch to 2.7.0 and demand CPython 3.11.
echo "==> creating $VENV (CPython 3.11)"
"$UV" venv --no-config --allow-existing --python 3.11 "$VENV"
pipi() { VIRTUAL_ENV="$VENV" "$UV" pip install --no-config --python "$PY" "$@"; }

echo "==> torch 2.5.1+cu124"
pipi torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# requirements.txt lines 1-2 are Tencent/Aliyun mirrors that are slow or
# unreachable from here; strip them and resolve against PyPI.  Also dropped, on
# the same "not needed for inference" argument the repo applies to nothing else:
# deepspeed (training-only, wants a compiler), and the demo/server stack
# (gradio, fastapi, uvicorn, pythreejs, tb_nightly) -- stage1_shape.py imports
# none of it, only gradio_app.py / api_server.py do, and we never run those.
echo "==> Hunyuan3D requirements (mirrors + demo stack stripped, bpy re-pinned)"
REQ="$(mktemp)"; trap 'rm -f "$REQ"' EXIT
grep -v -e '^--extra-index-url' -e '^deepspeed' -e '^bpy' \
        -e '^gradio' -e '^fastapi' -e '^uvicorn' -e '^pythreejs' -e '^tb_nightly' \
        "$HY/requirements.txt" > "$REQ"
pipi -r "$REQ"
pipi "bpy==${BPY_VERSION:-4.2.23}"        # see the header: bpy==4.0 is gone from PyPI
# (pybind11 and ninja are already pinned by requirements.txt lines 5-6 -- do NOT
# re-install them unpinned here, that risks drifting the C++ build's headers.)
# pytorch-lightning 1.9.5 -> lightning_fabric/__init__.py does
# `__import__("pkg_resources").declare_namespace(__name__)`.  uv venvs ship no
# setuptools, and declare_namespace is gone in setuptools >= 81, so pin below it.
pipi "setuptools<81"
# tencent/Hunyuan3D-2.1 is Xet-backed; without this the ~30 GB of weights falls
# back to plain HTTP on first run.
pipi hf_xet

echo "==> building custom_rasterizer (CUDA)"
( cd "$HY/hy3dpaint/custom_rasterizer" && VIRTUAL_ENV="$VENV" "$UV" pip install \
    --no-config --python "$PY" --no-build-isolation -e . )

# c++ -shared ... `python -m pybind11 --includes` -o mesh_inpaint_processor<ext>.
# Run it under the venv python so pybind11 headers and the ABI suffix match.
echo "==> building mesh_inpaint_processor (C++/pybind11)"
( cd "$HY/hy3dpaint/DifferentiableRenderer" \
  && c++ -O3 -Wall -shared -std=c++11 -fPIC \
       $("$PY" -m pybind11 --includes) \
       mesh_inpaint_processor.cpp \
       -o "mesh_inpaint_processor$("$PY"-config --extension-suffix 2>/dev/null \
          || "$PY" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')" )

echo "==> RealESRGAN weights"
mkdir -p "$HY/hy3dpaint/ckpt"
[ -f "$HY/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ] || wget -q --show-progress -O \
    "$HY/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# Both of these imports are swallowed by bare try/except in MeshRender.py and
# only surface as a confusing crash much later, so assert them loudly now.
echo "==> verifying the silent-failure imports"
( cd "$HY" && PYTHONPATH="$HY/hy3dpaint:$HY/hy3dshape" "$PY" - <<'PY'
import torch, bpy, custom_rasterizer
from DifferentiableRenderer import mesh_inpaint_processor
print(f"  torch              {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  bpy                {bpy.app.version_string}")
print("  custom_rasterizer  OK")
print("  mesh_inpaint_processor OK")
print("  hunyuan env OK")
PY
)
