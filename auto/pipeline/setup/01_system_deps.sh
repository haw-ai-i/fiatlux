#!/usr/bin/env bash
# System packages for the image -> mesh -> USD pipeline.  Needs sudo.
#
# The one genuinely unavoidable item is the CUDA toolkit: Hunyuan3D's texture
# stage rasterizes through `custom_rasterizer`, a CUDAExtension that must be
# compiled from source, and `MeshRender.py:372-377` hardcodes raster_mode="cr"
# with no CPU fallback.  No nvcc -> no texturing.
#
# The rest is what Blender 3.0.1 (SimArt's renderer) and bpy/OpenCV need to
# import and run headless.
set -euo pipefail

CUDA_VER="${CUDA_VER:-12-4}"

echo "==> apt: base build + GL/EGL runtime"
sudo apt-get update -qq
# g++-12 is not incidental: this image ships gcc-12 but only g++-11, so
# /usr/bin/gcc is 12 while /usr/bin/g++ is 11 and there is no cc1plus for 12.
# nvcc drives the host compiler as `gcc` for C++ translation units, which then
# dies with "cannot execute 'cc1plus'" when building custom_rasterizer.
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential cmake ninja-build pkg-config wget curl git \
    gcc-12 g++-12 \
    python3-dev \
    libgl1 libglx0 libegl1 libgles2 libglib2.0-0 \
    libxrender1 libxi6 libxxf86vm1 libxfixes3 libxkbcommon-x11-0 \
    libsm6 libxext6 \
    libeigen3-dev libcgal-dev

if command -v nvcc >/dev/null 2>&1; then
    echo "==> nvcc already present: $(nvcc --version | tail -1)"
    exit 0
fi

echo "==> apt: CUDA toolkit ${CUDA_VER} (for nvcc)"
KEYRING=/tmp/cuda-keyring_1.1-1_all.deb
wget -q -O "$KEYRING" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i "$KEYRING"
sudo apt-get update -qq
# toolkit only -- NOT `cuda`, which would drag in a second driver and can
# conflict with the running 580.126.09 one.
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "cuda-toolkit-${CUDA_VER}"

CUDA_HOME="/usr/local/cuda-${CUDA_VER/-/.}"
echo "==> installed: $("$CUDA_HOME/bin/nvcc" --version | tail -1)"
echo "    CUDA_HOME=$CUDA_HOME   (02_env_hunyuan.sh exports this itself)"
