#!/usr/bin/env bash
# Stage-4 environment: the URDF -> USD converter and its verifier.
#
# Deliberately tiny and GPU-free.  Kept separate from the two ML envs so the
# converter can be iterated in seconds without touching ~40 GB of torch wheels,
# and pinned to CPython 3.11 to match fiatlux's `requires-python = "==3.11.*"`,
# so `urdf_to_usd.py` stays importable from the benchmark itself later.
#
# usd-core 25.5.x is the same OpenUSD minor that Isaac Sim 5.1 embeds, so the
# crate files we write are exactly what Isaac will read.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HERE/.venvs/usd"
UV="${UV:-$HOME/.local/bin/uv}"

# --no-config: see the note in 03_env_simart.sh -- fiatlux's pyproject.toml
# would otherwise force its own torch/index overrides onto this env.
echo "==> creating $VENV (CPython 3.11)"
"$UV" venv --no-config --allow-existing --python 3.11 "$VENV"

echo "==> installing usd-core + numpy + trimesh"
VIRTUAL_ENV="$VENV" "$UV" pip install --no-config --python "$VENV/bin/python" \
    "usd-core==25.5.1" "numpy>=1.26,<2" "trimesh==4.5.1" "pillow>=10"

"$VENV/bin/python" - <<'PY'
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, Vt, UsdUtils
import numpy, trimesh, PIL
print(f"  pxr        {Usd.GetVersion()}")
print(f"  numpy      {numpy.__version__}")
print(f"  trimesh    {trimesh.__version__}")
print("  usd env OK")
PY
