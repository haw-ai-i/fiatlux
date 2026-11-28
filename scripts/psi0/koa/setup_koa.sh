#!/usr/bin/env bash
# Koa variant of scripts/psi0/setup.sh. It runs the stock script -- all six steps, unmodified --
# but splits it either side of one package pin that a fresh resolve needs:
#
#   pyproject.toml pins numpy<2.0.0 (line 16) and leaves pyarrow unbounded (line 124). Upstream's
#   uv.lock at 4f3720d does not parse, so setup.sh re-resolves from pyproject and lands on
#   pyarrow >= 21, which refuses to import against numpy 1.x:
#       ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4
#   datasets needs pyarrow >= 15, so the venv is held in the 15-20 window after step 1, before
#   anything imports pyarrow -- step 4, build_psi0_pack.py and the trainer all do.
#
# This is not in fact Koa-specific: any machine resolving today hits it. It lives here rather
# than in setup.sh so the stock path stays exactly as docs/psi0_reproduce.md describes it.
#
#   PSI0_KOA_REPO=/path/to/fiatlux HF_TOKEN=... scripts/psi0/koa/setup_koa.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
: "${HF_TOKEN:?set HF_TOKEN -- step 6 pulls the private teleop dataset}"
say() { echo "[koa] $*"; }

mkdir -p "$TOOLS" "$FT" "$ROOT/boot"

# uv bootstrap. The system python3.9 is only used to install uv; uv then fetches its own 3.11.
if [ ! -x "$UV" ]; then
    say "bootstrapping uv"
    python3 -m venv "$ROOT/boot"
    "$ROOT/boot/bin/python" -m pip -q install --upgrade pip
    "$ROOT/boot/bin/python" -m pip -q install uv "huggingface_hub[cli]"
fi
say "uv=$($UV --version)"

STEPS="1" bash "$REPO/scripts/psi0/setup.sh"

"$UV" pip install -q --python "$PSI0_REPO/.venv-psi/bin/python" "pyarrow<21"
"$PSI0_REPO/.venv-psi/bin/python" -c \
    "import numpy, pyarrow; print('[koa] pyarrow', pyarrow.__version__, '/ numpy', numpy.__version__)"

STEPS="2 3 4 5 6" bash "$REPO/scripts/psi0/setup.sh"

# Step 2 pre-fetches the frozen CLIP-L text encoder the trainer loads by name under
# HF_HUB_OFFLINE=1, with:
#     hf download openai/clip-vit-large-patch14 --include "*.json" "*.txt" "model.safetensors"
# This CLI's --include takes a single pattern, so "*.txt" and "model.safetensors" are parsed as
# positional filenames -- and when filenames are given, --include is ignored outright. The cache
# ends up with merges.txt and the weights but no vocab.json, and training dies ~70 s in with
#     TypeError: expected str, bytes or os.PathLike object, not NoneType
# out of CLIPTokenizer.from_pretrained, long after setup has reported success. Fetching the
# files by name avoids the ambiguity entirely. (Not Koa-specific either; the Qwen line above it
# is fine because it passes one --exclude pattern and no positionals.)
hf download openai/clip-vit-large-patch14 \
    config.json vocab.json tokenizer.json tokenizer_config.json \
    special_tokens_map.json preprocessor_config.json merges.txt model.safetensors > /dev/null

# Assert it, rather than trust it: this is the exact call that failed.
HF_HUB_OFFLINE=1 "$PSI0_REPO/.venv-psi/bin/python" -c \
    "from transformers import CLIPTokenizer as T; T.from_pretrained('openai/clip-vit-large-patch14'); print('[koa] CLIP tokenizer loads offline')"

say "setup done"
