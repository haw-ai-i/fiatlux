#!/usr/bin/env bash
# Step 0: fetch the two upstream model repos the pipeline drives.
#
# Both are pinned to the exact commits this pipeline was built and validated
# against.  Upstream moves -- SimArt is a 2026 SIGGRAPH release and Hunyuan3D
# ships breaking API changes between minor versions -- and the workarounds in
# stages/ and usd/ cite specific line numbers, so an unpinned clone would rot.
# Override with HUNYUAN_REF=main / SIMART_REF=main to track upstream instead.
#
# Also makes the weights untrackable *inside* each clone.  SimArt has no
# .gitignore at all, so a fresh `git add .` in that directory would stage the
# 19 GB checkpoints/ tree.  We write .git/info/exclude rather than editing the
# working tree, so the clones stay byte-identical to upstream apart from the one
# documented patch below.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTO="$(cd "$HERE/.." && pwd)"

HUNYUAN_URL="${HUNYUAN_URL:-https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git}"
HUNYUAN_REF="${HUNYUAN_REF:-82920d643c0dc2f7bfd7255f45f62d386edfe60c}"
SIMART_URL="${SIMART_URL:-https://github.com/ByteDance-Seed/SimArt.git}"
SIMART_REF="${SIMART_REF:-c8d0829f37153feda9e0792131078a47fc15010d}"

clone_at() {
    local url="$1" ref="$2" dest="$3"
    if [ -d "$dest/.git" ]; then
        local head
        head="$(git -C "$dest" rev-parse HEAD)"
        if [ "$head" = "$ref" ]; then
            echo "==> $(basename "$dest") already present at ${head:0:7}"
        else
            # The whole point of pinning is that stages/ and usd/ cite specific
            # upstream line numbers -- a drifted checkout is the failure mode
            # this guard exists for, so say so instead of shrugging.
            echo "==> WARNING: $(basename "$dest") is at ${head:0:7}, expected ${ref:0:7}."
            echo "    The pipeline was validated against the pinned commit; either"
            echo "    'git -C $dest checkout $ref' or export the matching *_REF."
        fi
        return
    fi
    echo "==> cloning $(basename "$dest") from $url"
    # blob:none fetches history metadata but no file contents until checkout --
    # about an order of magnitude less traffic than a full clone for a pinned ref.
    git clone --filter=blob:none "$url" "$dest"
    git -C "$dest" checkout --quiet "$ref"
    echo "    checked out $(git -C "$dest" rev-parse --short HEAD)"
}

# Keep model weights, caches and pipeline outputs untracked by the clone's own
# git.  .git/info/exclude is per-clone and never committed, which is what we want.
exclude_in() {
    local repo="$1"; shift
    local ex="$repo/.git/info/exclude"
    mkdir -p "$(dirname "$ex")"
    for pat in "$@"; do
        grep -qxF "$pat" "$ex" 2>/dev/null || echo "$pat" >> "$ex"
    done
}

clone_at "$HUNYUAN_URL" "$HUNYUAN_REF" "$AUTO/Hunyuan3D-2.1"
clone_at "$SIMART_URL"  "$SIMART_REF"  "$AUTO/SimArt"

echo "==> excluding model weights from each clone's own git"
exclude_in "$AUTO/SimArt" \
    "checkpoints/" "output/" "outputs/" "__pycache__/" "*.pt" "*.safetensors"
exclude_in "$AUTO/Hunyuan3D-2.1" \
    "hy3dpaint/ckpt/" "__pycache__/" "*.ckpt" "*.safetensors" \
    "hy3dpaint/custom_rasterizer/build/" "*.so"

# The SimArt patch makes the attention backend selectable (so a run does not
# hinge on a flash_attn wheel) and guards a bare json.loads over 24k tokens of
# model output.  Idempotent: --check fails once it is already applied, and the
# reverse --check confirms that rather than assuming it.
PATCH="$HERE/patches/simart_infer.patch"
if [ -f "$PATCH" ]; then
    if git -C "$AUTO/SimArt" apply --check "$PATCH" 2>/dev/null; then
        git -C "$AUTO/SimArt" apply "$PATCH"
        echo "==> applied patches/simart_infer.patch"
    elif git -C "$AUTO/SimArt" apply --check -R "$PATCH" 2>/dev/null; then
        echo "==> patches/simart_infer.patch already applied"
    else
        echo "==> WARNING: patches/simart_infer.patch does not apply cleanly."
        echo "    SimArt has probably moved past the pinned commit. Without the"
        echo "    patch, stage 3 needs a working flash_attn build (upstream"
        echo "    hardcodes attn_implementation=flash_attention_2) and loses the"
        echo "    SIMART_ATTN_IMPL / SIMART_SEED overrides."
    fi
fi

for r in Hunyuan3D-2.1 SimArt; do
    printf '    %-16s %s  %s\n' "$r" \
        "$(git -C "$AUTO/$r" rev-parse --short HEAD)" \
        "$(git -C "$AUTO/$r" remote get-url origin)"
done
echo "==> repos ready"
