#!/usr/bin/env bash
#
# Pull Fiatlux benchmark USD assets from the haw-ai-i/fiatlux-assets HF dataset into this folder.
#
# Usage:
#   ./download_assets.sh                   # robot + task assets
#   ./download_assets.sh --scene-dressing  # robot + task assets + scene dressing
#
# Robot assets (always synced):
#   unitree_g1/wholebody_inspire/  legged G1 + Inspire hands (env default)
#   unitree_g1/wholebody_{dex3,dex1}/  legged G1 + Dex3 / gripper
#   unitree_g1/{inspire,dex3,gripper}/  fixed-base G1 variants
#   Unitree's official pre-assembled USDs, mirrored from the HuggingFace dataset
#   unitreerobotics/unitree_sim_isaaclab_usds (Apache-2.0).
#
# Task assets (bulb/socket mechanic + ladder):
#   omniverse_bulb/           the graspable bulb + stock socket (BULB_USD/SOCKET_USD); this script
#                             authors the additive guide-sleeve layer locally, see below
#   omniverse_ladder/         step ladders; AlumStep_D is the one the benchmark climbs
#
# Room dressing (always synced -- defines the default look of every recorded
# run): table, warehouse backdrop + clutter props, and an HDRI sky, mirrored
# once from Isaac Sim's own Nucleus content library into our own dataset repo so nothing
# is fetched live from NVIDIA's CDN at sim launch.
#   isaac_packing_table/      table the lamp/bulb rest on
#   isaac_room/                room backdrop (walls/floor/windows)
#   isaac_skies/              PolyHaven HDRI sky for the dome light
#
# Scene dressing (opt-in via --scene-dressing; no benchmark preset spawns these):
#   omniverse_climb/          Mezzanine/OfficeSet elevated-platform climb structures
#   omniverse_lamp/           Omniverse residential lamps/fixtures
#
# Override the dataset repo with FIATLUX_ASSET_REPO env var.
# USD/mesh files are git-ignored.
set -euo pipefail

ASSET_REPO="${FIATLUX_ASSET_REPO:-haw-ai-i/fiatlux-assets}"
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_DRESSING=0

for arg in "$@"; do
    case "$arg" in
        --scene-dressing) SCENE_DRESSING=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if ! command -v uvx &> /dev/null; then
    echo "error: uvx not found. Install uv (https://docs.astral.sh/uv/)." >&2
    exit 1
fi

# Pinned floor: the "<group>/" subfolder form (expanded to "<group>/**") only exists in recent
# huggingface_hub, and older CLIs read it as a literal filename and 404. uvx otherwise takes latest.
HF_DOWNLOAD=(uvx --from 'huggingface_hub>=1.16' hf download "$ASSET_REPO" --repo-type dataset --local-dir "$TARGET_DIR")

ROBOT_ASSETS=(
    unitree_g1
)

TASK_ASSETS=(
    omniverse_ladder        # climb-ready ladders/platforms (convex-decomp / SDF collision)
    omniverse_bulb          # the separable LightBulb: BULB_USD + SOCKET_USD
)

ROOM_ASSETS=(
    isaac_packing_table
    isaac_room
    isaac_skies
)

# Opt-in dressing.
SCENE_DRESSING_ASSETS=(
    omniverse_climb         # Mezzanine/OfficeSet elevated-platform climb structures
    omniverse_lamp          # Omniverse residential lamps/fixtures
)

sync_group() {
    local group="$1"
    echo "  $group"
    if ! "${HF_DOWNLOAD[@]}" "${group}/"; then
        echo "error: could not fetch '${group}' from ${ASSET_REPO}." >&2
        if [[ -z "${FIATLUX_ASSET_REPO:-}" ]]; then
            echo "  If that was an auth failure: the default dataset is public and needs no login," >&2
            echo "  so this is likely a transient network/rate-limit issue -- retry, or check" >&2
            echo "  https://huggingface.co/datasets/${ASSET_REPO} directly." >&2
        else
            echo "  If that was an auth failure: FIATLUX_ASSET_REPO points at '${ASSET_REPO}', which" >&2
            echo "  may be private. Run 'uvx --from huggingface_hub hf auth login', or set HF_TOKEN" >&2
            echo "  to a token scoped to that repo, then retry." >&2
        fi
        exit 1
    fi
    # A pattern that matches nothing is a no-op for snapshot_download, NOT an error: without this
    # check a group that is missing from the dataset (never uploaded, renamed) reports a clean
    # sync and only fails much later, at scene build, as a missing-USD error.
    if [[ -z "$(ls -A "${TARGET_DIR}/${group}" 2>/dev/null)" ]]; then
        echo "error: '${group}' downloaded nothing -- is it still in ${ASSET_REPO}?" >&2
        exit 1
    fi
}

echo "Syncing robot assets from ${ASSET_REPO} ..."
for group in "${ROBOT_ASSETS[@]}"; do
    sync_group "$group"
done

echo "Syncing task assets from ${ASSET_REPO} ..."
for group in "${TASK_ASSETS[@]}"; do
    sync_group "$group"
done

# Author the socket's guide sleeve (issue #171): the dataset only ships the stock socket, and
# fiatlux_task/assets.py's SOCKET_USD points at the additive "_sleeve" layer this
# script generates on top of it (see its module docstring). Doing it here rather than baking the
# sleeve into the dataset keeps the synced asset as the stock socket, and means a fresh sync always
# leaves SOCKET_USD resolvable instead of failing at scene build with a missing-file error.
BULB_DIR="${TARGET_DIR}/omniverse_bulb"
if [[ -d "$BULB_DIR" ]]; then
    echo "Authoring the socket guide sleeve ..."
    if ! (cd "${TARGET_DIR}/.." && uv run python scripts/omniverse/omniverse_socket_guide_sleeve.py "$BULB_DIR"); then
        echo "  WARNING: could not author the socket guide sleeve. Until it is, SOCKET_USD" >&2
        echo "  (fiatlux_task/assets.py) points at a file that does not exist, and any" >&2
        echo "  task touching the socket will fail at scene build." >&2
        echo "  Re-run by hand from the repo root:" >&2
        echo "    uv run python scripts/omniverse/omniverse_socket_guide_sleeve.py $BULB_DIR" >&2
    fi
fi

echo "Syncing room dressing assets from ${ASSET_REPO} ..."
for group in "${ROOM_ASSETS[@]}"; do
    sync_group "$group"
done

if [[ "$SCENE_DRESSING" -eq 1 ]]; then
    echo "Syncing scene dressing assets ..."
    for group in "${SCENE_DRESSING_ASSETS[@]}"; do
        sync_group "$group"
    done
else
    echo "(skip scene dressing — pass --scene-dressing to include)"
fi

# Re-author the ladder colliders and their rigid overlays, in that order, BEFORE the platform box
# below (the collision pass rewrites <name>_collision.usd, which would drop the box).
#
# The dataset ships colliders authored as SDF. SDF is a signed distance field: its sign is the
# inside/outside of a CLOSED surface, and 92 of the 98 collected designs are open meshes, so the
# field is undefined wherever a hole is -- on AlumStep_D01 the holes cluster in the bottom 24 cm,
# the legs. omniverse_ladder_collision.py now measures that (_has_open_mesh) and routes an open
# design to convexDecomposition, which ignores topology; the 6 genuinely watertight designs keep
# SDF. Doing it here rather than re-baking the dataset keeps the synced asset as the vendor
# shipped it, and keeps a re-sync from silently reinstating the SDF colliders.
LADDER_DIR="${TARGET_DIR}/omniverse_ladder"
# The task ladder's real mass. Werner P400-4 catalogue net weight (24 lb); AlumStep_D01 matches
# that ladder on all four defining dimensions. Keep in sync with LADDER_MASS_KG in scene_cfg.py --
# the rigid overlay authors an inertia tensor for THIS mass, and PhysX does not rescale an
# authored inertia when a code-side override changes the mass.
TASK_LADDER_COLLISION="${LADDER_DIR}/AlumStep_D/AluminumStepLadder_D01_PR_NVD_01_collision.usd"
TASK_LADDER_MASS=10.9
if [[ -d "$LADDER_DIR" ]]; then
    echo "Authoring ladder colliders (open meshes -> convex, watertight -> SDF) ..."
    if ! (cd "${TARGET_DIR}/.." \
            && uv run python scripts/omniverse/omniverse_ladder_collision.py "$LADDER_DIR" \
            && uv run python scripts/omniverse/omniverse_ladder_rigid.py "$LADDER_DIR" \
            && uv run python scripts/omniverse/omniverse_ladder_rigid.py \
                "$TASK_LADDER_COLLISION" --mass "$TASK_LADDER_MASS"); then
        echo "  WARNING: could not author the ladder colliders. Until they are, the ladders keep" >&2
        echo "  the dataset's SDF colliders, whose fields are undefined where the meshes are open," >&2
        echo "  and their mass properties stay derived from collider volume." >&2
        echo "  Re-run by hand from the repo root:" >&2
        echo "    uv run python scripts/omniverse/omniverse_ladder_collision.py $LADDER_DIR" >&2
        echo "    uv run python scripts/omniverse/omniverse_ladder_rigid.py $LADDER_DIR" >&2
        echo "    uv run python scripts/omniverse/omniverse_ladder_rigid.py \\" >&2
        echo "        $TASK_LADDER_COLLISION --mass $TASK_LADDER_MASS" >&2
    fi
fi

# The step ladder's standing platform exists in its render mesh but not in its collider: the
# collision overlay's convexDecomposition leaves open air where the tread is, and a robot placed
# on it falls through the ladder. Author the box collider that fixes it here rather than baking
# it into the dataset, so the synced asset stays as the vendor shipped it. Idempotent -- the
# script replaces its own prim on every run.
LADDER_COLLISION_USD="${TARGET_DIR}/omniverse_ladder/AlumStep_D/AluminumStepLadder_D01_PR_NVD_01_collision.usd"
PLATFORM_ARGS=(--centre 0 10 116 --half-extent 24 20 2)
if [[ -f "$LADDER_COLLISION_USD" ]]; then
    echo "Authoring the step ladder's platform collider ..."
    if ! (cd "${TARGET_DIR}/.." && uv run python scripts/omniverse/omniverse_ladder_platform.py \
            "$LADDER_COLLISION_USD" "${PLATFORM_ARGS[@]}"); then
        echo "  WARNING: could not author the platform collider. Until it is, the at-height" >&2
        echo "  subtasks (S02-S04, S10-S12) drop the robot straight through the ladder." >&2
        echo "  Re-run by hand from the repo root:" >&2
        echo "    uv run python scripts/omniverse/omniverse_ladder_platform.py \\" >&2
        echo "        $LADDER_COLLISION_USD ${PLATFORM_ARGS[*]}" >&2
    fi
fi

echo "Done."
