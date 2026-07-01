#!/usr/bin/env bash
#
# Pull Fiatlux benchmark USD assets from the GCS bucket into this folder.
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
#   behavior1k_bulb/          light_bulb models (bulblampM Male plug)
#   behavior1k_bulb_broken/   broken_light_bulb models
#   behavior1k_lamp/          table_lamp models with bulblampF Female socket
#   behavior1k_ladder/        ladder models
#
# Room dressing (always synced -- defines the default look of every recorded
# run): table, warehouse backdrop + clutter props, and an HDRI sky, mirrored
# once from Isaac Sim's own Nucleus content library into our bucket so nothing
# is fetched live from NVIDIA's CDN at sim launch.
#   isaac_packing_table/      table the lamp/bulb rest on
#   isaac_warehouse/          warehouse room backdrop + clutter props
#   isaac_skies/              PolyHaven HDRI sky for the dome light
#
# Scene dressing (environment lighting, no bulb socket):
#   behavior1k_downlight/     recessed ceiling fixtures
#   behavior1k_room_light/    ceiling/pendant/wall fixtures
#   behavior1k_spotlight/     directional ceiling/track fixtures
#   behavior1k_square_light/  flat panel ceiling fixtures
#   behavior1k_rectangular_light/  fluorescent overhead panels
#   behavior1k_track_light/   track-mounted fixtures
#   behavior1k_wall_mounted_light/ wall sconces
#   behavior1k_chandelier/    ceiling-hanging multi-bulb fixtures
#   behavior1k_paper_lantern/ hanging pendant lanterns
#   behavior1k_lampshade/     shade housing components
#   behavior1k_floor_lamp/    standing lamps
#
# Override bucket with FIATLUX_ASSET_BUCKET env var.
# USD/mesh files are git-ignored.
set -euo pipefail

ASSET_BUCKET="${FIATLUX_ASSET_BUCKET:-gs://fiatlux/assets}"
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_DRESSING=0

for arg in "$@"; do
    case "$arg" in
        --scene-dressing) SCENE_DRESSING=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if ! command -v gsutil &> /dev/null; then
    echo "error: gsutil not found. Install the Google Cloud SDK and run 'gcloud auth login'."
    exit 1
fi

ROBOT_ASSETS=(
    unitree_g1
)

TASK_ASSETS=(
    behavior1k_bulb
    behavior1k_bulb_broken
    behavior1k_lamp
    behavior1k_ladder
    omniverse_ladder        # 98 climb-ready ladders/platforms (convex-decomp collision)
    omniverse_bulb          # separable LightBulb (bulb-swap candidate)
)

ROOM_ASSETS=(
    isaac_packing_table
    isaac_warehouse
    isaac_skies
)

SCENE_DRESSING_ASSETS=(
    behavior1k_downlight
    behavior1k_room_light
    behavior1k_spotlight
    behavior1k_square_light
    behavior1k_rectangular_light
    behavior1k_track_light
    behavior1k_wall_mounted_light
    behavior1k_chandelier
    behavior1k_paper_lantern
    behavior1k_lampshade
    behavior1k_floor_lamp
    omniverse_climb         # Mezzanine/OfficeSet elevated-platform climb structures
    omniverse_lamp          # Omniverse residential lamps/fixtures
)

sync_group() {
    local group="$1"
    echo "  $group"
    mkdir -p "${TARGET_DIR}/${group}"
    gsutil -m rsync -r "${ASSET_BUCKET}/${group}" "${TARGET_DIR}/${group}"
}

echo "Syncing robot assets from ${ASSET_BUCKET} ..."
for group in "${ROBOT_ASSETS[@]}"; do
    sync_group "$group"
done

echo "Syncing task assets from ${ASSET_BUCKET} ..."
for group in "${TASK_ASSETS[@]}"; do
    sync_group "$group"
done

echo "Syncing room dressing assets from ${ASSET_BUCKET} ..."
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

echo "Done."
