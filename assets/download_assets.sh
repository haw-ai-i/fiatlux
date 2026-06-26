#!/usr/bin/env bash
#
# Pull Fiatlux benchmark USD assets from the GCS bucket into this folder.
#
# Usage:
#   ./download_assets.sh                   # task assets only
#   ./download_assets.sh --scene-dressing  # task assets + scene dressing
#
# Task assets (bulb/socket mechanic + ladder):
#   behavior1k_bulb/          light_bulb models (bulblampM Male plug)
#   behavior1k_bulb_broken/   broken_light_bulb models
#   behavior1k_lamp/          table_lamp models with bulblampF Female socket
#   behavior1k_ladder/        ladder models
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

TASK_ASSETS=(
    behavior1k_bulb
    behavior1k_bulb_broken
    behavior1k_lamp
    behavior1k_ladder
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
)

sync_group() {
    local group="$1"
    echo "  $group"
    mkdir -p "${TARGET_DIR}/${group}"
    gsutil -m rsync -r "${ASSET_BUCKET}/${group}" "${TARGET_DIR}/${group}"
}

echo "Syncing task assets from ${ASSET_BUCKET} ..."
for group in "${TASK_ASSETS[@]}"; do
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
