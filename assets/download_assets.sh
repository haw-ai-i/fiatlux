#!/usr/bin/bash
#
# Pull the Fiatlux benchmark USD assets from the GCS bucket into this folder.
#
# Expected layout after sync (consumed by g1_bulb_env_cfg.py):
#   assets/unitree_g1/g1.usd          - Unitree G1 humanoid
#   assets/bulb_socket/bulb.usd       - graspable light bulb
#   assets/bulb_socket/socket.usd     - lamp / socket fixture
#   assets/ladder/ladder.usd          - ladder (climbing subtask, roadmap)
#
# Override the bucket with FIATLUX_ASSET_BUCKET. USD/mesh files are git-ignored.
set -euo pipefail

ASSET_BUCKET="${FIATLUX_ASSET_BUCKET:-gs://fiatlux/assets}"
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v gsutil &> /dev/null; then
    echo "Error: 'gsutil' not found. Install the Google Cloud SDK and authenticate"
    echo "       (gcloud auth login) to download assets from ${ASSET_BUCKET}."
    exit 1
fi

echo "Syncing Fiatlux assets from ${ASSET_BUCKET} -> ${TARGET_DIR} ..."
# Pull only the groups the benchmark needs (G1, bulb/socket, ladder).
for group in unitree_g1 bulb_socket ladder; do
    echo "  - ${group}"
    gsutil -m rsync -r "${ASSET_BUCKET}/${group}" "${TARGET_DIR}/${group}"
done
echo "Done."
