#!/usr/bin/bash
set -e

# Target directory is fiatlux_assets/models relative to the script
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
ASSET_URL="gs://fiatlux/assets"

echo "Checking for Fiatlux assets updates..."

if ! command -v gsutil &> /dev/null; then
    echo "Warning: 'gsutil' command not found. Assets will not be downloaded."
    echo "Please install Google Cloud SDK to sync assets from $ASSET_URL."
    echo "If you get an access error after installing, reach out to molybog@hawaii.edu"
    exit 0
fi

# Use rsync to only pull down changes
gsutil -m rsync -r "$ASSET_URL" "$TARGET_DIR" || {
    echo "Failed to sync assets."
    echo "If you get an access error (e.g. 403 Forbidden), please reach out to molybog@hawaii.edu"
}
