#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"

# Sync assets
ASSET_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/fiatlux_assets/scripts/download_assets.sh"
if [ -f "$ASSET_SCRIPT" ]; then
    bash "$ASSET_SCRIPT"
fi
