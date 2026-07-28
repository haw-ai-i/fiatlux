#!/usr/bin/env python3
"""Download NVIDIA Omniverse USD asset-pack ZIPs by name (public, no auth).

Source: https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html#d-openusd-asset-packs
These are the raw packs the Fiatlux Omniverse assets were sourced from; the workflow is
  download (this) -> scan (omniverse_pack_scan.py) -> extract -> author collision
  (omniverse_ladder_collision.py) -> rigid (omniverse_ladder_rigid.py).

Resumable (uses `curl -C -`). Defaults to ~/omniverse_packs/zips/.

Usage:
  python scripts/omniverse/omniverse_pack_download.py --list
  python scripts/omniverse/omniverse_pack_download.py Warehouse Residential
  python scripts/omniverse/omniverse_pack_download.py Warehouse --dest /data/packs
"""

import argparse
import os
import shutil
import subprocess
import sys

BASE = "https://d4i3qtqj3r0z5.cloudfront.net"

# friendly name -> CloudFront zip filename (the %40 is an encoded '@')
PACKS = {
    "Warehouse": "Warehouse_NVD%4010013.zip",
    "Residential": "Residential_NVD%4010012.zip",
    "Commercial": "Commercial_NVD%4010013.zip",
    "Industrial": "Industrial_NVD%4010012.zip",
    "SimReady_Warehouse_01": "SimReady_Warehouse_01_NVD%4010010.zip",
    "SimReady_Warehouse_02": "SimReady_Warehouse_02_NVD%4010010.zip",
    "SimReady_Furniture_Misc": "SimReady_Furniture_Misc_01_NVD%4010010.zip",
    "SimReady_Containers_Shipping_01": "SimReady_Containers_Shipping_01_NVD%4010010.zip",
    "SimReady_Containers_Shipping_02": "SimReady_Containers_Shipping_02_NVD%4010010.zip",
    "Sample_Scenes": "Sample_Scenes_NVD%4010013.zip",
    "Showcase": "Showcases_Content_NVD%4010011.zip",
    "Default_Scene_Templates": "Scene_Templates_NVD%4010011.zip",
    "Rigged_Characters": "Characters_NVD%4010012.zip",
    "Data_Center": "Datacenter_NVD%4010012.zip",
}
DEFAULT_DEST = os.path.expanduser("~/omniverse_packs/zips")


def download(name, dest):
    if name not in PACKS:
        print(f"  !! unknown pack '{name}' (use --list)", file=sys.stderr)
        return False
    if not shutil.which("curl"):
        print("  !! curl not found (required for resumable download)", file=sys.stderr)
        return False
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, f"{name}.zip")
    url = f"{BASE}/{PACKS[name]}"
    print(f"  downloading {name} -> {out}")
    rc = subprocess.call(["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o", out, url])
    if rc == 0:
        print(f"  done: {name} ({os.path.getsize(out) / 1e9:.2f} GB)")
    else:
        print(f"  !! curl failed for {name} (rc={rc})", file=sys.stderr)
    return rc == 0


def main():
    ap = argparse.ArgumentParser(description="Download Omniverse asset-pack ZIPs by name.")
    ap.add_argument("packs", nargs="*", help="pack names (see --list)")
    ap.add_argument("--dest", default=DEFAULT_DEST, help=f"output dir (default {DEFAULT_DEST})")
    ap.add_argument("--list", action="store_true", help="list available pack names")
    args = ap.parse_args()

    if args.list or not args.packs:
        print("Available packs:")
        for n in PACKS:
            print(f"  {n}")
        return
    ok = sum(download(n, args.dest) for n in args.packs)
    print(f"\n{ok}/{len(args.packs)} packs downloaded to {args.dest}")


if __name__ == "__main__":
    main()
