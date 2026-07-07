#!/usr/bin/env python3
"""Fix texture paths on the flat-layout BEHAVIOR-1K assets (no GPU).

Most BEHAVIOR-1K models store their USD nested as ``<id>/usd/<id>.usd`` with textures at
``<id>/material/``, so the in-USD texture paths ``../material/...`` resolve correctly. But
some models were flattened during intake to ``<id>/<id>.usd`` (USD directly in the model
folder) -- for those, ``../material/`` points one directory too high and Isaac logs
"asset can not be found" for every texture, so the asset renders untextured.

This rewrites ``../material/`` -> ``material/`` on the *flat* USDs only (nested ones are
already correct and are left untouched). Edits USDs in place; re-run after a fresh download.

Usage:
  python scripts/behavior1k_fix_flat_texpaths.py [ASSETS_DIR]            # dry-run
  python scripts/behavior1k_fix_flat_texpaths.py [ASSETS_DIR] --apply    # write changes
"""

import glob
import os
import sys

from pxr import Sdf, Usd


def flat_usds(assets_dir):
    """Primary USDs that sit flat in the model folder (`<id>/<id>.usd`, no `usd/` subdir)."""
    out = []
    for group in sorted(glob.glob(os.path.join(assets_dir, "behavior1k_*"))):
        for model in sorted(glob.glob(os.path.join(group, "*"))):
            if not os.path.isdir(model):
                continue
            name = os.path.basename(model)
            flat = os.path.join(model, f"{name}.usd")
            nested = os.path.join(model, "usd", f"{name}.usd")
            if os.path.exists(flat) and not os.path.exists(nested):
                out.append(flat)
    return out


def fix(usd, apply):
    """Rewrite `../material/` -> `material/` on every asset-valued attribute. Returns count."""
    stage = Usd.Stage.Open(usd)
    if not stage:
        return -1
    changed = 0
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            val = attr.Get()
            if isinstance(val, Sdf.AssetPath) and val.path.startswith("../material/"):
                if apply:
                    attr.Set(Sdf.AssetPath(val.path[len("../"):]))
                changed += 1
    if apply and changed:
        stage.GetRootLayer().Save()
    return changed


def main():
    args = [a for a in sys.argv[1:] if a != "--apply"]
    apply = "--apply" in sys.argv
    assets_dir = os.path.abspath(args[0]) if args else str(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")
    )

    files = flat_usds(assets_dir)
    print(f"[{len(files)} flat USDs under {assets_dir}; mode={'APPLY' if apply else 'dry-run'}]", flush=True)
    total = 0
    for f in files:
        n = fix(f, apply)
        total += max(n, 0)
        print(f"  {os.path.relpath(f, assets_dir):55s}  {n} texture paths", flush=True)
    print(f"[{'rewrote' if apply else 'would rewrite'} {total} texture paths across {len(files)} files]", flush=True)


if __name__ == "__main__":
    main()
