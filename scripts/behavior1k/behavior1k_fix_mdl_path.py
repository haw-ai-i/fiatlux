#!/usr/bin/env python3
"""Bake the OmniGibson vray material path into the BEHAVIOR-1K asset USDs (no GPU).

Every behavior1k object binds OmniGibson's ``OmniGibsonVRayMtl`` material via
``info:mdl:sourceAsset = @omnigibson_vray_mtl.mdl@`` -- a bare module name that only
resolves when OmniGibson's materials folder is on the MDL search path. This rewrites it to
a RELATIVE path pointing at the bundled ``assets/behavior1k_materials/``, so the material
resolves on its own -- no ``MDL_USER_PATH`` / launcher wiring, works in any tool that opens
the asset. The mdl's own ``vray_maps``/``vray_materials`` imports resolve from that folder.

Idempotent; edits USDs in place. Run ``behavior1k_fix_flat_texpaths.py`` for the companion
texture-path fix. Re-run after a fresh download.

Usage:
  python scripts/behavior1k/behavior1k_fix_mdl_path.py [ASSETS_DIR]            # dry-run
  python scripts/behavior1k/behavior1k_fix_mdl_path.py [ASSETS_DIR] --apply    # write changes
"""

import glob
import os
import sys

from pxr import Sdf, Usd, UsdShade

MDL_NAME = "omnigibson_vray_mtl.mdl"


def asset_usds(assets_dir):
    """One primary object USD per behavior1k model folder (skip collision + the materials dir)."""
    out = []
    for group in sorted(glob.glob(os.path.join(assets_dir, "behavior1k_*"))):
        if os.path.basename(group) == "behavior1k_materials":
            continue
        for model in sorted(glob.glob(os.path.join(group, "*"))):
            if not os.path.isdir(model):
                continue
            name = os.path.basename(model)
            for cand in (os.path.join(model, "usd", f"{name}.usd"), os.path.join(model, f"{name}.usd")):
                if os.path.exists(cand):
                    out.append(cand)
                    break
    return out


def fix(usd, materials_dir, apply):
    """Rewrite each OmniGibsonVRayMtl shader's sourceAsset to a relative mdl path. Returns count."""
    stage = Usd.Stage.Open(usd)
    if not stage:
        return -1
    rel = os.path.relpath(os.path.join(materials_dir, MDL_NAME), os.path.dirname(os.path.abspath(usd)))
    changed = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        sh = UsdShade.Shader(prim)
        cur = sh.GetSourceAsset("mdl")
        if cur is None:
            continue
        cur = str(cur).strip("@")
        if cur.endswith(MDL_NAME) and cur != rel:
            if apply:
                sub = sh.GetSourceAssetSubIdentifier("mdl")
                sh.SetSourceAsset(Sdf.AssetPath(rel), "mdl")
                if sub:
                    sh.SetSourceAssetSubIdentifier(sub, "mdl")
            changed += 1
    if apply and changed:
        stage.GetRootLayer().Save()
    return changed


def main():
    args = [a for a in sys.argv[1:] if a != "--apply"]
    apply = "--apply" in sys.argv
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assets_dir = os.path.abspath(args[0]) if args else os.path.join(repo_root, "assets")
    materials_dir = os.path.join(assets_dir, "behavior1k_materials")
    if not os.path.isdir(materials_dir):
        print(f"[error] {materials_dir} not found -- bundle the 3 OmniGibson mdls there first")
        sys.exit(1)

    files = asset_usds(assets_dir)
    print(f"[{len(files)} behavior1k object USDs; mode={'APPLY' if apply else 'dry-run'}]", flush=True)
    total = touched = 0
    for f in files:
        n = fix(f, materials_dir, apply)
        if n > 0:
            touched += 1
            total += n
    print(
        f"[{'rewrote' if apply else 'would rewrite'} {total} sourceAsset(s) across {touched}/{len(files)} files]",
        flush=True,
    )


if __name__ == "__main__":
    main()
