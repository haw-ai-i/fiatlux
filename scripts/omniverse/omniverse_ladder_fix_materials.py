#!/usr/bin/env python3
"""Fix the ladders' material references so they resolve + render cleanly in Isaac Sim.

Three fixes per ladder USD:
  1. Collapse the deep `(../)+Materials/` chain (the pack's original layout) to the
     local `../Materials/` -- on the mdl module path AND every texture-input asset
     attribute (on Shader and Material prims). Requires the materials extracted next
     to the ladders first (see omniverse_pack_extract.py).
  2. SimReady core materials referenced by a broken relative path (e.g.
     `../../../materials/SimPBR.mdl`) -> the bare name (`SimPBR.mdl`), which Isaac
     resolves from its own MDL library.
  3. Deactivate the SimReady `/Tagging` prim -- a thumbnail-rig payload that isn't
     shipped (harmless warnings only).

Only the primary <name>.usd files are edited; the _collision/_collision_rigid
overlays sublayer them and inherit the fix.

Usage: python scripts/omniverse/omniverse_ladder_fix_materials.py [ROOT]
       (default ROOT = ~/omniverse_packs/keep/Warehouse_ladders; expects the
        flattened layout ROOT/<design>/<file>.usd with Materials at ROOT/Materials)
"""
import glob
import os
import re
import sys

from pxr import Sdf, Usd, UsdShade

ROOT = (os.path.abspath(sys.argv[1]) if len(sys.argv) > 1
        else os.path.expanduser("~/omniverse_packs/keep/Warehouse_ladders"))


COLLAPSE = r'(\.\./)+Materials/'                         # deep ../..-chain to the local Materials/
CORE_MDLS = {"SimPBR.mdl", "OmniPBR.mdl"}               # ship with Isaac; reference by bare name


def fix(usd):
    s = Usd.Stage.Open(usd)
    n = 0
    for p in Usd.PrimRange(s.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        # 1) the mdl module path on shaders (special-cased to preserve the subIdentifier)
        if p.GetTypeName() == "Shader":
            sh = UsdShade.Shader(p)
            a = sh.GetSourceAsset("mdl")
            if a:
                old = str(a).strip("@")
                new = re.sub(COLLAPSE, '../Materials/', old)
                # SimReady core materials referenced by a broken relative path
                # (e.g. ../../../materials/SimPBR.mdl) -> bare name; Isaac resolves it
                base = new.rsplit("/", 1)[-1]
                if base in CORE_MDLS and new != base:
                    new = base
                if new != old:
                    sub = sh.GetSourceAssetSubIdentifier("mdl")
                    sh.SetSourceAsset(Sdf.AssetPath(new), "mdl")
                    if sub:
                        sh.SetSourceAssetSubIdentifier(sub, "mdl")
                    n += 1
        # 2) ANY asset-valued attribute (texture overrides on Shader OR Material prims,
        #    e.g. inputs:diffuse_texture authored directly on /World/Looks/<Material>)
        for a in p.GetAttributes():
            if a.GetName() == "info:mdl:sourceAsset":
                continue                                  # handled above
            v = a.Get()
            if isinstance(v, Sdf.AssetPath) and v.path:
                new = re.sub(COLLAPSE, '../Materials/', v.path)
                if new != v.path:
                    a.Set(Sdf.AssetPath(new))
                    n += 1
    # 3) drop the SimReady thumbnail-rig payload (dead ref -> harmless ThumbRig warnings)
    tag = s.GetPrimAtPath("/Tagging")
    if tag and tag.IsValid() and tag.IsActive():
        tag.SetActive(False)
        n += 1
    if n:
        s.GetRootLayer().Save()
    return n


def main():
    usds = [f for f in glob.glob(os.path.join(ROOT, "*", "*.usd"))
            if not f.endswith(("_collision.usd", "_collision_rigid.usd"))]
    total = files = 0
    for f in sorted(usds):
        try:
            k = fix(f)
            if k:
                files += 1
                total += k
        except Exception as e:
            print(f"  [FAIL] {os.path.basename(f)}: {e}")
    print(f"fixed {total} material paths across {files}/{len(usds)} Warehouse ladder USDs")


if __name__ == "__main__":
    main()
