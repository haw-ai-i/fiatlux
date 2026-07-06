#!/usr/bin/env python3
"""Author PhysX collision on EVERY collected ladder/platform (no GPU).

Walks each directory given and authors collision on every ladder found —
handles both the SimReady ladders (meters, /RootNode, entry.usd → _base/_inst
sublayers) and the Warehouse DigitalTwin ladders (centimetres, /World, geometry
inline). For each primary <name>.usd it writes <name>_collision.usd that
sublayers the original and adds CollisionAPI + MeshCollisionAPI(convexDecomposition)
on every mesh, binds a high-friction physics material, and stays static.

Usage: python scripts/omniverse/omniverse_ladder_collision.py <dir> [<dir> ...]
"""
import os
import sys
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

SF, DF, RES = 1.2, 1.0, 0.0
# The collision authoring below is universal (every mesh -> convex collider, any up-axis/unit);
# only this file *selection* is convention-based. These suffixes are the SimReady sublayer parts
# (_base/_inst/_inst_base, skipped so we author on the entry file, not its pieces) and our own
# _collision output (skipped for idempotent re-runs). Harmless on flat/standalone-USD sets
# (nothing matches). Only needs revisiting if another set names real models with these suffixes,
# or splits sublayers under different names (those parts wouldn't be skipped -> double-authored).
PRIMARY_SKIP = ("_base", "_inst", "_inst_base", "_collision")


def primary_usds(root):
    out = []
    for d, _, files in os.walk(root):
        if any(x in d.lower() for x in ("/materials/", "/material/", "/textures/", "/.thumbs/")):
            continue
        for f in files:
            if not f.lower().endswith(".usd"):
                continue
            stem = f[:-4]
            if stem.endswith(PRIMARY_SKIP):
                continue
            out.append(os.path.join(d, f))
    return sorted(out)


def author(src):
    d = os.path.dirname(src)
    stem = os.path.basename(src)[:-4]
    out = os.path.join(d, f"{stem}_collision.usd")
    if os.path.exists(out):
        os.remove(out)

    src_stage = Usd.Stage.Open(src)
    default = src_stage.GetDefaultPrim()
    if not default:
        return out, 0, "no default prim"
    up = UsdGeom.GetStageUpAxis(src_stage)
    mpu = UsdGeom.GetStageMetersPerUnit(src_stage)

    stage = Usd.Stage.CreateNew(out)
    stage.GetRootLayer().subLayerPaths.append(f"./{os.path.basename(src)}")
    UsdGeom.SetStageUpAxis(stage, up)
    UsdGeom.SetStageMetersPerUnit(stage, mpu)

    # high-friction physics material inside the default-prim scope (survives referencing)
    mat_scope = default.GetPath().AppendChild("PhysicsMaterials")
    mat_path = mat_scope.AppendChild("HighFriction")
    UsdGeom.Scope.Define(stage, mat_scope)
    mat = UsdShade.Material.Define(stage, mat_path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(SF)
    pm.CreateDynamicFrictionAttr(DF)
    pm.CreateRestitutionAttr(RES)

    # SimReady assets mark sub-trees instanceable -> meshes are instance proxies that
    # normal traversal skips and you can't author on. De-instance them via overrides.
    for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if p.IsInstance() or p.IsInstanceable():
            stage.OverridePrim(p.GetPath()).SetInstanceable(False)

    nmesh = 0
    for p in stage.Traverse():
        if p.GetTypeName() != "Mesh":
            continue
        over = stage.OverridePrim(p.GetPath())
        UsdPhysics.CollisionAPI.Apply(over)
        UsdPhysics.MeshCollisionAPI.Apply(over).CreateApproximationAttr(
            UsdPhysics.Tokens.convexDecomposition)
        UsdShade.MaterialBindingAPI.Apply(over).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        nmesh += 1

    stage.SetDefaultPrim(stage.GetPrimAtPath(default.GetPath()))
    stage.GetRootLayer().Save()
    return out, nmesh, f"mpu={mpu}"


def main():
    roots = sys.argv[1:]
    if not roots:
        print(__doc__)
        sys.exit(1)
    total = ok = 0
    for root in roots:
        for src in primary_usds(root):
            total += 1
            try:
                out, nm, info = author(src)
                status = "ok" if nm else "SKIP(no mesh)"
                if nm:
                    ok += 1
                print(f"  [{status}] {os.path.relpath(src, root)}  ({nm} meshes, {info})")
            except Exception as e:
                print(f"  [FAIL] {os.path.relpath(src, root)}: {e}")
    print(f"\nAuthored collision on {ok}/{total} ladder USDs.")


if __name__ == "__main__":
    main()
