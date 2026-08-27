#!/usr/bin/env python3
"""Author PhysX collision on EVERY collected ladder/platform (no GPU).

Walks each directory given and authors collision on every ladder found —
handles both the SimReady ladders (meters, /RootNode, entry.usd → _base/_inst
sublayers) and the Warehouse DigitalTwin ladders (centimetres, /World, geometry
inline). For each primary <name>.usd it writes <name>_collision.usd that
sublayers the original and adds CollisionAPI + MeshCollisionAPI (SDF exact-surface by default --
see _approximation_for) + a ~6 mm contact offset on every mesh, binds a high-friction physics
material, and stays static.

Usage: python scripts/omniverse/omniverse_ladder_collision.py <dir> [<dir> ...]
"""

import os
import sys

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

SF, DF, RES = 1.2, 1.0, 0.0
# Convex-decomposition tightness. Left at PhysX defaults the decomposition is a handful of fat
# hulls that bridge the open gaps between rungs and bulge proud of the surface (so a hand/foot is
# pushed off before it visually touches, and the rungs read as a solid ramp). More hulls + richer
# hull vertices + a finer voxelisation + shrink-wrap make the pieces separate the rungs and hug the
# rails, so the collider matches the visible ladder. convexDecomposition (not an exact triangle
# mesh) because these same files are also spawned DYNAMIC (the Replace / Carry manipuland), and
# PhysX forbids a triangle-mesh collider on a moving body.
CD_MAX_HULLS, CD_HULL_VERTS, CD_VOXEL_RES = 64, 64, 500_000
# SDF (signed distance field) grid resolution along the longest bbox axis. SDF stores the exact
# surface, so it keeps CONCAVE features open -- notably the fiberglass C-channel rail groove, which
# convexDecomposition fills by bridging convex pieces across the opening. res 256 ~ 7 mm cells over a
# 1.75 m ladder; a very thin/deep groove may want more.
SDF_RES = 256


def _approximation_for(src):
    """Which collision approximation to author for this design. SDF for every design (exact surface,
    concavities preserved). SDF works on dynamic bodies -- unlike an exact triangle mesh, which PhysX
    forbids on a moving body, and these files are also spawned dynamic (Replace / Carry manipuland).

    To go back to *selective* SDF (only the grooved fiberglass designs, cheaper for RL at scale),
    return ``"sdf" if ("frp" in src.lower() or "fiberglass" in src.lower()) else "convex"``.
    """
    return "sdf"


def _has_inverted_mesh(stage):
    """True if ANY single mesh is wound inside-out (world-space signed volume clearly negative).

    Winding matters only for SDF, whose inside/outside is defined by face orientation: an inverted
    mesh gives an inside-out SDF collider, and PhysX logs a "negative mass" error while attaching that
    *shape* (mass is derived per shape). So the check is **per mesh** -- summing volumes across a
    design would let one small inverted mesh hide behind larger correct ones and still error.

    - The WORLD transform is applied (some assets carry a mirror / negative-determinant transform that
      flips the sign; a local-space shortcut misidentifies the design).
    - "Clearly negative" is relative to the design's largest mesh, so flat decals (~0 volume) and unit
      scale (cm vs metre) don't matter.
    Points are transformed and volumes summed with numpy (vectorised matmul) so it stays fast."""
    import numpy as np

    vols = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        cnts = mesh.GetFaceVertexCountsAttr().Get()
        if not pts or not idx or not cnts:
            continue
        m = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
        p = (np.column_stack([np.asarray(pts, dtype=np.float64), np.ones(len(pts))]) @ m)[:, :3]
        idx = np.asarray(idx, dtype=np.int64)
        cnts = np.asarray(cnts, dtype=np.int64)
        if np.all(cnts == 3):  # already triangulated -> fully vectorised
            tri = idx.reshape(-1, 3)
            a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
            v = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0
        else:  # mixed polygons -> fan-triangulate per face (rare, slower)
            v = 0.0
            i = 0
            for c in cnts:
                for k in range(1, c - 1):
                    v += float(p[idx[i]] @ np.cross(p[idx[i + k]], p[idx[i + k + 1]])) / 6.0
                i += c
        vols.append(v)
    if not vols:
        return False
    max_abs = max(abs(v) for v in vols)
    return any(v < 0 and abs(v) > 0.01 * max_abs for v in vols)
# The collision authoring below is universal (every mesh -> convex collider, any up-axis/unit);
# only this file *selection* is convention-based. These suffixes are the SimReady sublayer parts
# (_base/_inst/_inst_base, skipped so we author on the entry file, not its pieces) and our own
# outputs -- both ``_collision`` and the ``_collision_rigid`` overlay ``omniverse_ladder_rigid.py``
# writes -- skipped for idempotent re-runs. Without ``_collision_rigid`` a re-run over a populated
# dir would treat each rigid overlay as a primary and emit ``_collision_rigid_collision.usd`` junk
# (its stem ends ``_rigid``, so the ``_collision`` entry alone does not catch it). Harmless on
# flat/standalone-USD sets (nothing matches). Only needs revisiting if another set names real
# models with these suffixes, or splits sublayers under different names (not skipped -> re-authored).
PRIMARY_SKIP = ("_base", "_inst", "_inst_base", "_collision", "_collision_rigid")


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

    approx = _approximation_for(src)
    if approx == "sdf" and _has_inverted_mesh(stage):
        approx = "convex"  # an inverted mesh -> SDF would be inside-out; convex ignores winding
    nmesh = 0
    for p in stage.Traverse():
        if p.GetTypeName() != "Mesh":
            continue
        over = stage.OverridePrim(p.GetPath())
        UsdPhysics.CollisionAPI.Apply(over)
        mca = UsdPhysics.MeshCollisionAPI.Apply(over)
        # Collision params are authored as raw attributes + an applied-schema token (not via
        # PhysxSchema) so this script keeps running in the plain-pxr env, which has no PhysX schema
        # bindings. PhysX reads them once the API is listed in the prim's schemas.
        if approx == "sdf":
            mca.CreateApproximationAttr("sdf")  # exact surface -> concave rail grooves stay open
            over.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
            over.CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(SDF_RES)
        else:
            mca.CreateApproximationAttr(UsdPhysics.Tokens.convexDecomposition)  # tightened, but fills concavities
            over.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
            over.CreateAttribute("physxConvexDecompositionCollision:maxConvexHulls", Sdf.ValueTypeNames.UInt).Set(
                CD_MAX_HULLS
            )
            over.CreateAttribute("physxConvexDecompositionCollision:hullVertexLimit", Sdf.ValueTypeNames.UInt).Set(
                CD_HULL_VERTS
            )
            over.CreateAttribute("physxConvexDecompositionCollision:voxelResolution", Sdf.ValueTypeNames.UInt).Set(
                CD_VOXEL_RES
            )
            over.CreateAttribute("physxConvexDecompositionCollision:shrinkWrap", Sdf.ValueTypeNames.Bool).Set(True)
        # Contact offset ~6 mm in world space (vs PhysX's ~2 cm default skin) so a hand contacts the
        # ladder at its surface, not a couple centimetres out. The value is a plain world-metre
        # distance -- NOT scaled by the asset's metersPerUnit or the spawn scale (verified: a 20 mm
        # value rests the ladder ~20 mm off the floor for both cm- and metre-authored designs), so
        # the same 0.006 is correct for every design. rest offset 0 keeps resting contact flush.
        over.AddAppliedSchema("PhysxCollisionAPI")
        over.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.006)
        over.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        UsdShade.MaterialBindingAPI.Apply(over).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        nmesh += 1

    stage.SetDefaultPrim(stage.GetPrimAtPath(default.GetPath()))
    stage.GetRootLayer().Save()
    return out, nmesh, f"{approx} mpu={mpu}"


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
