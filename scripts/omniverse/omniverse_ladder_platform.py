#!/usr/bin/env python3
"""Add an explicit box collider for a platform ladder's standing tread (no GPU).

``omniverse_ladder_collision.py`` authors a physics collider per mesh -- SDF by default,
convexDecomposition as its fallback for an open mesh, which AlumStep_D's frame is -- which is
right for the frame but loses the platform: on AlumStep_D the decomposed hulls leave open air
where the rendered tread is, and a G1 teleported onto it falls straight through. A box is exact,
cheap, and legal on a dynamic body (a triangle mesh is not), so the tread gets one of its own.

Writes into the existing ``<name>_collision.usd`` overlay, which ``<name>_collision_rigid.usd``
sublayers, so both the static and the dynamic ladder pick it up.

Usage:
    python scripts/omniverse/omniverse_ladder_platform.py <collision.usd> \
        --centre X Y Z --half-extent HX HY HZ

Coordinates are the asset's own units and frame (print them with ``--probe``).
"""

import argparse
import os
import sys

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ModuleNotFoundError:  # pxr only resolves once Kit has been bootstrapped
    from isaaclab.app import AppLauncher

    _APP = AppLauncher(headless=True).app
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

PLATFORM_PRIM = "PlatformCollider"


def horizontal_bands(stage, bands: int = 60):
    """Vertex-count profile by height, for locating the tread before authoring against it."""
    import numpy as np

    up = UsdGeom.GetStageUpAxis(stage)
    uz = {"Z": 2, "Y": 1, "X": 0}[up]
    ax = [i for i in range(3) if i != uz]
    xf = UsdGeom.XformCache()
    best = None
    for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if p.GetTypeName() != "Mesh":
            continue
        pts = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(), dtype=float)
        m = np.array(xf.GetLocalToWorldTransform(p), dtype=float).T
        w = (m[:3, :3] @ pts.T).T + m[:3, 3]
        if best is None or len(w) > len(best[1]):
            best = (str(p.GetPath()), w)
    if best is None:
        print("no meshes")
        return
    name, w = best
    h = w[:, uz]
    print(f"{name}: {len(w)} verts, height {h.min():.1f}..{h.max():.1f}, up={up}")
    edges = np.linspace(h.min(), h.max(), bands + 1)
    for i in range(bands):
        sel = (h >= edges[i]) & (h < edges[i + 1])
        if sel.sum() < 8:
            continue
        a, b = w[sel][:, ax[0]], w[sel][:, ax[1]]
        print(
            f"  {edges[i]:8.1f}..{edges[i+1]:8.1f}  {sel.sum():6d} verts  "
            f"a=[{a.min():7.1f},{a.max():7.1f}]  b=[{b.min():7.1f},{b.max():7.1f}]"
        )


def author(path, centre, half, visible=False):
    stage = Usd.Stage.Open(path)
    default = stage.GetDefaultPrim()
    if not default:
        sys.exit(f"{path}: no default prim")

    prim_path = default.GetPath().AppendChild(PLATFORM_PRIM)
    if stage.GetPrimAtPath(prim_path):
        stage.RemovePrim(prim_path)

    # A box MESH with a convexHull approximation, not a UsdGeom.Cube: every other collider on
    # these assets is a Mesh + MeshCollisionAPI, and that is the shape the spawner's property
    # pass and PhysX's parser both already handle here. An authored Cube prim appears in the
    # stage and in the bbox listing but never collides.
    hx, hy, hz = half
    cx, cy, cz = centre
    pts = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    faces = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    box = UsdGeom.Mesh.Define(stage, prim_path)
    box.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    box.CreateFaceVertexCountsAttr([4] * len(faces))
    box.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    box.CreateExtentAttr(
        [Gf.Vec3f(cx - hx, cy - hy, cz - hz), Gf.Vec3f(cx + hx, cy + hy, cz + hz)]
    )
    if not visible:
        UsdGeom.Imageable(box).MakeInvisible()

    prim = box.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    # The same ~6 mm contact offset the frame's colliders carry (omniverse_ladder_collision.py).
    # Without it this shape alone keeps PhysX's ~2 cm default skin, so a foot on the tread
    # contacts 2 cm above where the tread renders while the rest of the same body contacts at
    # 6 mm -- one rigid body with two different notions of where its surface is.
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.006)
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    mat_path = default.GetPath().AppendChild("PhysicsMaterials").AppendChild("HighFriction")
    mat = UsdShade.Material.Get(stage, mat_path)
    if mat:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    stage.GetRootLayer().Save()
    print(f"[ok] {os.path.basename(path)}: {PLATFORM_PRIM} at {centre}, half-extent {half}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usd", help="the *_collision.usd overlay to author into")
    ap.add_argument("--probe", action="store_true", help="print the height profile and exit")
    ap.add_argument("--centre", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ap.add_argument("--half-extent", nargs=3, type=float, metavar=("HX", "HY", "HZ"))
    ap.add_argument("--visible", action="store_true", help="leave the box rendered, for visual validation")
    args = ap.parse_args()

    if args.probe:
        horizontal_bands(Usd.Stage.Open(args.usd))
        return
    if not args.centre or not args.half_extent:
        ap.error("--centre and --half-extent are required unless --probe")
    author(args.usd, args.centre, args.half_extent, args.visible)


if __name__ == "__main__":
    main()
