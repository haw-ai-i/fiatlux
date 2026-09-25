#!/usr/bin/env python3
"""Author spawn-ready Z-up/metre variants of the Omniverse LightBulb halves, no GPU.

The source asset (`LightBulb.usda` and Yujin's `_collision` / `_bulb` / `_socket` layers)
is authored **Y-up in centimetres**; the Fiatlux scene is Z-up in metres. This writes two
thin wrapper layers that reference the halves and carry the frame conversion, so the env
cfgs can spawn them with an identity `init_state.rot` and all the existing preset
quaternion math (ceiling flip, wall mount) keeps working unchanged:

    LightBulb_bulb_z_rigid.usda    graspable bulb, RigidBodyAPI + MassAPI
    LightBulb_socket_z_static.usda fixture, colliders only (the cfg decides the body type)

Three things this deliberately does differently from ``omniverse_ladder_rigid.py``:

1. It is axis-aware. That script takes the Z extent as "height", which is wrong for a
   Y-up asset (its own docstring flags this); masses here are explicit, not size-derived.
2. The conversion (rotX+90 for Y-up -> Z-up, x0.01 for cm -> m) goes on a CHILD Xform,
   never the root: Isaac Lab's spawner calls ``create_prim()``, which overwrites
   translate/orient/scale on the spawned prim itself, so a transform baked on the root
   would be silently discarded.
3. It deactivates the asset's live runtime extras -- an OmniGraph that ticks every frame,
   a Camera, and the bulb's own SphereLight -- which would otherwise be duplicated per env.

Needs only ``pxr`` (no Isaac Sim). The project venv does not ship USD standalone; run with
any python that has it installed, e.g. one with the ``usd-core`` package.

Usage: python scripts/omniverse/omniverse_bulb_rigid.py [<omniverse_bulb dir>]
"""

import os
import sys

from pxr import Gf, Usd, UsdGeom, UsdPhysics

# Y-up (cm) -> Z-up (m). rotX(+90 deg) maps asset +Y onto world +Z.
_ROT_X_90 = Gf.Quatd(0.7071067811865476, Gf.Vec3d(0.7071067811865476, 0.0, 0.0))
_CM_TO_M = 0.01

# Explicit, documented masses -- NOT a size heuristic.
#   bulb   : a real A19 incandescent/LED is 30-45 g.
#   socket : Yujin's whole-assembly figure in LightBulb_collision_rigid.usda.
BULB_MASS_KG = 0.035
SOCKET_MASS_KG = 0.30

# Prims that must not be cloned per env: an OmniGraph that executes every tick, a spare
# camera, and the bulb's own SphereLight (the scene owns lighting).
_RUNTIME_EXTRAS = ("ActionGraph", "Camera")
_BULB_LIGHT = "Geom/BulbGrp/BulbLight"

# Expected extents after conversion (metres, Z-up), from the source geometry. The bulb's
# screw cap bottom is the mating point; the socket's origin is its floor-contact plane.
_EXPECT = {
    "bulb": {"z_min": 0.036259, "z_max": 0.192674, "radius": 0.039561},
    "socket": {"z_min": 0.0, "z_max": 0.070485},
}
_TOL = 1e-4  # m


def _author(src_dir, half, source_layer, out_name, mass, rigid):
    """Write one wrapper layer referencing ``source_layer``'s ``/World``."""
    src_path = os.path.join(src_dir, source_layer)
    if not os.path.isfile(src_path):
        raise FileNotFoundError(src_path)
    out = os.path.join(src_dir, out_name)
    if os.path.exists(out):
        os.remove(out)

    root_name = "Bulb" if half == "bulb" else "Socket"
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)

    # The conversion goes on its own clean Xform, NOT on the prim carrying the reference:
    # the referenced /World arrives with its own xformOpOrder, and adding ops there would
    # compose against the asset's authored transform instead of replacing the frame.
    convert = UsdGeom.Xform.Define(stage, f"/{root_name}/Convert")
    convert.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_ROT_X_90)
    convert.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(_CM_TO_M, _CM_TO_M, _CM_TO_M))

    inner_path = f"/{root_name}/Convert/LightBulb"
    inner = UsdGeom.Xform.Define(stage, inner_path)
    inner.GetPrim().GetReferences().AddReference(f"./{source_layer}", "/World")

    for extra in _RUNTIME_EXTRAS:
        p = stage.GetPrimAtPath(f"{inner_path}/{extra}")
        if p and p.IsValid():
            p.SetActive(False)
    if half == "bulb":
        p = stage.GetPrimAtPath(f"{inner_path}/{_BULB_LIGHT}")
        if p and p.IsValid():
            p.SetActive(False)

    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return out


def _verify(out, half):
    """Reopen the authored layer and check the converted extents against the source."""
    stage = Usd.Stage.Open(out)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    rng = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    if rng.IsEmpty():
        raise AssertionError(f"{out}: empty bbox -- the reference did not compose")
    mn, mx = rng.GetMin(), rng.GetMax()
    want = _EXPECT[half]
    problems = []
    if abs(mn[2] - want["z_min"]) > _TOL:
        problems.append(f"z_min {mn[2]:.6f} != {want['z_min']:.6f}")
    if abs(mx[2] - want["z_max"]) > _TOL:
        problems.append(f"z_max {mx[2]:.6f} != {want['z_max']:.6f}")
    if "radius" in want and abs(max(abs(mn[0]), abs(mx[0])) - want["radius"]) > _TOL:
        problems.append(f"radius {max(abs(mn[0]), abs(mx[0])):.6f} != {want['radius']:.6f}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        problems.append("up axis is not Z")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        problems.append("metersPerUnit is not 1")
    if problems:
        raise AssertionError(f"{os.path.basename(out)}: " + "; ".join(problems))
    return mn, mx


def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "omniverse_bulb"
    )
    if not os.path.isdir(src_dir):
        print(f"not a directory: {src_dir}\n")
        print(__doc__)
        sys.exit(1)

    jobs = [
        ("bulb", "LightBulb_bulb.usda", "LightBulb_bulb_z_rigid.usda", BULB_MASS_KG, True),
        ("socket", "LightBulb_socket.usda", "LightBulb_socket_z_static.usda", SOCKET_MASS_KG, False),
    ]
    for half, source_layer, out_name, mass, rigid in jobs:
        out = _author(src_dir, half, source_layer, out_name, mass, rigid)
        mn, mx = _verify(out, half)
        body = f"rigid, {mass} kg" if rigid else "colliders only (body type set by the cfg)"
        print(f"  [ok] {os.path.basename(out)}  ({body})")
        print(f"       z=[{mn[2]:+.6f}, {mx[2]:+.6f}] m  x=[{mn[0]:+.6f}, {mx[0]:+.6f}] m")

    print("\nAuthored 2 spawn-ready (Z-up, metre) LightBulb variants.")


if __name__ == "__main__":
    main()
