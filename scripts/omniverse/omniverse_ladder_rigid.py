#!/usr/bin/env python3
"""Author carry-able RIGID-BODY variants of the ladders (level-2 physics), no GPU.

For each `<name>_collision.usd` (the static collider), write a tiny
`<name>_collision_rigid.usd` that sublayers it and adds `RigidBodyAPI` + a size-based mass
on the default prim — turning the fixed climb target into a movable / carry-able
dynamic object. The geometry is NOT duplicated (overlay sublayers the collider),
so each file is only a few KB.

Load `_collision.usd` for a fixed ladder to climb; load `_collision_rigid.usd` for
a movable ladder to carry/knock over.

Usage: python scripts/omniverse/omniverse_ladder_rigid.py <dir> [<dir> ...]
"""
import os
import sys
from pxr import Usd, UsdGeom, UsdPhysics


def mass_for(height_m):
    """Rough, realistic-ish ladder mass from height (aluminium/fiberglass, hollow)."""
    return round(max(2.0, min(40.0, 2.0 + 3.0 * height_m)), 1)


def author(collision_usd):
    d = os.path.dirname(collision_usd)
    stem = os.path.basename(collision_usd)[:-len("_collision.usd")]
    out = os.path.join(d, f"{stem}_collision_rigid.usd")
    if os.path.exists(out):
        os.remove(out)

    src = Usd.Stage.Open(collision_usd)
    default = src.GetDefaultPrim()
    if not default:
        return out, None
    up = UsdGeom.GetStageUpAxis(src)
    mpu = UsdGeom.GetStageMetersPerUnit(src)
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    # NOTE: not universal — "height" is hardcoded to the Z extent (GetSize()[2]), which
    # assumes a Z-up asset (true for these ladders). A Y-up asset would use its depth here
    # and get the wrong mass; make this axis-aware (pick the component matching `up`) to reuse.
    dz = bb.ComputeWorldBound(src.GetPseudoRoot()).ComputeAlignedRange().GetSize()[2] * mpu
    mass = mass_for(dz)  # ladder-specific heuristic (2 + 3*h, clamped 2-40 kg), not general-purpose

    stage = Usd.Stage.CreateNew(out)
    stage.GetRootLayer().subLayerPaths.append(f"./{os.path.basename(collision_usd)}")
    UsdGeom.SetStageUpAxis(stage, up)
    UsdGeom.SetStageMetersPerUnit(stage, mpu)
    over = stage.OverridePrim(default.GetPath())
    UsdPhysics.RigidBodyAPI.Apply(over)
    UsdPhysics.MassAPI.Apply(over).CreateMassAttr(mass)
    # CCD (nice for thin ladders) is enabled at the PhysX scene level when used,
    # so no PhysxSchema dependency is baked into the asset.
    stage.SetDefaultPrim(stage.GetPrimAtPath(default.GetPath()))
    stage.GetRootLayer().Save()
    return out, mass


def find_collision(roots):
    out = []
    for root in roots:
        for d, _, files in os.walk(root):
            for f in files:
                if f.endswith("_collision.usd"):
                    out.append(os.path.join(d, f))
    return sorted(out)


def main():
    roots = sys.argv[1:]
    if not roots:
        print(__doc__)
        sys.exit(1)
    n = 0
    for c in find_collision(roots):
        out, mass = author(c)
        if mass is not None:
            n += 1
            print(f"  [ok] {os.path.basename(out)}  (mass {mass} kg)")
        else:
            print(f"  [SKIP] {os.path.basename(out)} (no default prim)")
    print(f"\nAuthored {n} rigid-body (carry-able) ladder variants.")


if __name__ == "__main__":
    main()
