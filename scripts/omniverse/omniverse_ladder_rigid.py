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

import argparse
import os

import numpy as np

from pxr import Gf, Usd, UsdGeom, UsdPhysics

# The collision helper omniverse_ladder_platform.py authors into the _collision overlay. It is a
# solid box standing in for a tread the render mesh has but the decomposed collider does not, so
# it is collision-only: counting it as material would put ~30% of a phantom solid at platform
# height. Excluded from the mass model below.
PLATFORM_PRIM = "PlatformCollider"


def mass_for(height_m):
    """Rough, realistic-ish ladder mass from height (aluminium/fiberglass, hollow).

    A shape-blind fallback for the 90-odd gallery designs nobody has looked up. Any design the
    benchmark actually scores should pass ``--mass`` with a real catalogue figure instead: for
    AlumStep_D01 this heuristic gives 7.6 kg against a measured 10.9 kg for the ladder it models
    (see ``LADDER_MASS_KG`` in ``scene_cfg.py``), so it is roughly 30% light.
    """
    return round(max(2.0, min(40.0, 2.0 + 3.0 * height_m)), 1)


def _triangles(stage, mpu):
    """Every collision triangle in world space, in METRES, minus the platform helper box."""
    xf = UsdGeom.XformCache()
    tris = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() != "Mesh" or prim.GetName() == PLATFORM_PRIM:
            continue
        mesh = UsdGeom.Mesh(prim)
        pts, idx, cnts = (
            mesh.GetPointsAttr().Get(),
            mesh.GetFaceVertexIndicesAttr().Get(),
            mesh.GetFaceVertexCountsAttr().Get(),
        )
        if not pts or not idx or not cnts:
            continue
        m = np.asarray(xf.GetLocalToWorldTransform(prim), dtype=np.float64)
        p = (np.column_stack([np.asarray(pts, dtype=np.float64), np.ones(len(pts))]) @ m)[:, :3] * mpu
        idx = np.asarray(idx, dtype=np.int64)
        i = 0
        for c in np.asarray(cnts, dtype=np.int64):
            f = idx[i : i + c]
            for k in range(1, c - 1):
                tris.append((p[f[0]], p[f[k]], p[f[k + 1]]))
            i += c
    return np.array(tris) if tris else None


def shell_mass_properties(stage, mpu, mass):
    """Centre of mass and inertia for a THIN SHELL of uniform areal density.

    PhysX derives both from the collider's enclosed VOLUME at uniform density, which models a
    ladder as a solid block. A ladder is not one: it is thin-walled extruded tube and folded
    sheet, so its mass follows SURFACE AREA, not volume. The two disagree about where the mass
    sits and, more sharply, about how far it sits from the axis -- mass-normalised, the shell's
    principal moments run ~20% higher than the solid's, because a shell puts its material at the
    outside. That difference is exactly what governs how the ladder responds to being shoved.

    Returns ``(com, principal_moments, principal_axes_quat)`` with the quaternion in USD's
    ``(w, x, y, z)`` order, or ``None`` if the design has no usable geometry.
    """
    tris = _triangles(stage, mpu)
    if tris is None:
        return None
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if total <= 0.0:
        return None
    com = (area[:, None] * ((a + b + c) / 3.0)).sum(0) / total

    # Each triangle is a uniform lamina: its second-moment about its own centroid plus the
    # parallel-axis shift to the body CoM.
    inertia = np.zeros((3, 3))
    for (p0, p1, p2), dm in zip(tris, mass * area / total):
        g = (p0 + p1 + p2) / 3.0
        local = np.array([p0, p1, p2]) - g
        d = g - com
        cov = (local.T @ local) / 12.0 + np.outer(d, d)
        inertia += dm * (np.trace(cov) * np.eye(3) - cov)

    moments, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:  # eigh may hand back a reflection; PhysX wants a rotation
        axes[:, 0] *= -1
    return com, moments, _quat_from_matrix(axes)


def _quat_from_matrix(m):
    """Rotation matrix -> (w, x, y, z), via the numerically stable branch-on-trace form."""
    t = np.trace(m)
    if t > 0.0:
        sq = np.sqrt(t + 1.0) * 2.0
        return (0.25 * sq, (m[2, 1] - m[1, 2]) / sq, (m[0, 2] - m[2, 0]) / sq, (m[1, 0] - m[0, 1]) / sq)
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        sq = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return ((m[2, 1] - m[1, 2]) / sq, 0.25 * sq, (m[0, 1] + m[1, 0]) / sq, (m[0, 2] + m[2, 0]) / sq)
    if i == 1:
        sq = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return ((m[0, 2] - m[2, 0]) / sq, (m[0, 1] + m[1, 0]) / sq, 0.25 * sq, (m[1, 2] + m[2, 1]) / sq)
    sq = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return ((m[1, 0] - m[0, 1]) / sq, (m[0, 2] + m[2, 0]) / sq, (m[1, 2] + m[2, 1]) / sq, 0.25 * sq)


def author(collision_usd, mass_override=None):
    d = os.path.dirname(collision_usd)
    stem = os.path.basename(collision_usd)[: -len("_collision.usd")]
    out = os.path.join(d, f"{stem}_collision_rigid.usd")
    if os.path.exists(out):
        os.remove(out)

    src = Usd.Stage.Open(collision_usd)
    default = src.GetDefaultPrim()
    if not default:
        return out, None, None
    up = UsdGeom.GetStageUpAxis(src)
    mpu = UsdGeom.GetStageMetersPerUnit(src)
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    # NOTE: not universal — "height" is hardcoded to the Z extent (GetSize()[2]), which
    # assumes a Z-up asset (true for these ladders). A Y-up asset would use its depth here
    # and get the wrong mass; make this axis-aware (pick the component matching `up`) to reuse.
    dz = bb.ComputeWorldBound(src.GetPseudoRoot()).ComputeAlignedRange().GetSize()[2] * mpu
    # ladder-specific heuristic (2 + 3*h, clamped 2-40 kg), not general-purpose; --mass wins.
    mass = mass_override if mass_override is not None else mass_for(dz)
    props = shell_mass_properties(src, mpu, mass)

    stage = Usd.Stage.CreateNew(out)
    stage.GetRootLayer().subLayerPaths.append(f"./{os.path.basename(collision_usd)}")
    UsdGeom.SetStageUpAxis(stage, up)
    UsdGeom.SetStageMetersPerUnit(stage, mpu)
    over = stage.OverridePrim(default.GetPath())
    UsdPhysics.RigidBodyAPI.Apply(over)
    mass_api = UsdPhysics.MassAPI.Apply(over)
    mass_api.CreateMassAttr(mass)
    # Author the centre of mass and the inertia tensor rather than letting PhysX derive them.
    # Derived, they come off the collider's enclosed volume at uniform density -- a ladder as a
    # solid block, and with the collision-only PlatformCollider box counted as material. Authored,
    # they come off the thin-shell model, which is what the object actually is.
    #
    # NOTE: PhysX does NOT rescale an authored inertia when something later overrides the mass, so
    # a code-side mass override (``LADDER_MASS_KG``) must match the mass authored here or the body
    # ends up with one object's mass and another's inertia.
    if props is not None:
        com, moments, quat = props
        # ``physics:centerOfMass`` is a POSITION, so it is expressed in the stage's own linear
        # units -- centimetres for these cm-authored assets -- while the shell model works in
        # metres. Verified by reading the value back through PhysX: authored in metres it arrived
        # 100x short (0.0086 m against the 0.8626 m intended). The inertia needs no such
        # conversion; it is consumed as kg*m^2 (also verified against PhysX's reported tensor).
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) / mpu for v in com)))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(v) for v in moments)))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(float(quat[0]), Gf.Vec3f(*(float(v) for v in quat[1:]))))
    # CCD (nice for thin ladders) is enabled at the PhysX scene level when used,
    # so no PhysxSchema dependency is baked into the asset.
    stage.SetDefaultPrim(stage.GetPrimAtPath(default.GetPath()))
    stage.GetRootLayer().Save()
    return out, mass, props


def find_collision(roots):
    """Every ``_collision.usd`` under the given dirs; a path to one such file is taken as-is.

    Accepting a single file matters for ``--mass``: the designs in one folder are a size ladder
    (AlumStep_D01..D06 climb from 1.86 m to 2.9 m), so a catalogue mass looked up for one of them
    is wrong for its siblings and must be appliable to that file alone.
    """
    out = []
    for root in roots:
        if os.path.isfile(root):
            if root.endswith("_collision.usd"):
                out.append(root)
            continue
        for d, _, files in os.walk(root):
            for f in files:
                if f.endswith("_collision.usd"):
                    out.append(os.path.join(d, f))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="dirs to walk for <name>_collision.usd, or such a file itself")
    ap.add_argument(
        "--mass",
        type=float,
        help="kg, overriding the height heuristic. Use a real catalogue figure for any design the "
        "benchmark scores; the heuristic is only a fallback for the gallery.",
    )
    args = ap.parse_args()
    n = 0
    for c in find_collision(args.dirs):
        out, mass, props = author(c, args.mass)
        if mass is None:
            print(f"  [SKIP] {os.path.basename(out)} (no default prim)")
            continue
        n += 1
        if props is None:
            print(f"  [ok] {os.path.basename(out)}  (mass {mass} kg, inertia DERIVED -- no geometry)")
        else:
            com, moments, _ = props
            print(
                f"  [ok] {os.path.basename(out)}  (mass {mass} kg, "
                f"com z={com[2]:.4f} m, principal I={[round(float(v), 4) for v in moments]})"
            )
    print(f"\nAuthored {n} rigid-body (carry-able) ladder variants.")


if __name__ == "__main__":
    main()
