#!/usr/bin/env python3
"""Check a generated USD without a GPU and without Isaac Sim.

Isaac Sim cannot run on this box (an A100 has no RT cores), so the only way to
know the stage-4 output is sound before shipping it to an RTX machine is to
re-derive its invariants from the layer itself.  Everything here is pure ``pxr``.

The highest-value check is ``joint frames coincide`` (see ``check_joints``): it
independently recomposes each joint's two body frames through ``XformCache`` and
asserts they land on the same world pose.  That single assertion catches the
whole absolute-vs-relative coordinate class of bugs -- which is exactly the trap
SimArt's own URDF falls into for parts nested more than one level deep.

Usage:
    verify_usd.py <stage.usd> [--expect-links N] [--expect-joints M]
                              [--size-range MIN MAX] [--json <path>]
Exit status is non-zero if any check FAILs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail=""):
        self.rows.append((status, name, detail))
        return status

    def ok(self, cond, name, detail="", warn_only=False):
        if cond:
            return self.add(PASS, name, detail)
        return self.add(WARN if warn_only else FAIL, name, detail)

    @property
    def failed(self):
        return [r for r in self.rows if r[0] == FAIL]

    def report(self):
        w = max(len(n) for _, n, _ in self.rows) if self.rows else 10
        for status, name, detail in self.rows:
            mark = {PASS: "ok  ", FAIL: "FAIL", WARN: "warn"}[status]
            print(f"  [{mark}] {name:<{w}}  {detail}")
        n_fail = len(self.failed)
        n_warn = sum(1 for r in self.rows if r[0] == WARN)
        print(f"\n  {len(self.rows) - n_fail - n_warn} passed, {n_warn} warnings, {n_fail} failed")
        return n_fail


def _finite(vals):
    """True if every component of every element is finite."""
    for v in vals:
        try:
            comps = list(v)
        except TypeError:
            comps = [v]
        for c in comps:
            if not math.isfinite(float(c)):
                return False
    return True


def check_stage(stage, c: Checks):
    root = stage.GetDefaultPrim()
    c.ok(bool(root and root.IsValid()), "default prim", str(root.GetPath()) if root else "MISSING")
    up = UsdGeom.GetStageUpAxis(stage)
    c.ok(up == UsdGeom.Tokens.z, "up axis is Z", f"upAxis={up}")
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    c.ok(abs(mpu - 1.0) < 1e-9, "metersPerUnit == 1", f"{mpu}")
    kpu = UsdPhysics.GetStageKilogramsPerUnit(stage)
    c.ok(abs(kpu - 1.0) < 1e-9, "kilogramsPerUnit == 1", f"{kpu}", warn_only=True)
    return root


def check_meshes(stage, c: Checks, approximation):
    meshes = [p for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
    c.ok(bool(meshes), "meshes present", f"{len(meshes)} mesh prim(s)")
    bad_idx, bad_ext, bad_sub, bad_nan, bad_st, no_coll, no_physmat = [], [], [], [], [], [], []
    for m in meshes:
        g = UsdGeom.Mesh(m)
        pts = g.GetPointsAttr().Get() or []
        idx = g.GetFaceVertexIndicesAttr().Get() or []
        cnt = g.GetFaceVertexCountsAttr().Get() or []
        name = m.GetPath().name

        if sum(cnt) != len(idx) or (idx and (min(idx) < 0 or max(idx) >= len(pts))):
            bad_idx.append(str(m.GetPath()))
        if not _finite(pts):
            bad_nan.append(str(m.GetPath()))

        sub = g.GetSubdivisionSchemeAttr().Get()
        # The default catmullClark would make Hydra render a smoothed limit
        # surface that no longer matches the shape PhysX cooked from `points`.
        if sub != UsdGeom.Tokens.none:
            bad_sub.append(f"{name}={sub}")

        ext = g.GetExtentAttr().Get()
        if ext and pts:
            lo = [min(p[i] for p in pts) for i in range(3)]
            hi = [max(p[i] for p in pts) for i in range(3)]
            if max(abs(ext[0][i] - lo[i]) for i in range(3)) > 1e-4 or \
               max(abs(ext[1][i] - hi[i]) for i in range(3)) > 1e-4:
                bad_ext.append(name)
        elif not ext:
            bad_ext.append(f"{name}=unauthored")

        pv = UsdGeom.PrimvarsAPI(m).GetPrimvar("st")
        if pv:
            interp = pv.GetInterpolation()
            vals = pv.Get() or []
            ind = pv.GetIndices() if pv.IsIndexed() else None
            if interp not in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.faceVarying):
                bad_st.append(f"{name}: interpolation={interp}")
            elif ind is not None and (len(ind) != sum(cnt) or (len(ind) and max(ind) >= len(vals))):
                bad_st.append(f"{name}: bad indices")
            elif interp == UsdGeom.Tokens.vertex and len(vals) != len(pts):
                bad_st.append(f"{name}: {len(vals)} uvs != {len(pts)} points")
            elif not _finite(vals):
                bad_st.append(f"{name}: non-finite uv")

        if approximation != "none":
            if not m.HasAPI(UsdPhysics.CollisionAPI) or not m.HasAPI(UsdPhysics.MeshCollisionAPI):
                no_coll.append(name)
            else:
                got = UsdPhysics.MeshCollisionAPI(m).GetApproximationAttr().Get()
                if got != approximation:
                    no_coll.append(f"{name}={got}")
            rel = UsdShade.MaterialBindingAPI(m).GetDirectBindingRel("physics")
            tgts = rel.GetTargets() if rel else []
            if not tgts or not any(stage.GetPrimAtPath(t).HasAPI(UsdPhysics.MaterialAPI) for t in tgts):
                no_physmat.append(name)

    c.ok(not bad_idx, "mesh indices consistent", "; ".join(bad_idx[:3]))
    c.ok(not bad_nan, "no NaN/Inf in points", "; ".join(bad_nan[:3]))
    c.ok(not bad_sub, "subdivisionScheme == none", "; ".join(bad_sub[:3]))
    c.ok(not bad_ext, "extent matches points", "; ".join(bad_ext[:3]))
    c.ok(not bad_st, "primvars:st well-formed", "; ".join(bad_st[:3]))
    if approximation != "none":
        c.ok(not no_coll, f"collider == {approximation}", "; ".join(no_coll[:3]))
        c.ok(not no_physmat, "physics material bound", "; ".join(no_physmat[:3]))
    return meshes


def check_no_scale(stage, c: Checks):
    """A scale op above a rigid body desynchronises PhysX joint frames.

    The converter bakes scale into mesh points precisely so this stays empty.
    """
    offenders = []
    for p in stage.Traverse():
        x = UsdGeom.Xformable(p)
        if not x:
            continue
        for op in x.GetOrderedXformOps():
            if op.GetOpType() in (UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.TypeTransform):
                offenders.append(f"{p.GetPath()}:{op.GetOpName()}")
    c.ok(not offenders, "no scale/transform xformOps", "; ".join(offenders[:3]))


def check_bodies(stage, c: Checks, expect_links):
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    paths = [p.GetPath() for p in bodies]
    if expect_links is not None:
        c.ok(len(bodies) == expect_links, "rigid body count",
             f"{len(bodies)} (expected {expect_links})")
    else:
        c.add(PASS, "rigid body count", str(len(bodies)))

    nested = [str(a) for a in paths for b in paths if a != b and str(a).startswith(str(b) + "/")]
    c.ok(not nested, "no nested rigid bodies", "; ".join(nested[:3]))

    massless = []
    for p in bodies:
        api = UsdPhysics.MassAPI(p)
        m = api.GetMassAttr().Get() if p.HasAPI(UsdPhysics.MassAPI) else None
        d = api.GetDensityAttr().Get() if p.HasAPI(UsdPhysics.MassAPI) else None
        if not ((m or 0) > 0 or (d or 0) > 0):
            massless.append(p.GetPath().name)
    c.ok(not massless, "every body has mass or density", "; ".join(massless[:3]))

    arts = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(bodies) > 1:
        c.ok(len(arts) == 1, "exactly one articulation root",
             ", ".join(str(a.GetPath()) for a in arts) or "none")
    else:
        c.ok(len(arts) == 0, "single body -> no articulation root", f"{len(arts)} found",
             warn_only=True)
    return bodies


JOINT_TYPES = ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint",
               "PhysicsFixedJoint", "PhysicsSphericalJoint")


def check_joints(stage, c: Checks, expect_joints):
    joints = [p for p in stage.Traverse() if p.GetTypeName() in JOINT_TYPES]
    if expect_joints is not None:
        c.ok(len(joints) == expect_joints, "joint count",
             f"{len(joints)} (expected {expect_joints})")
    else:
        c.add(PASS, "joint count", str(len(joints)))
    if not joints:
        return joints

    cache = UsdGeom.XformCache()
    bad_body, bad_axis, bad_quat, bad_frame, bad_limit = [], [], [], [], []

    for jp in joints:
        j = UsdPhysics.Joint(jp)
        name = jp.GetPath().name
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        if not b1:
            bad_body.append(f"{name}: no body1")
            continue
        if b0 and b0[0] == b1[0]:
            bad_body.append(f"{name}: body0 == body1")
        for t in list(b0) + list(b1):
            p = stage.GetPrimAtPath(t)
            if not p or not p.IsValid():
                bad_body.append(f"{name}: {t} unresolved")
            elif not p.HasAPI(UsdPhysics.RigidBodyAPI):
                bad_body.append(f"{name}: {t} is not a rigid body")

        lp0 = j.GetLocalPos0Attr().Get() or Gf.Vec3f(0)
        lp1 = j.GetLocalPos1Attr().Get() or Gf.Vec3f(0)
        lr0 = j.GetLocalRot0Attr().Get() or Gf.Quatf(1)
        lr1 = j.GetLocalRot1Attr().Get() or Gf.Quatf(1)
        for q, lbl in ((lr0, "localRot0"), (lr1, "localRot1")):
            n = Gf.Quatd(q).GetLength()
            if abs(n - 1.0) > 1e-5:
                bad_quat.append(f"{name}.{lbl}: |q|={n:.6f}")

        # --- the joint-frame coincidence check --------------------------------
        # Compose each side's joint frame into world space and require they agree.
        def world_frame(body_targets, lp, lr):
            T = Gf.Matrix4d(1.0)
            if body_targets:
                bp = stage.GetPrimAtPath(body_targets[0])
                if bp and bp.IsValid():
                    T = cache.GetLocalToWorldTransform(bp)
            local = Gf.Matrix4d(1.0)
            local.SetRotate(Gf.Quatd(lr))
            local.SetTranslateOnly(Gf.Vec3d(lp))
            return local * T

        F0 = world_frame(b0, lp0, lr0)
        F1 = world_frame(b1, lp1, lr1)
        dp = (F0.ExtractTranslation() - F1.ExtractTranslation()).GetLength()
        q0, q1 = F0.ExtractRotationQuat(), F1.ExtractRotationQuat()
        dq = min((Gf.Quatd(q0) - Gf.Quatd(q1)).GetLength(),
                 (Gf.Quatd(q0) + Gf.Quatd(q1)).GetLength())   # q and -q are the same rotation
        if dp > 1e-4 or dq > 1e-4:
            bad_frame.append(f"{name}: dpos={dp:.2e} drot={dq:.2e}")

        if jp.GetTypeName() in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            ax = jp.GetAttribute("physics:axis").Get()
            if ax not in ("X", "Y", "Z"):
                bad_axis.append(f"{name}: axis={ax}")
            lo = jp.GetAttribute("physics:lowerLimit").Get()
            hi = jp.GetAttribute("physics:upperLimit").Get()
            if lo is not None and hi is not None and lo <= hi:
                lim = 360.0 if jp.GetTypeName() == "PhysicsRevoluteJoint" else 100.0
                if not (math.isfinite(lo) and math.isfinite(hi)) or max(abs(lo), abs(hi)) > lim:
                    bad_limit.append(f"{name}: [{lo}, {hi}]")

    c.ok(not bad_body, "joint bodies resolve", "; ".join(bad_body[:3]))
    c.ok(not bad_quat, "joint quaternions are unit", "; ".join(bad_quat[:3]))
    c.ok(not bad_frame, "joint frames coincide", "; ".join(bad_frame[:3]))
    c.ok(not bad_axis, "joint axis token valid", "; ".join(bad_axis[:3]))
    c.ok(not bad_limit, "joint limits in range", "; ".join(bad_limit[:3]))
    return joints


def check_extent(stage, c: Checks, root, size_range, floor_align):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
    if rng.IsEmpty():
        return c.add(FAIL, "world bbox", "empty -- geometry did not compose")
    mn, mx = rng.GetMin(), rng.GetMax()
    size = max(mx[i] - mn[i] for i in range(3))
    detail = (f"x=[{mn[0]:+.3f},{mx[0]:+.3f}] y=[{mn[1]:+.3f},{mx[1]:+.3f}] "
              f"z=[{mn[2]:+.3f},{mx[2]:+.3f}] m  (max extent {size:.3f} m)")
    if size_range:
        c.ok(size_range[0] <= size <= size_range[1], "world size plausible", detail)
    else:
        c.add(PASS, "world bbox", detail)
    if floor_align:
        c.ok(abs(mn[2]) < 1e-3, "base sits at z=0", f"z_min={mn[2]:+.5f}", warn_only=True)
    return rng


def check_assets(path, c: Checks):
    """Texture/sublayer paths must resolve, layer-aware (not by naive globbing)."""
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(path))
    except Exception as exc:
        c.add(WARN, "asset dependencies", f"could not compute: {exc}")
        return
    c.ok(not unresolved, "no unresolved asset paths", "; ".join(map(str, unresolved[:3])))
    missing = [str(a) for a in assets if not os.path.isfile(str(a))]
    c.ok(not missing, "referenced assets exist on disk",
         f"{len(assets)} asset(s); missing: {missing[:3]}" if missing else f"{len(assets)} asset(s)")


def check_compliance(path, c: Checks):
    try:
        checker = UsdUtils.ComplianceChecker(arkit=False, skipARKitRootLayerCheck=True)
        checker.CheckCompliance(path)
    except Exception as exc:
        c.add(WARN, "USD compliance", f"checker unavailable: {exc}")
        return
    errs = list(checker.GetErrors()) + list(checker.GetFailedChecks())
    warns = list(checker.GetWarnings())
    c.ok(not errs, "USD compliance", "; ".join(errs[:2]) if errs else f"{len(warns)} warning(s)")


def main():
    ap = argparse.ArgumentParser(description="GPU-free structural check of a generated USD.")
    ap.add_argument("usd")
    ap.add_argument("--expect-links", type=int, default=None)
    ap.add_argument("--expect-joints", type=int, default=None)
    ap.add_argument("--size-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                    help="plausible real-world max extent in metres, e.g. 0.3 3.0")
    ap.add_argument("--collision", default="convexDecomposition")
    ap.add_argument("--no-floor-align", dest="floor_align", action="store_false", default=True)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.usd):
        print(f"ERROR: no such file: {args.usd}", file=sys.stderr)
        return 2

    print(f"\nverifying {args.usd}\n")
    c = Checks()
    stage = Usd.Stage.Open(args.usd)
    if not stage:
        c.add(FAIL, "stage opens", args.usd)
        c.report()
        return 1

    root = check_stage(stage, c)
    check_meshes(stage, c, args.collision)
    check_no_scale(stage, c)
    check_bodies(stage, c, args.expect_links)
    check_joints(stage, c, args.expect_joints)
    if root:
        check_extent(stage, c, root, args.size_range, args.floor_align)
    check_assets(args.usd, c)
    check_compliance(args.usd, c)

    n_fail = c.report()
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump([{"status": s, "check": n, "detail": d} for s, n, d in c.rows], fh, indent=2)
    print(f"\n  {'FAILED' if n_fail else 'OK'}: {args.usd}\n")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
