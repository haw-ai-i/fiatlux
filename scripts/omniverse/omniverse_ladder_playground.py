#!/usr/bin/env python3
"""Isaac Sim two-row playground for the Omniverse ladders.

  FRONT row  = <name>_collision.usd        (static)  -> a small box is dropped on
               EACH step (the boxes rest on the treads, proving collision at every
               step), PLUS one heavier box hovering above each ladder that drops onto
               it on play -- watch the ladder catch it (a missing collider would let
               it fall straight through to the floor).
  BACK row   = <name>_collision_rigid.usd  (dynamic) -> raised + tilted; on PLAY
               they topple/tumble, proving they are carry-able rigid bodies.

Ladders that can't free-stand -- extension ladders AND folded ladders (both tall
and thin) -- use a wall, since that is how they are actually climbed: the FRONT
collision one rests statically against the wall and holds a dropped box; the BACK
rigid one starts upright with a gap and TOPPLES onto the wall on play. One open
representative per design, plus a folded variant for every design that ships one.
Across all keep folders.

Run:  DISPLAY=:1001 python scripts/omniverse/omniverse_ladder_playground.py

Visual tool only -- physics plays automatically; you judge it by eye (a headless
pass/fail on "did a box rest on an open ladder" is too noisy to be meaningful).
"""
import os
import math
import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, PhysxSchema  # noqa: E402

# Point at the downloaded asset group (from download_assets.sh). It's flattened:
# all design folders + the shared Materials/ live directly under omniverse_ladder/.
# Override with FIATLUX_LADDER_DIR; defaults to the repo assets/ location.
LADDER_DIR = os.environ.get(
    "FIATLUX_LADDER_DIR",
    # this script lives in scripts/omniverse/, so climb 3 levels: omniverse -> scripts -> repo
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "assets", "omniverse_ladder"))
ROOTS = [LADDER_DIR]
SPACING = 1.8
ROW_Y_COLLISION = -2.5
ROW_Y_RIGID = 2.5
BOX = 0.10               # box edge (m)
DROP_ABOVE = 0.05        # how far above each step the box starts (small = less bounce)
DROP_SZ = 0.15           # drop-box cube edge (m)
DROP_H = 0.6             # height a drop-box starts above the ladder top (m)


def bbox_dims(usd):
    """(width_x, depth_y, height_z) of the asset in metres."""
    s = Usd.Stage.Open(usd)
    mpu = UsdGeom.GetStageMetersPerUnit(s)
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    d = bb.ComputeWorldBound(s.GetPseudoRoot()).ComputeAlignedRange().GetSize()
    return d[0] * mpu, d[1] * mpu, d[2] * mpu


def bbox_range(usd):
    """(ymin, ymax, zmin, zmax) of the asset in metres."""
    s = Usd.Stage.Open(usd)
    mpu = UsdGeom.GetStageMetersPerUnit(s)
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        s.GetPseudoRoot()).ComputeAlignedRange()
    return r.GetMin()[1] * mpu, r.GetMax()[1] * mpu, r.GetMin()[2] * mpu, r.GetMax()[2] * mpu


def footprint(usd):
    dx, dy, _ = bbox_dims(usd)
    return min(dx, dy)


FOLDED_FOOT = 0.35       # min horizontal footprint below this = folded/thin (leans on a wall)


def is_leaning(usd):
    """Tall + thin footprint -> can't free-stand; must lean on a wall.
    Catches extension ladders (and any very-narrow ladder). Folded variants are
    flagged explicitly by the caller, so keep this threshold tight to avoid
    walling self-standing ladders like the ~0.39 m-deep residential one."""
    dx, dy, dz = bbox_dims(usd)
    return min(dx, dy) < 0.30 and dz > 1.6


def make_wall(st, path, x, y_face, height, width=0.9):
    """A static wall whose front face is at y_face (facing -y), sitting on the floor."""
    thick = 0.12
    # guard: PhysX rejects a box with a ~zero/negative dimension (degenerate wall_h from an
    # off-centre ladder bbox). Clamp to a valid minimum so shape creation never fails.
    if height < 0.2 or width < 0.05:
        print(f"  [wall guard] {path}: clamped height={height:.3f} width={width:.3f}", flush=True)
    height = max(float(height), 0.2)
    width = max(float(width), 0.05)
    w = UsdGeom.Cube.Define(st, path)
    w.CreateSizeAttr(1.0)
    w.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(
        Gf.Vec3d(x, y_face - thick / 2, height / 2))
    w.AddXformOp(UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(width, thick, height))
    w.CreateDisplayColorAttr([(0.72, 0.72, 0.75)])
    UsdPhysics.CollisionAPI.Apply(w.GetPrim())


def gather_items():
    """One OPEN representative per design, plus one FOLDED representative for designs
    that ship a folded variant (folded ladders lean on a wall too). Returns
    (label, collision_usd, rigid_usd) tuples."""
    items = []
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for design in sorted(os.listdir(root)):
            dd = os.path.join(root, design)
            if not os.path.isdir(dd) or design.startswith("."):   # skip .SubUSDs/.thumbs stub folders
                continue
            rigids = sorted(f for f in os.listdir(dd) if f.endswith("_collision_rigid.usd"))
            if not rigids:
                continue
            opens, foldeds = [], []
            for r in rigids:
                orig = os.path.join(dd, r.replace("_collision_rigid.usd", ".usd"))
                try:
                    (opens if os.path.exists(orig) and footprint(orig) >= FOLDED_FOOT else foldeds).append(r)
                except Exception:
                    opens.append(r)
            def entry(r, label, folded):
                rigid = os.path.join(dd, r)
                coll = os.path.join(dd, r.replace("_collision_rigid.usd", "_collision.usd"))
                return (label, coll, rigid, folded)
            if opens:
                items.append(entry(opens[0], design, False))
            elif foldeds:
                items.append(entry(foldeds[0], design, True))
            if opens and foldeds:                       # also show the folded one, leaned on a wall
                items.append(entry(foldeds[0], design + " (folded)", True))
    return items


def detect_steps(usd_path, max_steps=8):
    """Geometrically find the step/tread heights of a ladder and a box position on each.
    Returns list of (x, y, z_top) in metres, relative to the ladder origin."""
    s = Usd.Stage.Open(usd_path)
    mpu = UsdGeom.GetStageMetersPerUnit(s)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    pts = []
    for pr in Usd.PrimRange(s.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if pr.GetTypeName() == "Mesh":
            P = UsdGeom.Mesh(pr).GetPointsAttr().Get()
            if not P:
                continue
            xf = xc.GetLocalToWorldTransform(pr)
            pts.extend(xf.Transform(p) for p in P)
    if not pts:
        return []
    A = np.array([[p[0], p[1], p[2]] for p in pts]) * mpu
    z = A[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    h = zmax - zmin
    if h <= 0.05:
        return [(float(np.median(A[:, 0])), float(np.median(A[:, 1])), zmax)]
    nb = max(12, int(h / 0.02))
    hist, edges = np.histogram(z, bins=nb)
    centers = (edges[:-1] + edges[1:]) / 2
    lo, hi = zmin + 0.10 * h, zmax        # include the very top (platform/step-stand surface)
    thr = hist.mean()
    peaks = [(centers[i], hist[i]) for i in range(1, len(hist) - 1)
             if lo <= centers[i] <= hi and hist[i] >= hist[i - 1]
             and hist[i] >= hist[i + 1] and hist[i] > thr]
    peaks.sort()
    merged = []
    for zc, cnt in peaks:                      # merge peaks within 8 cm (same tread)
        if merged and zc - merged[-1][0] < 0.08:
            if cnt > merged[-1][1]:
                merged[-1] = (zc, cnt)
        else:
            merged.append((zc, cnt))
    merged.sort(key=lambda t: -t[1])
    merged = sorted(merged[:max_steps])
    steps = []
    cell = 0.04
    for zc, _ in merged:
        near = A[np.abs(A[:, 2] - zc) < 0.02]     # just the top surface at this height
        if len(near) < 10:
            continue
        # A flat TREAD is a SOLID horizontal patch (wide AND deep). A round rung is a
        # thin strip. Rasterize into a 4 cm occupancy grid and require a solid
        # box-sized (3x3 ≈ 12 cm) patch — that rules out rungs/rails you can't rest on.
        xs, ys = near[:, 0], near[:, 1]
        x0, y0 = xs.min(), ys.min()
        nx = int((xs.max() - x0) / cell) + 1
        ny = int((ys.max() - y0) / cell) + 1
        if nx < 3 or ny < 3:                       # too thin in a direction -> not a tread
            continue
        occ = np.zeros((nx, ny), bool)
        occ[((xs - x0) / cell).astype(int), ((ys - y0) / cell).astype(int)] = True
        best = None
        for a in range(nx - 2):
            for b in range(ny - 2):
                fill = int(occ[a:a + 3, b:b + 3].sum())
                if best is None or fill > best[0]:
                    best = (fill, a, b)
        if best is None or best[0] < 6:            # need a mostly-solid 3x3 patch
            continue
        _, a, b = best
        mx = x0 + (a + 1.5) * cell
        my = y0 + (b + 1.5) * cell
        # the solid patch sits at ~zc; that's the surface to rest a box on
        steps.append((float(mx), float(my), float(zc)))
    if not steps:                              # fallback: one box on the top
        steps = [(float(np.median(A[:, 0])), float(np.median(A[:, 1])), zmax)]
    return steps


def make_box(st, path, pos, mat=None, color=(1.0, 0.55, 0.1)):
    from pxr import UsdShade
    c = UsdGeom.Cube.Define(st, path)
    c.CreateSizeAttr(BOX)
    c.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(*pos))
    c.CreateDisplayColorAttr([color])
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(c.GetPrim())
    UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(0.15)
    if mat is not None:                       # grip so it doesn't slide off the tread
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return c


def make_dropbox(st, path, pos, mat=None, color=(0.9, 0.15, 0.15)):
    """A heavier CCD cube dropped from above a ladder -- the visual drop test.
    On play it falls onto the ladder; a solid collider catches it, no collider
    lets it fall through to the floor."""
    from pxr import UsdShade
    c = UsdGeom.Cube.Define(st, path)
    c.CreateSizeAttr(DROP_SZ)
    c.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(*pos))
    c.CreateDisplayColorAttr([color])
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(c.GetPrim())
    PhysxSchema.PhysxRigidBodyAPI.Apply(c.GetPrim()).CreateEnableCCDAttr(True)  # don't tunnel thin rungs
    UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(0.5)
    if mat is not None:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return c


def build(st):
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdGeom.Xform.Define(st, "/World")
    sc = UsdPhysics.Scene.Define(st, "/World/PhysicsScene")
    sc.CreateGravityMagnitudeAttr(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim()).CreateEnableCCDAttr(True)
    UsdLux.DistantLight.Define(st, "/World/Sun").CreateIntensityAttr(2500)
    UsdLux.DomeLight.Define(st, "/World/Sky").CreateIntensityAttr(1000)
    g = UsdGeom.Cube.Define(st, "/World/Ground")
    g.CreateSizeAttr(1.0)
    g.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(0, 0, -0.05))
    g.AddXformOp(UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(120, 20, 0.1))
    g.CreateDisplayColorAttr([(0.4, 0.42, 0.45)])
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())

    from pxr import UsdShade
    bmat = UsdShade.Material.Define(st, "/World/BoxFriction")
    bpm = UsdPhysics.MaterialAPI.Apply(bmat.GetPrim())
    bpm.CreateStaticFrictionAttr(1.0)
    bpm.CreateDynamicFrictionAttr(0.9)
    bpm.CreateRestitutionAttr(0.0)

    items = gather_items()
    n = len(items)
    x0 = -(n - 1) * SPACING / 2.0
    boxes = []    # (prim_path, target_step_z)  -- box dropped on each step
    drops = []    # (prim_path, label, top_z)   -- a box hovering above each ladder top
    rigids = []   # (label, prim_path, start_z) -- free-standing rigid ladders (should fall)
    leaners = []  # (label, prim_path)          -- leaned rigid ladders (should STAY up)
    for i, (design, coll, rigid, folded) in enumerate(items):
        x = x0 + i * SPACING
        mpu = UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(coll))
        dx, dy, dz = bbox_dims(coll)
        lean = folded or is_leaning(coll)     # folded ladders always lean on a wall

        if lean:
            # Rotate about +X: the top tips toward -y; the wall sits at the RESTING lean
            # (wall_lean) so the top corner (ymin, zmax) just touches it. The static
            # collision ladder rests on the wall (and holds a dropped box); the RIGID
            # ladder starts more upright (start_lean, a gap) and TOPPLES onto the wall.
            # A box tips once its centre passes the base edge, at atan(footprint/height);
            # start each rigid leaner a few degrees PAST its own tip angle so it reliably
            # falls TOWARD the wall (short/wide folded ones have a big tip angle).
            foot = min(dx, dy)
            tip = math.degrees(math.atan2(foot, dz)) if dz > 0 else 12.0
            start_lean = max(10.0, tip + 4.0)
            wall_lean = start_lean + 8.0
            sw, cw = math.sin(math.radians(wall_lean)), math.cos(math.radians(wall_lean))
            ymn, ymx, zmn, zmx = bbox_range(coll)
            ext = ymn * cw - zmx * sw                        # world -y of the tipped-top corner
            wall_h = zmx * cw + ymx * sw + 0.3
            for row, tag, ref, ang in ((ROW_Y_COLLISION, "C", coll, wall_lean),
                                       (ROW_Y_RIGID, "R", rigid, start_lean)):
                slot = UsdGeom.Xform.Define(st, f"/World/{tag}_{i}")
                slot.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(x, row, 0.02))
                slot.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ).Set(Gf.Vec3f(ang, 0, 0))
                slot.AddXformOp(UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(mpu, mpu, mpu))
                UsdGeom.Xform.Define(st, f"/World/{tag}_{i}/a").GetPrim().GetReferences().AddReference(ref)
                make_wall(st, f"/World/wall_{tag}_{i}", x, row + ext - 0.01, wall_h)
            # gentle nudge toward the wall so it topples that way promptly (deg/s about +X)
            UsdPhysics.RigidBodyAPI.Apply(st.GetPrimAtPath(f"/World/R_{i}/a")).CreateAngularVelocityAttr(
                Gf.Vec3f(6.0, 0.0, 0.0))
            # a box dropped onto the (static) collision ladder as it leans on its wall
            cxm, cym, czm = bbox_center(coll)
            a = math.radians(wall_lean)
            yc = ROW_Y_COLLISION + (cym * math.cos(a) - czm * math.sin(a))
            zc = 0.02 + (cym * math.sin(a) + czm * math.cos(a))
            dp = f"/World/drop_{i}"
            make_dropbox(st, dp, (x + cxm, yc, zc + DROP_SZ / 2 + DROP_H), mat=bmat)
            drops.append((dp, design, zc))
            leaners.append((design, f"/World/R_{i}/a"))
            print(f"  {i:2} {design:28} LEAN (rigid topples onto wall)", flush=True)
        else:
            # ---- FRONT: static collision ladder + a box on every step ----
            cslot = UsdGeom.Xform.Define(st, f"/World/C_{i}")
            cslot.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(x, ROW_Y_COLLISION, 0))
            cslot.AddXformOp(UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(mpu, mpu, mpu))
            UsdGeom.Xform.Define(st, f"/World/C_{i}/a").GetPrim().GetReferences().AddReference(coll)
            steps = detect_steps(coll)
            for k, (mx, my, mz) in enumerate(steps):
                bp = f"/World/box_{i}_{k}"
                # mz is the true top surface at (mx,my); spawn just above it -> lands ON
                # TOP with a tiny drop (no bounce, never inside a leg/bar)
                make_box(st, bp, (x + mx, ROW_Y_COLLISION + my, mz + BOX / 2 + DROP_ABOVE), mat=bmat)
                boxes.append((bp, mz))
            # a heavier box hovering above the ladder top -- on play it drops onto the
            # ladder (watch it get caught, or fall through if there were no collider)
            cx, cy, tz = bbox_center_top(coll)
            dp = f"/World/drop_{i}"
            make_dropbox(st, dp, (x + cx, ROW_Y_COLLISION + cy, tz + DROP_SZ / 2 + DROP_H), mat=bmat)
            drops.append((dp, design, tz))
            # ---- BACK: dynamic collision+rigid ladder (raised + tilted) ----
            rslot = UsdGeom.Xform.Define(st, f"/World/R_{i}")
            rslot.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(x, ROW_Y_RIGID, 0.35))
            rslot.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ).Set(Gf.Vec3f(12, 6, 0))
            rslot.AddXformOp(UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(mpu, mpu, mpu))
            UsdGeom.Xform.Define(st, f"/World/R_{i}/a").GetPrim().GetReferences().AddReference(rigid)
            rigids.append((design, f"/World/R_{i}/a", 0.35))
            print(f"  {i:2} {design:28} steps={len(steps)}", flush=True)
    return items, boxes, drops, rigids, leaners


def bbox_center_top(usd):
    """(cx, cy, top_z) of the asset in metres -- where to hover a drop-box above it."""
    s = Usd.Stage.Open(usd)
    mpu = UsdGeom.GetStageMetersPerUnit(s)
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        s.GetPseudoRoot()).ComputeAlignedRange()
    cx = (r.GetMin()[0] + r.GetMax()[0]) / 2 * mpu
    cy = (r.GetMin()[1] + r.GetMax()[1]) / 2 * mpu
    return cx, cy, r.GetMax()[2] * mpu


def bbox_center(usd):
    """(cx, cy, cz) centre of the asset in metres."""
    s = Usd.Stage.Open(usd)
    mpu = UsdGeom.GetStageMetersPerUnit(s)
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        s.GetPseudoRoot()).ComputeAlignedRange()
    return tuple((r.GetMin()[k] + r.GetMax()[k]) / 2 * mpu for k in range(3))


def main():
    print(f"[ladder source: {LADDER_DIR}]", flush=True)
    ctx = omni.usd.get_context()
    ctx.new_stage()
    st = ctx.get_stage()
    items, boxes, drops, rigids, leaners = build(st)
    print(f"\n[{len(items)} ladders.  FRONT=collision ({len(boxes)} step boxes + {len(drops)} "
          f"drop-boxes above the tops) | BACK=collision+rigid (topple on play).  {len(leaners)} "
          f"lean-type on walls.]\n", flush=True)

    from isaacsim.core.api import SimulationContext
    sim = SimulationContext(physics_dt=1 / 120, rendering_dt=1 / 120, stage_units_in_meters=1.0)
    sim.reset()
    while simulation_app.is_running():   # physics runs live: boxes drop, rigids topple, leaners hold
        sim.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
