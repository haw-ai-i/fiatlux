"""The benchmark ladder's tread geometry, and the check that it still holds.

Importable WITHOUT launching Isaac: replay_clip_kinematic.py builds a SimulationApp at
module level, so anything that merely wants the tread table must not import it (doing so
silently starts Kit and blocks on its EULA prompt).
"""

from __future__ import annotations

import math

import numpy as np


def quat_rot(q, v):
    w, u = q[0], np.asarray(q[1:])
    v = np.asarray(v, float)
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def mesh_inventory(stage, ladder_prim_path, scale):
    """World AABB of every mesh under the ladder prim. The step treads are NOT named prims --
    they come out of the convex decomposition -- so they are found by shape, not by name."""
    from pxr import Gf, Usd, UsdGeom

    out = []
    root = stage.GetPrimAtPath(ladder_prim_path)
    if not root or not root.IsValid():
        print(f"[WARN  ] no prim at {ladder_prim_path}", flush=True)
        return out
    for prim in Usd.PrimRange(root):
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            continue
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        P = np.array([[*m.Transform(Gf.Vec3d(q[0], q[1], q[2]))] for q in pts], float) * scale
        lo, hi = P.min(0), P.max(0)
        out.append(
            {
                "name": prim.GetName(),
                "path": str(prim.GetPath()),
                "lo": lo.tolist(),
                "hi": hi.tolist(),
                "npts": len(pts),
            }
        )
    out.sort(key=lambda t: t["hi"][2])
    return out


# Tread geometry in the LADDER'S OWN frame, measured off this asset's vertex cloud
# (AlumStep_D D01, 41849-vert single body mesh -- treads are not prims and cannot be found by
# name). The asset is fixed; only its POSE randomises, and the pose is read live every run.
# verify_treads() re-derives this from the live mesh so a changed asset fails loudly instead
# of silently aiming the foot at thin air.
TREADS_LOCAL = [
    {"top": 0.2360, "y_front": -0.4173, "y_back": -0.3435},
    {"top": 0.4700, "y_front": -0.3367, "y_back": -0.2629},
    {"top": 0.7060, "y_front": -0.2511, "y_back": -0.1774},
    {"top": 0.9400, "y_front": -0.1656, "y_back": -0.0919},
]
# The PLATFORM: the ladder's top standing surface. Much deeper than a tread (0.393 m vs 0.074)
# and the place a climb actually ends, so it is kept separate rather than as a fifth tread.
PLATFORM_LOCAL = {"top": 1.1520, "y_front": -0.1040, "y_back": 0.2890}
PLATFORM_WIDTH = 0.4764

TREAD_WIDTH = 0.4946
TREAD_DEPTH = 0.0738
ZBINS = 0.002  # top = highest 2 mm bin (anchored at 0) holding >=100 verts; ONE definition,
#       shared by the table above and verify_treads below
TREAD_X_CENTRE = 0.0


def to_ladder_frame(verts_world, ladder_pos, ladder_yaw):
    c, s_ = math.cos(-ladder_yaw), math.sin(-ladder_yaw)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    return (np.asarray(verts_world, float) - np.asarray(ladder_pos, float)) @ R.T


def verify_treads(verts_world, ladder_pos, ladder_yaw, tol=0.006):
    """Re-derive each tread's top from the live mesh and compare with TREADS_LOCAL."""
    L = to_ladder_frame(verts_world, ladder_pos, ladder_yaw)
    ok = True
    for i, t in enumerate(TREADS_LOCAL, 1):
        yc = (t["y_front"] + t["y_back"]) / 2
        m = (np.abs(L[:, 1] - yc) < 0.055) & (np.abs(L[:, 2] - t["top"]) < 0.015)
        sub = L[m]
        if len(sub) < 50:
            print(f"[VERIFY] tread {i}: only {len(sub)} verts near the expected surface", flush=True)
            ok = False
            continue
        # Top = the highest 2 mm bin holding >=100 verts, on a grid ANCHORED AT ZERO so the
        # answer does not drift with the data's own min. Each tread carries a sparse lip ~14 mm
        # above its walking surface (12-18 verts); that lip is not what a foot stands on.
        h, e = np.histogram(sub[:, 2], bins=np.arange(0.0, 1.4, ZBINS))
        dense = [e[k + 1] for k, n in enumerate(h) if n >= 100]
        if not dense:
            print(f"[VERIFY] tread {i}: no dense top surface found", flush=True)
            ok = False
            continue
        top = max(dense)
        d = top - t["top"]
        flag = "" if abs(d) <= tol else "   <-- MISMATCH"
        print(f"[VERIFY] tread {i}: live top {top:.4f} vs table {t['top']:.4f}  ({d * 1000:+.1f} mm){flag}", flush=True)
        ok = ok and abs(d) <= tol
    return ok


def tread_world(t, ladder_pos, ladder_yaw):
    """Tread centre and front-edge midpoint in WORLD coordinates."""
    c, s_ = math.cos(ladder_yaw), math.sin(ladder_yaw)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    ctr = R @ np.array([t["x_centre"], (t["y_front"] + t["y_back"]) / 2, t["top"]]) + ladder_pos
    fr = R @ np.array([t["x_centre"], t["y_front"], t["top"]]) + ladder_pos
    return ctr, fr
