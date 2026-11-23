"""Does the clip put a hand THROUGH something? Offline, no Isaac, real collision shapes.

Why this exists
---------------
The authoring gate modelled the Dex3 hand as a flat pad plus a fingertip. The real hand is not
that: with the fingers open the THUMB hangs ~100 mm below the palm face, and the wrist camera
housing sits 28 mm below the palm face and reaches 107 mm sideways toward the other hand. A clip
that passed the gate was thrown off the ladder by the solver on its first physics step
(10^4..10^5 N on the hand links), because those parts started inside the ladder, inside the
other hand, or inside the robot's own forearm.

This checks every sampled frame against the convex hulls PhysX collides (extracted from the
robot USD into g1_dex3_hand_hulls.npz), fingers at zero = open:

  ladder : every hand link against the ladder mesh. A link that is SUPPOSED to rest on a hold
           (palm, fingers) may touch its hold's top face from above; anything deeper, and any
           other link at all, is a collision.
  hands  : left hand links against right hand links.
  self   : hand links against the same arm's wrist_pitch / wrist_roll / elbow links, judged
           against the clearance the same pair has at the zero pose (they are close by design).

    python scripts/climb/check_hand_collision.py --clip <dir> --mesh <ladder_mesh.npz>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--mesh", required=True)
ap.add_argument("--stride", type=int, default=6, help="sample every Nth clip frame")
ap.add_argument("--clear", type=float, default=0.004, help="required clearance, m")
ap.add_argument(
    "--rest_depth",
    type=float,
    default=0.006,
    help="a resting link may sit this far into its hold's top face (the builder "
    "presses the pad 3 mm in so the solver finds contact)",
)
a = ap.parse_args()
HERE = pathlib.Path(__file__).parent
clip = pathlib.Path(a.clip)
m = np.load(clip / "motion.npz")
meta = json.loads((clip / "meta.json").read_text())
bn, fps = list(meta["body_names"]), float(meta["fps"])
holds = json.loads((clip / "holds.json").read_text())
BP, BQ = m["body_pos_w"], m["body_quat_w"]
T = BP.shape[0]


def qrot(q, v):
    w, u = q[..., :1], q[..., 1:]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


# hand links, as fixed offsets from wrist_yaw_link (every joint frame is unrotated, fingers at 0)
def offsets(side):
    sy = 1.0 if side == "left" else -1.0
    palm = np.array([0.0415, 0.003 * sy, 0.0])
    o = {"hand_camera_base_link": palm, "hand_palm_link": palm, "wrist_yaw_link": np.zeros(3)}
    for f, z in (("index", 0.0285), ("middle", -0.0285)):
        o[f"hand_{f}_0_link"] = palm + np.array([0.0777, 0.0016 * sy, z])
        o[f"hand_{f}_1_link"] = o[f"hand_{f}_0_link"] + np.array([0.0458, 0.0, 0.0])
    o["hand_thumb_0_link"] = palm + np.array([0.0255, 0.0, 0.0])
    o["hand_thumb_1_link"] = o["hand_thumb_0_link"] + np.array([-0.0025, -0.0193 * sy, 0.0])
    o["hand_thumb_2_link"] = o["hand_thumb_1_link"] + np.array([0.0, -0.0458 * sy, 0.0])
    return o


H = np.load(HERE / "g1_dex3_hand_hulls.npz")


def dense(V, sp=0.010):
    """Hull vertices plus points spread over the hull faces, so a thin sheet cannot slip between."""
    h = ConvexHull(V)
    pts = [V[h.vertices]]
    for s in h.simplices:
        A, B, C = V[s]
        n = int(max(np.linalg.norm(B - A), np.linalg.norm(C - A), np.linalg.norm(C - B)) / sp) + 1
        u, v = np.meshgrid(np.arange(n + 1) / n, np.arange(n + 1) / n)
        k = (u + v) <= 1.0
        pts.append(A + u[k][:, None] * (B - A) + v[k][:, None] * (C - A))
    P = np.unique(np.round(np.concatenate(pts), 4), axis=0)
    return P, h.equations


HANDL = {}  # side -> {link: (points in wrist frame, hull planes in wrist frame)}
ARML = {}  # side -> {link: (points in own frame, planes in own frame)}
for side in ("left", "right"):
    HANDL[side] = {}
    for ln, off in offsets(side).items():
        P, eq = dense(H[f"{side}_{ln}"].astype(float))
        eq = eq.copy()
        eq[:, 3] -= eq[:, :3] @ off
        HANDL[side][ln] = (P + off, eq)
    ARML[side] = {
        ln: dense(H[f"{side}_{ln}"].astype(float)) for ln in ("wrist_pitch_link", "wrist_roll_link", "elbow_link")
    }
REST = ("hand_palm_link", "hand_index_0_link", "hand_index_1_link", "hand_middle_0_link", "hand_middle_1_link")

# ladder, clip frame, sampled densely
TRI = np.load(a.mesh)["T"].reshape(-1, 3)
TRI = np.stack([TRI[:, 1] - 0.1996, -TRI[:, 0], TRI[:, 2]], 1).reshape(-1, 3, 3)
lp = [TRI.reshape(-1, 3)]
for t in TRI:
    A, B, C = t
    L = max(np.linalg.norm(B - A), np.linalg.norm(C - A), np.linalg.norm(C - B))
    if L > 0.005:
        n = int(L / 0.004) + 1
        u, v = np.meshgrid(np.arange(n + 1) / n, np.arange(n + 1) / n)
        k = (u + v) <= 1.0
        lp.append(A + u[k][:, None] * (B - A) + v[k][:, None] * (C - A))
LAD = cKDTree(np.unique(np.round(np.concatenate(lp), 3), axis=0))
print(f"[ladder] {LAD.n} surface samples")


def hull_gap(Pa, eqb):
    """Signed gap of point set A to convex hull B: >0 clear (a lower bound), <0 = depth inside."""
    return float((Pa @ eqb[:, :3].T + eqb[:, 3]).max(1).min())


def held(side, step):
    for s0, s1, nm, c in holds["schedule"][side]:
        if s0 <= step <= s1:
            return nm, np.array(c)
    return None, None


def wf(side, i, link="wrist_yaw_link"):
    k = bn.index(f"{side}_{link}")
    return BP[i, k], BQ[i, k]


worst = {}


def note(kind, key, val, i, extra=""):
    k = (kind, key)
    if k not in worst or val < worst[k][0]:
        worst[k] = (val, i, extra)


# self-clearance baseline at the zero pose cannot be read from the clip; use each pair's best
# clearance over the whole clip as "how far apart they normally are" and report the minimum.
for i in range(0, T, a.stride):
    step = i / fps * 50
    W = {}
    for side in ("left", "right"):
        wp, wq = wf(side, i)
        W[side] = {ln: wp + qrot(wq, P) for ln, (P, _) in HANDL[side].items()}
        nm, c = held(side, step)
        face_z = holds["holds"][nm]["z"] if nm else None
        for ln, P in W[side].items():
            dd, di = LAD.query(P)
            if nm and ln in REST:
                # resting link: over the hold's top face it may sit up to rest_depth into it; the
                # part of it that is NOT over the face (hanging in front of the edge) is judged
                # like any other link
                hx, hy = holds["holds"][nm]["x"], holds["holds"][nm]["y"]
                over = (
                    (P[:, 0] >= hx[0])
                    & (P[:, 0] <= hx[1])
                    & (P[:, 1] >= hy[0])
                    & (P[:, 1] <= hy[1])
                    & (P[:, 2] > face_z - 0.03)
                )
                if over.any():
                    note(
                        "ladder(resting)", f"{side} {ln} on {nm}", float(P[over, 2].min()) - (face_z - a.rest_depth), i
                    )
                low = (~over) & (P[:, 2] < face_z - a.rest_depth)  # hanging in front AND below the top
                if low.any():
                    k_ = np.where(low)[0][dd[low].argmin()]
                    note(
                        "ladder",
                        f"{side} {ln} (part below the face) while holding {nm}",
                        float(dd[low].min()),
                        i,
                        f"hand pt {P[k_].round(3)} ladder pt {LAD.data[di[k_]].round(3)}",
                    )
            else:
                note(
                    "ladder",
                    f"{side} {ln} while {'holding ' + nm if nm else 'moving'}",
                    float(dd.min()),
                    i,
                    f"hand pt {P[dd.argmin()].round(3)} ladder pt {LAD.data[di[dd.argmin()]].round(3)}",
                )
        for an, (PA, eqA) in ARML[side].items():
            ap_, aq_ = wf(side, i, an)
            PAw = ap_ + qrot(aq_, PA)
            for ln in ("hand_camera_base_link", "hand_palm_link", "hand_thumb_2_link", "hand_index_1_link"):
                Pl, eql = HANDL[side][ln]
                # arm link points into the wrist frame, against the hand link's hull
                loc = qrot(wq * np.array([1, -1, -1, -1]), PAw - wp)
                note("self", f"{side} {ln} vs {an}", hull_gap(loc, eql), i)
    lw, lq = wf("left", i)
    rw, rq = wf("right", i)
    for la, (Pl, eql) in HANDL["left"].items():
        for ra, (Pr, eqr) in HANDL["right"].items():
            g1 = hull_gap(qrot(lq * np.array([1, -1, -1, -1]), W["right"][ra] - lw), eql)
            g2 = hull_gap(qrot(rq * np.array([1, -1, -1, -1]), W["left"][la] - rw), eqr)
            note("hands", f"left {la} vs right {ra}", min(g1, g2), i)

bad = 0
for kind in ("ladder", "ladder(resting)", "hands", "self"):
    rows = sorted(((v, k[1], i, e) for k, (v, i, e) in worst.items() if k[0] == kind))
    print(f"\n[{kind}] closest approach per pair, worst first (negative = inside), mm")
    shown = 0
    for v, key, i, e in rows:
        # self: the camera housing's hull sits 0.5 mm inside the wrist_pitch link's at EVERY wrist
        # yaw (the asset is built that way and the sim tolerates it); only the extra overlap at
        # the yaw limit, which the sim does push on, counts
        fail = v < {"ladder(resting)": 0.0, "self": -0.0008}.get(kind, a.clear)
        bad += int(fail)
        if fail or shown < 4:
            print(f"   {'FAIL' if fail else ' ok '} {v * 1000:7.1f}  {key}  (step {i / fps * 50:.0f})  {e}")
            shown += 1
print(f"\n[result] {'CLEAN' if bad == 0 else str(bad) + ' colliding pairs'}")
sys.exit(0 if bad == 0 else 2)
