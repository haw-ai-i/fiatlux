"""Do the shins or thighs sit inside the ladder anywhere in a clip?

Each shin (knee_link) and thigh (hip_yaw_link) is taken as the convex hull of its collision mesh,
as the simulator does, and tested against a dense sampling of the ladder's surface. Only ladder
parts more than 10 cm above that leg's own ankle count, so a foot resting on its tread is not
reported. Depth is how far the deepest ladder sample sits inside the hull.

    python scripts/climb/check_leg_collision.py --clip <dir> --mesh <ladder_mesh.npz> [--detail]
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--mesh", required=True, help="ladder collision triangles, npz with T [n,3,3] in the ladder frame")
ap.add_argument("--stride", type=int, default=6)
ap.add_argument("--detail", action="store_true", help="list every episode")
a = ap.parse_args()
clip = pathlib.Path(a.clip)
m = np.load(clip / "motion.npz")
meta = json.loads((clip / "meta.json").read_text())
bn, fps = list(meta["body_names"]), float(meta["fps"])
BP, BQ = m["body_pos_w"], m["body_quat_w"]
H = np.load(pathlib.Path(__file__).parent / "g1_body_hulls.npz")
TRI = np.load(a.mesh)["T"].reshape(-1, 3)
TRI = np.stack([TRI[:, 1] - 0.1996, -TRI[:, 0], TRI[:, 2]], 1).reshape(-1, 3, 3)  # ladder -> clip frame
lp = []
for A, B, C in TRI:
    n = int(max(np.linalg.norm(B - A), np.linalg.norm(C - A), np.linalg.norm(C - B)) / 0.005) + 1
    u, v = np.meshgrid(np.arange(n + 1) / n, np.arange(n + 1) / n)
    k = (u + v) <= 1
    lp.append(A + u[k][:, None] * (B - A) + v[k][:, None] * (C - A))
LAD = np.unique(np.round(np.concatenate(lp), 3), axis=0)
tree = cKDTree(LAD)


def qrot(q, v):
    w, u = q[..., :1], q[..., 1:]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


clean = True
for ln in ("left_knee_link", "right_knee_link", "left_hip_yaw_link", "right_hip_yaw_link"):
    V = H[ln].astype(float)
    eq = ConvexHull(V).equations
    k = bn.index(ln)
    ka = bn.index(ln.split("_")[0] + "_ankle_roll_link")
    r = np.linalg.norm(V, axis=1).max()
    ev = []
    for i in range(0, BP.shape[0], a.stride):
        idx = [j for j in tree.query_ball_point(BP[i, k], r) if LAD[j, 2] > BP[i, ka, 2] + 0.10]
        if not idx:
            continue
        L = qrot(BQ[i, k] * np.array([1, -1, -1, -1.0]), LAD[idx] - BP[i, k])
        dmin = float((L @ eq[:, :3].T + eq[:, 3]).max(1).min())
        if dmin < -0.003:
            ev.append((int(i / fps * 50), -dmin * 1000))
    dd = np.array([e[1] for e in ev]) if ev else np.zeros(0)
    sec = lambda th: (dd > th).sum() * a.stride / fps
    clean &= len(dd) == 0
    print(
        f"[{'PASS' if len(dd) == 0 else 'FAIL'}] {ln:20s} seconds deeper than 3 mm {sec(3):.1f}, 10 mm {sec(10):.1f}, "
        f"20 mm {sec(20):.1f}; deepest {dd.max() if len(dd) else 0:.0f} mm"
    )
    if a.detail and ev:
        runs = []
        for s_, dp in ev:
            if runs and s_ - runs[-1][1] <= 8:
                runs[-1][1] = s_
                runs[-1][2] = max(runs[-1][2], dp)
            else:
                runs.append([s_, s_, dp])
        for s0, s1, dp in runs:
            print(f"       steps {s0}-{s1}: up to {dp:.0f} mm")
print("[result]", "CLEAN" if clean else "LEG INSIDE LADDER")
