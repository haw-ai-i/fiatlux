"""How close does the carried bulb come to the ladder along a bulb clip?

The bulb is placed where the benchmark seats it in the right hand (measured in S10 by
measure_bulb_in_hand.py: root at (0.1192, 0.0540, -0.1310) in the right wrist_yaw_link frame, axis
along the frame's +z) and modelled as its two parts from grasp_poses.py: the glass, a 40 mm sphere
131 mm along the axis, and the cap, a 21 mm cylinder from the root to the glass. Each is measured
against a 5 mm sampling of the ladder's collision triangles, so numbers are good to about 4 mm.
This is the REFERENCE pose: it says how much tracking error the bulb can take, not what a
sagging or swaying robot does.

    python check_bulb_clearance.py --clip <dir> --mesh <ladder_mesh.npz> [--warn 0.02]
"""
import argparse, json, pathlib
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--mesh", required=True)
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--warn", type=float, default=0.02, help="list steps with less clearance than this, m")
ap.add_argument("--seat", type=float, nargs=3, default=[0.1192, 0.0540, -0.1310])
a = ap.parse_args()
clip = pathlib.Path(a.clip)
m = np.load(clip / "motion.npz")
meta = json.loads((clip / "meta.json").read_text())
bn, fps = list(meta["body_names"]), float(meta["fps"])
BP, BQ = m["body_pos_w"], m["body_quat_w"]
TRI = np.load(a.mesh)["T"].reshape(-1, 3)
TRI = np.stack([TRI[:, 1] - 0.1996, -TRI[:, 0], TRI[:, 2]], 1).reshape(-1, 3, 3)   # ladder -> clip frame
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


def part(p):
    side = "right" if p[1] < -0.15 else "left" if p[1] > 0.15 else "centre"
    return f"{side} x {p[0]:+.3f} y {p[1]:+.3f} z {p[2]:.3f}"


w = bn.index("right_wrist_yaw_link")
seat = np.array(a.seat)
glass = seat + np.array([0, 0, 0.131])
cap = [seat + np.array([0, 0, z]) for z in np.arange(0.0, 0.0911, 0.013)]
rows = []
for i in range(0, BP.shape[0], a.stride):
    g = BP[i, w] + qrot(BQ[i, w], glass)
    dg, jg = tree.query(g)
    c = BP[i, w] + qrot(BQ[i, w][None], np.array(cap))
    dc, jc = tree.query(c)
    kc = int(np.argmin(dc))
    rows.append((i / fps * 50, dg - 0.040, LAD[jg], dc[kc] - 0.021, LAD[jc[kc]]))
st = np.array([r[0] for r in rows]); G = np.array([r[1] for r in rows]); C = np.array([r[3] for r in rows])
oc = m["object_contact"][:, 1]
print(f"[bulb] {len(rows)} frames. Glass closest {G.min()*1000:+.1f} mm at step {st[G.argmin()]:.0f} ({part(rows[int(G.argmin())][2])}); "
      f"cap closest {C.min()*1000:+.1f} mm at step {st[C.argmin()]:.0f} ({part(rows[int(C.argmin())][4])})")
for lo in range(0, int(st.max()) + 1, 50):
    k = (st >= lo) & (st < lo + 50)
    if not k.any():
        continue
    ig = int(np.flatnonzero(k)[G[k].argmin()]); ic = int(np.flatnonzero(k)[C[k].argmin()])
    lab = sorted(set(int(x) for x in oc[lo:lo + 50]))
    flag = "  <-- under %.0f mm" % (a.warn * 1000) if min(G[ig], C[ic]) < a.warn else ""
    print(f"[bulb] steps {lo:4d}-{lo+49:4d} right-hand label {lab}: glass {G[ig]*1000:+6.1f} mm ({part(rows[ig][2])}); cap {C[ic]*1000:+6.1f} mm ({part(rows[ic][4])}){flag}")
print("[result] " + ("BULB INSIDE THE LADDER" if min(G.min(), C.min()) < 0 else f"closest {min(G.min(), C.min())*1000:.1f} mm"))
