"""Does the arm run into the robot's own torso anywhere in a clip?

The simulator treats each link as the convex hull of its collision mesh, so this does the same:
the hull of the elbow link and of the wrist-roll link (the forearm's two halves) against the hull
of the torso link, at every `--stride`-th frame. Negative = the two hulls overlap by that depth.

    python scripts/climb/check_arm_torso.py --clip <dir> [--stride 5]
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.spatial import ConvexHull

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--stride", type=int, default=5)
ap.add_argument("--gate", type=float, default=0.0, help="smallest allowed gap, m")
a = ap.parse_args()
HERE = pathlib.Path(__file__).parent
clip = pathlib.Path(a.clip)
m = np.load(clip / "motion.npz")
meta = json.loads((clip / "meta.json").read_text())
bn, fps = list(meta["body_names"]), float(meta["fps"])
BP, BQ = m["body_pos_w"], m["body_quat_w"]
H = np.load(HERE / "g1_body_hulls.npz")


def qrot(q, v):
    w, u = q[..., :1], q[..., 1:]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def hull(name):
    V = H[name].astype(float)
    h = ConvexHull(V)
    return V[h.vertices], h.equations


TV, TEQ = hull("torso_link")
kt = bn.index("torso_link")
worst = {}
bad = {}
for i in range(0, BP.shape[0], a.stride):
    step = int(i / fps * 50)
    for side in ("left", "right"):
        for ln in ("elbow_link", "wrist_roll_link"):
            V, _ = hull(f"{side}_{ln}") if (side, ln) not in worst else worst[(side, ln)][3]
            k = bn.index(f"{side}_{ln}")
            W = BP[i, k] + qrot(BQ[i, k], V)
            # into the torso frame
            qi = BQ[i, kt] * np.array([1, -1, -1, -1.0])
            L = qrot(qi, W - BP[i, kt])
            g = float((L @ TEQ[:, :3].T + TEQ[:, 3]).max(1).min())
            w = worst.get((side, ln))
            if w is None or g < w[0]:
                worst[(side, ln)] = (g, step, None, (V, None))
            elif w[3] is None:
                pass
            if g < a.gate:
                bad.setdefault((side, ln), []).append((step, g))
ok = True
for (side, ln), (g, step, _, _) in sorted(worst.items()):
    b = bad.get((side, ln), [])
    ok &= not b
    msg = f"closest {g * 1000:+.1f} mm at step {step}"
    if b:
        st = sorted({s for s, _ in b})
        runs, r0, p = [], st[0], st[0]
        for s in st[1:]:
            if s - p > 12:
                runs.append((r0, p))
                r0 = s
            p = s
        runs.append((r0, p))
        msg += "; inside the torso during steps " + ", ".join(f"{x}-{y}" for x, y in runs)
    print(f"[{'FAIL' if b else 'PASS'}] {side} {ln:16s} vs torso: {msg}")
print("[result]", "CLEAN" if ok else "ARM INSIDE TORSO")
