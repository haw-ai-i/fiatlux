"""Grid search over fixed right-arm poses for carrying the bulb through a climb clip.
Keeps poses whose hand+bulb sphere and forearm clear the ladder boxes by --clear at every sampled
frame and stay off the robot's own torso and right thigh; ranks by how far FORWARD the hand is
(moving 4 kg of arm behind the hips shifts the body's weight back off the treads)."""

import argparse
import itertools
import pathlib
import runpy
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--clear", type=float, default=0.06)
ap.add_argument("--hand_r", type=float, default=0.085)
ap.add_argument("--hand_fwd", type=float, default=0.11)
a = ap.parse_args()
sys.argv = ["x", "--clip", a.clip, "--pose", "none:0,0,0,0,0,0,0"]
g = runpy.run_path(str(pathlib.Path(__file__).parent / "scan_carry_pose.py"))  # reuse its boxes, FK inputs
B, box_dist, PP, PQ, waist, st, step50 = (g[k] for k in ("B", "box_dist", "PP", "PQ", "waist", "st", "step50"))
lim = np.array([K.limits(t.format(s="right")) for t in K.ARM])
res = []
for sp, sr, sy, el, wr in itertools.product(
    np.linspace(-1.6, 1.2, 15), np.linspace(-1.5, 0.1, 9), (-0.6, 0.0, 0.6), np.linspace(0.0, 1.8, 7), (1.57, 0.0)
):
    q = np.array([sp, sr, sy, el, wr, 0.0, 0.0])
    if (q < lim[:, 0] + 0.03).any() or (q > lim[:, 1] - 0.03).any():
        continue
    o = K.fk(PP, PQ, waist, np.tile(q, (len(st), 1)), "right")
    w, wq, e_ = o["right_wrist_yaw_link"], o["right_wrist_yaw_link_q"], o["right_elbow_link"]
    hc = w + K.qrot(wq, np.array([a.hand_fwd, 0.0, 0.0]))
    hd = np.stack([box_dist(hc, c, h, p) for _, c, h, p in B], 1).min(1) - a.hand_r
    if hd.min() < a.clear:
        continue
    fd = (
        min(np.stack([box_dist(e_ + u * (w - e_), c, h, p) for _, c, h, p in B], 1).min() for u in (0, 0.33, 0.66, 1))
        - 0.05
    )
    if fd < 0.03:
        continue
    T_ = o["torso_p"]
    loc = K.qrot(K.qconj(PQ), hc - PP)  # hand in the pelvis frame
    x, y, z = np.median(loc, 0)
    tors = (
        (np.abs(loc[:, 1]) < 0.17 + a.hand_r)
        & (loc[:, 0] > -0.15 - a.hand_r)
        & (loc[:, 0] < 0.20 + a.hand_r)
        & (loc[:, 2] > -0.25)
        & (loc[:, 2] < 0.50)
    )
    thigh = (loc[:, 0] > -0.05) & (loc[:, 0] < 0.45) & (loc[:, 2] < 0.05) & (loc[:, 1] > -0.26)
    if tors.any() or thigh.any():
        continue
    res.append((x, y, z, hd.min(), fd, q))
print(f"{len(res)} poses clear the ladder by {a.clear * 1000:.0f} mm and the body at every frame")
res.sort(key=lambda r: -r[0])
print("hand in pelvis frame x,y,z mm | min ladder clearance hand, forearm mm | sp sr sy el wr")
for x, y, z, c, f, q in res[:10]:
    print(
        f"  {x * 1000:+5.0f} {y * 1000:+5.0f} {z * 1000:+5.0f} | {c * 1000:4.0f} {f * 1000:4.0f} | "
        + " ".join(f"{v:+.2f}" for v in q[:5])
    )
print("narrowest sideways (|y| smallest):")
for x, y, z, c, f, q in sorted(res, key=lambda r: abs(r[1]))[:5]:
    print(
        f"  {x * 1000:+5.0f} {y * 1000:+5.0f} {z * 1000:+5.0f} | {c * 1000:4.0f} {f * 1000:4.0f} | "
        + " ".join(f"{v:+.2f}" for v in q[:5])
    )
print("beside the right hip (x -150..+60, |y| < 340):")
for x, y, z, c, f, q in sorted([r for r in res if -0.15 < r[0] < 0.06 and abs(r[1]) < 0.34], key=lambda r: -r[3])[:8]:
    print(
        f"  {x * 1000:+5.0f} {y * 1000:+5.0f} {z * 1000:+5.0f} | {c * 1000:4.0f} {f * 1000:4.0f} | "
        + " ".join(f"{v:+.2f}" for v in q[:5])
    )
