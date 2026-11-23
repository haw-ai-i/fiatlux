"""Where can the bulb arm be carried through a climb clip without meeting the ladder?

S10 climbs with the fresh bulb in the RIGHT hand, so that arm is not available as a hold. This
keeps the clip's body and left arm, puts the right arm in a fixed joint pose for the whole clip,
and reports the clearance between that arm and the ladder (the trainer's box model: treads,
platform, tray, stringers, rear legs).

The hand with the bulb is a sphere of --hand_r centred --hand_fwd ahead of the wrist (curled
fingers plus a 60 mm bulb); the forearm is a capsule of radius 0.05 from elbow to wrist.

    python scan_carry_pose.py --clip <dir>
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--hand_r", type=float, default=0.085)
ap.add_argument("--hand_fwd", type=float, default=0.11)
ap.add_argument("--pose", action="append", default=None, help="name:sp,sr,sy,el,wr,wp,wy  (repeatable)")
a = ap.parse_args()

POSES = {  # shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw
    "cradle (S10 spawn)": [-0.35, 0.00, 0.00, 0.35, 1.57, 0.0, 0.0],
    "hang at the side": [0.30, -0.25, 0.00, 0.30, 1.57, 0.0, 0.0],
    "hang, out to the side": [0.30, -0.60, 0.00, 0.30, 1.57, 0.0, 0.0],
    "tucked to the chest": [0.20, -0.10, 0.30, 1.60, 1.57, 0.0, 0.0],
    "raised out to the side": [0.00, -1.20, 0.00, 0.60, 1.57, 0.0, 0.0],
    "behind the hip": [0.90, -0.30, 0.00, 0.40, 1.57, 0.0, 0.0],
}
if a.pose:
    POSES = {p.split(":")[0]: [float(x) for x in p.split(":")[1].split(",")] for p in a.pose}

# ---- ladder as oriented boxes: centre, half sizes, pitch about y (deg) ----
TOPS = (0.2397, 0.4742, 0.7092, 0.9441)
X4 = (-0.3650, -0.2973)
DX = 0.0857
B = []


def slab(x0, x1, hw, top, bot, name):
    B.append(
        (name, np.array([(x0 + x1) / 2, 0.0, (top + bot) / 2]), np.array([(x1 - x0) / 2, hw, (top - bot) / 2]), 0.0)
    )


for k, t in enumerate(TOPS):
    slab(X4[0] - (3 - k) * DX, X4[1] - (3 - k) * DX, 0.2, t, t - 0.0356, f"tread {k + 1}")
slab(-0.3036, 0.0762, 0.2, 1.1517, 1.1517 - 0.0428, "platform")
slab(-0.0421, 0.0701, 0.247, 1.8609, 1.8105, "tray")
ang, zc = 70.08, 0.915
for sy in (1, -1):
    B.append(
        (
            f"stringer {'L' if sy > 0 else 'R'}",
            np.array([-0.598 + (zc - 0.30) * 0.3623 + 0.0405, 0.225 * sy, zc]),
            np.array([1.83 / math.sin(math.radians(ang)) / 2, 0.020, 0.0381]),
            -ang,
        )
    )
ang2, zc2 = 80.03, 0.75
for sy in (1, -1):
    B.append(
        (
            f"rear leg {'L' if sy > 0 else 'R'}",
            np.array([0.165 - (zc2 - 0.30) * 0.1755 + 0.037, 0.225 * sy, zc2]),
            np.array([1.50 / math.sin(math.radians(ang2)) / 2, 0.013, 0.03645]),
            -(180.0 - ang2),
        )
    )


def box_dist(P, c, h, pitch):
    t = math.radians(pitch)
    cs, sn = math.cos(t), math.sin(t)
    d = P - c
    L = np.stack([cs * d[..., 0] - sn * d[..., 2], d[..., 1], sn * d[..., 0] + cs * d[..., 2]], -1)  # world -> box
    q = np.abs(L) - h
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)


clip = pathlib.Path(a.clip)
d = np.load(clip / "motion.npz")
m = json.loads((clip / "meta.json").read_text())
jn, bn = m["joint_names"], m["body_names"]
st = np.arange(0, len(d["joint_pos"]), 6)
PP, PQ = d["body_pos_w"][st, bn.index("pelvis")], d["body_quat_w"][st, bn.index("pelvis")]
waist = d["joint_pos"][st][:, [jn.index(n) for n in K.WAIST]]
step50 = st / m["fps"] * 50
print(
    f"{clip.name}: {len(st)} sampled frames; bulb hand = sphere r {a.hand_r * 1000:.0f} mm, {a.hand_fwd * 1000:.0f} mm ahead of the wrist"
)
print(
    f"{'right-arm pose':26s} | hand+bulb to ladder: min mm (what, step) | frames < 30 mm | forearm min mm (what, step) | hand vs torso centre-line y, pelvis-frame x (median) mm"
)
for name, q in POSES.items():
    o = K.fk(PP, PQ, waist, np.tile(np.array(q), (len(st), 1)), "right")
    w, wq, el = o["right_wrist_yaw_link"], o["right_wrist_yaw_link_q"], o["right_elbow_link"]
    hc = w + K.qrot(wq, np.array([a.hand_fwd, 0.0, 0.0]))
    hd = np.stack([box_dist(hc, c, h, p) for _, c, h, p in B], 1) - a.hand_r
    i, j = np.unravel_index(hd.argmin(), hd.shape)
    seg = [el + u * (w - el) for u in (0.0, 0.33, 0.66, 1.0)]
    fd = np.stack([np.stack([box_dist(s_, c, h, p) for _, c, h, p in B], 1) for s_ in seg], 0).min(0) - 0.05
    fi, fj = np.unravel_index(fd.argmin(), fd.shape)
    loc = K.qrot(K.qconj(PQ), hc - PP)
    print(
        f"{name:26s} | {hd.min() * 1000:+6.0f} ({B[j][0]}, step {step50[i]:.0f}) | {int((hd.min(1) < 0.03).sum()):4d} of {len(st)} | "
        f"{fd.min() * 1000:+6.0f} ({B[fj][0]}, step {step50[fi]:.0f}) | x {np.median(loc[:, 0]) * 1000:+.0f} y {np.median(loc[:, 1]) * 1000:+.0f} z {np.median(loc[:, 2]) * 1000:+.0f}"
    )
