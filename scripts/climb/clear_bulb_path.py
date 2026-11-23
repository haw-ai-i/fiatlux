"""Give the carried bulb more room from the ladder along a bulb clip, where the hand is free to move.

check_bulb_clearance.py measures the bulb's glass against the ladder with the bulb at a seat in the
right wrist's frame. Measured with the hand closed to curl 0.9 (physics_segment_test_bulb.py
--follow_root), the bulb rides 22 mm higher out of the palm than the benchmark's loose-cup seat:
root at (0.112, 0.076, -0.128). With that seat `bulb20` has the glass 2 mm INSIDE the platform's
front lip on the approach to the tread-4 rest and 8-10 mm from it just before and after the rest, and
the bulb touched the ladder there in physics (up to 23 N).

This moves the right WRIST, orientation unchanged, straight away from the ladder's nearest point by
whatever the glass is short of `--target`, in frames where the right hand is NOT labelled on a rest,
fading to nothing over `--taper` steps next to a rest so the rest poses themselves are untouched.
The seven arm joints are solved for the moved wrist (position and orientation, damped least
squares, all frames at once). Only the right arm changes, only in the frames printed.

What it cannot do: a rest whose own pose leaves the glass under `--target` stays as it is. Those are
reported at the end.

    python clear_bulb_path.py --clip <in> --out <out> --mesh <ladder_mesh.npz> [--target 0.030]
"""

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mesh", required=True)
ap.add_argument(
    "--seat",
    type=float,
    nargs=3,
    default=[0.112, 0.076, -0.128],
    help="bulb root in the right wrist_yaw_link frame, m (default: measured at curl 0.9)",
)
ap.add_argument("--target", type=float, default=0.030, help="clearance wanted between the glass and the ladder, m")
ap.add_argument(
    "--taper", type=int, default=20, help="50 Hz steps next to a rest over which the correction fades to zero"
)
ap.add_argument("--smooth", type=float, default=6.0, help="50 Hz steps of smoothing on the correction")
ap.add_argument("--max_shift", type=float, default=0.06, help="largest wrist displacement allowed, m")
a = ap.parse_args()
src, dst = pathlib.Path(a.clip), pathlib.Path(a.out)
d = dict(np.load(src / "motion.npz"))
meta = json.loads((src / "meta.json").read_text())
jn, bn, fps = list(meta["joint_names"]), list(meta["body_names"]), float(meta["fps"])
per = fps / 50.0
N = d["joint_pos"].shape[0]
TRI = np.load(a.mesh)["T"].reshape(-1, 3)
TRI = np.stack([TRI[:, 1] - 0.1996, -TRI[:, 0], TRI[:, 2]], 1).reshape(-1, 3, 3)  # ladder -> clip frame
lp = []
for A_, B_, C_ in TRI:
    n = int(max(np.linalg.norm(B_ - A_), np.linalg.norm(C_ - A_), np.linalg.norm(C_ - B_)) / 0.005) + 1
    u, v = np.meshgrid(np.arange(n + 1) / n, np.arange(n + 1) / n)
    k = (u + v) <= 1
    lp.append(A_ + u[k][:, None] * (B_ - A_) + v[k][:, None] * (C_ - A_))
LAD = np.unique(np.round(np.concatenate(lp), 3), axis=0)
tree = cKDTree(LAD)
GLASS = np.array(a.seat) + np.array([0.0, 0.0, 0.131])
R_GLASS = 0.040
w = bn.index("right_wrist_yaw_link")
pel = bn.index("pelvis")
J = [
    jn.index(f"right_{k}_joint")
    for k in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
]
lab = d["object_contact"].reshape(len(d["object_contact"]), -1)[:, 1] != 0
lab_f = lab[np.clip((np.arange(N) / per).astype(int), 0, len(lab) - 1)]
# weight: 0 on a rest, rising to 1 over the taper away from it
dist = np.full(N, 1e9)
idx = np.flatnonzero(lab_f)
if len(idx):
    dist = (
        np.abs(np.arange(N)[:, None] - idx[None, np.searchsorted(idx, np.arange(N)).clip(0, len(idx) - 1)])
        .min(1)
        .astype(float)
    )
    dist = np.minimum(
        dist,
        np.abs(np.arange(N)[:, None] - idx[None, (np.searchsorted(idx, np.arange(N)) - 1).clip(0, len(idx) - 1)]).min(
            1
        ),
    )
u = np.clip(dist / (a.taper * per), 0, 1)
W = u * u * (3 - 2 * u)


def margin(P, Q):
    g = P + K.qrot(Q, np.tile(GLASS, (len(P), 1)))
    dd, jj = tree.query(g)
    return dd - R_GLASS, g - LAD[jj]


q0 = d["joint_pos"].copy()
q = q0.copy()
bp, bq = d["body_pos_w"], d["body_quat_w"]
Pw0, Qw0 = bp[:, w].copy(), bq[:, w].copy()
m0, _ = margin(Pw0, Qw0)
target = Pw0.copy()
for it in range(6):  # the direction changes as the wrist moves: a few passes
    o = K.fk_all(bp[:, pel], bq[:, pel], q, jn)["right_wrist_yaw_link"]
    m_, n_ = margin(o[0], o[1])
    need = np.clip(a.target - m_, 0, None) * W
    step = need[:, None] * n_ / (np.linalg.norm(n_, axis=1, keepdims=True) + 1e-9)
    step = gaussian_filter1d(step, a.smooth * per, axis=0, mode="nearest") * (W[:, None] > 0)
    if np.abs(step).max() < 2e-4:
        break
    target = target + step
    sh = target - Pw0
    nrm = np.linalg.norm(sh, axis=1, keepdims=True)
    target = Pw0 + sh * np.minimum(1.0, a.max_shift / (nrm + 1e-9))
    rows = np.flatnonzero(np.linalg.norm(target - o[0], axis=1) > 1e-5)
    for _ in range(25):  # arm IK for the moved wrist, orientation held
        o = K.fk_all(bp[rows, pel], bq[rows, pel], q[rows], jn)["right_wrist_yaw_link"]
        dq_ = K.qmul(Qw0[rows], K.qconj(o[1]))
        dq_ = dq_ * np.sign(dq_[:, :1] + 1e-12)
        e = np.concatenate([target[rows] - o[0], 2.0 * dq_[:, 1:] * 0.1], 1)  # [n, 6]; 0.1 m lever on the rotation
        if np.abs(e).max() < 2e-5:
            break
        Jm = np.zeros((len(rows), 6, 7))
        for c, j in enumerate(J):
            qd = q[rows].copy()
            qd[:, j] += 1e-5
            o2 = K.fk_all(bp[rows, pel], bq[rows, pel], qd, jn)["right_wrist_yaw_link"]
            dq2 = K.qmul(Qw0[rows], K.qconj(o2[1]))
            dq2 = dq2 * np.sign(dq2[:, :1] + 1e-12)
            Jm[:, :, c] = (np.concatenate([target[rows] - o2[0], 2.0 * dq2[:, 1:] * 0.1], 1) - e) / 1e-5
        JT = Jm.transpose(0, 2, 1)
        dj = np.linalg.solve(JT @ Jm + 1e-6 * np.eye(7)[None], -(JT @ e[:, :, None]))[:, :, 0]
        for c, j in enumerate(J):
            lo, hi = K.limits(jn[j])
            q[rows, j] = np.clip(q[rows, j] + dj[:, c], lo + 1e-4, hi - 1e-4)
o = K.fk_all(bp[:, pel], bq[:, pel], q, jn)
m1, _ = margin(o["right_wrist_yaw_link"][0], o["right_wrist_yaw_link"][1])
dq_ = K.qmul(Qw0, K.qconj(o["right_wrist_yaw_link"][1]))
ori = np.degrees(2 * np.abs(dq_[:, 1:]).max())
ch = np.flatnonzero(np.abs(q - q0).max(1) > 1e-7)
if not len(ch):
    raise SystemExit(
        f"nothing to do: the glass is never under {a.target * 1000:.0f} mm off a rest (closest {m0.min() * 1000:.1f} mm)"
    )
arm = [b for b in bn if b.startswith("right_") and any(s in b for s in ("shoulder", "elbow", "wrist"))]
nbp, nbq = bp.copy(), bq.copy()
for b in arm:
    nbp[:, bn.index(b)], nbq[:, bn.index(b)] = o[b]
keep = np.setdiff1d(np.arange(N), ch)
chk = max(float(np.abs(nbp[keep][:, bn.index(b)] - bp[keep][:, bn.index(b)]).max()) for b in arm) if len(keep) else 0.0
lv, av, jv = d["body_lin_vel_w"].copy(), d["body_ang_vel_w"].copy(), d["joint_vel"].copy()
g = np.zeros(N, bool)
g[np.clip(np.concatenate([ch - 1, ch, ch + 1]), 0, N - 1)] = True
for j in J:
    jv[g, j] = np.gradient(q[:, j], 1.0 / fps)[g]
for b in arm:
    i = bn.index(b)
    lv[g, i] = np.gradient(nbp[:, i], 1.0 / fps, axis=0)[g]
    dqq = K.qmul(np.roll(nbq[:, i], -1, axis=0), K.qconj(nbq[:, i]))
    dqq = dqq * np.sign(dqq[..., :1] + 1e-12)
    av[g, i] = (2.0 * dqq[..., 1:] * fps)[g]
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "motion.npz":
        shutil.copy(f, dst / f.name)
np.savez(
    dst / "motion.npz",
    **dict(d, joint_pos=q, joint_vel=jv, body_pos_w=nbp, body_quat_w=nbq, body_lin_vel_w=lv, body_ang_vel_w=av),
)
st = lambda f: f / per
sh = np.linalg.norm(o["right_wrist_yaw_link"][0] - Pw0, axis=1)
e_ = np.flatnonzero(np.diff(np.r_[0, (sh > 5e-4).astype(int), 0]))
runs = [(st(e_[i]), st(e_[i + 1] - 1), sh[e_[i] : e_[i + 1]].max() * 1000) for i in range(0, len(e_), 2)]
print(f"[seat  ] bulb root {a.seat} in the wrist frame; glass clearance wanted {a.target * 1000:.0f} mm")
print(
    "[moved ] wrist moved in steps "
    + ", ".join(f"{x:.0f}-{y:.0f} (up to {z:.0f} mm)" for x, y, z in runs)
    + f"; orientation held to {ori:.3f} deg"
)
free = ~lab_f
print(
    f"[margin] off the rests: closest {m0[free].min() * 1000:+.1f} mm at step {st(np.flatnonzero(free)[m0[free].argmin()]):.0f} -> {m1[free].min() * 1000:+.1f} mm at step {st(np.flatnonzero(free)[m1[free].argmin()]):.0f}"
)
if lab_f.any():
    print(
        f"[margin] ON the rests (untouched): closest {m1[lab_f].min() * 1000:+.1f} mm at step {st(np.flatnonzero(lab_f)[m1[lab_f].argmin()]):.0f}"
        + ("  <-- under the target: the rest pose itself is too close" if m1[lab_f].min() < a.target else "")
    )
bad = [jn[j] for j in J if q[:, j].min() < K.limits(jn[j])[0] - 1e-6 or q[:, j].max() > K.limits(jn[j])[1] + 1e-6]
print(
    f"[check ] joint limits: {'PASS' if not bad else 'FAIL ' + str(bad)}; unchanged frames reproduce the clip's arm bodies to {chk * 1000:.3f} mm; "
    f"arm joint speed peak {np.abs(jv[:, J]).max():.2f} rad/s (was {np.abs(d['joint_vel'][:, J]).max():.2f}); largest joint change {np.degrees(np.abs(q - q0).max()):.1f} deg"
)
print(f"[out   ] {dst}")
