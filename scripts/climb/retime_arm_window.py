"""Stretch ONE arm's move over a longer stretch of a clip, and change nothing else.

For the bulb arm's reach at the mount (S10). The clip takes the right wrist from the carry pose to
its rest on tread 4's near edge in 0.6 s, peaking at 0.83 m/s; a teacher that tracks it loosely
arrives early at 1.8 m/s and the landing ends the episode (`fresh_bulb_struck` / `bulb_crushed`,
88 N on the hand in one step). This replays the arm's own joint path from `--src A B` over the
longer `--dst C B` (same end step), uniformly slowed, so every pose of the reach is one the clip
already had (to within the small offset below) and the arm ends on the same frame. Before C the arm
must already be within 3 deg of the pose it had at A (the tool checks; the offset is faded out over
the window), so start the walk-in's arm move early enough.

Only that arm's seven joints and its bodies change, only inside C..B. The pelvis is where the clip
has it at each step, so during the stretched reach the hand's WORLD path differs from the original
whenever the pelvis was still moving: rerun the clearance checks.

    python retime_arm_window.py --clip <in> --out <out> --side right --src 250 300 --dst 150 300
"""
import argparse, json, pathlib, shutil, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--side", default="right", choices=["left", "right"])
ap.add_argument("--src", type=int, nargs=2, required=True, help="50 Hz steps A B of the arm move as the clip has it")
ap.add_argument("--dst", type=int, nargs=2, required=True, help="50 Hz steps C B to play it over (C < A, same B)")
a = ap.parse_args()
src, dst = pathlib.Path(a.clip), pathlib.Path(a.out)
d = dict(np.load(src / "motion.npz")); meta = json.loads((src / "meta.json").read_text())
jn, bn, fps = list(meta["joint_names"]), list(meta["body_names"]), float(meta["fps"])
(A, B), (C, B2) = a.src, a.dst
if B2 != B or not C < A < B:
    raise SystemExit("need --dst C B with the same end step B as --src A B, and C < A < B")
h = 0 if a.side == "left" else 1
oc = d["object_contact"].reshape(len(d["object_contact"]), -1)
if (oc[C:B, h] != 0).any():
    raise SystemExit(f"the {a.side} hand is labelled on a hold inside steps {C}-{B}; not touching it")
fr = lambda s: int(round(s * fps / 50))
fA, fB, fC = fr(A), fr(B), fr(C)
J = [jn.index(n) for n in jn if n.startswith(a.side + "_") and any(k in n for k in ("shoulder", "elbow", "wrist"))]
q0 = d["joint_pos"]; q = q0.copy()
drift = float(np.degrees(np.abs(q0[fC:fA + 1][:, J] - q0[fA, J]).max()))
if drift > 3.0:
    raise SystemExit(f"between steps {C} and {A} the arm is up to {drift:.2f} deg away from its pose at step {A}: "
                     f"it must already be holding that pose from step {C} (start the walk-in's arm move earlier)")
t_dst = np.arange(fC, fB + 1)
t_src = fA + (t_dst - fC) * (fB - fA) / (fB - fC)
for j in J:
    u = (t_dst - fC) / (fB - fC); fade = 1 - u * u * u * (u * (6 * u - 15) + 10)
    # the arm at C may sit a fraction of a degree from its pose at A (a level-palm walk-in keeps solving the wrist):
    # carry that offset in at C and fade it out by B, so the window joins the clip exactly at both ends
    q[fC:fB + 1, j] = np.interp(t_src, np.arange(fA, fB + 1), q0[fA:fB + 1, j]) + (q0[fC, j] - q0[fA, j]) * fade
# a uniform slow-down has a slope step at C and at B; round both with a short moving average
k = max(3, int(round(0.06 * fps)) | 1); ker = np.ones(k) / k
for j in J:
    for c in (fC, fB):
        g = slice(c - 2 * k, c + 2 * k + 1)
        sm = np.convolve(np.pad(q[g, j], k, mode="edge"), ker, mode="same")[k:-k]
        w = np.hanning(len(sm) + 2)[1:-1]
        q[g, j] = (1 - w) * q[g, j] + w * sm
bad = [jn[j] for j in J if q[:, j].min() < K.limits(jn[j])[0] - 1e-6 or q[:, j].max() > K.limits(jn[j])[1] + 1e-6]
if bad:
    raise SystemExit(f"outside joint limits: {bad}")
bp, bq = d["body_pos_w"].copy(), d["body_quat_w"].copy()
pel = bn.index("pelvis"); e0, e1 = fC - 2 * k, fB + 2 * k; sl = slice(e0, e1 + 1)
allb = K.fk_all(bp[sl, pel], bq[sl, pel], q[sl], jn)
arm = [b for b in bn if b.startswith(a.side + "_") and any(s in b for s in ("shoulder", "elbow", "wrist"))]
chk = max(float(np.abs(allb[b][0][[0, -1]] - bp[sl, bn.index(b)][[0, -1]]).max()) for b in arm)
print(f"[fk] the window's two end frames reproduce the clip's own arm bodies to {chk * 1000:.2f} mm")
for b in arm:
    bp[sl, bn.index(b)], bq[sl, bn.index(b)] = allb[b]
lv, av, jv = d["body_lin_vel_w"].copy(), d["body_ang_vel_w"].copy(), d["joint_vel"].copy()
g = slice(e0 - 1, e1 + 2)
for j in J:
    jv[g, j] = np.gradient(q[:, j], 1.0 / fps)[g]
for b in arm:
    i = bn.index(b)
    lv[g, i] = np.gradient(bp[:, i], 1.0 / fps, axis=0)[g]
    dq = K.qmul(np.roll(bq[:, i], -1, axis=0), K.qconj(bq[:, i])); dq = dq * np.sign(dq[..., :1] + 1e-12)
    av[g, i] = (2.0 * dq[..., 1:] * fps)[g]
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "motion.npz": shutil.copy(f, dst / f.name)
out = dict(d, joint_pos=q, joint_vel=jv, body_pos_w=bp, body_quat_w=bq, body_lin_vel_w=lv, body_ang_vel_w=av)
np.savez(dst / "motion.npz", **out)
wi = bn.index(f"{a.side}_wrist_yaw_link"); W = slice(fC, fB + 1)
rel = lambda v: np.linalg.norm(v[W, wi] - v[W, pel], axis=1).max()
ch = sorted({int(x * 50 / fps) for x in np.flatnonzero((np.abs(q - q0) > 1e-9).any(1))})
print(f"[retime] {a.side} arm: steps {A}-{B} now played over {C}-{B} ({(B - A) / 50:.2f} s -> {(B - C) / 50:.2f} s)")
print(f"[speed ] wrist relative to the pelvis, peak in the window: {rel(lv):.2f} m/s (was {rel(d['body_lin_vel_w']):.2f}); "
      f"arm joint peak {np.abs(jv[W][:, J]).max():.2f} rad/s (was {np.abs(d['joint_vel'][W][:, J]).max():.2f})")
print(f"[out   ] steps changed: {ch[0]}-{ch[-1]}; last frame of the window against the clip: "
      f"{np.degrees(np.abs(q[fB + 2 * k + 1] - q0[fB + 2 * k + 1]).max()):.4f} deg -> {dst}")
