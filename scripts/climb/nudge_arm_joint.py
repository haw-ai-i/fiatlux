"""Add a small, smooth bump to ONE arm joint over a short window of a clip, and nothing else.

For a brief overlap a checker finds in the middle of a free arm move (rebuild 73: the right
wrist_roll link 3 mm inside the torso for clip steps 741-743). The bump is a raised cosine, zero
with zero slope at both ends; bodies are recomputed with g1_arm_fk for that arm only, and only
inside the window. Every other array entry is copied unchanged, and the tool refuses a window in
which that hand is labelled on a hold.

    python nudge_arm_joint.py --clip <in> --out <out> --joint right_shoulder_roll_joint --deg -4 --steps 722 764
"""
import argparse, json, pathlib, shutil, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--joint", required=True); ap.add_argument("--deg", type=float, required=True)
ap.add_argument("--steps", type=int, nargs=2, required=True, help="window in 50 Hz clip steps")
a = ap.parse_args()
src, dst = pathlib.Path(a.clip), pathlib.Path(a.out)
d = dict(np.load(src / "motion.npz")); meta = json.loads((src / "meta.json").read_text())
jn, bn, fps = list(meta["joint_names"]), list(meta["body_names"]), float(meta["fps"])
side = a.joint.split("_")[0]; h = 0 if side == "left" else 1
s0, s1 = a.steps
if (d["object_contact"][s0:s1 + 1, h] != 0).any():
    raise SystemExit(f"the {side} hand is labelled on a hold inside steps {s0}-{s1}; not touching it")
f0, f1 = int(round(s0 * fps / 50)), int(round(s1 * fps / 50))
q = d["joint_pos"].copy()
w = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(f0, f1 + 1) - f0) / (f1 - f0)))
j = jn.index(a.joint); q[f0:f1 + 1, j] += np.radians(a.deg) * w
lo, hi = K.limits(a.joint)
if q[f0:f1 + 1, j].min() < lo or q[f0:f1 + 1, j].max() > hi:
    raise SystemExit("the bump leaves the joint's limits")
bp, bq = d["body_pos_w"].copy(), d["body_quat_w"].copy()
pel = bn.index("pelvis"); sl = slice(f0, f1 + 1)
allb = K.fk_all(bp[sl, pel], bq[sl, pel], q[sl], jn)
arm = [b for b in bn if b.startswith(side + "_") and any(k in b for k in ("shoulder", "elbow", "wrist"))]
chk = max(float(np.abs(allb[b][0][[0, -1]] - bp[sl, bn.index(b)][[0, -1]]).max()) for b in arm)
print(f"[fk] window ends reproduce the clip's own arm bodies to {chk*1000:.2f} mm")
for b in arm:
    bp[sl, bn.index(b)], bq[sl, bn.index(b)] = allb[b]
lv, av, jv = d["body_lin_vel_w"].copy(), d["body_ang_vel_w"].copy(), d["joint_vel"].copy()
g = slice(max(f0 - 1, 0), f1 + 2)
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
hj = dst / "holds.json"
if hj.exists():
    hh = json.loads(hj.read_text()); hh.setdefault("nudges", []).append(dict(joint=a.joint, deg=a.deg, steps=a.steps)); hj.write_text(json.dumps(hh, indent=1))
ch = {k: int((out[k] != d[k]).reshape(len(d[k]), -1).any(1).sum()) for k in d}
print(f"[out] frames changed per array: {ch}")
print(f"[speed] {a.joint} peak in the window {np.abs(jv[g, j]).max():.2f} rad/s (was {np.abs(d['joint_vel'][g, j]).max():.2f}); whole clip arm peak "
      f"{np.abs(np.gradient(q, 1.0 / fps, axis=0)[:, [jn.index(n) for n in jn if 'shoulder' in n or 'elbow' in n or 'wrist' in n]]).max():.2f} rad/s")
