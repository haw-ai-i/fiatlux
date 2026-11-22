"""Move the pelvis back a little during chosen windows, with both feet and both wrists held where they are.

The left shin brushes the nose of the next tread as the left foot lands on each tread (up to 26 mm
in the reference; 0.7-1.8 kN in the staged scan). The foot is flat and fixed, the hip is fixed by
the pelvis, so the knee has nowhere to go: the only way to stand the shin up is to move the hip.
This shifts the pelvis by --dx (clip x, negative = away from the ladder) over each window with a
smooth ramp in and out, and re-solves all four limbs so the two ankle links and the two
wrist_yaw links keep their original world poses (damped least squares on g1_arm_fk.fk_all; the
arms have seven joints for six constraints, so the solve takes the smallest joint change). The
waist is not touched.

    python shift_pelvis_windows.py --clip <in> --out <out> --dx -0.02 --window 780 825 --window 1195 1235 ...
"""
import argparse, json, pathlib, shutil, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dx", type=float, default=-0.02)
ap.add_argument("--dz", type=float, default=0.0)
ap.add_argument("--window", type=int, nargs=2, action="append", required=True, metavar=("FROM", "TO"), help="50 Hz steps at full shift")
ap.add_argument("--ramp", type=float, default=0.5, help="seconds to ramp the shift in before FROM and out after TO")
a = ap.parse_args()
src, dst = pathlib.Path(a.clip), pathlib.Path(a.out)
d = dict(np.load(src / "motion.npz")); meta = json.loads((src / "meta.json").read_text())
jn, bn, fps = list(meta["joint_names"]), list(meta["body_names"]), float(meta["fps"])
T = d["joint_pos"].shape[0]; PEL = bn.index("pelvis")
sm = lambda u: u * u * u * (u * (6 * u - 15) + 10)
w = np.zeros(T); nr = int(a.ramp * fps); t = np.arange(T)
for s0, s1 in a.window:
    f0, f1 = int(s0 * fps / 50), int(s1 * fps / 50)
    w = np.maximum(w, sm(np.clip((t - (f0 - nr)) / nr, 0, 1)) * sm(np.clip(((f1 + nr) - t) / nr, 0, 1)))
act = np.flatnonzero(w > 1e-6); sl = slice(int(act.min()), int(act.max()) + 1)
q = d["joint_pos"].copy(); bp, bq = d["body_pos_w"].copy(), d["body_quat_w"].copy()
pp = bp[:, PEL].copy(); pp[:, 0] += a.dx * w; pp[:, 2] += a.dz * w
pq = bq[:, PEL]
LIMBS = {}
for side in ("left", "right"):
    LIMBS[f"{side} leg"] = (f"{side}_ankle_roll_link", [jn.index(f"{side}_{k}_joint") for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")])
    LIMBS[f"{side} arm"] = (f"{side}_wrist_yaw_link", [jn.index(f"{side}_{k}_joint") for k in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")])
for name, (body, js) in LIMBS.items():
    k = bn.index(body); tp, tq = bp[sl, k].copy(), bq[sl, k].copy()

    def err(qq):
        o = K.fk_all(pp[sl], pq[sl], qq, jn)[body]
        e_ = K.qmul(tq, K.qconj(o[1])); e_ = e_ * np.sign(e_[:, :1] + 1e-12)
        return np.concatenate([tp - o[0], 2.0 * e_[:, 1:]], 1)

    qs = q[sl].copy(); n = len(js)
    for it in range(80):
        e = err(qs)
        if np.abs(e).max() < 1e-5:
            break
        Jm = np.zeros((qs.shape[0], 6, n))
        for c, j in enumerate(js):
            qp = qs.copy(); qp[:, j] += 1e-4
            Jm[:, :, c] = -(err(qp) - e) / 1e-4
        JJt = np.einsum("nij,nkj->nik", Jm, Jm) + 1e-6 * np.eye(6)
        qs[:, js] += np.clip(np.einsum("nji,nj->ni", Jm, np.linalg.solve(JJt, e)), -0.2, 0.2)
    e = err(qs)
    q[sl.start:sl.stop, js] = qs[:, js]
    lim = [(jn[j], round(float(np.degrees(q[:, j].min())), 1), round(float(np.degrees(q[:, j].max())), 1)) for j in js
           if q[:, j].min() < K.limits(jn[j])[0] - 1e-6 or q[:, j].max() > K.limits(jn[j])[1] + 1e-6]
    print(f"[{name:9s}] held at its original pose to {np.abs(e[:, :3]).max()*1000:.2f} mm / {np.degrees(np.abs(e[:, 3:]).max()):.2f} deg ({it + 1} iterations); "
          f"largest joint change {np.degrees(np.abs(q[sl][:, js] - d['joint_pos'][sl][:, js]).max()):.1f} deg; outside limits: {lim or 'none'}")
allb = K.fk_all(pp, pq, q, jn)
for i, b in enumerate(bn):
    bp[:, i], bq[:, i] = allb[b]
chk = max(float(np.abs(bp[~(w > 1e-6)] - d["body_pos_w"][~(w > 1e-6)]).max()), 0.0)
qd = np.gradient(q, 1.0 / fps, axis=0)
print(f"[shift] pelvis moved {a.dx*1000:+.0f} mm in x, {a.dz*1000:+.0f} mm in z over {len(a.window)} windows ({int((w > 1e-6).sum())} frames); "
      f"frames outside the windows reproduce the clip to {chk*1000:.3f} mm; peak joint speed {np.abs(qd).max():.2f} rad/s (was {np.abs(d['joint_vel']).max():.2f})")
dq_ = K.qmul(np.roll(bq, -1, axis=0), K.qconj(bq)); dq_[-1] = dq_[-2]; dq_ = dq_ * np.sign(dq_[..., :1] + 1e-12)
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "motion.npz": shutil.copy(f, dst / f.name)
np.savez(dst / "motion.npz", joint_pos=q, joint_vel=qd, body_pos_w=bp, body_quat_w=bq,
         body_lin_vel_w=np.gradient(bp, 1.0 / fps, axis=0), body_ang_vel_w=2.0 * dq_[..., 1:] * fps, object_contact=d["object_contact"])
print(f"[out  ] {dst}")
