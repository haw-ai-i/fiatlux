"""Bring a clip's leg joints back inside their limits, and say where it mattered.

Rebuild 67's legs come from the base clip untouched, and the builder's joint-limit gate looks at
the arms only. Measured 2026-11-21: on every left step up, the left ankle pitch runs to -50.6..-56.1
deg against a -50 deg limit (steps 99-112, 537-552, 950-965, 1,127-1,142, 1,306-1,319). The real
joint stops at the limit, so the robot's foot cannot be where the reference says it is.

Each offending joint is clamped to `limit -/+ --margin` with a smooth knee (so the joint velocity
has no step), and the bodies below it in that leg are recomputed with g1_arm_fk. Everything else
is copied. The report says whether the foot was planted while it was clamped: clamping a planted
foot moves a foot that should not move, and the tool refuses unless --allow_planted is given.

    python clamp_leg_limits.py --clip <in> --out <out> [--margin 1.0]
"""
import argparse, json, pathlib, shutil, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--margin", type=float, default=1.0, help="deg inside the limit")
ap.add_argument("--knee", type=float, default=3.0, help="deg over which the clamp eases in")
ap.add_argument("--allow_planted", action="store_true")
ap.add_argument("--keep_toe", action="store_true",
                help="after clamping, lift the foot by what the toe lost, so the swing clears what it cleared before "
                     "(the leg is re-solved for the clamped foot orientation at the raised position)")
ap.add_argument("--toe", type=float, nargs=3, default=[0.135, 0.0, -0.035], help="toe point in the ankle link's frame")
a = ap.parse_args()
src, dst = pathlib.Path(a.clip), pathlib.Path(a.out)
d = dict(np.load(src / "motion.npz")); meta = json.loads((src / "meta.json").read_text())
jn, bn, fps = list(meta["joint_names"]), list(meta["body_names"]), float(meta["fps"])
q = d["joint_pos"].copy(); bp, bq = d["body_pos_w"].copy(), d["body_quat_w"].copy()
mg, kn = np.radians(a.margin), np.radians(a.knee)


def soft_clip(x, lo, hi):
    """x where it is more than `kn` inside [lo, hi]; eases onto the bound (C1) over the last `kn`."""
    y = x.copy()
    for bound, sgn in ((lo, 1.0), (hi, -1.0)):
        u = sgn * (x - bound)                               # distance inside the bound; < 0 is outside
        v = np.where(u >= kn, u, np.where(u <= -kn, 0.0, (u + kn) ** 2 / (4 * kn)))   # C1 at u = +-kn, never below 0
        y = np.where(u < kn, bound + sgn * v, y)
    return y


changed = []
for i, n in enumerate(jn):
    if not any(s in n for s in ("hip", "knee", "ankle")):
        continue
    lo, hi = K.limits(n)
    if q[:, i].min() >= lo - 1e-9 and q[:, i].max() <= hi + 1e-9:
        continue
    out = np.flatnonzero((q[:, i] < lo) | (q[:, i] > hi))
    q[:, i] = soft_clip(q[:, i], lo + mg, hi - mg)
    side = n.split("_")[0]
    foot = bn.index(f"{side}_ankle_roll_link")
    z = d["body_pos_w"][:, foot, 2]
    moving = np.linalg.norm(d["body_lin_vel_w"][:, foot], axis=1) > 0.05
    planted = int((~moving[out]).sum())
    br = np.flatnonzero(np.diff(out) > 1); st = np.r_[out[0], out[br + 1]]; en = np.r_[out[br], out[-1]]
    print(f"[limit] {n}: outside {np.degrees(lo):.0f}..{np.degrees(hi):.0f} deg on {len(out)} frames, worst {np.degrees(d['joint_pos'][:, i].min()):.1f}/{np.degrees(d['joint_pos'][:, i].max()):.1f}; "
          f"steps " + ", ".join(f"{int(s_ * 50 / fps)}-{int(e_ * 50 / fps)}" for s_, e_ in zip(st, en)) + f"; foot planted on {planted} of those frames")
    if planted and not a.allow_planted:
        raise SystemExit("the foot is planted while the joint is outside its limit: clamping would move a planted foot (--allow_planted to do it anyway)")
    changed.append((i, n, side))
if not changed:
    print("[limit] every leg joint is inside its limits; nothing to do")
PEL = bn.index("pelvis")
allb = K.fk_all(bp[:, PEL], bq[:, PEL], q, jn)
for i, n, side in changed:
    for k, b in enumerate(bn):
        if b.startswith(side + "_") and any(s in b for s in ("hip", "knee", "ankle")):
            bp[:, k], bq[:, k] = allb[b]
    foot = bn.index(f"{side}_ankle_roll_link")
    mv = np.linalg.norm(bp[:, foot] - d["body_pos_w"][:, foot], axis=1)
    print(f"[limit] {n}: now {np.degrees(q[:, i].min()):.1f}..{np.degrees(q[:, i].max()):.1f} deg; the {side} ankle link moves by at most {mv.max()*1000:.1f} mm; "
          f"joint speed peak {np.abs(np.gradient(q[:, i], 1.0 / fps)).max():.2f} rad/s (was {np.abs(d['joint_vel'][:, i]).max():.2f})")
if a.keep_toe:
    TOE = np.array(a.toe)
    for i, n, side in changed:
        leg = [jn.index(f"{side}_{k}_joint") for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
        fb = f"{side}_ankle_roll_link"; k = bn.index(fb)
        toe0 = d["body_pos_w"][:, k, 2] + K.qrot(d["body_quat_w"][:, k], TOE[None])[:, 2]
        toe1 = bp[:, k, 2] + K.qrot(bq[:, k], TOE[None])[:, 2]
        dz = np.clip(toe0 - toe1, 0.0, None)
        w = int(0.08 * fps); ker = np.hanning(2 * w + 1); ker /= ker.sum()
        dz = np.maximum(np.convolve(dz, ker, mode="same"), dz * 0.0)       # smooth in and out
        dz = np.maximum(dz, np.clip(toe0 - toe1, 0.0, None) * 0.0)
        act = np.flatnonzero(dz > 1e-5)
        if not len(act):
            continue
        sl = slice(max(act.min() - 2, 0), min(act.max() + 3, q.shape[0]))
        tp = bp[sl, k].copy(); tp[:, 2] += dz[sl]; tq = bq[sl, k].copy()
        pp, pq = bp[sl, PEL], bq[sl, PEL]

        def err(qq):
            o = K.fk_all(pp, pq, qq, jn)[fb]
            e_ = K.qmul(tq, K.qconj(o[1])); e_ = e_ * np.sign(e_[:, :1] + 1e-12)
            return np.concatenate([tp - o[0], 2.0 * e_[:, 1:]], 1)

        qs = q[sl].copy()
        for it in range(60):
            e = err(qs)
            if np.abs(e).max() < 1e-5:
                break
            Jm = np.zeros((qs.shape[0], 6, 6))
            for c, j in enumerate(leg):
                qp = qs.copy(); qp[:, j] += 1e-4
                Jm[:, :, c] = -(err(qp) - e) / 1e-4
            qs[:, leg] += np.clip(np.linalg.solve(np.einsum("nij,nik->njk", Jm, Jm) + 1e-6 * np.eye(6), np.einsum("nij,ni->nj", Jm, e)), -0.2, 0.2)
        q[sl][:, leg] = qs[:, leg]; q[sl.start:sl.stop, leg] = qs[:, leg]
        allb = K.fk_all(bp[:, PEL], bq[:, PEL], q, jn)
        for kk, b in enumerate(bn):
            if b.startswith(side + "_") and any(s_ in b for s_ in ("hip", "knee", "ankle")):
                bp[:, kk], bq[:, kk] = allb[b]
        toe2 = bp[:, k, 2] + K.qrot(bq[:, k], TOE[None])[:, 2]
        lo, hi = K.limits(n)
        worst = {m_: (float(np.degrees(q[:, jn.index(m_)].min())), float(np.degrees(q[:, jn.index(m_)].max()))) for m_ in [jn[j] for j in leg]
                 if q[:, jn.index(m_)].min() < K.limits(m_)[0] - 1e-6 or q[:, jn.index(m_)].max() > K.limits(m_)[1] + 1e-6}
        print(f"[toe  ] {side}: foot lifted by up to {dz.max()*1000:.1f} mm over {len(act)} frames, leg re-solved ({it + 1} iterations, error {np.abs(err(qs)[:, :3]).max()*1000:.2f} mm); "
              f"toe now at most {np.clip(toe0 - toe2, 0, None).max()*1000:.1f} mm below where the clip had it (was {np.clip(toe0 - toe1, 0, None).max()*1000:.1f}); "
              f"{n} {np.degrees(q[:, i].min()):.1f}..{np.degrees(q[:, i].max()):.1f} deg; leg joints outside limits: {worst or 'none'}")
dq_ = K.qmul(np.roll(bq, -1, axis=0), K.qconj(bq)); dq_[-1] = dq_[-2]; dq_ = dq_ * np.sign(dq_[..., :1] + 1e-12)
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "motion.npz": shutil.copy(f, dst / f.name)
np.savez(dst / "motion.npz", joint_pos=q, joint_vel=np.gradient(q, 1.0 / fps, axis=0), body_pos_w=bp, body_quat_w=bq,
         body_lin_vel_w=np.gradient(bp, 1.0 / fps, axis=0), body_ang_vel_w=2.0 * dq_[..., 1:] * fps, object_contact=d["object_contact"])
print(f"[out  ] {dst}: {int((np.abs(q - d['joint_pos']).max(1) > 1e-9).sum())} of {q.shape[0]} frames changed")
