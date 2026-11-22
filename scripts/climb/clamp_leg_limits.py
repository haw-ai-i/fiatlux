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
dq_ = K.qmul(np.roll(bq, -1, axis=0), K.qconj(bq)); dq_[-1] = dq_[-2]; dq_ = dq_ * np.sign(dq_[..., :1] + 1e-12)
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "motion.npz": shutil.copy(f, dst / f.name)
np.savez(dst / "motion.npz", joint_pos=q, joint_vel=np.gradient(q, 1.0 / fps, axis=0), body_pos_w=bp, body_quat_w=bq,
         body_lin_vel_w=np.gradient(bp, 1.0 / fps, axis=0), body_ang_vel_w=2.0 * dq_[..., 1:] * fps, object_contact=d["object_contact"])
print(f"[out  ] {dst}: {int((np.abs(q - d['joint_pos']).max(1) > 1e-9).sum())} of {q.shape[0]} frames changed")
