"""Make the last step onto the platform LAND where the clip ends, instead of landing short and sliding.

The base clip's top-out is a blend: each foot comes down with its ankle about 5 cm outside the
platform's open edge, and over the clip's last 2 s both planted feet travel 15 cm along the
platform at constant height. A robot cannot slide a loaded foot, so a trained teacher stays where
the feet landed -- 7 cm of sole on the platform, heels over the edge -- and falls behind the
reference by exactly that slide.

For each foot this finds its last swing (ankle from below the platform to standing on it) and
rewrites the foot's path from the moment the sole is above the platform: it travels to the
clip's FINAL foot pose while still in the air (kept above --clear until it is over its spot),
touches down there, and stays. The pelvis, waist and arms are the clip's own, untouched; only
that leg's six joints are re-solved (damped least squares on g1_arm_fk.fk_all) and that leg's
bodies recomputed.

    python fix_topout_landing.py --clip <in> --out <out> [--platform_top 1.152]
"""

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--platform_top", type=float, default=1.152)
ap.add_argument("--sole_drop", type=float, default=0.035, help="ankle link above the sole's underside, m")
ap.add_argument(
    "--clear", type=float, default=0.03, help="sole kept this far above the platform while it travels over it, m"
)
ap.add_argument("--settle", type=float, default=0.25, help="seconds to come down onto the final spot")
ap.add_argument(
    "--travel",
    type=float,
    default=0.30,
    help="least seconds for the travel over the platform; the clip's own swing is used when it leaves more",
)
ap.add_argument(
    "--land_shift",
    type=float,
    nargs=2,
    default=[0.0, 0.0],
    metavar=("LEFT", "RIGHT"),
    help="m added to each foot's landing x (negative = nearer the open edge than the clip's final stance)",
)
ap.add_argument(
    "--low_travel",
    action="store_true",
    help="travel over the platform at the clearance height instead of at the clip's own swing height: with the "
    "foot far forward AND high, the thigh is driven into the pelvis (6.3 kN of self-contact at the left step)",
)
ap.add_argument(
    "--end_at",
    type=int,
    default=None,
    help="drop the clip after this 50 Hz step first (its straight-legged ending cannot reach a landing nearer the edge; a tail tool carries on from here)",
)
ap.add_argument(
    "--check_to",
    type=int,
    default=None,
    help="report joint limits only up to this 50 Hz step (when a later tool cuts the clip there)",
)
a = ap.parse_args()
src, dst = pathlib.Path(a.clip), pathlib.Path(a.out)
d = dict(np.load(src / "motion.npz"))
meta = json.loads((src / "meta.json").read_text())
jn, bn, fps = list(meta["joint_names"]), list(meta["body_names"]), float(meta["fps"])
if a.end_at is not None:
    _f = int(round(a.end_at * fps / 50)) + 1
    d = {k: (v[: a.end_at + 1] if k == "object_contact" else v[:_f]) for k, v in d.items()}
    print(f"[cut  ] clip ends after step {a.end_at}")
J = {n: i for i, n in enumerate(jn)}
PEL = bn.index("pelvis")
T = d["joint_pos"].shape[0]
_full = np.load(src / "motion.npz")
FINAL_P, FINAL_Q = _full["body_pos_w"][-1].copy(), _full["body_quat_w"][-1].copy()
q = d["joint_pos"].copy()
bp, bq = d["body_pos_w"].copy(), d["body_quat_w"].copy()
pel_p, pel_q = bp[:, PEL].copy(), bq[:, PEL].copy()
sm = lambda u: u * u * u * (u * (6 * u - 15) + 10)
stand_z = a.platform_top + a.sole_drop  # ankle height when the sole is on the platform
air_z = stand_z + a.clear
step = lambda f: f * 50.0 / fps

for side in ("left", "right"):
    leg = [J[f"{side}_{k}_joint"] for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    fb = f"{side}_ankle_roll_link"
    k = bn.index(fb)
    z = bp[:, k, 2]
    below = np.flatnonzero(z < stand_z - 0.10)  # last frame the foot is still down on tread 4
    f_up = int(below.max())
    f_clear = f_up + int(np.argmax(z[f_up:] >= air_z))  # sole first clear of the platform
    f_peak = f_clear + int(np.argmax(z[f_clear : f_clear + int(1.0 * fps)]))
    f_down = f_peak + int(
        np.argmax(z[f_peak:] <= FINAL_P[k, 2] + 0.006)
    )  # the clip's own touchdown: first time back at standing height
    n_set = int(round(a.settle * fps))
    f_arrive = max(f_down - n_set, f_clear + int(round(a.travel * fps)))
    f_touch = f_arrive + n_set
    p0, q0 = bp[f_clear, k].copy(), bq[f_clear, k].copy()
    p1, q1 = FINAL_P[k].copy(), FINAL_Q[k].copy()
    p1[0] += a.land_shift[0 if side == "left" else 1]
    slid = np.linalg.norm((FINAL_P[k] - bp[f_down, k])[:2])
    print(
        f"[{side:5s}] last swing: leaves tread 4 at step {step(f_up):.0f}, sole clear of the platform at {step(f_clear):.0f}, "
        f"the clip puts it down at {step(f_down):.0f} at ({bp[f_down, k, 0]:+.3f}, {bp[f_down, k, 1]:+.3f}) and then slides it "
        f"{slid * 1000:.0f} mm to ({p1[0]:+.3f}, {p1[1]:+.3f})"
    )
    if q0 @ q1 < 0:
        q1 = -q1
    fp, fq = bp[:, k].copy(), bq[:, k].copy()
    n_tr = f_arrive - f_clear
    for i in range(f_clear, T):
        if i <= f_arrive:
            w = sm((i - f_clear) / max(n_tr, 1))
            fp[i, :2] = p0[:2] + w * (p1[:2] - p0[:2])
            fp[i, 2] = max(z[i], air_z) if i < f_arrive else max(min(z[i], z[f_clear : f_arrive + 1].max()), air_z)
            if a.low_travel:
                fp[i, 2] = air_z
        elif i <= f_touch:
            w = sm((i - f_arrive) / max(n_set, 1))
            fp[i, :2] = p1[:2]
            fp[i, 2] = fp[f_arrive, 2] + w * (p1[2] - fp[f_arrive, 2])
        else:
            fp[i] = p1
        wq = sm(min(1.0, (i - f_clear) / max(n_tr, 1)))
        qq = (1 - wq) * q0 + wq * q1
        fq[i] = qq / np.linalg.norm(qq)
    print(
        f"[{side:5s}] now: travels {np.linalg.norm((p1 - p0)[:2]) * 1000:.0f} mm over the platform in steps {step(f_clear):.0f}-{step(f_arrive):.0f} "
        f"(peak {np.abs(np.gradient(fp[f_clear : f_arrive + 1, :2], 1.0 / fps, axis=0)).max():.2f} m/s), down by step {step(f_touch):.0f}, planted from there"
    )
    sl = slice(f_clear, T)

    def err(qq):
        o = K.fk_all(pel_p[sl], pel_q[sl], qq, jn)[fb]
        dq = K.qmul(fq[sl], K.qconj(o[1]))
        dq = dq * np.sign(dq[:, :1] + 1e-12)
        return np.concatenate([fp[sl] - o[0], 2.0 * dq[:, 1:]], 1)

    qs = q[sl].copy()
    for it in range(80):
        e = err(qs)
        if np.abs(e).max() < 1e-5:
            break
        Jm = np.zeros((qs.shape[0], 6, 6))
        for c, j in enumerate(leg):
            qp = qs.copy()
            qp[:, j] += 1e-4
            Jm[:, :, c] = -(err(qp) - e) / 1e-4
        dq_ = np.linalg.solve(np.einsum("nij,nik->njk", Jm, Jm) + 1e-6 * np.eye(6), np.einsum("nij,ni->nj", Jm, e))
        qs[:, leg] += np.clip(dq_, -0.2, 0.2)
    e = err(qs)
    print(
        f"[{side:5s}] leg solved in {it + 1} iterations: worst foot error {np.abs(e[:, :3]).max() * 1000:.2f} mm, {np.degrees(np.abs(e[:, 3:]).max()):.2f} deg; "
        f"knee {np.degrees(qs[:, leg[3]].min()):.0f}..{np.degrees(qs[:, leg[3]].max()):.0f} deg"
    )
    q[sl][:, leg] = qs[:, leg]
    q[f_clear:, leg] = qs[:, leg]

_chg = np.flatnonzero(np.abs(q - d["joint_pos"]).max(1) > 1e-9)
_lo = int(_chg.min()) if len(_chg) else 0
_hi = T if a.check_to is None else int(round(a.check_to * fps / 50)) + 1
bad = [
    (n, round(float(np.degrees(q[_lo:_hi, i].min())), 1), round(float(np.degrees(q[_lo:_hi, i].max())), 1))
    for i, n in enumerate(jn)
    if q[_lo:_hi, i].min() < K.limits(n)[0] - 1e-6 or q[_lo:_hi, i].max() > K.limits(n)[1] + 1e-6
]
print(
    f"[gate ] joint limits over the frames this changes (steps {step(_lo):.0f}-{step(_hi - 1):.0f}): "
    + ("PASS" if not bad else f"FAIL {bad}")
)
_was = [
    (n, round(float(np.degrees(d["joint_pos"][:_lo, i].min())), 1))
    for i, n in enumerate(jn)
    if d["joint_pos"][:_lo, i].min() < K.limits(n)[0] - 1e-6 or d["joint_pos"][:_lo, i].max() > K.limits(n)[1] + 1e-6
]
if _was:
    print(f"[gate ] NOT this tool's: the clip is already outside its limits before step {step(_lo):.0f}: {_was}")
qd = np.gradient(q, 1.0 / fps, axis=0)
legs_all = [i for i, n in enumerate(jn) if any(s in n for s in ("hip", "knee", "ankle"))]
print(
    f"[gate ] leg joint speed: peak {np.abs(qd[:, legs_all]).max():.2f} rad/s ({jn[legs_all[int(np.abs(qd[:, legs_all]).max(0).argmax())]]}); "
    f"was {np.abs(d['joint_vel'][:, legs_all]).max():.2f}"
)
allb = K.fk_all(pel_p, pel_q, q, jn)
for i, b in enumerate(bn):
    if b.startswith(("left_", "right_")) and any(s in b for s in ("hip", "knee", "ankle")):
        bp[:, i], bq[:, i] = allb[b]
for side in ("left", "right"):
    k = bn.index(f"{side}_ankle_roll_link")
    tail = bp[-int(1.5 * fps) :, k]
    print(
        f"[feet ] {side}: over the clip's last 1.5 s the ankle moves {np.linalg.norm(tail.max(0) - tail.min(0)) * 1000:.2f} mm (was {np.linalg.norm(np.ptp(d['body_pos_w'][-int(1.5 * fps) :, k], axis=0)) * 1000:.0f})"
    )
dq_ = K.qmul(np.roll(bq, -1, axis=0), K.qconj(bq))
dq_[-1] = dq_[-2]
dq_ = dq_ * np.sign(dq_[..., :1] + 1e-12)
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "motion.npz":
        shutil.copy(f, dst / f.name)
np.savez(
    dst / "motion.npz",
    joint_pos=q,
    joint_vel=qd,
    body_pos_w=bp,
    body_quat_w=bq,
    body_lin_vel_w=np.gradient(bp, 1.0 / fps, axis=0),
    body_ang_vel_w=2.0 * dq_[..., 1:] * fps,
    object_contact=d["object_contact"],
)
ch = int((np.abs(q - d["joint_pos"]).max(1) > 1e-9).sum())
print(
    f"[out  ] {dst}: {ch} of {T} frames changed (steps {step(np.flatnonzero(np.abs(q - d['joint_pos']).max(1) > 1e-9).min()):.0f} to the end); pelvis, waist and arms untouched"
)
