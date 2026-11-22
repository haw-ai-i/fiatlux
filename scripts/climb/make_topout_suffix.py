"""Append a short tail to a climb clip so it ENDS in the next stage's start pose.

The climb ends standing on the platform. The benchmark's next stage (S03, and every on-ladder
stage) starts standing on the platform too, but not in the same place or pose: measured
2026-11-21, 9 cm further forward, 2 cm higher, centred, legs in the default pose, arms at zero.
This adds a quasi-static shuffle from the clip's last frame to that pose:

    per swing in --steps (default left then right, all the way):
      shift  pelvis over the stance foot        swing  the other foot to its place, lifted
    home     pelvis, arms and every joint onto the target pose

Same machinery as make_approach_prefix.py (legs by damped least squares on g1_arm_fk.fk_all).
The hands hold nothing in the tail, so its contact labels are 0.

    python make_topout_suffix.py --clip <in> --out <out> --target <spawn_S03.json>

target json: {"root_pos_clip": [x, y, z], "joint_pos": {joint: rad}} from record_climb.py
--task FIATLUX-S03-RemoveOldBulb-v0 --dump_spawn (root position in the CLIP's frame).
"""
import argparse, json, pathlib, shutil, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--phases", type=float, nargs=4, default=[0.0, 0.7, 0.7, 0.8], help="(unused), seconds per weight shift, per swing, to home")
ap.add_argument("--steps", nargs="+", default=["left:1.0", "right:1.0"],
                help="FOOT:FRACTION in order -- how far along its spawn->clip path each swing takes that foot")
ap.add_argument("--ahead", type=float, default=0.03, help="pelvis this far ahead of the stance ankle, m")
ap.add_argument("--walk_z", type=float, default=None, help="pelvis height while stepping, m (default: the clip's last frame)")
ap.add_argument("--lift", type=float, default=0.04, help="swing foot clearance, m")
ap.add_argument("--keep_yaw", action="store_true", help="keep the clip's final pelvis yaw instead of the target's")
ap.add_argument("--over", type=float, default=0.85, help="how far the pelvis goes toward the stance foot's y (1 = over the ankle)")
a = ap.parse_args()
a_steps = [(x.split(":")[0], float(x.split(":")[1])) for x in a.steps]
src, dst = pathlib.Path(a.clip), pathlib.Path(a.out)
d = dict(np.load(src / "motion.npz")); meta = json.loads((src / "meta.json").read_text())
jn, bn, fps = list(meta["joint_names"]), list(meta["body_names"]), float(meta["fps"])
sp = json.loads(pathlib.Path(a.target).read_text())
J = {n: i for i, n in enumerate(jn)}
LEG = {s: [J[f"{s}_{k}_joint"] for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")] for s in ("left", "right")}
FOOT = {s: f"{s}_ankle_roll_link" for s in LEG}
PEL = bn.index("pelvis")

# ---- the two ends ---------------------------------------------------------------------------
q_start = d["joint_pos"][-1].copy()
pp_start, pq_start = d["body_pos_w"][-1, PEL].copy(), d["body_quat_w"][-1, PEL].copy()
foot_start = {s: (d["body_pos_w"][-1, bn.index(FOOT[s])].copy(), d["body_quat_w"][-1, bn.index(FOOT[s])].copy()) for s in LEG}
q_end = np.array([float(sp["joint_pos"].get(n, 0.0)) for n in jn])
pp_end = np.array(sp["root_pos_clip"], float)
pq_end = np.array(sp.get("root_quat_clip", [1.0, 0, 0, 0]), float) if a.keep_yaw is False else pq_start.copy()
o1 = K.fk_all(pp_end, pq_end, q_end, jn)
foot_end = {s: (o1[FOOT[s]][0].copy(), o1[FOOT[s]][1].copy()) for s in LEG}
dz = np.mean([foot_start[s][0][2] - foot_end[s][0][2] for s in LEG])          # stand on the SAME surface the clip stands on
pp_end[2] += dz
for s in LEG:
    foot_end[s][0][2] += dz
if a.walk_z is None:
    a.walk_z = float(pp_start[2])
for s in LEG:
    print(f"[ends ] {s} foot: clip end {np.round(foot_start[s][0], 3)} -> target {np.round(foot_end[s][0], 3)}  ({np.linalg.norm((foot_end[s][0] - foot_start[s][0])[:2])*1000:.0f} mm)")
print(f"[ends ] pelvis: clip end {np.round(pp_start, 3)} -> target {np.round(pp_end, 3)} (target raised {dz*1000:+.1f} mm so the feet stay on the platform)")

# ---- timeline: keyframes, smooth between them ------------------------------------------------
sm = lambda u: u * u * u * (u * (6 * u - 15) + 10)                 # zero velocity and acceleration at both ends
T_SETTLE, T_SHIFT, T_SWING, T_HOME = a.phases
other = {"left": "right", "right": "left"}
fp = {s: foot_start[s][0].copy() for s in LEG}; fq = {s: foot_start[s][1].copy() for s in LEG}
pel = pp_start.copy()
segs = []                                                          # (seconds, pelvis0, pelvis1, swing foot or None, p0, p1, q0, q1)
for s, frac in a_steps:
    st = other[s]                                                  # stance foot for this swing
    tgt = np.array([fp[st][0] + a.ahead, a.over * fp[st][1] + (1 - a.over) * pel[1], a.walk_z])
    segs.append((T_SHIFT, pel.copy(), tgt, None, None, None, None, None)); pel = tgt.copy()
    p1 = foot_start[s][0] + frac * (foot_end[s][0] - foot_start[s][0])
    w1 = np.array([frac]); q1 = (foot_start[s][1] * (1 - frac) + foot_end[s][1] * frac); q1 = q1 / np.linalg.norm(q1)
    segs.append((T_SWING, pel.copy(), pel.copy(), s, fp[s].copy(), p1.copy(), fq[s].copy(), q1.copy())); fp[s], fq[s] = p1.copy(), q1.copy()
segs.append((T_HOME, pel.copy(), pp_end.copy(), None, None, None, None, None))
n_seg = [int(round(t * fps)) for t, *_ in segs]
unit = int(fps / np.gcd(int(fps), 50))                             # frames per whole number of 50 Hz steps
n_seg[-1] += (-(sum(n_seg)) % unit)                                # prefix (without the shared last frame) = whole steps
N = sum(n_seg) + 1
pel_p = np.zeros((N, 3)); foot_p = {s: np.zeros((N, 3)) for s in LEG}; foot_q = {s: np.zeros((N, 4)) for s in LEG}
swing = {s: np.zeros(N, bool) for s in LEG}
cur_p = {s: foot_start[s][0].copy() for s in LEG}; cur_q = {s: foot_start[s][1].copy() for s in LEG}
i0 = 0
for (t, c0, c1, s, p0, p1, q0, q1), n in zip(segs, n_seg):
    w = sm(np.arange(n + 1) / n)
    sl = slice(i0, i0 + n + 1)
    pel_p[sl] = c0[None] + w[:, None] * (c1 - c0)[None]
    for ft in LEG:
        if ft == s:
            pp_ = p0[None] + w[:, None] * (p1 - p0)[None]; pp_[:, 2] += a.lift * np.sin(np.pi * w) ** 2
            qq_ = (1 - w)[:, None] * q0 + w[:, None] * q1
            foot_p[ft][sl], foot_q[ft][sl] = pp_, qq_ / np.linalg.norm(qq_, axis=1, keepdims=True)
            swing[ft][i0 + 1:i0 + n] = True
            cur_p[ft], cur_q[ft] = p1.copy(), q1.copy()
        else:
            foot_p[ft][sl], foot_q[ft][sl] = cur_p[ft][None], cur_q[ft][None]
    i0 += n
wH = sm(np.clip((np.arange(N) - (N - 1 - n_seg[-1])) / n_seg[-1], 0, 1))
w_all = sm(np.arange(N) / (N - 1))
pq1 = pq_end if np.dot(pq_start, pq_end) >= 0 else -pq_end
pel_q = (1 - w_all)[:, None] * pq_start + w_all[:, None] * pq1; pel_q /= np.linalg.norm(pel_q, axis=1, keepdims=True)
F = wH
# every joint that is not a leg: clip -> target while coming home (arms, waist); legs are solved below
q = q_start[None] + wH[:, None] * (q_end - q_start)[None]
print("[plan ] " + "; ".join(f"{'home' if k == len(segs) - 1 else ('swing ' + sg[3]) if sg[3] else 'shift'} {n / fps:.2f}s" for k, (sg, n) in enumerate(zip(segs, n_seg))))

# ---- legs: damped least squares, all frames at once -----------------------------------------
def foot_err(qq, s):
    o = K.fk_all(pel_p, pel_q, qq, jn)[FOOT[s]]
    dq = K.qmul(foot_q[s], K.qconj(o[1])); dq = dq * np.sign(dq[:, :1] + 1e-12)
    return np.concatenate([foot_p[s] - o[0], 2.0 * dq[:, 1:]], 1)                      # [N, 6]


for s in LEG:
    for it in range(60):
        e = foot_err(q, s)
        if np.abs(e).max() < 1e-5:
            break
        Jm = np.zeros((N, 6, 6))
        for c, j in enumerate(LEG[s]):
            qp = q.copy(); qp[:, j] += 1e-4
            Jm[:, :, c] = -(foot_err(qp, s) - e) / 1e-4
        dq_ = np.linalg.solve(np.einsum("nij,nik->njk", Jm, Jm) + 1e-6 * np.eye(6), np.einsum("nij,ni->nj", Jm, e))
        q[:, LEG[s]] += np.clip(dq_, -0.2, 0.2)
    e = foot_err(q, s)
    print(f"[ik   ] {s} leg: {it + 1} iterations, worst foot position error {np.abs(e[:, :3]).max()*1000:.2f} mm, orientation {np.degrees(np.abs(e[:, 3:]).max()):.2f} deg")
# land exactly on the clip's frame: spread what is left over phase F
q += F[:, None] * (q_end - q[-1])[None]
q[0] = q_start                                                      # first tail frame IS the clip's last
bad = [(n, float(np.degrees(q[:, i].min())), float(np.degrees(q[:, i].max()))) for i, n in enumerate(jn)
       if q[:, i].min() < K.limits(n)[0] - 1e-6 or q[:, i].max() > K.limits(n)[1] + 1e-6]
print("[gate ] joint limits: " + ("PASS" if not bad else f"FAIL {bad}"))
qd = np.gradient(q, 1.0 / fps, axis=0)
print(f"[gate ] joint speed: peak {np.abs(qd).max():.2f} rad/s ({jn[int(np.abs(qd).max(0).argmax())]}); pelvis speed peak {np.linalg.norm(np.gradient(pel_p, 1.0 / fps, axis=0), axis=1).max():.2f} m/s")

# ---- bodies, and the joined clip -------------------------------------------------------------
allb = K.fk_all(pel_p, pel_q, q, jn)
bp = np.stack([allb[b][0] for b in bn], 1); bq = np.stack([allb[b][1] for b in bn], 1)
seam = max(float(np.abs(bp[0] - d["body_pos_w"][-1]).max()), float(np.abs(q[0] - q_start).max()))
print(f"[join ] first tail frame against the clip's last frame: worst difference {seam*1000:.3f} mm / mrad; last tail frame against the target joints {np.abs(q[-1] - q_end).max()*1000:.3f} mrad")
for s in LEG:
    k = bn.index(FOOT[s]); pl = ~swing[s]
    both = pl[1:] & pl[:-1]
    print(f"[feet ] {s}: lowest ankle {bp[:, k, 2].min():.4f} m (floor stance {foot_start[s][0][2]:.4f}); slides while planted {np.abs(np.diff(bp[:, k, :2], axis=0))[both].max()*1000:.3f} mm/frame")
Q = np.concatenate([d["joint_pos"], q[1:]]); BP = np.concatenate([d["body_pos_w"], bp[1:]]); BQ = np.concatenate([d["body_quat_w"], bq[1:]])
n_pre = (N - 1) * 50 / fps
if abs(n_pre - round(n_pre)) > 1e-9:
    raise SystemExit(f"prefix is {n_pre} steps, not a whole number")
n_pre = int(round(n_pre))
n_tot = int(round(Q.shape[0] * 50 / fps + 0.49))
oc = np.concatenate([d["object_contact"], np.zeros((n_tot - len(d["object_contact"]), 2), d["object_contact"].dtype)])
dq_ = K.qmul(np.roll(BQ, -1, axis=0), K.qconj(BQ)); dq_[-1] = dq_[-2]; dq_ = dq_ * np.sign(dq_[..., :1] + 1e-12)
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "motion.npz": shutil.copy(f, dst / f.name)
np.savez(dst / "motion.npz", joint_pos=Q, joint_vel=np.gradient(Q, 1.0 / fps, axis=0), body_pos_w=BP, body_quat_w=BQ,
         body_lin_vel_w=np.gradient(BP, 1.0 / fps, axis=0), body_ang_vel_w=2.0 * dq_[..., 1:] * fps, object_contact=oc)
hj = dst / "holds.json"
if hj.exists():
    h = json.loads(hj.read_text()); h["topout_suffix"] = dict(steps=n_pre, phases=a.phases, swings=a.steps, target=sp); hj.write_text(json.dumps(h, indent=1))
print(f"[out  ] {dst}: {Q.shape[0]} frames = {len(oc)} steps ({len(d['object_contact'])} of the clip + {len(oc) - len(d['object_contact'])} tail)")
