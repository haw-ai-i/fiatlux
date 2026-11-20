"""Author the climb clip's arms as a PLANNED motion, and refuse to write one that is not physical.

build_topholds_motion.py stays runnable. It solved every frame on its own from the base pose,
with every joint clamped to +-2.6 rad, and judged the result only by hand-to-ladder distance.
Audited at the joint level its output had arm speeds of 133 rad/s at the 99th percentile, 229
frames outside the real joint limits (the elbow bent 89 degrees backwards) and the wrist behind
the robot for 10% of the clip. None of that showed in the metric it was tuned on.

This one is built the other way round:

* HOLDS are solved in time order, each frame warm-started from the one before, with the USD's own
  joint limits and a cap on how far a joint may move per frame. Continuity is by construction.
* TRANSITS are interpolated in JOINT space between two valid poses, through a lifted waypoint
  above each hold, with a monotone spline -- which cannot overshoot, so a move between two
  in-limit poses stays in limits. A transit is widened until its speed is under the cap.
* The TORSO leans (waist pitch, inside its limit) where the hands change holds, because with the
  base clip's upright torso neither hold is reachable there for 2.3 s.
* An ACCEPTANCE GATE measures the finished clip -- limits, speed, wrist behind the torso, arm
  clearance to the real mesh, hands below the face they hold -- and the clip is only written as
  accepted when every line passes.

Runs offline in seconds on g1_arm_fk (validated to 0.3 mm against Isaac), so it can be iterated.

    python scripts/climb/build_keyframe_motion.py --base <clip> --out <dir> --mesh <npz>
"""
import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mesh", required=True)
ap.add_argument("--reach", type=float, default=0.4161, help="Dex3 shoulder-to-palm, FK-measured")
ap.add_argument("--palm_off", type=float, default=0.0416)
ap.add_argument("--hold_margin", type=float, default=0.010)
ap.add_argument("--grip_up", type=float, default=0.025)
ap.add_argument("--min_hold", type=float, default=0.50)
ap.add_argument("--lean", type=float, default=0.30, help="extra waist pitch through the handover, rad")
ap.add_argument("--lean_on", type=float, nargs=2, default=[270, 320], help="50 Hz steps: ramp in")
ap.add_argument("--lean_off", type=float, nargs=2, default=[440, 480], help="50 Hz steps: ramp out")
ap.add_argument("--crouch", type=float, default=0.0,
                help="metres the pelvis drops on step 2 before the step-up onto step 3, feet "
                     "held where they are by re-solving hip pitch, knee and ankle pitch. Keeps "
                     "the platform under the hands until the push-off.")
ap.add_argument("--crouch_on", type=float, nargs=2, default=[290, 335])
ap.add_argument("--crouch_off", type=float, nargs=2, default=[425, 452])
ap.add_argument("--plan_only", action="store_true", help="print the reach schedule and stop")
ap.add_argument("--vcap", type=float, default=4.0, help="arm joint speed cap in a transit, rad/s")
ap.add_argument("--lift", type=float, default=0.08, help="waypoint height above a hold, m")
ap.add_argument("--approach", type=float, default=0.05,
                help="a hold only counts as reachable when the wrist can also get THIS far above "
                     "it. A palm presses on top of a face, so it must arrive from above; from "
                     "below, maximum reach touches the bar's FRONT, not its top.")
ap.add_argument("--front_min", type=float, default=0.15,
                help="a hold is only kept while the wrist stays this far in FRONT of the chest. "
                     "Late in the platform hold the body moves over the hand; held on, the hand "
                     "ends up under and behind the shoulder and the next move swings the arm "
                     "backward over the top. A climber lets go before that.")
ap.add_argument("--edge_in", type=float, default=0.03, help="palm lands this far in from a face's near edge")
ap.add_argument("--retreat", type=float, default=0.09, help="a moving hand first backs this far toward the robot")
ap.add_argument("--min_transit", type=float, default=0.90)
ap.add_argument("--hold_tol", type=float, default=0.015)
ap.add_argument("--gate_speed", type=float, default=6.0)
ap.add_argument("--gate_clear", type=float, default=0.020)
ap.add_argument("--start_blend", type=float, default=0.40,
                help="seconds over which the opening is re-timed. The base clip holds a standing "
                     "stub for 12 frames and then JUMPS to the climb pose in one frame (pelvis "
                     "+55 mm, 0.2 rad); this eases between the two instead.")
ap.add_argument("--rail_keepout", type=float, default=0.13,
                help="|y| the hands stay inside. Above the platform the ladder is a corridor: "
                     "side rails at |y| 0.20-0.25 m run up to the handrail, and the robot's "
                     "shoulders are wider than the gap, so the arms go up BETWEEN the rails.")
a = ap.parse_args()

base = pathlib.Path(a.base)
meta = json.loads((base / "meta.json").read_text())
d = dict(np.load(base / "motion.npz"))
fps = float(meta["fps"])
jn, bn = list(meta["joint_names"]), list(meta["body_names"])
T = d["joint_pos"].shape[0]
step50 = np.arange(T) / fps * 50
pel = bn.index("pelvis")
PP, PQ = d["body_pos_w"][:, pel], d["body_quat_w"][:, pel]
SIDES = ("left", "right")
print(f"[base] {T} frames @ {fps:.0f} fps")

# ---- repair the base clip's opening jump ---------------------------------------------------
_dq0 = np.abs(np.diff(d["joint_pos"][:60], axis=0)).max(1)
_j = int(np.argmax(_dq0))
if _dq0[_j] > 0.1:
    nb = int(a.start_blend * fps)
    u = np.linspace(0.0, 1.0, nb + 1)
    w = (10 * u**3 - 15 * u**4 + 6 * u**5)[:, None]                   # minimum jerk
    d["joint_pos"][:nb + 1] = (1 - w) * d["joint_pos"][0] + w * d["joint_pos"][nb]
    PP[:nb + 1] = (1 - w) * PP[0] + w * PP[nb]
    qa, qb = PQ[0].copy(), PQ[nb].copy()
    if (qa * qb).sum() < 0:
        qb = -qb
    qq = (1 - w) * qa + w * qb
    PQ[:nb + 1] = qq / np.linalg.norm(qq, axis=1, keepdims=True)
    print(f"[base] one-frame jump of {_dq0[_j]:.2f} rad at frame {_j + 1} eased over {nb} frames")

# ---- crouch: lower the pelvis, keep the feet -----------------------------------------------
if a.crouch > 0:
    cr = (np.clip((step50 - a.crouch_on[0]) / (a.crouch_on[1] - a.crouch_on[0]), 0, 1)
          * np.clip((a.crouch_off[1] - step50) / (a.crouch_off[1] - a.crouch_off[0]), 0, 1))
    cr = 0.5 - 0.5 * np.cos(np.pi * cr)
    ref = K.fk_all(PP, PQ, d["joint_pos"], jn)
    PP[:, 2] -= a.crouch * cr
    fwd = np.array([1.0, 0.0, 0.0])
    for sd in SIDES:
        cols = [jn.index(f"{sd}_{t}_joint") for t in ("hip_pitch", "knee", "ankle_pitch")]
        foot = f"{sd}_ankle_roll_link"
        tp, tq = ref[foot]
        tf = K.qrot(tq, fwd)

        def resid(jp):
            o = K.fk_all(PP, PQ, jp, jn)[foot]
            return np.stack([o[0][:, 0] - tp[:, 0], o[0][:, 2] - tp[:, 2],
                             K.qrot(o[1], fwd)[:, 2] - tf[:, 2]], 1)

        jp = d["joint_pos"]
        for _ in range(25):
            r0 = resid(jp)
            if np.abs(r0).max() < 2e-5:
                break
            J = np.zeros((T, 3, 3))
            for c_, col in enumerate(cols):
                jq = jp.copy(); jq[:, col] += 1e-4
                J[:, :, c_] = (resid(jq) - r0) / 1e-4
            dq = np.linalg.solve(J + 1e-6 * np.eye(3), -r0[:, :, None])[:, :, 0]
            jp[:, cols] += np.clip(dq, -0.2, 0.2)
        kn = jp[:, cols[1]]
        lo_k, hi_k = K.limits(f"{sd}_knee_joint")
        print(f"[crouch] {sd}: foot held to {np.abs(resid(jp)[:, :2]).max()*1000:.2f} mm; knee peak "
              f"{kn.max():.2f} rad (limit {hi_k:.2f}), knee speed peak "
              f"{np.abs(np.gradient(kn, 1/fps)).max():.1f} rad/s")

# ---- torso lean ---------------------------------------------------------------------------
wi = [jn.index(n) for n in K.WAIST]
waist = d["joint_pos"][:, wi].copy()
lo_w, hi_w = K.limits("waist_pitch_joint")
ramp = (np.clip((step50 - a.lean_on[0]) / (a.lean_on[1] - a.lean_on[0]), 0, 1)
        * np.clip((a.lean_off[1] - step50) / (a.lean_off[1] - a.lean_off[0]), 0, 1))
ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)
waist[:, 2] = np.clip(waist[:, 2] + a.lean * ramp, lo_w + 0.02, hi_w - 0.02)
print(f"[lean] waist pitch peak {waist[:, 2].max():.2f} rad (limit {hi_w:.2f}); base peak "
      f"{d['joint_pos'][:, wi[2]].max():.2f}")

# ---- ladder: mesh, true face tops, exact distance ------------------------------------------
_TRI = np.load(a.mesh)["T"]
_TV = _TRI.reshape(-1, 3)
_TC = np.stack([_TV[:, 1] - 0.1996, -_TV[:, 0], _TV[:, 2]], 1)          # clip frame
_TCL = _TC.reshape(-1, 3, 3)
_TLO, _THI = _TCL.min(1), _TCL.max(1)


def _exact(p, ids):
    t = _TCL[ids]
    A, B, C = t[:, 0], t[:, 1], t[:, 2]
    ab, ac = B - A, C - A
    nn = np.cross(ab, ac)
    den = np.maximum((nn * nn).sum(1), 1e-20)
    apv = p - A
    u = (np.cross(apv, ac) * nn).sum(1) / den
    v = (np.cross(ab, apv) * nn).sum(1) / den
    ins = (u >= 0) & (v >= 0) & (u + v <= 1)
    q = A + u[:, None] * ab + v[:, None] * ac
    best = np.where(ins, ((q - p) ** 2).sum(1), np.inf)
    for p0_, ed in ((A, ab), (A, ac), (B, C - B)):
        w_ = np.clip(((p - p0_) * ed).sum(1) / np.maximum((ed * ed).sum(1), 1e-20), 0, 1)
        best = np.minimum(best, ((p0_ + w_[:, None] * ed - p) ** 2).sum(1))
    return float(np.sqrt(best.min()))


def mesh_dist(p, k=2048):
    """Exact distance to the ladder (AABB lower-bound prune, rechecked, so it stays exact)."""
    lb = np.sqrt((np.maximum(np.maximum(_TLO - p, p - _THI), 0.0) ** 2).sum(1))
    part = np.argpartition(lb, k)
    dist = _exact(p, part[:k])
    return dist if dist <= lb[part[k]] else _exact(p, np.arange(lb.size))


def lad_to_clip(x, y, z):
    return np.array([y - 0.1996, -x, z])


HOLDS = {}
for _k, (_z, _yc) in enumerate([(0.237, -0.387), (0.472, -0.301), (0.707, -0.216), (0.942, -0.130)]):
    _a, _b = lad_to_clip(-0.247, _yc - 0.038, _z), lad_to_clip(0.247, _yc + 0.038, _z)
    HOLDS[f"step{_k+1}"] = dict(x=(min(_a[0], _b[0]), max(_a[0], _b[0])), y=(-0.247, 0.247), z=_z)
HOLDS["platform"] = dict(x=(-0.302, 0.075), y=(-0.208, 0.206), z=1.1494)
HOLDS["handrail"] = dict(x=(-0.030, 0.066), y=(-0.241, 0.241), z=1.8424)
ORDER = ["step1", "step2", "step3", "step4", "platform", "handrail"]
LABEL = {n: i + 1 for i, n in enumerate(ORDER)}
for _n in ORDER:                       # the TRUE top of each footprint, never a triangle mean
    _h = HOLDS[_n]
    _m = ((_TC[:, 0] >= _h["x"][0] - 0.005) & (_TC[:, 0] <= _h["x"][1] + 0.005)
          & (_TC[:, 1] >= _h["y"][0] - 0.005) & (_TC[:, 1] <= _h["y"][1] + 0.005)
          & (np.abs(_TC[:, 2] - _h["z"]) < 0.06))
    _h["z"] = float(_TC[_m, 2].max())

# ---- shoulders (after the lean) and hold targets -------------------------------------------
SH = {s: K.fk(PP, PQ, waist, np.zeros((T, 7)), s)[f"{s}_shoulder_pitch_link"] for s in SIDES}


_TQ = K.fk(PP, PQ, waist, np.zeros((T, 7)), "left")
TORSO_P, TORSO_Q = _TQ["torso_p"], _TQ["torso_q"]


def chest_x(i, p):
    """How far in front of the chest plane a world point is, at frame i."""
    return float(K.qrot(K.qconj(TORSO_Q[i]), p - TORSO_P[i])[0])


# The Dex3 hand, in the wrist_yaw_link frame, read off the USD: the palm pad is 87 mm long (x) by
# 88 mm wide (z); its FACE is the -y side on the left hand (the fingers curl toward -y and the
# thumb sits there) and the +y side on the right. PC is the centre of that face.
HAND = {"left": dict(pc=np.array([0.0865, -0.017, 0.0]), n=np.array([0.0, -1.0, 0.0]),
                     tip=np.array([0.21, -0.005, 0.0])),
        "right": dict(pc=np.array([0.0865, 0.017, 0.0]), n=np.array([0.0, 1.0, 0.0]),
                      tip=np.array([0.21, 0.005, 0.0]))}
PALM_HALF = 0.045


def contact_on(name, sp, side):
    """Where the CENTRE of the palm face rests on this hold, nearest to this shoulder."""
    h = HOLDS[name]
    ylo, yhi = h["y"]
    if name in ("handrail", "platform"):                   # between the rails, each hand its half
        ylo, yhi = (0.03, a.rail_keepout) if side == "left" else (-a.rail_keepout, -0.03)
    xlo, xhi = h["x"][0] + PALM_HALF, h["x"][1] - PALM_HALF
    if xlo > xhi:                                          # a face shallower than the palm: centre it
        xlo = xhi = 0.5 * (h["x"][0] + h["x"][1])
    return np.array([np.clip(sp[0], xlo, xhi), np.clip(sp[1], ylo, yhi), h["z"]])


def target_on(name, sp, side):
    """Where the WRIST sits when the palm lies flat on the hold: 20 mm above the face and one
    palm-length (86 mm) back from the contact centre, toward the shoulder. Used to judge reach
    and to place a move's waypoints; the hold itself is solved on the palm face (solve_palm)."""
    c = contact_on(name, sp, side)
    hv = (sp - c) * np.array([1.0, 1.0, 0.0])
    hn = float(np.linalg.norm(hv))
    back = hv / hn * min(HAND[side]["pc"][0], hn) if hn > 1e-6 else np.zeros(3)
    return c + back + np.array([0.0, 0.0, 0.020])


def schedule(side):
    raw = []
    for i in range(T):
        ok = []
        wmax = a.reach - a.palm_off - a.hold_margin
        for n in ORDER:
            tg = target_on(n, SH[side][i], side)
            r = np.linalg.norm(tg - SH[side][i])
            r_up = np.linalg.norm(tg + np.array([0.0, 0.0, a.approach]) - SH[side][i])
            if r <= wmax and r_up <= a.reach - a.palm_off and chest_x(i, tg) >= a.front_min:
                ok.append((n, r))
        raw.append(min(ok, key=lambda nr: nr[1])[0] if ok else None)
    hi_seen = 0
    for i in range(T):                                     # a climb only goes UP the holds
        if raw[i] is not None:
            if LABEL[raw[i]] < hi_seen:
                raw[i] = None
            hi_seen = max(hi_seen, LABEL[raw[i]] if raw[i] else hi_seen)
    for i in range(T):                                     # ... and not back after moving on
        if raw[i] is not None and any(r is not None and LABEL[r] > LABEL[raw[i]] for r in raw[max(0, i - 1):i]):
            raw[i] = None
    segs, cur, st = [], raw[0], 0
    for i in range(1, T):
        if raw[i] != cur:
            segs.append([st, i, cur]); cur, st = raw[i], i
    segs.append([st, T, cur])
    minf = int(a.min_hold * fps)
    out = []
    for s in segs:                                         # short runs become part of a transit
        if s[2] is not None and s[1] - s[0] < minf:
            s[2] = None
        if out and out[-1][2] == s[2]:
            out[-1][1] = s[1]
        else:
            out.append(s)
    # the SAME hold either side of a short gap is one hold -- the reach test flickers when the
    # body lunges, and letting go to re-place a hand on the bar it is already on is nonsense
    j = 1
    while j < len(out) - 1:
        if (out[j][2] is None and out[j - 1][2] is not None and out[j - 1][2] == out[j + 1][2]
                and out[j][1] - out[j][0] < int(0.8 * fps)):
            out[j - 1][1] = out[j + 1][1]
            del out[j:j + 2]
        else:
            j += 1
    # a brief return to the hold just left is a flicker of the reach test, not a real re-grip
    seen, fin = None, []
    for s in out:
        if s[2] is not None and s[2] == seen and s[1] - s[0] < int(1.0 * fps):
            s[2] = None
        if s[2] is not None:
            seen = s[2]
        if fin and fin[-1][2] == s[2]:
            fin[-1][1] = s[1]
        else:
            fin.append(s)
    return fin


if a.plan_only:
    for s_ in SIDES:
        print(f"[sched] {s_[0].upper()} reachable: " + "  ".join(
            f"{step50[x0]:.0f}-{step50[x1-1]:.0f}:{nm or '-'}" for x0, x1, nm in schedule(s_)))
    sys.exit(0)

# ---- IK on the offline model ---------------------------------------------------------------
ARMJ = {s_: [t.format(s=s_) for t in K.ARM] for s_ in SIDES}
LIM = {s_: np.array([K.limits(n) for n in ARMJ[s_]]) for s_ in SIDES}
# The USD limits allow the arm to swing backward over the shoulder and to fold across the body.
# A climb uses neither, and a solver given the freedom takes it whenever it is the shorter path.
# Forward working range for [pitch, roll, yaw, elbow], mirrored for the right arm; wrists keep
# their own limits.
_FUNC = np.array([[-3.00, 0.40], [-0.35, 1.30], [-1.20, 1.20], [-0.90, 1.90]])
for _s in SIDES:
    f = _FUNC.copy()
    if _s == "right":
        f[1] = [-1.30, 0.35]
    LIM[_s][:4, 0] = np.maximum(LIM[_s][:4, 0], f[:, 0])
    LIM[_s][:4, 1] = np.minimum(LIM[_s][:4, 1], f[:, 1])
MARG = 0.03
REST = {"left": np.array([-0.6, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0]),
        "right": np.array([-0.6, -0.2, 0.0, 0.3, 0.0, 0.0, 0.0])}
DOWN = np.array([0.0, 0.0, -1.0])


def hand_of(i, side, q):
    """-> wrist pos, palm-face centre, palm normal, fingertip, elbow (world)."""
    o = K.fk(PP[i], PQ[i], waist[i], q, side)
    wp, wq = o[f"{side}_wrist_yaw_link"], o[f"{side}_wrist_yaw_link_q"]
    h = HAND[side]
    return wp, wp + K.qrot(wq, h["pc"]), K.qrot(wq, h["n"]), wp + K.qrot(wq, h["tip"]), o[f"{side}_elbow_link"]


def wrist_of(i, side, q):
    return K.fk(PP[i], PQ[i], waist[i], q, side)[f"{side}_wrist_yaw_link"]


def _clip(side, q):
    return np.clip(q, LIM[side][:, 0] + MARG, LIM[side][:, 1] - MARG)


def solve(i, side, tgt, q0, iters=25, dq_cap=None):
    """WRIST position to `tgt` with the four shoulder/elbow joints; wrist joints stay as given."""
    q = q0.copy()
    for _ in range(iters):
        w = wrist_of(i, side, q)
        e = tgt - w
        if np.linalg.norm(e) < 1e-4:
            break
        J = np.zeros((3, 4))
        for c in range(4):
            qp = q.copy(); qp[c] += 1e-4
            J[:, c] = (wrist_of(i, side, qp) - w) / 1e-4
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-5 * np.eye(3))
        dq = Jp @ e + 0.05 * (np.eye(4) - Jp @ J) @ (REST[side][:4] - q[:4])
        q[:4] += np.clip(dq, -0.2, 0.2)
        q = _clip(side, q)
    if dq_cap is not None:
        q = q0 + np.clip(q - q0, -dq_cap, dq_cap)
    return q, float(np.linalg.norm(tgt - wrist_of(i, side, q)))


def _palm_err(i, side, q, contact):
    _, pc, n, _, _ = hand_of(i, side, q)
    return np.concatenate([contact + np.array([0.0, 0.0, 0.003]) - pc, 0.25 * np.cross(n, DOWN)]), pc, n


def solve_palm(i, side, contact, q0, iters=25, dq_cap=None):
    """All seven arm joints: put the centre of the PALM FACE on `contact`, palm facing down.
    The hand's heading about the vertical is left free. -> (q, position error, tilt in degrees)."""
    q = q0.copy()
    for _ in range(iters):
        e, _, _ = _palm_err(i, side, q, contact)
        if np.linalg.norm(e) < 2e-4:
            break
        J = np.zeros((6, 7))
        for c in range(7):
            qp = q.copy(); qp[c] += 1e-4
            J[:, c] = (_palm_err(i, side, qp, contact)[0] - e) / -1e-4
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-5 * np.eye(6))
        dq = Jp @ e + 0.05 * (np.eye(7) - Jp @ J) @ (REST[side] - q)
        q = _clip(side, q + np.clip(dq, -0.2, 0.2))
    if dq_cap is not None:
        q = q0 + np.clip(q - q0, -dq_cap, dq_cap)
    e, pc, n = _palm_err(i, side, q, contact)
    tilt = float(np.degrees(np.arccos(np.clip(n @ DOWN, -1, 1))))
    return q, float(np.linalg.norm(e[:3])), tilt


def arm_clear(i, side, q):
    w, _, _, _, e = hand_of(i, side, q)
    return min(mesh_dist(e), mesh_dist(0.5 * (e + w)))


def steer_clear(i, side, contact, q, want=0.05):
    """Swing the elbow along the arm's free motion if the elbow or forearm is near the ladder."""
    c0 = arm_clear(i, side, q)
    if c0 >= want:
        return q
    e0 = _palm_err(i, side, q, contact)[0]
    J = np.zeros((6, 7))
    for c in range(7):
        qp = q.copy(); qp[c] += 1e-4
        J[:, c] = (_palm_err(i, side, qp, contact)[0] - e0) / -1e-4
    nvec = np.linalg.svd(J)[2][-1]                         # leaves the palm where it is
    best = (c0, q)
    for sgn in (1.0, -1.0):
        q2, e2, t2 = solve_palm(i, side, contact, _clip(side, q + sgn * 0.05 * nvec), iters=3)
        if e2 < 0.01:
            c2 = arm_clear(i, side, q2)
            if c2 > best[0]:
                best = (c2, q2)
    return best[1]


def _seeds(side):
    sg = 1.0 if side == "left" else -1.0
    return [REST[side][:4], np.array([-1.2, 0.3 * sg, 0.0, 0.2]), np.array([-1.8, 0.4 * sg, 0.0, 0.6]),
            np.array([-0.3, 0.3 * sg, 0.3 * sg, 0.9]), np.array([-2.2, 0.2 * sg, 0.0, 0.0]),
            np.array([-1.0, 0.8 * sg, -0.5 * sg, 0.5])]


def solve_seeded(i, side, tgt, wr, extra=()):
    """A waypoint: several starts, keep the reachable one nearest a natural pose."""
    best = None
    for s0 in [np.concatenate([x, wr]) for x in _seeds(side)] + [np.concatenate([x[:4], wr]) for x in extra]:
        q, e = solve(i, side, tgt, s0, iters=60)
        cost = (e * 50.0 + 0.2 * np.linalg.norm(q[:4] - REST[side][:4])
                + 6.0 * max(0.0, 0.06 - arm_clear(i, side, q)))
        if best is None or cost < best[0]:
            best = (cost, q, e)
    return best[1], best[2]


def solve_palm_seeded(i, side, contact):
    """First frame of a hold: arm starts x wrist starts; keep the flattest, most natural, clearest."""
    sg = 1.0 if side == "left" else -1.0
    wrists = [np.zeros(3), np.array([1.4 * sg, 0.0, 0.0]), np.array([-1.4 * sg, 0.0, 0.0]),
              np.array([1.4 * sg, 0.8, 0.0]), np.array([1.4 * sg, -0.8, 0.0]),
              np.array([-1.4 * sg, 0.8, 0.0]), np.array([-1.4 * sg, -0.8, 0.0]),
              np.array([0.0, 1.2, 0.0]), np.array([0.0, -1.2, 0.0])]
    best = None
    for a4 in _seeds(side):
        for wr in wrists:
            q, e, tilt = solve_palm(i, side, contact, np.concatenate([a4, wr]), iters=60)
            cost = (e * 50.0 + tilt / 30.0 + 0.2 * np.linalg.norm(q[:4] - REST[side][:4])
                    + 0.05 * np.linalg.norm(q[4:]) + 6.0 * max(0.0, 0.06 - arm_clear(i, side, q)))
            if best is None or cost < best[0]:
                best = (cost, q, e, tilt)
    return best[1], best[2], best[3]


def plan_arm(side):
    segs = schedule(side)
    q4_base = d["joint_pos"][:, [jn.index(n) for n in ARMJ[side]]]
    Q = np.full((T, 7), np.nan)
    err = np.zeros(T)
    tilt = np.zeros(T)
    lab = np.zeros(T, dtype=int)
    cap = 5.5 / fps        # a fixed hand under a body rising at 1 m/s needs ~4 rad/s; no more
    holds = [s for s in segs if s[2] is not None]
    for s, e, name in holds:
        Q[s], err[s], tilt[s] = solve_palm_seeded(s, side, contact_on(name, SH[side][s], side))
        for i in range(s + 1, e):
            ct = contact_on(name, SH[side][i], side)
            Q[i], err[i], tilt[i] = solve_palm(i, side, ct, Q[i - 1], iters=4, dq_cap=cap)
            if i % 2 == 0:
                qs = steer_clear(i, side, ct, Q[i])
                Q[i] = Q[i - 1] + np.clip(qs - Q[i - 1], -cap, cap)
                ee, _, nn = _palm_err(i, side, Q[i], ct)
                err[i] = float(np.linalg.norm(ee[:3]))
                tilt[i] = float(np.degrees(np.arccos(np.clip(nn @ DOWN, -1, 1))))
        lab[s:e] = LABEL[name]
        print(f"[hold ] {side[0].upper()} {name:9s} {step50[s]:.0f}-{step50[e-1]:.0f}: start err {err[s]*1000:.0f} mm, "
              f"max {err[s:e].max()*1000:.0f} mm; palm tilt start {tilt[s]:.0f} deg, max {tilt[s:e].max():.0f} deg; "
              f"wrist q0 {np.round(Q[s][4:], 2)}")

    def waypoint(i, name, prev):
        t = target_on(name, SH[side][i], side) + np.array([0.0, 0.0, a.lift])
        return solve(i, side, t, np.concatenate([prev[:4], Q[i][4:]]), iters=40)[0]

    def route(lo_, hi_, A_, B_, t_front, t_above, back, wait, chain):
        """Knots for a move from hold A_ (frame lo_) to hold B_ (frame hi_): lift off; optionally
        WAIT in a natural pose in front of the chest; then rise in front of the new face, come
        over it and settle from above. Every waypoint is pulled onto the reach sphere, so each
        is a pose the arm can take -- a solver chasing an unreachable point wanders."""
        n = hi_ - lo_
        sg = 1.0 if side == "left" else -1.0
        lim = (a.reach - a.palm_off) * 0.97

        def reachable(i_, pt):
            v = pt - SH[side][i_]
            dn = float(np.linalg.norm(v))
            return SH[side][i_] + v / dn * lim if dn > lim else pt

        def ready(i_):
            return TORSO_P[i_] + K.qrot(TORSO_Q[i_], np.array([0.17, 0.09 * sg, 0.10]))

        tb = target_on(B_, SH[side][hi_], side)
        # A flat hand reaches 21 cm ahead of the wrist, so on a face its fingertips sit UNDER
        # whatever overhangs it (the platform lip above step 4, the handrail above the platform).
        # Draw the hand back before it rises, as from under a shelf.
        ta = target_on(A_, SH[side][lo_], side)
        # ... and LEVEL: with the wrist joints frozen the hand pitches nose-down as the arm
        # retracts, and the fingertip was measured dropping 9 mm INTO the platform and dragging
        # along it. So the first two waypoints are flat-palm poses: straight up, then back.
        knots = [(lo_ + min(int(0.22 * fps), n // 6),
                  ("palm", lambda i_: contact_on(A_, SH[side][i_], side) + np.array([0.0, 0.0, 0.07]))),
                 (lo_ + min(int(0.50 * fps), n // 3),
                  ("palm", lambda i_: contact_on(A_, SH[side][i_], side) + np.array([-back, 0.0, 0.10])))]
        if wait and n > int((t_front + 1.2) * fps):
            knots += [(lo_ + int(1.00 * fps), ready), (hi_ - int((t_front + 0.35) * fps), ready)]
        knots += [(hi_ - int(t_front * fps),
                   lambda i_: np.array([min(ta[0] - back, tb[0] - back), tb[1], tb[2] + a.lift])),
                  # ... and ARRIVE level too: over the new face from behind and above, then down.
                  # Turning flat while still moving forward swept the fingertip through the bar.
                  (hi_ - int((t_above + 0.24) * fps),
                   ("dst", lambda i_: contact_on(B_, SH[side][i_], side) + np.array([-0.09, 0.0, 0.11]))),
                  (hi_ - int(t_above * fps),
                   ("dst", lambda i_: contact_on(B_, SH[side][i_], side) + np.array([0.0, 0.0, 0.05])))]
        kt, kq, prev = [lo_], [Q[lo_]], Q[lo_]
        for kk, (i_, fn) in enumerate(knots):
            if not (kt[-1] < i_ < hi_):
                continue
            if isinstance(fn, tuple):                      # a level, palm-down waypoint
                seed = Q[hi_] if fn[0] == "dst" else prev  # same wrist branch as the hold it serves
                prev = solve_palm(i_, side, fn[1](i_), seed, iters=40)[0]
                kt.append(i_); kq.append(prev)
                continue
            tg = reachable(i_, fn(i_))
            # Lift off as held; travel with the hand IN LINE with the forearm, the way an arm
            # reaches; turn palm-down only over the new face. Carrying a flat hand the whole way
            # sweeps a rigid 21 cm paddle through the gap under the handrail.
            # (In line only for the rise in FRONT of the new face, where the forearm points up.
            # Straight after release the forearm points down at a low hold, and an in-line hand
            # would dip its fingertips into the face it just left -- so it stays as held there.)
            wr = np.zeros(3) if kk == len(knots) - 3 else prev[4:]   # in line only for the rise
            if chain:
                prev = solve(i_, side, tg, np.concatenate([prev[:4], wr]), iters=40)[0]
            else:
                prev = solve_seeded(i_, side, tg, wr, extra=[prev])[0]
            kt.append(i_); kq.append(prev)
        kt.append(hi_); kq.append(Q[hi_])
        return kt, np.array(kq)

    def all_clear(i_, q_):
        w_, pc_, _, tip_, e_ = hand_of(i_, side, q_)
        return min(mesh_dist(w_), mesh_dist(pc_), mesh_dist(tip_), mesh_dist(e_), mesh_dist(0.5 * (e_ + w_)))

    def repair(seg, lo_, want=0.034):
        """Push every too-close pose of a chosen move along the joint direction that opens its
        clearance fastest, then re-smooth. Hand-placed waypoints kept clipping a different point
        of the handrail each time one was moved; this works on the path itself instead."""
        n_ = len(seg)
        idx = sorted(set(list(range(0, n_, 6)) + [n_ - 1]))
        kq = seg[idx].copy()
        for _ in range(14):
            moved = 0
            for k_, j in enumerate(idx):
                if not (0.14 * n_ < j < 0.90 * n_):        # the ends rest on a face by design
                    continue
                c = all_clear(lo_ + j, kq[k_])
                if c >= want:
                    continue
                g = np.zeros(7)
                for c_ in range(7):
                    qp = kq[k_].copy(); qp[c_] += 0.01
                    g[c_] = (all_clear(lo_ + j, qp) - c) / 0.01
                gn = float(np.linalg.norm(g))
                if gn > 1e-6:
                    kq[k_] = _clip(side, kq[k_] + g / gn * min(0.06, (want - c) / gn + 0.01))
                    moved += 1
            if not moved:
                break
        return PchipInterpolator(idx, kq, axis=0)(np.arange(n_))

    def judge(seg, lo_):
        """Score a candidate move with the gate's own measures. 0 = passes."""
        v = float(np.abs(np.diff(seg, axis=0)).max() * fps)
        bad = max(0.0, v - a.vcap) / a.vcap
        if bad > 1.0:
            return bad + 10.0                              # hopeless: skip the mesh queries
        cmin, xmin = 1e9, 1e9
        for k in range(0, len(seg), 8):
            i_ = lo_ + k
            w_, pc_, _, tip_, e_ = hand_of(i_, side, seg[k])
            xmin = min(xmin, chest_x(i_, w_))
            if 0.15 < k / len(seg) < 0.88:                 # the ends rest ON a face by design
                cmin = min(cmin, mesh_dist(w_), mesh_dist(0.5 * (e_ + w_)), mesh_dist(e_),
                           mesh_dist(pc_), mesh_dist(tip_))
        return bad + 2.0 * max(0.0, 0.028 - cmin) / 0.028 + 20.0 * max(0.0, 0.03 - xmin)

    # transits: joint-space monotone spline, widened until under the speed cap
    edges = [(None, holds[0])] + list(zip(holds[:-1], holds[1:])) + [(holds[-1], None)]
    for A, B in edges:
        if A is None:
            s0, s1 = 0, B[0]
            if s1 == 0:
                continue
            grow = 0
            while True:
                b = min(B[0] + grow, B[1] - 2)
                knots_t = [0, max(1, int(0.6 * b)), b]
                knots_q = [q4_base[0], waypoint(b, B[2], Q[b]), Q[b]]
                f = PchipInterpolator(knots_t, np.array(knots_q), axis=0)
                seg = f(np.arange(0, b + 1))
                if np.abs(np.diff(seg, axis=0)).max() * fps <= a.vcap or b >= B[1] - 2:
                    break
                grow += int(0.1 * fps)
            Q[0:b + 1] = seg; lab[0:b] = 0
            continue
        if B is None:
            if A[1] < T:                                   # nothing left to reach: ease to rest
                n = T - A[1]
                f = PchipInterpolator([0, max(1, int(1.2 * fps))], np.array([Q[A[1] - 1], REST[side]]), axis=0)
                Q[A[1]:] = f(np.minimum(np.arange(1, n + 1), int(1.2 * fps)))
            continue
        keep = int(a.min_hold * fps)
        best = None
        for grow in (0, int(0.15 * fps), int(0.30 * fps), int(0.50 * fps), int(0.75 * fps)):
            lo_ = min(A[1] - 1, max(A[0] + keep, A[1] - 1 - grow))
            hi_ = max(B[0], min(B[1] - keep, B[0] + grow))
            if hi_ - lo_ < int(a.min_transit * fps):
                continue
            for t_front in (0.80, 1.05):
                for t_above in (0.22,):
                    for back in (0.12, 0.18):
                        for wait in (True, False):
                            for chain in (True, False):
                                kt, kq = route(lo_, hi_, A[2], B[2], t_front, t_above, back, wait, chain)
                                seg = PchipInterpolator(kt, kq, axis=0)(np.arange(lo_, hi_ + 1))
                                sc = judge(seg, lo_)
                                if best is None or sc < best[0]:
                                    best = (sc, lo_, hi_, seg)
                                if sc == 0.0:
                                    break
                            if best[0] == 0.0: break
                        if best[0] == 0.0: break
                    if best[0] == 0.0: break
                if best[0] == 0.0: break
            if best is not None and best[0] == 0.0:
                break
        sc, lo_, hi_, seg = best
        if sc > 0.0:
            seg2 = repair(seg, lo_)
            sc2 = judge(seg2, lo_)
            if sc2 < sc:
                print(f"[move ] {side[0].upper()} {A[2]}->{B[2]}: repaired {sc:.2f} -> {sc2:.2f}", flush=True)
                sc, seg = sc2, seg2
        print(f"[move ] {side[0].upper()} {A[2]}->{B[2]}: steps {step50[lo_]:.0f}-{step50[hi_]:.0f}, "
              f"{'clean' if sc == 0.0 else f'best score {sc:.2f} (NOT clean)'}", flush=True)
        Q[lo_:hi_ + 1] = seg
        lab[lo_ + 1:hi_] = 0; err[lo_ + 1:hi_] = 0.0
    assert not np.isnan(Q).any(), "unplanned frames"
    lab[(lab > 0) & ((err > a.hold_tol) | (tilt > 25.0))] = 0   # never claim a hold not made flat
    return Q, lab, err, segs, tilt


plan = {s: plan_arm(s) for s in SIDES}
for s in SIDES:
    print(f"[sched] {s[0].upper()} reachable: " + "  ".join(
        f"{step50[x0]:.0f}-{step50[x1-1]:.0f}:{nm or '-'}" for x0, x1, nm in plan[s][3]))
    e_ = plan[s][2]; big = np.where(e_ > a.hold_tol)[0]
    if len(big):
        print(f"[sched] {s[0].upper()} hold frames over {a.hold_tol*1000:.0f} mm: {len(big)}, steps "
              f"{step50[big[0]]:.0f}..{step50[big[-1]]:.0f}, worst {e_.max()*1000:.0f} mm")
for s in SIDES:
    lab = plan[s][1]
    chg = [0] + [i for i in range(1, T) if lab[i] != lab[i - 1]] + [T]
    inv = {v: k for k, v in LABEL.items()}
    print(f"[plan] {s[0].upper()}: " + "  ".join(
        f"{step50[chg[k]]:.0f}-{step50[chg[k+1]-1]:.0f}:{inv.get(int(lab[chg[k]]), 'move')}"
        for k in range(len(chg) - 1) if chg[k + 1] - chg[k] > 6))

# ---- assemble the clip ---------------------------------------------------------------------
q_out = d["joint_pos"].copy()
q_out[:, wi] = waist
bp, bq = d["body_pos_w"].copy(), d["body_quat_w"].copy()
bp[:, pel], bq[:, pel] = PP, PQ
FKO = {}
for s in SIDES:
    aq = plan[s][0]
    for k, t in enumerate(K.ARM):
        q_out[:, jn.index(t.format(s=s))] = aq[:, k]
    FKO[s] = K.fk(PP, PQ, waist, aq, s)
    for t in K.ARM:
        b = K.CHAIN[t.format(s=s)]["b1"]
        bp[:, bn.index(b)], bq[:, bn.index(b)] = FKO[s][b], FKO[s][b + "_q"]
# The base clip does not agree with itself: its stored body positions sit up to 138 mm from the
# pose its own joint angles describe. Rebuild EVERY body from the joints so one pose is tracked.
_all = K.fk_all(PP, PQ, q_out, jn)
_dis = max(float(np.linalg.norm(_all[b][0] - d["body_pos_w"][:, i], axis=1).max()) for i, b in enumerate(bn)
           if any(k in b for k in ("hip", "knee", "ankle")))
print(f"[base] leg bodies stored up to {_dis*1000:.0f} mm from the pose the joints describe; rebuilt from joints")
for i, b in enumerate(bn):
    bp[:, i], bq[:, i] = _all[b]
bv = np.gradient(bp, 1.0 / fps, axis=0)
jv = np.gradient(q_out, 1.0 / fps, axis=0)
dq_ = K.qmul(np.roll(bq, -1, axis=0), K.qconj(bq)); dq_[-1] = dq_[-2]
dq_ = dq_ * np.sign(dq_[..., :1] + 1e-12)
bw = 2.0 * dq_[..., 1:] * fps

# ---- THE GATE ------------------------------------------------------------------------------
print("\n[gate] ---------------------------------------------------------------")
fails = []


def check(name, ok, detail):
    print(f"[gate] {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        fails.append(name)


viol = 0
vmax = 0.0
for s in SIDES:
    for k, t in enumerate(K.ARM):
        x = q_out[:, jn.index(t.format(s=s))]
        lo, hi = K.limits(t.format(s=s))
        viol += int(((x < lo) | (x > hi)).sum())
        vmax = max(vmax, float(np.abs(np.gradient(x, 1 / fps)).max()))
wv = float(np.abs(np.gradient(waist, 1 / fps, axis=0)).max())
check("joint limits", viol == 0, f"{viol} arm frames outside the USD limits")
check("arm joint speed", vmax <= a.gate_speed, f"peak {vmax:.2f} rad/s (gate {a.gate_speed})")
check("waist speed", wv <= 3.0, f"peak {wv:.2f} rad/s")
_leg = [i for i, n in enumerate(jn) if any(k in n for k in ("hip", "knee", "ankle"))]
_lv = np.abs(np.gradient(q_out[:, _leg], 1 / fps, axis=0)).max(1)
# Judged against the actuators, not a round number: the G1's leg motors are rated 20 rad/s
# (hip roll, knee) to 37 rad/s (ankles). 60% of the slowest is the bar.
_b = np.where(_lv > 12.0)[0]
check("leg joint speed (base clip's legs, unchanged)", len(_b) == 0,
      f"peak {_lv.max():.1f} rad/s against a 20 rad/s knee rating; {len(_b)} frames over 12")
_ps = np.linalg.norm(np.gradient(PP, 1 / fps, axis=0), axis=1)
print(f"[gate] info  pelvis speed peak {_ps.max():.2f} m/s")
behind, wsp = 0, 0.0
for s in SIDES:
    wl = K.qrot(K.qconj(FKO[s]["torso_q"]), FKO[s][f"{s}_wrist_yaw_link"] - FKO[s]["torso_p"])
    behind += int((wl[:, 0] < 0.0).sum())
    wsp = max(wsp, float(np.linalg.norm(np.gradient(FKO[s][f"{s}_wrist_yaw_link"], 1 / fps, axis=0), axis=1).max()))
check("wrist in front of the torso", behind == 0, f"{behind} frames with the wrist behind the chest plane")
check("wrist speed", wsp <= 2.0, f"peak {wsp:.2f} m/s")
cl_min, cl_where, hd_min, hd_where, below, tilt_max = 1e9, None, 1e9, None, 0, 0.0
inv = {v: k for k, v in LABEL.items()}
near = int(0.20 * fps)
for s in SIDES:
    lab = plan[s][1]
    W, E = FKO[s][f"{s}_wrist_yaw_link"], FKO[s][f"{s}_elbow_link"]
    WQ = FKO[s][f"{s}_wrist_yaw_link_q"]
    PC = W + K.qrot(WQ, HAND[s]["pc"])
    TIP = W + K.qrot(WQ, HAND[s]["tip"])
    held_near = np.convolve((lab > 0).astype(float), np.ones(2 * near + 1), mode="same") > 0
    tilt_max = max(tilt_max, float(plan[s][4][lab > 0].max()) if (lab > 0).any() else 0.0)
    for i in range(0, T, 3):
        for nm, p_ in (("forearm", 0.5 * (W[i] + E[i])), ("elbow", E[i])):
            dd = mesh_dist(p_)
            if dd < cl_min:
                cl_min, cl_where = dd, f"{s} {nm} at step {step50[i]:.0f}"
        if not held_near[i]:                               # a hand in the air, away from any hold
            for nm, p_ in (("wrist", W[i]), ("palm", PC[i]), ("fingertip", TIP[i])):
                dd = mesh_dist(p_)
                if dd < hd_min:
                    hd_min, hd_where = dd, f"{s} {nm} at step {step50[i]:.0f}"
        if lab[i] and PC[i, 2] < HOLDS[inv[int(lab[i])]]["z"] - 0.004:
            below += 1
check("arm clear of the ladder", cl_min >= a.gate_clear,
      f"elbow/forearm closest {cl_min*1000:.1f} mm ({cl_where}); gate {a.gate_clear*1000:.0f} mm")
check("moving hand clear of the ladder", hd_min >= a.gate_clear,
      f"wrist/palm/fingertip closest {hd_min*1000:.1f} mm ({hd_where}) while not on a hold")
check("palm flat on its hold", tilt_max <= 25.0, f"worst palm tilt while holding {tilt_max:.0f} deg")
check("hand above its hold", below == 0, f"{below} sampled frames below the face they hold")
held = np.stack([plan[s][1] > 0 for s in SIDES], 1)
# Reported, not gated: how much of the clip has a hold is set by the BODY trajectory (see the
# reach schedule above), and a threshold here would only reward claiming holds that are not made.
print(f"[gate] info  hands on holds: at least one hand holding {held.any(1).mean()*100:.1f}% of the clip, both {held.all(1).mean()*100:.1f}%; "
      f"both off {(~held.any(1)).mean()*100:.1f}%")
herr = max(float(plan[s][2][plan[s][1] > 0].max()) for s in SIDES)
print(f"[gate] info  worst held-hand position error {herr*1000:.1f} mm; "
      f"hold counts " + ", ".join(f"{n} {int(sum((plan[s][1] == LABEL[n]).sum() for s in SIDES))}" for n in ORDER[3:]))

out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
n50 = d["object_contact"].shape[0]
oc = np.zeros_like(d["object_contact"])
for i in range(n50):
    f = min(int(i / 50 * fps), T - 1)
    oc[i] = [plan["left"][1][f], plan["right"][1][f]]
np.savez(out / "motion.npz", joint_pos=q_out, joint_vel=jv, body_pos_w=bp, body_quat_w=bq,
         body_lin_vel_w=bv, body_ang_vel_w=bw, object_contact=oc)
shutil.copy(base / "meta.json", out / "meta.json")
(out / "holds.json").write_text(json.dumps(dict(holds=HOLDS, label=LABEL, contact_idx=0,
                                                objects=["dummy"] + ORDER, gate_failed=fails), indent=1))
print(f"\n[out ] {out/'motion.npz'}  --  {'ACCEPTED' if not fails else 'REJECTED: ' + ', '.join(fails)}")
sys.exit(0 if not fails else 2)
