"""Author the climb clip's arms PHASE BY PHASE: a hand only moves while both feet are planted.

build_keyframe_motion.py stays runnable. It chose holds by reach alone, so the hands changed
holds whenever the geometry allowed -- including in the middle of a step up, which left the climb
to step 3 with no hand on anything. This variant takes the footwork as the clock:

* The base clip is RETIMED: a smooth pause is inserted wherever the hands need time, so a hand
  move happens with the body standing still on a tread.
* The SCHEDULE is explicit. Every climb (first foot leaves .. second foot lands) has both hands on
  a fixed hold; hands change holds one at a time inside a pause, never both off together.
* A hold is a FIXED point on the ladder, chosen once for the whole hold. (The earlier builder let
  the palm follow the shoulder along the face, i.e. slide.)
* Standing on step 2 the handrail is out of palm reach, and standing on step 3 the platform is;
  so from step 2 the hand goes up to the handrail's near top edge on its finger pads ("hook"),
  and is re-seated palm-down once both feet are on step 3.
* The gate adds: both hands hold through every foot step.

    python scripts/climb/build_phased_motion.py --base <clip> --out <dir> --mesh <npz>
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
ap.add_argument("--lean_on", type=float, nargs=2, default=[270, 298], help="BASE-clip 50 Hz steps: ramp in")
ap.add_argument("--lean_off", type=float, nargs=2, default=[330, 370], help="BASE-clip 50 Hz steps: ramp out")
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
ap.add_argument("--pause", action="append", default=None,
                help="STEP:SECONDS -- hold the body still at this BASE-clip step for this long. "
                     "Three are expected, in order: before the first foot lifts, on step 1, on step 2.")
ap.add_argument("--climb", type=float, nargs="+", default=[47, 138, 219, 300, 386, 468, 575, 645, 752, 820],
                help="BASE-clip step pairs: first foot leaves .. second foot has landed, per tread")
ap.add_argument("--hook_pitch", type=float, default=40.0, help="hand pitch on the handrail edge from step 2, deg")
ap.add_argument("--roll_z", type=float, nargs=2, default=[1.575, 1.665],
                help="shoulder heights between which the hand rolls from finger pads to flat palm")
ap.add_argument("--release_at", type=float, default=821.0, help="BASE-clip step: let go of the handrail on top")
a = ap.parse_args()
if a.pause is None:
    a.pause = ["44:0.5", "146:3.6", "302:4.4"]
PAUSES = sorted((float(x.split(":")[0]), float(x.split(":")[1])) for x in a.pause)

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

# ---- retime: insert the pauses ----------------------------------------------------------------
RAMP = int(0.30 * fps)
_rate, _cur, PLATEAU = [], 0.0, []
for _ps, _pd in PAUSES:
    _pf = _ps / 50.0 * fps
    _n1 = int(round(_pf - RAMP / 2.0 - _cur))
    assert _n1 > 0, "pauses too close together"
    _u = (np.arange(RAMP) + 0.5) / RAMP
    _dn = 0.5 * (1 + np.cos(np.pi * _u))
    _dn *= (RAMP / 2.0) / _dn.sum()
    _rate += [1.0] * _n1 + list(_dn)
    PLATEAU.append((len(_rate), len(_rate) + int(_pd * fps)))
    _rate += [0.0] * int(_pd * fps) + list(_dn[::-1])
    _cur += _n1 + RAMP
_rate += [1.0] * int(T - 1 - _cur)
OLD = np.clip(np.concatenate([[0.0], np.cumsum(_rate)]), 0, T - 1)       # base frame of each new frame
_i0 = np.floor(OLD).astype(int); _i1 = np.minimum(_i0 + 1, T - 1); _w = OLD - _i0


def _warp(x, quat=False):
    wv = _w.reshape((-1,) + (1,) * (x.ndim - 1))
    x0, x1 = x[_i0], x[_i1]
    if quat:
        x1 = x1 * np.sign((x0 * x1).sum(-1, keepdims=True) + 1e-12)
    y = (1 - wv) * x0 + wv * x1
    return y / np.linalg.norm(y, axis=-1, keepdims=True) if quat else y


d["joint_pos"] = _warp(d["joint_pos"])
d["body_pos_w"] = _warp(d["body_pos_w"])
d["body_quat_w"] = _warp(d["body_quat_w"], quat=True)
T_BASE, T = T, len(OLD)
PP, PQ = d["body_pos_w"][:, pel].copy(), d["body_quat_w"][:, pel].copy()
step50 = np.arange(T) / fps * 50
ostep = OLD / fps * 50                                    # the BASE-clip step each frame shows


def new_of(base_step):
    """First new frame that shows this base-clip step."""
    return int(np.searchsorted(ostep, base_step, side="left"))


print(f"[time] {len(PAUSES)} pauses inserted: {T_BASE} -> {T} frames ({T/fps:.1f} s); body still during new steps "
      + ", ".join(f"{step50[p0]:.0f}-{step50[p1]:.0f}" for p0, p1 in PLATEAU))
CLIMBS = [(new_of(a.climb[k]), new_of(a.climb[k + 1])) for k in range(0, len(a.climb), 2)]
print("[time] foot steps (new steps): " + ", ".join(f"{step50[c0]:.0f}-{step50[c1]:.0f}" for c0, c1 in CLIMBS))

# ---- crouch: lower the pelvis, keep the feet -----------------------------------------------
if a.crouch > 0:
    cr = (np.clip((ostep - a.crouch_on[0]) / (a.crouch_on[1] - a.crouch_on[0]), 0, 1)
          * np.clip((a.crouch_off[1] - ostep) / (a.crouch_off[1] - a.crouch_off[0]), 0, 1))
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
ramp = (np.clip((ostep - a.lean_on[0]) / (a.lean_on[1] - a.lean_on[0]), 0, 1)
        * np.clip((a.lean_off[1] - ostep) / (a.lean_off[1] - a.lean_off[0]), 0, 1))
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
# From step 2 the handrail is beyond PALM reach, so the hand goes up to its near top edge on the
# finger pads. Same object, same label; a different point of the hand and a pitched hand.
HOLDS["hook"] = dict(HOLDS["handrail"])
LABEL["hook"] = LABEL["handrail"]
_HP = np.radians(a.hook_pitch)
HOOK_F = np.array([np.cos(_HP), 0.0, np.sin(_HP)])        # where the fingers point
HOOK_N = np.array([np.sin(_HP), 0.0, -np.cos(_HP)])       # where the palm faces
HOOK_U = 0.17                                             # finger-pad point, metres ahead of the wrist

# ---- shoulders (after the lean) and hold targets -------------------------------------------
SH = {s: K.fk(PP, PQ, waist, np.zeros((T, 7)), s)[f"{s}_shoulder_pitch_link"] for s in SIDES}


LAM = {}
for _s in SIDES:
    _x = np.clip((SH[_s][:, 2] - a.roll_z[0]) / (a.roll_z[1] - a.roll_z[0]), 0, 1)
    LAM[_s] = np.maximum.accumulate(_x * _x * (3 - 2 * _x))
_TQ = K.fk(PP, PQ, waist, np.zeros((T, 7)), "left")
TORSO_P, TORSO_Q = _TQ["torso_p"], _TQ["torso_q"]


def chest_x(i, p):
    """How far in front of the chest plane a world point is, at frame i."""
    return float(K.qrot(K.qconj(TORSO_Q[i]), p - TORSO_P[i])[0])


def front_of_body(i, p, side):
    """'In front' for a LEANING torso. The chest plane tips back over the head when the torso
    leans forward, so a hand reaching straight up to the handrail reads as behind it while it is
    plainly ahead of the shoulder. A point is behind the body only if it is behind the chest
    plane AND behind the shoulder along the robot's horizontal heading."""
    h = K.qrot(TORSO_Q[i], np.array([1.0, 0.0, 0.0])); h[2] = 0.0
    h = h / np.linalg.norm(h)
    return max(chest_x(i, p), float((p - SH[side][i]) @ h))


# The Dex3 hand, in the wrist_yaw_link frame, read off the USD: the palm pad is 87 mm long (x) by
# 88 mm wide (z); its FACE is the -y side on the left hand (the fingers curl toward -y and the
# thumb sits there) and the +y side on the right. PC is the centre of that face.
HAND = {"left": dict(pc=np.array([0.0865, -0.017, 0.0]), n=np.array([0.0, -1.0, 0.0]),
                     tip=np.array([0.21, -0.005, 0.0])),
        "right": dict(pc=np.array([0.0865, 0.017, 0.0]), n=np.array([0.0, 1.0, 0.0]),
                      tip=np.array([0.21, 0.005, 0.0]))}
PALM_HALF = 0.045
CUR = {}


HOOK_U1 = HAND["left"]["pc"][0] - PALM_HALF               # the palm point that ends up on the edge


def roll_of(side, i):
    return 0.0 if i is None else float(LAM[side][i])


def hook_contact(c, lam):
    """The pivot: the rounded near top edge of the bar; flush with the top once the palm is flat."""
    return c + lam * np.array([0.006, 0.0, 0.006])


def set_spec(name, side, i=None):
    """Which point of the hand rests on the hold, and how the hand is turned there. On the
    handrail the hand ROLLS over the near top edge as the body comes up: finger pads and a
    pitched hand from step 2 (lam 0), the palm flat on top by the time the robot stands on
    step 3 (lam 1). The edge stays under the hand throughout; nothing is let go."""
    if name == "hook":
        lam = roll_of(side, i)
        th = (1.0 - lam) * _HP
        CUR.update(pt=np.array([(1 - lam) * HOOK_U + lam * HOOK_U1, HAND[side]["pc"][1] * (0.8 + 0.2 * lam), 0.0]),
                   n=np.array([np.sin(th), 0.0, -np.cos(th)]), f=np.array([np.cos(th), 0.0, np.sin(th)]),
                   name=name, lam=lam)
    else:
        CUR.update(pt=HAND[side]["pc"], n=np.array([0.0, 0.0, -1.0]), f=None, name=name)


def proxy_at(name, c, sp, side):
    """Where the WRIST sits with the hand on fixed contact `c`, seen from shoulder `sp`."""
    if name == "hook":
        return c - HOOK_U * HOOK_F - 0.014 * HOOK_N      # as first placed, from step 2
    hv = (sp - c) * np.array([1.0, 1.0, 0.0])
    hn = float(np.linalg.norm(hv))
    back = hv / hn * min(HAND[side]["pc"][0], hn) if hn > 1e-6 else np.zeros(3)
    return c + back + np.array([0.0, 0.0, 0.020])


def fixed_contact(name, side, s, e):
    """ONE point for the whole hold: the candidate whose worst reach over the hold is smallest."""
    best = None
    for i in np.unique(np.linspace(s, e - 1, 14).astype(int)):
        c = contact_on(name, SH[side][i], side)
        worst = max(float(np.linalg.norm(proxy_at(name, c, SH[side][j], side) - SH[side][j]))
                    for j in range(s, e, 6))
        if best is None or worst < best[0]:
            best = (worst, c)
    return best[1], best[0]


def contact_on(name, sp, side):
    """Where the CENTRE of the palm face rests on this hold, nearest to this shoulder."""
    h = HOLDS[name]
    ylo, yhi = h["y"]
    if name == "hook":                                     # the rounded near top edge of the bar
        return np.array([h["x"][0] - 0.006, np.clip(sp[1], *((0.03, a.rail_keepout) if side == "left"
                                                             else (-a.rail_keepout, -0.03))), h["z"] - 0.006])
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


def phased():
    """Hold list per side, [start, end, name]. Hands move one at a time inside a pause (left
    first), or -- for the re-seat on step 3 -- inside the planted window, and at no other time."""
    (p0a, p0b), (p1a, p1b), (p2a, p2b) = PLATEAU
    gap = int(0.08 * fps)
    out = {}
    rel = new_of(a.release_at)
    for k, sd in enumerate(SIDES):
        h1 = (p1b - p1a - 3 * gap) // 2                    # each hand's share of pause 1
        h2 = (p2b - p2a - 3 * gap) // 2
        m1 = (p1a + gap + k * (h1 + gap), p1a + gap + k * (h1 + gap) + h1)
        m2 = (p2a + gap + k * (h2 + gap), p2a + gap + k * (h2 + gap) + h2)
        out[sd] = [[p0a, m1[0], "step4"], [m1[1], m2[0], "platform"], [m2[1], rel, "hook"]]
    return out


SCHED = phased()
for s_ in SIDES:
    print(f"[sched] {s_[0].upper()}: " + "  ".join(f"{step50[x0]:.0f}-{step50[x1-1]:.0f}:{nm}" for x0, x1, nm in SCHED[s_]))
    for x0, x1, nm in SCHED[s_]:
        c_, w_ = fixed_contact(nm, s_, x0, x1 if nm != "hook" else min(x1, x0 + int(2 * fps)))
        print(f"[reach] {s_[0].upper()} {nm:9s} fixed at ({c_[0]:+.3f},{c_[1]:+.3f},{c_[2]:.3f}); wrist needs at most "
              f"{w_:.3f} m of {a.reach - a.palm_off:.3f}")
if a.plan_only:
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


_XH = np.array([1.0, 0.0, 0.0])


def _palm_err(i, side, q, contact):
    o = K.fk(PP[i], PQ[i], waist[i], q, side)
    wp, wq = o[f"{side}_wrist_yaw_link"], o[f"{side}_wrist_yaw_link_q"]
    pc = wp + K.qrot(wq, CUR["pt"])
    n = K.qrot(wq, HAND[side]["n"])
    rot = np.cross(n, CUR["n"])
    if CUR["f"] is not None:                               # a pitched hand: heading matters too
        rot = 0.5 * (rot + np.cross(K.qrot(wq, _XH), CUR["f"]))
    return np.concatenate([contact - 0.003 * CUR["n"] - pc, 0.25 * rot]), pc, n


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
    tilt = float(np.degrees(np.arccos(np.clip(n @ CUR["n"], -1, 1))))
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
    segs = [list(x) for x in SCHED[side]]
    q4_base = d["joint_pos"][:, [jn.index(n) for n in ARMJ[side]]]
    Q = np.full((T, 7), np.nan)
    err = np.zeros(T)
    tilt = np.zeros(T)
    lab = np.zeros(T, dtype=int)
    cap = 5.5 / fps        # a fixed hand under a body rising at 1 m/s needs ~4 rad/s; no more
    holds = [s + [fixed_contact(s[2], side, s[0], s[1] if s[2] != "hook" else min(s[1], s[0] + int(2 * fps)))[0]]
             for s in segs]
    for s, e, name, ct0 in holds:
        set_spec(name, side, s)
        ct = ct0
        Q[s], err[s], tilt[s] = solve_palm_seeded(s, side, ct)
        for i in range(s + 1, e):
            set_spec(name, side, i)
            if name == "hook":
                ct = hook_contact(ct0, CUR["lam"])
                if CUR["lam"] >= 1.0:
                    CUR["f"] = None                        # flat on top: heading is free again
            Q[i], err[i], tilt[i] = solve_palm(i, side, ct, Q[i - 1], iters=4, dq_cap=cap)
            if i % 2 == 0:
                qs = steer_clear(i, side, ct, Q[i])
                Q[i] = Q[i - 1] + np.clip(qs - Q[i - 1], -cap, cap)
                ee, _, nn = _palm_err(i, side, Q[i], ct)
                err[i] = float(np.linalg.norm(ee[:3]))
                tilt[i] = float(np.degrees(np.arccos(np.clip(nn @ CUR["n"], -1, 1))))
        lab[s:e] = LABEL[name]
        print(f"[hold ] {side[0].upper()} {name:9s} {step50[s]:.0f}-{step50[e-1]:.0f}: start err {err[s]*1000:.0f} mm, "
              f"max {err[s:e].max()*1000:.0f} mm; palm tilt start {tilt[s]:.0f} deg, max {tilt[s:e].max():.0f} deg; "
              f"wrist q0 {np.round(Q[s][4:], 2)}")

    def waypoint(i, name, prev, c_):
        t = proxy_at(name, c_, SH[side][i], side) + np.array([0.0, 0.0, a.lift])
        return solve(i, side, t, np.concatenate([prev[:4], Q[i][4:]]), iters=40)[0]

    def route(lo_, hi_, A_, B_, t_front, t_above, back, wait, chain, CA, CB):
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

        tb = proxy_at(B_, CB, SH[side][hi_], side)
        if B_ == "hook":                                   # arrive along the fingers, from below-behind
            arr = [-0.10 * HOOK_F - 0.04 * HOOK_N, -0.05 * HOOK_F - 0.02 * HOOK_N]
            rise = np.array([-0.04, 0.0, -0.14])
        else:
            arr = [np.array([-0.09, 0.0, 0.11]), np.array([0.0, 0.0, 0.05])]
            rise = None
        # A flat hand reaches 21 cm ahead of the wrist, so on a face its fingertips sit UNDER
        # whatever overhangs it (the platform lip above step 4, the handrail above the platform).
        # Draw the hand back before it rises, as from under a shelf.
        ta = proxy_at(A_, CA, SH[side][lo_], side)
        # ... and LEVEL: with the wrist joints frozen the hand pitches nose-down as the arm
        # retracts, and the fingertip was measured dropping 9 mm INTO the platform and dragging
        # along it. So the first two waypoints are flat-palm poses: straight up, then back.
        knots = [(lo_ + min(int(0.22 * fps), n // 6),
                  ("palm", lambda i_: CA + np.array([0.0, 0.0, 0.07]))),
                 (lo_ + min(int(0.50 * fps), n // 3),
                  ("palm", lambda i_: CA + np.array([-back, 0.0, 0.10])))]
        if wait and n > int((t_front + 1.2) * fps):
            knots += [(lo_ + int(1.00 * fps), ready), (hi_ - int((t_front + 0.35) * fps), ready)]
        knots += [(hi_ - int(t_front * fps),
                   lambda i_: (tb + rise) if rise is not None else
                   np.array([min(ta[0] - back, tb[0] - back), tb[1], tb[2] + a.lift])),
                  # ... and ARRIVE level too: over the new face from behind and above, then down.
                  # Turning flat while still moving forward swept the fingertip through the bar.
                  (hi_ - int((t_above + 0.24) * fps),
                   ("dst", lambda i_: CB + arr[0])),
                  (hi_ - int(t_above * fps),
                   ("dst", lambda i_: CB + arr[1]))]
        kt, kq, prev = [lo_], [Q[lo_]], Q[lo_]
        for kk, (i_, fn) in enumerate(knots):
            if not (kt[-1] < i_ < hi_):
                continue
            if isinstance(fn, tuple):                      # a level, palm-down waypoint
                seed = Q[hi_] if fn[0] == "dst" else prev  # same wrist branch as the hold it serves
                set_spec(B_ if fn[0] == "dst" else A_, side)
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
            wr = prev[4:]
            if kk == len(knots) - 3:                       # the rise in front of the new face
                wr = Q[hi_][4:] if B_ == "hook" else np.zeros(3)
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
            xmin = min(xmin, front_of_body(i_, w_, side))
            if int(0.21 * fps) < k < len(seg) - int(0.21 * fps):   # the ends rest ON a face by design
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
                # Tread 4 sits under the platform's lip: come in level from the robot's side,
                # not down from above (the fingertips passed 2 mm from the lip that way).
                set_spec(B[2], side, b)
                k1, k2 = max(1, int(0.55 * b)), max(2, int(0.85 * b))
                knots_t = [0, k1, k2, b]
                # The base clip opens with the Dex3 fingertips 18 mm from tread 4's nose; open
                # instead with the hands level, a hand's length short of the tread.
                knots_q = [solve_palm(0, side, B[3] + np.array([-0.20, 0.0, 0.05]), Q[b], iters=60)[0],
                           solve_palm(k1, side, B[3] + np.array([-0.14, 0.0, 0.05]), Q[b], iters=40)[0],
                           solve_palm(k2, side, B[3] + np.array([-0.04, 0.0, 0.03]), Q[b], iters=40)[0], Q[b]]
                f = PchipInterpolator(knots_t, np.array(knots_q), axis=0)
                seg = f(np.arange(0, b + 1))
                if np.abs(np.diff(seg, axis=0)).max() * fps <= a.vcap or b >= B[1] - 2:
                    break
                grow += int(0.1 * fps)
            Q[0:b + 1] = seg; lab[0:b] = 0
            B[0] = b
            continue
        if B is None:
            if A[1] < T:                                   # nothing left to reach: ease to rest
                n = T - A[1]
                f = PchipInterpolator([0, max(1, int(1.2 * fps))], np.array([Q[A[1] - 1], REST[side]]), axis=0)
                Q[A[1]:] = f(np.minimum(np.arange(1, n + 1), int(1.2 * fps)))
            continue
        keep = int(a.min_hold * fps)
        best = None
        for grow in (0,):                                  # the window is the schedule's, not ours
            lo_, hi_ = A[1] - 1, B[0]
            for t_front in (0.55, 0.70, 0.85):
                for t_above in (0.20,):
                    for back in (0.12, 0.18, 0.24, 0.30):
                        for wait in (True, False):
                            for chain in (True, False):
                                kt, kq = route(lo_, hi_, A[2], B[2], t_front, t_above, back, wait, chain, A[3], B[3])
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
    return Q, lab, err, holds, tilt


plan = {s: plan_arm(s) for s in SIDES}
for s in SIDES:
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
    behind += int(sum(front_of_body(i, FKO[s][f"{s}_wrist_yaw_link"][i], s) < 0.0 for i in range(T)))
    wsp = max(wsp, float(np.linalg.norm(np.gradient(FKO[s][f"{s}_wrist_yaw_link"], 1 / fps, axis=0), axis=1).max()))
check("wrist in front of the body", behind == 0, f"{behind} frames with the wrist behind both the chest plane and the shoulder")
check("wrist speed", wsp <= 2.0, f"peak {wsp:.2f} m/s")
cl_min, cl_where, hd_min, hd_where, below, tilt_max = 1e9, None, 1e9, None, 0, 0.0
wr_min, wr_where = 1e9, None
HOOKED = {}
for s in SIDES:
    HOOKED[s] = np.zeros(T, dtype=bool)
    for x0, x1, nm, _c in plan[s][3]:
        if nm == "hook":
            HOOKED[s][x0:x1] = LAM[s][x0:x1] < 1.0
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
        if lab[i] and not HOOKED[s][i] and PC[i, 2] < HOLDS[inv[int(lab[i])]]["z"] - 0.004:
            below += 1
        if lab[i]:
            dd = mesh_dist(W[i])
            if dd < wr_min:
                wr_min, wr_where = dd, f"{s} wrist at step {step50[i]:.0f}"
check("arm clear of the ladder", cl_min >= a.gate_clear,
      f"elbow/forearm closest {cl_min*1000:.1f} mm ({cl_where}); gate {a.gate_clear*1000:.0f} mm")
check("moving hand clear of the ladder", hd_min >= a.gate_clear,
      f"wrist/palm/fingertip closest {hd_min*1000:.1f} mm ({hd_where}) while not on a hold")
check("palm flat on its hold", tilt_max <= 25.0, f"worst palm tilt while holding {tilt_max:.0f} deg")
check("hand above its hold", below == 0, f"{below} sampled frames below the face they hold")
check("holding wrist clear of the ladder", wr_min >= 0.012, f"closest {wr_min*1000:.1f} mm ({wr_where})")
# THE RULE: a hand may leave its hold only while both feet are planted. So through every foot
# step -- first foot leaving to second foot landed -- both hands must be on a hold.
_both = np.stack([plan[s][1] > 0 for s in SIDES], 1).all(1)
_one = np.stack([plan[s][1] > 0 for s in SIDES], 1).any(1)
_bad, _txt = 0, []
for c0, c1 in CLIMBS:
    nb_ = int((~_both[c0:c1]).sum()); _bad += nb_
    _txt.append(f"{step50[c0]:.0f}-{step50[c1]:.0f}: {'both hands on' if nb_ == 0 else f'{nb_/fps:.2f} s without both hands'}")
check("both hands hold through every foot step", _bad == 0, "; ".join(_txt))
_first = min(plan[s][3][0][0] for s in SIDES); _last = min(plan[s][3][-1][1] for s in SIDES)
_none = int((~_one[_first:_last]).sum())
check("never both hands off between first grip and top-out", _none == 0, f"{_none/fps:.2f} s with no hand on a hold")
_an = [bn.index(f"{s}_ankle_roll_link") for s in SIDES]
_fs = np.linalg.norm(np.gradient(bp[:, _an], 1 / fps, axis=0), axis=2).max(1)
_mv = ~_both[_first:_last]
print(f"[gate] info  foot speed while a hand is off its hold: peak {(_fs[_first:_last][_mv].max() if _mv.any() else 0.0):.3f} m/s")
held = np.stack([plan[s][1] > 0 for s in SIDES], 1)
# Reported, not gated: how much of the clip has a hold is set by the BODY trajectory (see the
# reach schedule above), and a threshold here would only reward claiming holds that are not made.
print(f"[gate] info  hands on holds: at least one hand holding {held.any(1).mean()*100:.1f}% of the clip, both {held.all(1).mean()*100:.1f}%; "
      f"both off {(~held.any(1)).mean()*100:.1f}%")
herr = max(float(plan[s][2][plan[s][1] > 0].max()) for s in SIDES)
print(f"[gate] info  worst held-hand position error {herr*1000:.1f} mm; "
      f"hold counts " + ", ".join(f"{n} {int(sum((plan[s][1] == LABEL[n]).sum() for s in SIDES))}" for n in ORDER[3:]))

out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
n50 = int(T / fps * 50)
oc = np.zeros((n50,) + d["object_contact"].shape[1:], dtype=d["object_contact"].dtype)
for i in range(n50):
    f = min(int(i / 50 * fps), T - 1)
    oc[i] = [plan["left"][1][f], plan["right"][1][f]]
np.savez(out / "motion.npz", joint_pos=q_out, joint_vel=jv, body_pos_w=bp, body_quat_w=bq,
         body_lin_vel_w=bv, body_ang_vel_w=bw, object_contact=oc)
shutil.copy(base / "meta.json", out / "meta.json")
(out / "holds.json").write_text(json.dumps(dict(holds=HOLDS, label=LABEL, contact_idx=0, pauses=PAUSES,
                                                schedule={s: [[float(step50[x0]), float(step50[x1 - 1]), nm, [float(v) for v in c_]]
                                                              for x0, x1, nm, c_ in plan[s][3]] for s in SIDES},
                                                objects=["dummy"] + ORDER, gate_failed=fails), indent=1))
print(f"\n[out ] {out/'motion.npz'}  --  {'ACCEPTED' if not fails else 'REJECTED: ' + ', '.join(fails)}")
sys.exit(0 if not fails else 2)
