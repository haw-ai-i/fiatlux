"""S10 climb clip, second design: the bulb arm helps, on its WRIST.

The right hand holds the bulb palm-up, so its palm is never a hold. Instead the back of the right
wrist rests on the near edge of tread 4, then the platform, then (from tread 3) the handrail, with
the hand pitched fingers-up over the hold so that only the wrist touches. Between rests the arm
moves in joint space. Pelvis, legs, waist and the left arm are the input clip's.

Reach decides the schedule (straight arm 0.382 m, measured on the rebuild 72 motion): the handrail
needs 0.55 m from tread 2, so the wrist leaves the platform before the step to tread 3 and only
goes to the handrail once both feet are on tread 3.

Reported for every frame: position and tilt error of each rest, and the clearance of a hand-plus-
bulb proxy (bulb glass r 40 mm at the palm, in any orientation) and of the back of the hand.

    python make_bulb_wristrest_clip.py --clip <in> --out <out>
"""
import argparse, json, math, pathlib, shutil, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--insets", type=float, nargs=3, default=[0.0, 0.045, 0.02], help="wrist this far past the near edge of tread 4, platform, handrail, m")
ap.add_argument("--y", type=float, default=-0.14, help="world y of the right wrist on its holds")
ap.add_argument("--inset", type=float, default=0.030, help="wrist this far past a hold's near edge, m")
ap.add_argument("--pitch", type=float, nargs=3, default=[0.0, 8.0, 8.0], help="fingers-up pitch of the hand on tread 4, platform, handrail, deg")
ap.add_argument("--rests", type=float, nargs=6, default=[52, 190, 620, 890, 0, 0],
                help="clip steps: tread-4 rest start end, platform rest start end, handrail rest start end; start >= end skips one")
ap.add_argument("--carry", type=float, nargs=3, default=[-0.03, -0.31, 0.06], help="carried wrist position in the pelvis frame, m")
ap.add_argument("--chest", type=float, nargs=3, default=[0.20, -0.13, 0.24], help="waypoint in front of the chest, pelvis frame, m")
ap.add_argument("--elbow_out", type=float, default=-0.9, help="shoulder roll the arm leans toward where the task leaves it free, rad")
ap.add_argument("--elbow_w", type=float, default=0.15)
ap.add_argument("--carry_pitch", type=float, default=0.0, help="fingers-up pitch of the hand while carried, deg. In physics the wrist sags "
                "13-33 deg fingers-DOWN under the benchmark's wrist gains and the bulb can roll off the fingers.")
ap.add_argument("--carry_z_early", type=float, default=None, help="carry height (pelvis frame, m) before --carry_switch; default: the same as --carry")
ap.add_argument("--carry_switch", type=float, default=920.0, help="clip step around which the carry height changes")
ap.add_argument("--move", type=float, default=2.4, help="s for the arm to go between its carry pose and a rest")
ap.add_argument("--settle", type=float, default=0.4, help="s after arriving before a rest is labelled a hold")
ap.add_argument("--lift", type=float, default=0.07, help="m the wrist rises mid-way through a move between rests")
a = ap.parse_args()

S = "right"
PC = np.array([0.005, -0.030, 0.0])            # back of the right wrist, wrist_yaw_link frame
PALM_N = np.array([0.0, 1.0, 0.0])             # right palm face is the +y side
BACK_OF_HAND = np.array([0.10, -0.024, 0.0])
# hand + bulb, MEASURED in the benchmark (measure_bulb_in_hand.py, S10 after its settle event): the bulb LIES
# ACROSS the palm. In the right wrist_yaw_link frame its root is at (0.119, 0.054..0.076, -0.131) with its
# axis along +z: glass (r 40 mm) centred at (0.117, 0.058, 0.0), cap (r 21 mm) from z -0.058 to -0.095.
# Proxy: glass and cap a little oversize, plus the curled fingertips around the glass.
BULB = [(np.array([0.117, 0.060, 0.0]), 0.046), (np.array([0.116, 0.066, -0.055]), 0.028), (np.array([0.115, 0.070, -0.085]), 0.028),
        (np.array([0.165, 0.060, 0.030]), 0.028), (np.array([0.165, 0.060, -0.030]), 0.028)]
BULB_IN_WRIST = (np.array([0.1192, 0.0540, -0.1310]), np.array([0.70710678, 0.0, 0.0, 0.70710678]))   # root pos, quat (wxyz)
NOSE = {"step4": (-0.3693, 0.9441), "platform": (-0.3036, 1.1517), "handrail": (-0.0395, 1.8609)}
LAB = {"step4": 4, "platform": 5, "handrail": 6}

# ---- ladder boxes (the trainer's model) ----
TOPS = (0.2397, 0.4742, 0.7092, 0.9441); X4 = (-0.3650, -0.2973); DX = 0.0857
B = []
def slab(x0, x1, hw, top, bot, name): B.append((name, np.array([(x0 + x1) / 2, 0.0, (top + bot) / 2]), np.array([(x1 - x0) / 2, hw, (top - bot) / 2]), 0.0))
for k, t in enumerate(TOPS): slab(X4[0] - (3 - k) * DX, X4[1] - (3 - k) * DX, 0.2, t, t - 0.0356, f"tread {k+1}")
slab(-0.3036, 0.0762, 0.2, 1.1517, 1.1517 - 0.0428, "platform")
slab(-0.0421, 0.0701, 0.247, 1.8419, 1.8105, "tray floor"); slab(-0.0395, -0.0280, 0.247, 1.8609, 1.8419, "tray rim"); slab(0.06, 0.0701, 0.247, 1.8609, 1.8419, "tray back rim")
for sy in (1, -1):
    B.append((f"stringer {'L' if sy > 0 else 'R'}", np.array([-0.598 + (0.915 - 0.30) * 0.3623 + 0.0405, 0.225 * sy, 0.915]),
              np.array([1.83 / math.sin(math.radians(70.08)) / 2, 0.020, 0.0381]), -70.08))
    B.append((f"rear leg {'L' if sy > 0 else 'R'}", np.array([0.165 - (0.75 - 0.30) * 0.1755 + 0.037, 0.225 * sy, 0.75]),
              np.array([1.50 / math.sin(math.radians(80.03)) / 2, 0.013, 0.03645]), -(180.0 - 80.03)))
def box_dist(P, c, h, pitch):
    t = math.radians(pitch); cs, sn = math.cos(t), math.sin(t); d = P - c
    L = np.stack([cs * d[..., 0] - sn * d[..., 2], d[..., 1], sn * d[..., 0] + cs * d[..., 2]], -1)
    q = np.abs(L) - h
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
def ladder_dist(P):
    dd = [float(box_dist(P, c, h, p)) for _, c, h, p in B]; j = int(np.argmin(dd)); return dd[j], B[j][0]

src, dst = pathlib.Path(a.clip), pathlib.Path(a.out)
d = dict(np.load(src / "motion.npz")); m = json.loads((src / "meta.json").read_text())
jn, bn, fps = m["joint_names"], m["body_names"], float(m["fps"])
T = len(d["joint_pos"]); pel = bn.index("pelvis")
PP, PQ = d["body_pos_w"][:, pel], d["body_quat_w"][:, pel]
waist = d["joint_pos"][:, [jn.index(n) for n in K.WAIST]]
aj = [jn.index(n.format(s=S)) for n in K.ARM]
Q0 = d["joint_pos"][:, aj].copy()
bp0, bq0 = d["body_pos_w"], d["body_quat_w"]
LIM = np.array([K.limits(n.format(s=S)) for n in K.ARM])
fr = lambda s: int(round(s / 50.0 * fps))
def set_pitch(deg):
    global F_W, N_W
    th = math.radians(deg)
    F_W = np.array([math.cos(th), 0.0, math.sin(th)])    # where the fingers point
    N_W = np.array([-math.sin(th), 0.0, math.cos(th)])   # where the palm faces: up
set_pitch(0.0)

def pose(i, q):
    o = K.fk(PP[i], PQ[i], waist[i], q, S)
    return o[f"{S}_wrist_yaw_link"], o[f"{S}_wrist_yaw_link_q"]
def err(i, q, tgt):
    wp, wq = pose(i, q)
    rot = 0.5 * (np.cross(K.qrot(wq, PALM_N), N_W) + np.cross(K.qrot(wq, np.array([1.0, 0, 0])), F_W))
    return np.concatenate([tgt - (wp + K.qrot(wq, PC)), 0.25 * rot])
def solve(i, tgt, q0, iters=40):
    q = q0.copy()
    for _ in range(iters):
        e = err(i, q, tgt)
        if np.linalg.norm(e) < 2e-4: break
        J = np.zeros((6, 7))
        for c in range(7):
            qp = q.copy(); qp[c] += 1e-4; J[:, c] = (err(i, qp, tgt) - e) / -1e-4
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-5 * np.eye(6))
        pref = q.copy(); pref[1] = a.elbow_out                  # the one spare freedom: swing the elbow out, off the torso
        dq = Jp @ e + a.elbow_w * (np.eye(7) - Jp @ J) @ (pref - q)
        q = np.clip(q + np.clip(dq, -0.2, 0.2), LIM[:, 0] + 0.02, LIM[:, 1] - 0.02)
    wp, wq = pose(i, q)
    return q, float(np.linalg.norm(tgt - (wp + K.qrot(wq, PC)))), float(np.degrees(np.arccos(np.clip(K.qrot(wq, PALM_N) @ N_W, -1, 1))))

Q = np.full((T, 7), np.nan); lab = np.zeros(T, int); rest_err = {}; TGT = {}
names = ["step4", "platform", "handrail"]
active = []
for k, nm in enumerate(names):
    s0, s1 = fr(a.rests[2 * k]), fr(a.rests[2 * k + 1])
    if s1 <= s0:
        continue
    active.append((nm, s0, s1, k))
    x, z = NOSE[nm]; TGT[nm] = np.array([x + a.insets[k], a.y, z + 0.002])

# ---- moves: the wrist's contact point follows waypoints (world), hand kept palm-up and level ----
def path(i0, i1, pts, q_seed):
    """Frames i0..i1-1 along the polyline through pts (first = where the wrist is at i0), minimum-jerk in arc length."""
    pts = [np.asarray(p, float) for p in pts]
    seg = np.array([np.linalg.norm(pts[k + 1] - pts[k]) for k in range(len(pts) - 1)]); cum = np.concatenate([[0], np.cumsum(seg)])
    q = q_seed.copy(); worst = 0.0
    for k in range(i1 - i0):
        u = k / max(i1 - i0 - 1, 1); sarc = (10 * u**3 - 15 * u**4 + 6 * u**5) * cum[-1]
        j = min(int(np.searchsorted(cum, sarc, side="right")) - 1, len(seg) - 1)
        tg = pts[j] + (pts[j + 1] - pts[j]) * ((sarc - cum[j]) / max(seg[j], 1e-9))
        q, e, _ = solve(i0 + k, tg, q, iters=40); Q[i0 + k] = q; worst = max(worst, e)
    return worst
def contact_at(i, q):
    wp, wq = pose(i, q); return wp + K.qrot(wq, PC)
# The palm faces UP for the whole clip (the bulb stands on it). The arm is CARRIED with the wrist at a point
# fixed in the pelvis frame, beside the right hip, and every move is a path of the wrist's contact point.
ap_carry = np.array(a.carry)
def carry_pt(i):
    c = ap_carry.copy()
    if a.carry_z_early is not None:
        u = np.clip((i * 50.0 / fps - (a.carry_switch - 30.0)) / 60.0, 0.0, 1.0); w_ = 10 * u**3 - 15 * u**4 + 6 * u**5
        c[2] = (1 - w_) * a.carry_z_early + w_ * ap_carry[2]
    return PP[i] + K.qrot(PQ[i], c)
def chest_pt(i): return carry_pt(i)                       # the carry IS in front of the chest; no separate waypoint
set_pitch(a.carry_pitch)
q = min((solve(0, carry_pt(0), sd, iters=200) for sd in [np.array([0.3, -0.3, 0.0, 0.9, 1.57, 0.0, 0.0]), np.array([0.0, -0.4, 0.3, 1.2, 1.5, 0.0, 0.0]),
                                                      np.array([0.6, -0.5, -0.3, 0.8, 1.6, 0.0, 0.0])]), key=lambda x: x[1] + 0.002 * x[2])[0]
ce = 0.0; QC = np.zeros((T, 7))
for i in range(T):
    q, e, t = solve(i, carry_pt(i), q, iters=10); QC[i] = q; ce = max(ce, e)
Q[:] = QC
print(f"[carry] wrist held at {np.round(ap_carry*1000)} mm in the pelvis frame, palm up: worst miss {ce*1000:.1f} mm")
nmv = fr(a.move * 50)
PRE = {"step4": np.array([-0.23, 0.0, 0.05]),      # the platform overhangs tread 4: slide in and out level, from behind
       "platform": np.array([-0.10, 0.0, 0.14]), "handrail": np.array([-0.10, 0.0, 0.10])}
def up_first(c, pre):
    """A point over the chest waypoint at the height of the point above the hold: the hand rises before it
    goes in over an edge (a straight line from chest height clipped the platform's nose in the S10 scan)."""
    return np.array([c[0], c[1], max(c[2], pre[2])]) if pre[2] > c[2] else np.array([pre[0], pre[1], c[2]])


def rest_frames(nm, s0, s1, k):
    set_pitch(a.pitch[k]); q = Q[s0 - 1].copy(); es, ts = [], []
    for i in range(s0, s1):
        q, e, t = solve(i, TGT[nm], q); Q[i] = q; es.append(e); ts.append(t)
    lab[s0 + fr(a.settle * 50):s1] = LAB[nm]
    print(f"[rest ] {nm:9s} steps {a.rests[2*k]:.0f}-{a.rests[2*k+1]:.0f}: wrist off its spot by at most {max(es)*1000:.1f} mm, "
          f"hand tilt off by at most {max(ts):.1f} deg")
for n_, (nm, s0, s1, k) in enumerate(active):
    tg = TGT[nm]; pre = tg + PRE[nm]; seg = {}
    prev = active[n_ - 1] if n_ > 0 else None
    if prev is not None and s0 - prev[2] < 2 * nmv:                  # from the previous rest, by way of the chest
        ptg = TGT[prev[0]]; ppre = ptg + PRE[prev[0]]; g = s0 - prev[2]
        i_a = prev[2] + int(0.20 * g); i_b = prev[2] + int(0.45 * g); i_c2 = prev[2] + int(0.80 * g)
        set_pitch(a.pitch[prev[3]])
        seg[f"out of {prev[0]}"] = path(prev[2], i_a, [ptg, ppre], Q[prev[2] - 1])
        seg["->chest"] = path(i_a, i_b, [ppre, up_first(chest_pt(i_b), ppre), chest_pt(i_b)], Q[i_a - 1])
        set_pitch(a.pitch[k])
        seg["chest->pre"] = path(i_b, i_c2, [chest_pt(i_b), up_first(chest_pt(i_b), pre), pre], Q[i_b - 1])
        seg["in"] = path(i_c2, s0, [pre, tg], Q[i_c2 - 1])
    else:                                                             # from the carry, in front of the chest, then in
        i0 = max(s0 - fr(20) - nmv, 0); i_c = i0 + (s0 - fr(20) - i0) // 2
        set_pitch(a.carry_pitch)
        if s0 - fr(20) - nmv < 0:
            # no time to come round from the hip before the first rest: the clip STARTS with the hand in front
            # of the chest, palm up (S10 spawns with the bulb cradled in front of the body anyway)
            i_c = fr(6)
            qc = min((solve(0, chest_pt(0), sd, iters=200) for sd in (QC[0], np.array([-0.3, -0.2, 0.0, 0.9, 1.57, 0.0, 0.0]))), key=lambda x: x[1] + 0.002 * x[2])[0]
            for i in range(i_c):
                qc, e_, _ = solve(i, chest_pt(i), qc, iters=20); Q[i] = qc
            seg["start at chest"] = e_
        else:
            seg["carry->chest"] = path(i0, i_c, [carry_pt(i0), chest_pt(i_c)], QC[i0])
        set_pitch(a.pitch[k])
        seg["chest->pre"] = path(i_c, s0 - fr(20), [chest_pt(i_c), up_first(chest_pt(i_c), pre), pre], Q[i_c - 1])
        seg["in"] = path(s0 - fr(20), s0, [pre, tg], Q[s0 - fr(20) - 1])
    rest_frames(nm, s0, s1, k)
    nxt = active[n_ + 1] if n_ + 1 < len(active) else None
    if nxt is None or nxt[1] - s1 >= 2 * nmv:                         # back to the carry
        seg["out"] = path(s1, s1 + fr(20), [tg, pre], Q[s1 - 1])
        i_d = s1 + fr(20) + nmv // 2; i1 = s1 + fr(20) + nmv
        seg["pre->chest"] = path(s1 + fr(20), i_d, [pre, up_first(chest_pt(i_d), pre), chest_pt(i_d)], Q[s1 + fr(20) - 1])
        set_pitch(a.carry_pitch)
        seg["chest->carry"] = path(i_d, i1, [chest_pt(i_d), carry_pt(i1)], Q[i_d - 1])
        nb = fr(15)
        for j in range(nb):
            w = (j + 1) / nb; Q[i1 + j] = (1 - w) * Q[i1 - 1] + w * QC[i1 + j]
    print(f"[move ] {nm}: misses mm: " + ", ".join(f"{k_} {v*1000:.0f}" for k_, v in seg.items()))
assert not np.isnan(Q).any()
# take the kinks out where path segments join, then put each rest back exactly on its spot
from scipy.ndimage import gaussian_filter1d
Q = gaussian_filter1d(Q, sigma=0.06 * fps, axis=0, mode="nearest")
for nm, s0, s1, k in active:
    set_pitch(a.pitch[k]); q = Q[s0].copy(); n_in = fr(10)
    for i in range(s0, s1):
        q2, e, t = solve(i, TGT[nm], Q[i], iters=8)
        w = min(1.0, (i - s0 + 1) / n_in, (s1 - i) / n_in)          # ease into and out of the exact solution
        Q[i] = (1 - w) * Q[i] + w * q2
Q = np.clip(Q, LIM[:, 0] + 0.02, LIM[:, 1] - 0.02)

# ---- report: clearances ----
from scipy.spatial import ConvexHull as _CH
_TH = _CH(np.load(pathlib.Path(__file__).parent / "g1_body_hulls.npz")["torso_link"]).equations     # torso hull, torso frame
def torso_dist(o, P):
    L = K.qrot(K.qconj(o["torso_q"]), P - o["torso_p"]); return float((L @ _TH[:, :3].T + _TH[:, 3]).max())
tw = (9.0, 0)
for i in range(0, T, 3):
    o = K.fk(PP[i], PQ[i], waist[i], Q[i], S); el, wr_ = o[f"{S}_elbow_link"], o[f"{S}_wrist_roll_link"]
    c = min(torso_dist(o, el) - 0.045, torso_dist(o, wr_) - 0.040, torso_dist(o, 0.5 * (el + wr_)) - 0.040)
    if c < tw[0]: tw = (c, i)
_HB = np.load(pathlib.Path(__file__).parent / "g1_body_hulls.npz")
lw = (9.0, "", 0)
for ln in ("right_knee_link", "right_hip_yaw_link", "left_knee_link", "left_hip_yaw_link"):
    eq = _CH(_HB[ln]).equations; bi = bn.index(ln)
    for i in range(0, T, 3):
        o = K.fk(PP[i], PQ[i], waist[i], Q[i], S); wp, wq, el = o[f"{S}_wrist_yaw_link"], o[f"{S}_wrist_yaw_link_q"], o[f"{S}_elbow_link"]
        pts = [(wp + K.qrot(wq, c), rr) for c, rr in BULB] + [(wp + K.qrot(wq, BACK_OF_HAND), 0.02), (wp, 0.035), (0.5 * (wp + el), 0.04), (el, 0.045)]
        for P_, rr in pts:
            L_ = K.qrot(K.qconj(bq0[i, bi]), P_ - bp0[i, bi]); c = float((L_ @ eq[:, :3].T + eq[:, 3]).max()) - rr
            if c < lw[0]: lw = (c, ln, i)
print(f"[clear] bulb hand / forearm closest to the robot's own legs: {lw[0]*1000:+.0f} mm ({lw[1]}, step {lw[2]*50/fps:.0f})")
print(f"[clear] elbow / forearm closest to the robot's own torso: {tw[0]*1000:+.0f} mm (step {tw[1]*50/fps:.0f})")
worst = {"bulb": (9.0, "", 0), "back of hand": (9.0, "", 0), "forearm": (9.0, "", 0)}
low = 0
for i in range(0, T, 3):
    o = K.fk(PP[i], PQ[i], waist[i], Q[i], S); wp, wq, el = o[f"{S}_wrist_yaw_link"], o[f"{S}_wrist_yaw_link_q"], o[f"{S}_elbow_link"]
    cb = min((ladder_dist(wp + K.qrot(wq, c))[0] - rr, ladder_dist(wp + K.qrot(wq, c))[1]) for c, rr in BULB)
    ch = ladder_dist(wp + K.qrot(wq, BACK_OF_HAND))
    cf = min((ladder_dist(el + u * (wp - el))[0] - 0.045, ladder_dist(el + u * (wp - el))[1]) for u in (0.0, 0.35, 0.7))
    for key, c in (("bulb", cb), ("back of hand", ch), ("forearm", cf)):
        if c[0] < worst[key][0]: worst[key] = (c[0], c[1], i)
    low += cb[0] < 0.010
for key, (c, what, i) in worst.items():
    print(f"[clear] {key:13s} closest to the ladder: {c*1000:+.0f} mm ({what}, step {i*50/fps:.0f})")
print(f"[clear] sampled frames with the bulb proxy closer than 10 mm: {low} of {len(range(0, T, 3))}")
for nm, s0, s1, k in active:                               # forearm against the hold's edge, through the rest
    row = []
    for b0 in range(int(a.rests[2 * k]), int(a.rests[2 * k + 1]), 80):
        cm = 9.0
        for st_ in range(b0, min(b0 + 80, int(a.rests[2 * k + 1])), 4):
            i = fr(st_); o = K.fk(PP[i], PQ[i], waist[i], Q[i], S); wp, el = o[f"{S}_wrist_yaw_link"], o[f"{S}_elbow_link"]
            cm = min(cm, min(ladder_dist(el + u * (wp - el))[0] - 0.045 for u in (0.0, 0.35, 0.7)))
        row.append(f"{b0}:{cm*1000:+.0f}")
    print(f"[clear] forearm vs ladder during the {nm} rest, by 80 steps (mm): " + "  ".join(row))
print(f"[speed] peak right-arm joint speed {np.abs(np.gradient(Q, 1.0 / fps, axis=0)).max():.2f} rad/s")

# ---- write ----
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "motion.npz": shutil.copy(f, dst / f.name)
q = d["joint_pos"].copy(); q[:, aj] = Q
allb = K.fk_all(PP, PQ, q, jn); bp, bq = d["body_pos_w"].copy(), d["body_quat_w"].copy()
for i, b in enumerate(bn): bp[:, i], bq[:, i] = allb[b]
dq_ = K.qmul(np.roll(bq, -1, axis=0), K.qconj(bq)); dq_[-1] = dq_[-2]; dq_ = dq_ * np.sign(dq_[..., :1] + 1e-12)
oc = d["object_contact"].copy()
for s50 in range(len(oc)): oc[s50, 1] = lab[min(fr(s50), T - 1)]
np.savez(dst / "motion.npz", joint_pos=q, joint_vel=np.gradient(q, 1.0 / fps, axis=0), body_pos_w=bp, body_quat_w=bq,
         body_lin_vel_w=np.gradient(bp, 1.0 / fps, axis=0), body_ang_vel_w=2.0 * dq_[..., 1:] * fps, object_contact=oc)
hj = dst / "holds.json"
if hj.exists():
    h = json.loads(hj.read_text()); h["bulb_wrist_rest"] = dict(rests=list(a.rests), insets=list(a.insets), bulb_in_wrist=[BULB_IN_WRIST[0].tolist(), BULB_IN_WRIST[1].tolist()], pitch=list(a.pitch), y=a.y, contact_point=PC.tolist())
    hj.write_text(json.dumps(h, indent=1))
print(f"[out  ] {dst}/motion.npz; right-hand labels: " + ", ".join(f"{nm} {int((oc[:,1]==LAB[nm]).sum())} steps" for nm in names))
