"""Phase 2: give the swing foot a clearance arc, so it goes OVER the tread instead of through it.

Phase 1 measured the failure: replaying LadderMan's step onto our tread 1 lands the foot dead
centre (36.9 mm from each edge) but drives it THROUGH the 80 mm slab on the way -- at the moment
the foot crosses the tread's front-edge plane it is 165 mm BELOW the tread top. LadderMan's rung
is a bar a foot passes under and around; our tread is a slab that must be cleared from above.

So the swing foot gets a new path and the leg is re-solved to follow it:

  lift      rise to (tread top + sole offset + margin) with the foot still OUT in front
  traverse  move rearward over the tread at that height
  place     descend onto the landing point

Everything else -- pelvis trajectory, stance leg, arms, waist -- is left exactly as the clip has
it. Only the swing leg's six joints change, and only between takeoff and touchdown.

FK/IK run offline in numpy against Unitree's own URDF chain; the sim replay then measures the
foot independently, so a wrong IK shows up as a gate failure rather than a plausible-looking lie.
"""
from __future__ import annotations
import argparse, json, math, os
import numpy as np

# right leg, from g1_body29_hand14.urdf (parent -> child, origin xyz/rpy, axis, limits)
RIGHT_LEG = [
    ("right_hip_pitch_joint",   (0.0, -0.064452, -0.1027),    (0.0,  0.0,    0.0), (0, 1, 0), (-2.5307,  2.8798)),
    ("right_hip_roll_joint",    (0.0, -0.052,    -0.030465),  (0.0, -0.1749, 0.0), (1, 0, 0), (-2.9671,  0.5236)),
    ("right_hip_yaw_joint",     (0.025001, 0.0,  -0.12412),   (0.0,  0.0,    0.0), (0, 0, 1), (-2.7576,  2.7576)),
    ("right_knee_joint",        (-0.078273, -0.0021489, -0.17734), (0.0, 0.1749, 0.0), (0, 1, 0), (-0.087267, 2.8798)),
    ("right_ankle_pitch_joint", (0.0, 9.4445e-05, -0.30001),  (0.0,  0.0,    0.0), (0, 1, 0), (-0.87267, 0.5236)),
    ("right_ankle_roll_joint",  (0.0, 0.0,       -0.017558),  (0.0,  0.0,    0.0), (1, 0, 0), (-0.2618,  0.2618)),
]
JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
LEG_COLS = [6, 7, 8, 9, 10, 11]          # indices of these joints in the clip's 29-dof JOINT_ORDER


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])


def axis_mat(axis, q):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q)*K + (1 - math.cos(q))*(K @ K)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                     [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def leg_fk(qleg, chain=RIGHT_LEG):
    """Foot pose in the PELVIS frame, plus every joint's origin and axis (for the Jacobian)."""
    R = np.eye(3); p = np.zeros(3)
    origins, axes = [], []
    for (nm, xyz, rpy, ax, _lim), q in zip(chain, qleg):
        p = p + R @ np.asarray(xyz, float)
        R = R @ rpy_mat(*rpy)
        origins.append(p.copy())
        axes.append(R @ np.asarray(ax, float))
        R = R @ axis_mat(ax, float(q))
    return R, p, np.array(origins), np.array(axes)


def leg_jacobian(qleg, chain=RIGHT_LEG):
    R, p, origins, axes = leg_fk(qleg, chain)
    J = np.zeros((6, len(chain)))
    for i in range(len(chain)):
        J[:3, i] = np.cross(axes[i], p - origins[i])
        J[3:, i] = axes[i]
    return J, R, p


def so3_err(R_cur, R_des):
    E = R_des @ R_cur.T
    w = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) * 0.5
    s = np.linalg.norm(w)
    c = (np.trace(E) - 1.0) * 0.5
    ang = math.atan2(s, max(-1.0, min(1.0, c)))
    return w / s * ang if s > 1e-9 else np.zeros(3)


DEFAULT_LEG = np.array([-0.312, 0.0, 0.0, 0.669, -0.363, 0.0])


def ik_leg(q0, p_des, R_des, chain=RIGHT_LEG, iters=120, lam=0.08, w_rot=0.45, restarts=True,
           alt_seeds=None):
    """Damped least squares onto a 6-DoF foot pose in the pelvis frame, limits respected.

    Warm-starting from the previous frame is fast but can pin the solve against a joint limit it
    cannot climb back off (seen as a large POSITION error while the rotation error also stalls).
    When the warm start does not converge, retry from fixed seeds -- the default stance and two
    knee-bent poses -- and keep the best. Deterministic: no random restarts.
    """
    seeds = [np.array(q0, float)]
    if restarts:
        seeds += [np.array(s, float) for s in (alt_seeds if alt_seeds is not None else
                  (DEFAULT_LEG, DEFAULT_LEG + np.array([-0.5, 0.0, 0.0, 1.0, -0.5, 0.0]),
                   DEFAULT_LEG + np.array([0.4, 0.0, 0.0, 0.8, -1.0, 0.0])))]
    best = None
    for seed in seeds:
        q, ep, er = _ik_once(seed, p_des, R_des, chain, iters, lam, w_rot)
        if best is None or (ep + 0.05 * er) < (best[1] + 0.05 * best[2]):
            best = (q, ep, er)
        if ep < 1e-3 and er < 0.05:
            break
    return best


def _ik_once(q0, p_des, R_des, chain, iters, lam, w_rot):
    q = np.array(q0, float)
    lo = np.array([c[4][0] for c in chain]); hi = np.array([c[4][1] for c in chain])
    for _ in range(iters):
        J, R, p = leg_jacobian(q, chain)
        e = np.concatenate([p_des - p, w_rot * so3_err(R, R_des)])
        if np.linalg.norm(e[:3]) < 2e-5 and np.linalg.norm(e[3:]) < 2e-4:
            break
        JT = J.T
        dq = JT @ np.linalg.solve(J @ JT + (lam ** 2) * np.eye(6), e)
        q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
    J, R, p = leg_jacobian(q, chain)
    return q, float(np.linalg.norm(p_des - p)), float(np.linalg.norm(so3_err(R, R_des)))


def smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def arc_path(p0, p1, z_clear, n, f_lift=0.35, f_place=0.80, out_dir=None, out_bulge=0.0):
    """lift straight up -> traverse rearward at z_clear -> descend onto the landing point.

    ``out_bulge`` swings the foot OUT along ``out_dir`` through the middle of the swing and
    back to zero at touchdown. Lifting straight up off the tread below tucks the knee under the
    tread above; swinging out first is what gives the knee room, and is what a person does.
    """
    out = np.zeros((n, 3))
    for k in range(n):
        s = k / max(n - 1, 1)
        if s <= f_lift:
            z = p0[2] + (z_clear - p0[2]) * smoothstep(s / f_lift)
            xy = p0[:2]
        elif s <= f_place:
            z = z_clear
            xy = p0[:2] + (p1[:2] - p0[:2]) * smoothstep((s - f_lift) / (f_place - f_lift))
        else:
            z = z_clear + (p1[2] - z_clear) * smoothstep((s - f_place) / (1.0 - f_place))
            xy = p1[:2]
        out[k] = (xy[0], xy[1], z)
        if out_bulge and out_dir is not None:
            out[k] = out[k] + np.asarray(out_dir, float) * out_bulge * math.sin(math.pi * min(1.0, s / f_place))
    out[-1] = p1
    return out


# ---------------------------------------------------------------------------------------------
def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def yaw_quat(t):
    return np.array([math.cos(t/2), 0.0, 0.0, math.sin(t/2)])


def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="/home/yujin-chen/climb_motion/ladderman_climb.npz")
    ap.add_argument("--phase1", default="/home/yujin-chen/psi0_ft/climb/phase1_tread1/replay.json")
    ap.add_argument("--out-clip", required=True)
    ap.add_argument("--margin", type=float, default=0.040, help="clearance above the tread top, m")
    ap.add_argument("--f-lift", type=float, default=0.35)
    ap.add_argument("--f-place", type=float, default=0.80)
    ap.add_argument("--takeoff", type=int, default=None, help="clip frame the swing starts (default: auto)")
    a = ap.parse_args()

    P = json.load(open(a.phase1))
    meta, tread = P["meta"], P["tread"]
    lp = np.array(meta["ladder_pos"]); lyaw = math.radians(meta["ladder_yaw_deg"])
    sole = meta["sole_offset_m"]
    shift = np.array(P["shift"])
    start = P["frames"][0]["frame"]; end = P["frames"][-1]["frame"]

    clip = np.load(a.clip)
    dof = clip["dof"].astype(np.float64).copy()
    rrot = clip["root_rot"].astype(np.float64)
    rtra = clip["root_trans_offset"].astype(np.float64)
    fps = int(clip["fps"])

    # reproduce Phase 1's alignment exactly
    lq = np.array(meta["ladder_quat"])
    Rl = quat_to_mat(lq)
    face_w = Rl @ np.array([0.0, -1.0, 0.0])
    desired_yaw = math.atan2(-face_w[1], -face_w[0])
    qy = yaw_quat(desired_yaw - quat_yaw(rrot[end]))
    Rqy = quat_to_mat(qy)

    def root_of(i):
        return Rqy @ rtra[i] + shift, quat_mul(qy, rrot[i])

    # ---- verify the offline FK against what the sim measured in Phase 1 -----------------------
    errs = []
    for fr in P["frames"]:
        i = fr["frame"]
        rp, rq = root_of(i)
        _, pf, _, _ = leg_fk(dof[i, LEG_COLS])
        errs.append(np.linalg.norm(rp + quat_to_mat(rq) @ pf - np.array(fr["swing_foot"])))
    errs = np.array(errs)
    print(f"[FK    ] offline vs sim-measured swing foot: mean {errs.mean()*1000:.3f} mm, "
          f"max {errs.max()*1000:.3f} mm over {len(errs)} frames")
    if errs.max() > 0.003:
        print("[FATAL] offline FK does not reproduce the sim; IK built on it cannot be trusted")
        return 2

    # ---- the swing window ---------------------------------------------------------------------
    fw = np.array([fr["swing_foot"] for fr in P["frames"]])
    spd = np.r_[0.0, np.linalg.norm(np.diff(fw, axis=0), axis=1) * fps]
    k0 = a.takeoff - start if a.takeoff is not None else int(np.argmax(spd > 0.05))
    k1 = len(P["frames"]) - 1
    print(f"[SWING ] frames {start+k0}-{start+k1}  (t {(start+k0)/fps:.2f}-{(start+k1)/fps:.2f}s)")

    p0 = fw[k0].copy(); p1 = fw[k1].copy()
    z_clear = tread["top"] + sole + a.margin
    # tread top is a LADDER-frame z and the ladder sits on the floor, so this is also world z
    print(f"[ARC   ] takeoff {np.round(p0,4)} -> landing {np.round(p1,4)}")
    print(f"[ARC   ] clear height {z_clear:.4f} m  (tread top {tread['top']:.4f} + sole {sole:.4f} + margin {a.margin:.3f})")
    path = arc_path(p0, p1, z_clear, k1 - k0 + 1, a.f_lift, a.f_place)

    # ---- solve the leg --------------------------------------------------------------------------
    worst_p = worst_r = 0.0
    for k in range(k0, k1 + 1):
        i = start + k
        rp, rq = root_of(i)
        Rr = quat_to_mat(rq)
        p_des_pelvis = Rr.T @ (path[k - k0] - rp)
        R_cur, _, _, _ = leg_fk(dof[i, LEG_COLS])          # keep the clip's foot ORIENTATION
        q, ep, er = ik_leg(dof[i, LEG_COLS], p_des_pelvis, R_cur)
        dof[i, LEG_COLS] = q
        worst_p = max(worst_p, ep); worst_r = max(worst_r, er)
    print(f"[IK    ] worst residual: position {worst_p*1000:.3f} mm, rotation {math.degrees(worst_r):.3f} deg")

    os.makedirs(os.path.dirname(a.out_clip) or ".", exist_ok=True)
    np.savez(a.out_clip, dof=dof.astype(np.float32), root_rot=rrot,
             root_trans_offset=rtra.astype(np.float32), fps=np.int64(fps))
    print(f"[OK    ] wrote {a.out_clip}  (only right-leg cols {LEG_COLS} changed, frames {start+k0}-{start+k1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------------------------
# Arms. Generated from Unitree's own g1_body29_hand14.urdf, not transcribed by hand.
#
# The shoulders hang off `torso_link`, which IS the child of waist_pitch_joint -- so the arm base
# moves with the waist and the caller has to compose WAIST before the arm chain. Each arm carries
# a trailing zero-range `hand_palm` entry so that leg_fk/leg_jacobian run FK all the way to the
# palm: the solver gets a column it cannot move (limits 0..0), which costs nothing and means the
# IK target is the palm rather than the wrist.
#
# The climb needs these because the feet alone cannot hold the robot. Measured: a knee-clearing
# posture puts the CoM 331 mm behind the support edge and demands 125 N m from a 50 N m ankle.
# Hands on the rails add support, which is how a person climbs a step ladder.
# ---------------------------------------------------------------------------------------------
LEFT_ARM = [
    ('left_shoulder_pitch_joint', (0.0039563, 0.10022, 0.24778), (0.27931, 5.4949e-05, -0.00019159), (0, 1, 0), (-3.0892, 2.6704)),
    ('left_shoulder_roll_joint', (0, 0.038, -0.013831), (-0.27925, 0, 0), (1, 0, 0), (-1.5882, 2.2515)),
    ('left_shoulder_yaw_joint', (0, 0.00624, -0.1032), (0, 0, 0), (0, 0, 1), (-2.6180, 2.6180)),
    ('left_elbow_joint', (0.015783, 0, -0.080518), (0, 0, 0), (0, 1, 0), (-1.0472, 2.0944)),
    ('left_wrist_roll_joint', (0.1, 0.00188791, -0.01), (0, 0, 0), (1, 0, 0), (-1.9722, 1.9722)),
    ('left_wrist_pitch_joint', (0.038, 0, 0), (0, 0, 0), (0, 1, 0), (-1.6144, 1.6144)),
    ('left_wrist_yaw_joint', (0.046, 0, 0), (0, 0, 0), (0, 0, 1), (-1.6144, 1.6144)),
    ('left_hand_palm_joint', (0.0415, 0.003, 0), (0, 0, 0), (0, 0, 1), (0.0, 0.0)),  # fixed: carries FK to the palm
]
RIGHT_ARM = [
    ('right_shoulder_pitch_joint', (0.0039563, -0.10021, 0.24778), (-0.27931, 5.4949e-05, 0.00019159), (0, 1, 0), (-3.0892, 2.6704)),
    ('right_shoulder_roll_joint', (0, -0.038, -0.013831), (0.27925, 0, 0), (1, 0, 0), (-2.2515, 1.5882)),
    ('right_shoulder_yaw_joint', (0, -0.00624, -0.1032), (0, 0, 0), (0, 0, 1), (-2.6180, 2.6180)),
    ('right_elbow_joint', (0.015783, 0, -0.080518), (0, 0, 0), (0, 1, 0), (-1.0472, 2.0944)),
    ('right_wrist_roll_joint', (0.1, -0.00188791, -0.01), (0, 0, 0), (1, 0, 0), (-1.9722, 1.9722)),
    ('right_wrist_pitch_joint', (0.038, 0, 0), (0, 0, 0), (0, 1, 0), (-1.6144, 1.6144)),
    ('right_wrist_yaw_joint', (0.046, 0, 0), (0, 0, 0), (0, 0, 1), (-1.6144, 1.6144)),
    ('right_hand_palm_joint', (0.0415, -0.003, 0), (0, 0, 0), (0, 0, 1), (0.0, 0.0)),  # fixed: carries FK to the palm
]
WAIST = [
    ('waist_yaw_joint', (0, 0, 0), (0, 0, 0), (0, 0, 1), (-2.6180, 2.6180)),
    ('waist_roll_joint', (-0.0039635, 0, 0.044), (0, 0, 0), (1, 0, 0), (-0.5200, 0.5200)),
    ('waist_pitch_joint', (0, 0, 0), (0, 0, 0), (0, 1, 0), (-0.5200, 0.5200)),
]

ARM_CHAIN = {"left": LEFT_ARM, "right": RIGHT_ARM}

# arm down and slightly forward -- the posture the climb used before the hands had a job
DEFAULT_ARM = np.array([0.25, 0.22, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0])
ARM_SEEDS = (DEFAULT_ARM,
             np.array([0.0, 0.35, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0]),    # elbow up, reaching out
             np.array([-0.6, 0.25, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0]))   # reaching forward and high


def mirror_arm_seed(q, side):
    """Shoulder roll and the yaw joints flip sign on the right arm."""
    q = np.array(q, float).copy()
    if side == "right":
        q[1] = -q[1]; q[2] = -q[2]; q[4] = -q[4]; q[6] = -q[6]
    return q


def torso_frame(waist_q):
    """Pose of torso_link in the PELVIS frame, given (waist_yaw, waist_roll, waist_pitch)."""
    R = np.eye(3); p = np.zeros(3)
    for (_nm, xyz, rpy, ax, _lim), q in zip(WAIST, waist_q):
        p = p + R @ np.asarray(xyz, float)
        R = R @ rpy_mat(*rpy) @ axis_mat(ax, float(q))
    return R, p


def ik_arm(q0, p_des_torso, chain, iters=160, lam=0.10):
    """Place the PALM at a point in the torso frame. Position only -- a hand closing on a round
    rail does not care how the wrist is rolled, and spending the orientation DoF on a grip angle
    we have not measured would only fight the reach."""
    best = None
    for seed in [np.array(q0, float)] + [mirror_arm_seed(s, "right" if chain is RIGHT_ARM else "left")
                                         for s in ARM_SEEDS]:
        q = np.array(seed, float)
        lo = np.array([c[4][0] for c in chain]); hi = np.array([c[4][1] for c in chain])
        for _ in range(iters):
            J, R, p = leg_jacobian(q, chain)
            e = p_des_torso - p
            if np.linalg.norm(e) < 1e-4:
                break
            Jp = J[:3]
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + (lam ** 2) * np.eye(3), e)
            q = np.clip(q + np.clip(dq, -0.25, 0.25), lo, hi)
        _, _, p = leg_jacobian(q, chain)
        ep = float(np.linalg.norm(p_des_torso - p))
        if best is None or ep < best[1]:
            best = (q, ep)
        if ep < 1e-3:
            break
    return best


# The grip line, fitted from the 55 mm collider maps themselves (the geometry physics actually
# sees) rather than from render-mesh vertices -- that mesh carries vertices only at feature
# corners, so sampling it by height finds rail where there is a corner and nothing in between.
#
# It is PIECEWISE. Below the platform the front rails slope with the tread stack: the fitted
# 0.3694 matches the tread run/rise (0.0839/0.2347 = 0.357). Above it the guard posts carry on to
# z = 1.85 at a shallower 0.3041, so a single line extrapolated from the lower rail misses the top
# of the structure by 37 mm. The two fits agree to 9 mm where they meet.
# Worst residual 51 mm, mean 14 mm, most of which is the 25-50 mm probe grid.
RAIL_Z_KNEE = 1.15
RAIL_LOW = (-0.4618, 0.3694, 0.220)      # y0, dy/dz, |x|
RAIL_HIGH = (-0.3778, 0.3041, 0.225)
RAIL_Z_MIN, RAIL_Z_MAX = 0.30, 1.80

# kept for callers that want the single-line form
RAIL_X, RAIL_Y0, RAIL_DYDZ = RAIL_LOW[2], RAIL_LOW[0], RAIL_LOW[1]


def rail_point(z, side):
    """Grip point on the ladder's front rail / guard post, in LADDER-local coordinates."""
    y0, dydz, x = RAIL_LOW if z <= RAIL_Z_KNEE else RAIL_HIGH
    return np.array([x if side == "right" else -x, y0 + dydz * z, z])


def rail_nearest_z(p_local):
    """Height on the grip line closest to a point, solved per segment and clamped to the rail."""
    best = None
    for y0, dydz, _x, lo, hi in ((*RAIL_LOW, RAIL_Z_MIN, RAIL_Z_KNEE),
                                 (*RAIL_HIGH, RAIL_Z_KNEE, RAIL_Z_MAX)):
        z = (p_local[2] + dydz * (p_local[1] - y0)) / (1.0 + dydz ** 2)
        z = float(min(max(z, lo), hi))
        d = math.hypot(y0 + dydz * z - p_local[1], z - p_local[2])
        if best is None or d < best[1]:
            best = (z, d)
    return best[0]
