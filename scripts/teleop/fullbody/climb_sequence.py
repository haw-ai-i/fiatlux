"""Phase 2b: generate a FULL kinematic climb of this benchmark's ladder.

LadderMan's clip climbs two rungs of ITS ladder, so it cannot be replayed up our four treads --
the pelvis alone has to rise 4 x 0.2347 = 0.94 m and move 4 x 0.0839 = 0.34 m rearward. The clip
is therefore used as a POSTURE source, not a trajectory: its waist/arm pose and its pelvis-to-foot
offset (a very steady 0.740 m above the mean foot, leaning 0.085 m into the ladder, measured at
four separate both-feet-planted moments) define how a G1 holds itself on a ladder. The climb
itself is generated against our own tread table.

Each step reuses the Phase 2 clearance arc that passed the gate: lift to tread_top + sole +
margin while still out in front, traverse rearward at that height, descend onto the tread. Both
legs are solved by the same URDF-exact IK, the stance foot held, the swing foot on the arc.

Output is a clip ALREADY IN WORLD COORDINATES for the ladder it was generated against, so it is
replayed with --no-align. Per-step frame ranges go to a sidecar JSON so the gate can be run on
every step, not just the first.
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget_clearance import (ARM_CHAIN, DEFAULT_ARM, ik_arm, rail_nearest_z,  # noqa: E402
                                mirror_arm_seed, rail_point, torso_frame,
                                RIGHT_LEG, leg_fk, ik_leg, arc_path, quat_to_mat,  # noqa: E402
                                quat_mul, yaw_quat, smoothstep)

LEFT_LEG = [
    ("left_hip_pitch_joint",   (0.0, 0.064452, -0.1027),      (0.0,  0.0,    0.0), (0, 1, 0), (-2.5307,  2.8798)),
    ("left_hip_roll_joint",    (0.0, 0.052,    -0.030465),    (0.0, -0.1749, 0.0), (1, 0, 0), (-0.5236,  2.9671)),
    ("left_hip_yaw_joint",     (0.025001, 0.0, -0.12412),     (0.0,  0.0,    0.0), (0, 0, 1), (-2.7576,  2.7576)),
    ("left_knee_joint",        (-0.078273, 0.0021489, -0.17734), (0.0, 0.1749, 0.0), (0, 1, 0), (-0.087267, 2.8798)),
    ("left_ankle_pitch_joint", (0.0, -9.4445e-05, -0.30001),  (0.0,  0.0,    0.0), (0, 1, 0), (-0.87267, 0.5236)),
    ("left_ankle_roll_joint",  (0.0, 0.0,      -0.017558),    (0.0,  0.0,    0.0), (1, 0, 0), (-0.2618,  0.2618)),
]
LEG_COLS = {"left": [0, 1, 2, 3, 4, 5], "right": [6, 7, 8, 9, 10, 11]}
LEG_CHAIN = {"left": LEFT_LEG, "right": RIGHT_LEG}

# Climbing posture, read off the clip at four both-feet-planted moments (t = 8.0/10.0/16.0/16.5):
# the pelvis-to-mean-foot offset is steady to within 4 mm, so it is a property of the pose and
# not of that particular rung.
PELVIS_ABOVE_FOOT = 0.740
# LadderMan climbs a RUNG ladder, where the body hugs the structure -- its +0.085 m lean INTO
# the ladder is right there and wrong here. On this step ladder the treads project toward the
# climber and the rails slope back, so leaning in drives the THIGH into the frame (measured:
# -27 mm at +0.085, -6 mm at 0.0, +22 mm at -0.060). The body belongs slightly OUT from the
# treads, which is also how a person stands on a step ladder.
PELVIS_INTO_LADDER = -0.060
# The G1 leg spans 0.6495 m from the hip-pitch origin to the ankle-roll link, and the hip sits
# 0.1027 m below the pelvis, so the pelvis can never be more than ~0.753 m above a PLANTED foot.
# 0.740 is 98% of that -- nearly straight legs, which is how the clip climbs and how people do.
# It follows that the pelvis must track the LOWEST PLANTED foot; driving it from the mean of both
# feet asks the stance leg for 0.88 m mid-swing and the IK simply cannot deliver it.
PELVIS_MAX_ABOVE_FOOT = 0.753
# LadderMan's arms reach forward to grip RUNGS. This ladder has no rungs at that height -- its
# rails slope away and its platform overhangs -- so that pose puts both hands and elbows inside
# the structure (measured: elbows -35/-42 mm, hand links to -37 mm). Replaced with a tucked
# climbing pose: arms down and slightly forward, clear of the frame. Waist pitch is kept, since
# the forward lean of the torso is what the clip actually contributes here.
POSTURE_WAIST_ARMS = [
    0.0, 0.0, 0.26462,                                          # waist yaw/roll/pitch
    0.25, 0.22, 0.0, 0.45, 0.0, 0.0, 0.0,                       # left arm  (down, slightly fwd)
    0.25, -0.22, 0.0, 0.45, 0.0, 0.0, 0.0,                      # right arm
]
FOOT_HALF_STANCE = 0.090      # lateral half-separation on the tread (tread is 0.4946 wide)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="/home/yujin-chen/psi0_ft/climb/phase2_arc/ladder_geometry.json")
    ap.add_argument("--treads", default="/home/yujin-chen/psi0_ft/climb/phase1_tread1/replay.json",
                    help="any Phase 1/2 replay.json -- its meta carries the sole offset")
    ap.add_argument("--out-clip", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--margin", type=float, default=0.040)
    ap.add_argument("--stance", type=float, default=None,
                    help="override FOOT_HALF_STANCE (lateral half-separation on the tread)")
    ap.add_argument("--lean", type=float, default=None,
                    help="override PELVIS_INTO_LADDER (negative = stand further out)")
    ap.add_argument("--rail-hands", action="store_true",
                    help="grip the ladder's front rails. The feet alone cannot hold a posture "
                         "that clears the knee: measured, that needs 125 N m from a 50 N m ankle. "
                         "Hands on the rails put the ladder in the support set, which is how a "
                         "person climbs one.")
    ap.add_argument("--waist-yaw", type=float, default=0.0,
                    help="degrees of torso rotation toward the GRIPPING hand, which is the one "
                         "opposite the stepping foot -- how a person climbs. Yaw carries that "
                         "shoulder forward by about 0.10*sin(yaw), which is what buys back the "
                         "reach a knee-clearing lean costs.")
    ap.add_argument("--grip-offset", type=float, default=0.120,
                    help="m to stand the palm off the rail's AXIS, toward the robot. The grip "
                         "line was fitted from the 55 mm occupancy centroid, which is the centre "
                         "of the tube, not its surface -- commanding the palm there drives the "
                         "hand through the rail (measured: 110 kN on the palm link).")
    ap.add_argument("--rail-zmin", type=float, default=0.30)
    ap.add_argument("--rail-zmax", type=float, default=1.80,
                    help="highest grip on the front rail, ladder-local z")
    ap.add_argument("--waist-pitch", type=float, default=None,
                    help="rad of forward torso pitch (G1 waist_pitch limit +-0.52). Leaning the "
                         "pelvis back buys knee clearance and spends balance one-for-one, because "
                         "the torso rides along with it. Pitching the torso FORWARD puts the upper "
                         "body's mass back over the feet, which is how a person climbs: hips back, "
                         "chest forward, hands on the rails.")
    ap.add_argument("--toe-out", type=float, default=0.0,
                    help="degrees of toe-out per foot. The leg's ONLY yaw DOF is hip_yaw (the "
                         "ankle has roll and pitch, no yaw), so commanding foot yaw forces the "
                         "hip to rotate the whole leg plane -- which swings the knee's bow out "
                         "of the ladder instead of into it. Costs nothing in balance.")
    ap.add_argument("--out-bulge", type=float, default=0.0,
                    help="swing the foot this far OUT from the ladder mid-step, m")
    ap.add_argument("--check", action="store_true", help="measure leg-link clearance offline")
    ap.add_argument("--step-frames", type=int, default=60)
    ap.add_argument("--settle-frames", type=int, default=25)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--top-tread", type=int, default=4)
    ap.add_argument("--platform", action="store_true", default=True,
                    help="finish by stepping onto the platform (the ladder's real top)")
    ap.add_argument("--no-platform", dest="platform", action="store_false")
    a = ap.parse_args()

    global PELVIS_INTO_LADDER, FOOT_HALF_STANCE, POSTURE_WAIST_ARMS
    if a.waist_pitch is not None:
        POSTURE_WAIST_ARMS = list(POSTURE_WAIST_ARMS)
        POSTURE_WAIST_ARMS[2] = a.waist_pitch          # waist_yaw, waist_roll, waist_PITCH
    if a.lean is not None:
        PELVIS_INTO_LADDER = a.lean
    if a.stance is not None:
        FOOT_HALF_STANCE = a.stance
    G = json.load(open(a.geometry))
    sole = json.load(open(a.treads))["meta"]["sole_offset_m"]
    lp = np.array(G["ladder_pos"]); lyaw = math.radians(G["ladder_yaw_deg"])
    from ladder_treads import TREADS_LOCAL, PLATFORM_LOCAL
    SURFACES = list(TREADS_LOCAL) + [PLATFORM_LOCAL]   # the platform is the real top
    Rl = np.array([[math.cos(lyaw), -math.sin(lyaw), 0.0],
                   [math.sin(lyaw),  math.cos(lyaw), 0.0], [0.0, 0.0, 1.0]])

    def world(x_local, y_local, z_local):
        return Rl @ np.array([x_local, y_local, z_local]) + lp

    # robot faces INTO the ladder: ladder-local +y, which Phase 1 solved as yaw 0 for this pose
    face_w = Rl @ np.array([0.0, -1.0, 0.0])
    robot_yaw = math.atan2(-face_w[1], -face_w[0])
    qroot = yaw_quat(robot_yaw)
    Rroot = quat_to_mat(qroot)

    def foot_target(tread_i, side):
        """World foot position standing on tread_i (0 = floor in front of tread 1).

        The lateral offset is taken along the ROBOT's own left axis, not the ladder's x: with
        this ladder yawed -90 deg, ladder-local +x maps to world -y, which is the robot's RIGHT,
        so using the ladder's axis silently crosses the legs.
        """
        lat = +FOOT_HALF_STANCE if side == "left" else -FOOT_HALF_STANCE
        if tread_i == 0:
            t0 = SURFACES[0]
            base = world(0.0, t0["y_front"] - 0.16, sole)
        else:
            t = SURFACES[tread_i - 1]
            # on the deep platform, stand a little back from its front edge rather than dead
            # centre of a 0.39 m slab, which would put the robot against the ladder's top cap
            y = (t["y_front"] + t["y_back"]) / 2.0 if t["y_back"] - t["y_front"] < 0.15 \
                else t["y_front"] + 0.11
            base = world(0.0, y, t["top"] + sole)
        return base + Rroot @ np.array([0.0, lat, 0.0])

    def clear_height(tread_i):
        return SURFACES[tread_i - 1]["top"] + sole + a.margin

    # ---- the step plan: both feet onto each tread in turn -------------------------------------
    # Both feet onto each tread, EXCEPT the one below the platform: the platform's front edge
    # sits 25 mm behind that tread's foot centre, so parking both feet there puts the shins
    # against it. One foot transitions through it, as a person does.
    plan = []
    for ti in range(1, a.top_tread + 1):
        plan.append(("right", ti))
        if ti < a.top_tread:
            plan.append(("left", ti))
    if a.platform:
        plan.append(("left", len(SURFACES)))
        plan.append(("right", len(SURFACES)))
    print(f"[PLAN  ] {len(plan)} steps: " + " -> ".join(f"{s[0][0].upper()}{s[1]}" for s in plan))

    foot_now = {"left": foot_target(0, "left"), "right": foot_target(0, "right")}
    q_now = {"left": np.array([-0.312, 0.0, 0.0, 0.669, -0.363, 0.0]),
             "right": np.array([-0.312, 0.0, 0.0, 0.669, -0.363, 0.0])}

    dof_rows, root_p_rows, root_q_rows, steps_meta = [], [], [], []

    # The leg has ONE budget, and a lean spends it. Standing 0.18 m out while still asking for
    # 0.740 m of height needs sqrt(0.180^2 + 0.740^2) = 0.762 m from a leg that spans 0.753: the
    # solver cannot reach it, saturates, and returns the SAME residual whatever else is asked of
    # it -- which is exactly what the first sweep produced (9.30 mm in every lean -0.18 row,
    # identical across stance, margin and bulge). Those rows scored well only because a robot at
    # full stretch barely touches the ladder. Lower the pelvis until both feet are inside the
    # budget, so a lean costs height, which is real, instead of reachability, which is a lie.
    REACH_USABLE = 0.98
    clamp_mm = [0.0]

    def pelvis_for(fL, fR):
        """Pelvis over the LOWEST foot, then lowered until both feet are within leg reach."""
        low = min(fL[2], fR[2])
        mean_xy = (fL[:2] + fR[:2]) / 2.0
        p = np.array([mean_xy[0], mean_xy[1], low + PELVIS_ABOVE_FOOT])
        p = p + Rroot @ np.array([PELVIS_INTO_LADDER, 0.0, 0.0])
        lim = PELVIS_MAX_ABOVE_FOOT * REACH_USABLE
        z_want = p[2]
        for f in (fL, fR):
            dxy = float(np.linalg.norm(p[:2] - f[:2]))
            p[2] = min(p[2], f[2] + math.sqrt(max(lim * lim - dxy * dxy, 1e-6)))
        clamp_mm[0] = max(clamp_mm[0], (z_want - p[2]) * 1000.0)
        return p

    # ---- hands on the rails ------------------------------------------------------------------
    # The torso is where the arms hang from, and it is the child of waist_pitch -- so pitching the
    # chest forward both moves the CoM toward the feet AND carries the shoulders toward the rails.
    waist_q = list(POSTURE_WAIST_ARMS[:3])     # mutated per frame; yaw alternates with the step
    q_arm = {s_: mirror_arm_seed(DEFAULT_ARM, s_) for s_ in ("left", "right")}
    hand_err = {"left": 0.0, "right": 0.0}
    grip_ok = {"left": 0, "right": 0}
    grip_n = {"left": 0, "right": 0}
    grip_z = {"left": [], "right": []}

    def place_hands(pelvis):
        """Grip the nearest reachable point on each front rail.

        The rail is a straight line in the ladder's (y, z) plane, so the best grip for a given
        shoulder is the foot of the perpendicular from that shoulder onto the line -- solved in
        closed form, then clamped to where the rail actually exists. Which rail is which is taken
        from the sign of the shoulder's own ladder-local x, not from a hard-coded hand of the
        ladder's yaw.
        """
        R_t_p, p_t_p = torso_frame(waist_q)
        Rtw = Rroot @ R_t_p
        ptw = pelvis + Rroot @ p_t_p
        for side in ("left", "right"):
            sh_w = ptw + Rtw @ np.asarray(ARM_CHAIN[side][0][1], float)
            sh_l = Rl.T @ (sh_w - lp)
            z = float(np.clip(rail_nearest_z(sh_l), a.rail_zmin, a.rail_zmax))
            tgt_l = rail_point(z, "right" if sh_l[0] > 0 else "left")
            tgt_l[1] -= a.grip_offset          # stand off the tube toward the climber
            q, ep = ik_arm(q_arm[side], Rtw.T @ ((Rl @ tgt_l + lp) - ptw), ARM_CHAIN[side])
            q_arm[side] = q
            hand_err[side] = max(hand_err[side], ep)
            if ep < 0.015:
                grip_ok[side] += 1
            grip_n[side] += 1
            grip_z[side].append(z)
        return q_arm

    def emit(pelvis, qL, qR):
        row = np.zeros(29)
        row[LEG_COLS["left"]] = qL
        row[LEG_COLS["right"]] = qR
        row[12:29] = POSTURE_WAIST_ARMS
        row[12:15] = waist_q
        if a.rail_hands:
            arms = place_hands(pelvis)
            row[15:22] = arms["left"][:7]
            row[22:29] = arms["right"][:7]
        dof_rows.append(row); root_p_rows.append(pelvis); root_q_rows.append(qroot)

    # A foot standing on a tread wants to be LEVEL. Seeding the target orientation from the
    # foot's current pose instead lets it drift, and the solver then trades position away to
    # honour it -- which is exactly what 230 mm of position error with 0.3 deg of rotation error
    # looks like. The pelvis is yaw-only here, so identity in the pelvis frame IS level, and it
    # matches the G1's default stance (hip -0.312 + knee 0.669 + ankle -0.363 = -0.006 rad).
    R_LEVEL = np.eye(3)

    # Toe-out. The knee does not sit on the hip-ankle line, it bows off it -- 83 mm at the
    # nominal stance, 152 mm on a bent swing leg -- and with the foot square that bow points
    # straight at the tread. Yawing the foot rotates the bow out to the side instead, and it
    # costs nothing in balance, which is the one currency a lean cannot spend (measured: at
    # lean -0.42 the CoM sits 251 mm outside the support and the ankles are 2.5x over).
    # A yaw about pelvis z keeps the sole level, so the foot is still flat on the tread.
    TOE = math.radians(a.toe_out)

    def R_foot(side):
        t = TOE if side == "left" else -TOE      # toes away from the midline
        c, sn = math.cos(t), math.sin(t)
        return np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])

    def solve(pelvis, side, p_des, q_seed):
        p_pelvis = Rroot.T @ (p_des - pelvis)
        q, ep, er = ik_leg(q_seed, p_pelvis, R_foot(side), LEG_CHAIN[side])
        return q, ep, er

    pel_now = pelvis_for(foot_now["left"], foot_now["right"])
    for _ in range(a.settle_frames):
        emit(pel_now, q_now["left"], q_now["right"])
        for side in ("left", "right"):
            q_now[side], _, _ = solve(pel_now, side, foot_now[side], q_now[side])
        dof_rows[-1][LEG_COLS["left"]] = q_now["left"]
        dof_rows[-1][LEG_COLS["right"]] = q_now["right"]

    worst_p = worst_r = 0.0
    knee_fwd = {"left": -9.9, "right": -9.9}      # max +x (toward the ladder) in the pelvis frame
    knee_lat = {"left": 0.0, "right": 0.0}

    def track_knee(side):
        _, _, orig, _ = leg_fk(q_now[side], LEG_CHAIN[side])
        k = orig[3]                                # knee joint origin, pelvis frame
        knee_fwd[side] = max(knee_fwd[side], float(k[0]))
        knee_lat[side] = max(knee_lat[side], abs(float(k[1])))

    for side, ti in plan:
        p0 = foot_now[side].copy(); p1 = foot_target(ti, side)
        path = arc_path(p0, p1, clear_height(ti), a.step_frames,
                        out_dir=-(Rl @ np.array([0.0, 1.0, 0.0])), out_bulge=a.out_bulge)
        f0 = len(dof_rows)
        stance = "left" if side == "right" else "right"
        # pelvis anchor AFTER this step: over whichever foot is then lowest
        fL_end = p1 if side == "left" else foot_now["left"]
        fR_end = p1 if side == "right" else foot_now["right"]
        pel_end = pelvis_for(fL_end, fR_end)
        pel_start = pel_now.copy()
        step_worst = (0.0, "", "", -1, 0.0, 0.0)
        yaw_t = math.radians(a.waist_yaw) * (1.0 if side == "left" else -1.0)
        for k in range(a.step_frames):
            # ease in over the first third, hold, ease out over the last third
            f = k / max(a.step_frames - 1, 1)
            ramp = smoothstep(min(f / 0.33, 1.0)) * (1.0 - smoothstep(max((f - 0.67) / 0.33, 0.0)))
            waist_q[0] = yaw_t * ramp
            fsw, fst = path[k], foot_now[stance]
            pel = pel_start + (pel_end - pel_start) * smoothstep(k / max(a.step_frames - 1, 1))
            emit(pel, q_now["left"], q_now["right"])
            for s2, tgt in ((side, fsw), (stance, fst)):
                q_now[s2], ep, er = solve(pel, s2, tgt, q_now[s2])
                reach = np.linalg.norm(tgt - pel)
                if ep > step_worst[0]:
                    step_worst = (ep, s2, "swing" if s2 == side else "stance", k, reach, er)
                worst_p = max(worst_p, ep); worst_r = max(worst_r, er)
            dof_rows[-1][LEG_COLS["left"]] = q_now["left"]
            dof_rows[-1][LEG_COLS["right"]] = q_now["right"]
            track_knee("left"); track_knee("right")
        waist_q[0] = 0.0
        foot_now[side] = p1
        pel_now = pel_end
        steps_meta.append({"swing": side, "tread": ti, "f0": f0, "f1": len(dof_rows) - 1,
                           "clear_height": clear_height(ti), "target": p1.tolist()})
        ep, s2, role, k, reach, er = step_worst
        print(f"[STEP  ] {side:5s} -> tread {ti}  frames {f0}-{len(dof_rows)-1}  "
              f"worst {ep*1000:7.1f} mm on the {s2:5s} leg ({role:6s}) at k={k:2d}, "
              f"pelvis-foot {reach:.3f} m, rot {math.degrees(er):.1f} deg", flush=True)

    for _ in range(a.settle_frames):
        emit(pel_now, q_now["left"], q_now["right"])
        for side in ("left", "right"):
            q_now[side], _, _ = solve(pel_now, side, foot_now[side], q_now[side])
        dof_rows[-1][LEG_COLS["left"]] = q_now["left"]
        dof_rows[-1][LEG_COLS["right"]] = q_now["right"]

    if a.rail_hands:
        gl, gr = grip_z["left"], grip_z["right"]
        fl = 100.0 * grip_ok["left"] / max(grip_n["left"], 1)
        fr = 100.0 * grip_ok["right"] / max(grip_n["right"], 1)
        both = 100.0 * max(grip_ok["left"], grip_ok["right"]) / max(grip_n["left"], 1)
        print(f"[HANDS ] rail grip: worst palm residual {hand_err['left']*1000:.1f} mm (L) / "
              f"{hand_err['right']*1000:.1f} mm (R); grip height {min(gl+gr):.3f}..{max(gl+gr):.3f} m")
        print(f"[HANDS ] on the rail (within 15 mm): left {fl:.0f}%, right {fr:.0f}%, "
              f"at least one hand {both:.0f}% of frames")
    print(f"[POSE  ] waist pitch {POSTURE_WAIST_ARMS[2]:+.3f} rad "
          f"({math.degrees(POSTURE_WAIST_ARMS[2]):+.1f} deg forward)")
    print(f"[KNEE  ] toe-out {a.toe_out:.0f} deg -> knee reaches {knee_fwd['left']*1000:+.0f} mm "
          f"(L) / {knee_fwd['right']*1000:+.0f} mm (R) toward the ladder, "
          f"lateral {knee_lat['left']*1000:.0f}/{knee_lat['right']*1000:.0f} mm")
    print(f"[PELVIS] reach clamp lowered the pelvis by at most {clamp_mm[0]:.1f} mm "
          f"(lean {PELVIS_INTO_LADDER:+.3f} m costs height out of the same leg budget)")
    print(f"[IK    ] worst residual over the whole climb: position {worst_p*1000:.2f} mm, "
          f"rotation {math.degrees(worst_r):.2f} deg")
    dof = np.array(dof_rows, np.float32)
    np.savez(a.out_clip, dof=dof, root_rot=np.array(root_q_rows, np.float64),
             root_trans_offset=np.array(root_p_rows, np.float32), fps=np.int64(a.fps))
    json.dump({"steps": steps_meta, "n_frames": len(dof_rows), "fps": a.fps,
               "ladder_pos": G["ladder_pos"], "ladder_yaw_deg": G["ladder_yaw_deg"],
               "sole_offset_m": sole, "treads": TREADS_LOCAL, "surfaces": SURFACES},
              open(a.out_meta, "w"), indent=2)
    if a.check:
        from scipy.spatial import cKDTree
        import glob
        vz = sorted(glob.glob("/home/yujin-chen/psi0_ft/climb/*/ladder_vertices.npz"))[-1]
        Vz = np.load(vz); Pc = np.concatenate([Vz[k] for k in Vz.files]).astype(float)
        tree = cKDTree(Pc)
        RADS = (("thigh", 2, 3, 0.075), ("knee", 3, 3, 0.065), ("shin", 3, 4, 0.055),
                ("foot", 4, 5, 0.055))
        worst = {k: (9.9, -1) for k, *_ in RADS}
        for fi in range(len(dof_rows)):
            pel = root_p_rows[fi]
            for side in ("left", "right"):
                _, _, orig, _ = leg_fk(dof_rows[fi][LEG_COLS[side]], LEG_CHAIN[side])
                W = np.array([pel + Rroot @ o for o in orig])
                for nm, i0_, i1_, r in RADS:
                    pts = np.array([W[i0_] * (1 - u) + W[i1_] * u for u in np.linspace(0, 1, 9)])
                    d = tree.query(pts)[0].min() - r
                    if d < worst[nm][0]:
                        worst[nm] = (float(d), fi)
        print("[CHECK ] leg-link clearance against the live ladder mesh (negative = penetrating):")
        for nm, _, _, _ in RADS:
            g, fi = worst[nm]
            print(f"         {nm:6s} {g*1000:+8.1f} mm at frame {fi:4d} (t {fi/a.fps:5.2f}s)"
                  + ("   PENETRATES" if g < 0 else ""))
    print(f"[OK    ] wrote {a.out_clip} ({len(dof_rows)} frames = {len(dof_rows)/a.fps:.1f} s) "
          f"and {a.out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
