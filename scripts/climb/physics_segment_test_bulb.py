"""physics_segment_test.py for the BULB climb (S10): a sibling, not an edit.

Runs in FIATLUX-S10-ClimbWithBulb-v0. The right hand keeps S10's cupped grip, the bulb is placed in it
at every staged pose (seat measured by measure_bulb_in_hand.py), and each test also reports the force
between the bulb and the ladder, the hardest squeeze on the bulb by a hand link (glass limit 50 N), and
how far the bulb ends from its seat. Everything else is the original test.
"""

"""Play segments of a reference clip under REAL physics, to test whether a hand hold bears load.

Why a new script
----------------
The authoring gate is kinematic: it proves a hand reaches its hold and clears the ladder, not
that the hold supports anything. This drops the robot into the clip's own pose at the start of a
segment (free base, real contacts, gravity) and then either freezes the joint targets ("hold")
or lets them follow the clip ("track"), and records what the hands, the feet and the ladder do.

There is NO balance controller here: the joints are plain PD on the clip's joint angles. So a
fall does not by itself condemn a hold -- run the same test on a segment whose hold is trusted
(a flat palm) and compare. What the test can show directly is the hand load, whether a hand
slides off its hold, and whether the open fingers get bent back.

measure_clip_clearance.py and trace_climb.py stay runnable; this is a variant, not an edit.
"""
import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description="Physics playback of clip segments.")
ap.add_argument("--task", default="FIATLUX-S10-ClimbWithBulb-v0")
ap.add_argument("--motion", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--robot", default="dex3", choices=["dex3", "inspire"])
ap.add_argument("--layout_seed", type=int, default=0)
ap.add_argument("--segment", action="append", required=True, help="NAME:START:END in 50 Hz clip steps; repeatable")
ap.add_argument(
    "--hold_steps", type=int, default=100, help="length of the frozen-target test at each segment's start pose"
)
ap.add_argument(
    "--lift",
    type=float,
    action="append",
    default=None,
    help="stage the robot this many mm above the clip pose and let it settle onto its "
    "holds; repeatable. The clip puts palms and soles AT the surface, and the "
    "collision hulls are fatter than the visual mesh, so lift 0 starts in "
    "interpenetration and the solver throws the robot off.",
)
ap.add_argument(
    "--fingers",
    default="open",
    choices=["open", "spawn"],
    help="open = every finger joint at 0, the pose the clip was authored for",
)
ap.add_argument(
    "--foot_sole", action="store_true", help="add the flat sole collider the real foot has (see foot_sole.py)"
)
ap.add_argument(
    "--yaw_sweep",
    action="store_true",
    help="no playback: robot in the clip's first pose (hands in free air), sweep both wrist "
    "yaw joints across their range and print what the solver pushes on. Only the yaw "
    "joint lies between the camera housing and the wrist_pitch link, so this shows at "
    "which angles the asset collides with itself.",
)
ap.add_argument(
    "--body_scan",
    type=int,
    default=0,
    help="no playback: every N clip steps, stage the pose, take ONE physics step and list "
    "EVERY robot body the solver pushes on (a contact sensor over the whole robot is "
    "added for this). Finds parts that start inside the ladder -- shins, thighs, "
    "elbows -- which the hand and foot sensors cannot see.",
)
ap.add_argument(
    "--probe",
    action="store_true",
    help="no playback: stage each segment's start pose at a ladder of lifts, take ONE "
    "physics step, and print which bodies the solver pushes on. The lift at which "
    "the push vanishes is how deep the collision shapes overlap at the clip pose.",
)
ap.add_argument("--modes", default="hold,track")
ap.add_argument("--scan_min", type=float, default=400.0, help="body scan: report pushes over this many newtons")
ap.add_argument("--scan_only", default="", help="body scan: only bodies whose name contains this")
ap.add_argument(
    "--wrist_offset",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.0],
    help="DIAGNOSTIC: radians added to the right wrist roll, pitch, yaw at every staged pose and target",
)
ap.add_argument(
    "--pin_root",
    action="store_true",
    help="test fixture: fix the robot's root at the reference pose, so the grip is tested without the body sagging or falling",
)
ap.add_argument(
    "--follow_root",
    action="store_true",
    help="test fixture for the BULB, not for balance: the robot is a puppet of the clip. Its gravity is off, the ladder is "
    "kinematic, and every control step the root pose and velocity and the 29 joints' positions and velocities are "
    "written from the clip, so the hand moves through the world exactly as the clip has it (speed, acceleration, "
    "tilt) while the bulb and the fingers stay dynamic. --pin_root freezes the root where the segment starts, which "
    "is only right while the body barely moves; with it a climbing segment read the palm 65 deg off level where the "
    "clip has 28 at most.",
)
ap.add_argument(
    "--right_curl",
    type=float,
    default=None,
    help="close the right hand's fingers to this curl (rad; the benchmark's cup is 0.667) AFTER the bulb is seated",
)
ap.add_argument("--curl_ramp", type=int, default=15, help="control steps over which --right_curl is reached")
ap.add_argument("--lm_gains", action="store_true", help="LadderMan's kp/kd instead of fiatlux's")
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.headless = True
a.enable_cameras = True
app = AppLauncher(a).app

import fiatlux_task.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ladderman_runtime import LadderManObs  # noqa: E402

set_layout_seed(a.layout_seed)
cfg = parse_env_cfg(a.task, num_envs=1)
if a.robot != "inspire":
    from fiatlux_task.robots.g1 import swap_robot_variant

    swap_robot_variant(cfg, a.robot)
if a.foot_sole:
    from foot_sole import add_foot_sole

    add_foot_sole(cfg)
    print("[foot] full-sole collider added to both feet")
if a.body_scan:
    from isaaclab.sensors import ContactSensorCfg

    cfg.scene.whole_body_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1)
# ---- bulb variant: the fresh bulb is in the right hand; watch what touches it ----
from isaaclab.sensors import ContactSensorCfg as _CSC

try:
    cfg.scene.fresh_bulb.spawn.activate_contact_sensors = True
except Exception as _e:
    print("[bulb] could not enable contact reporting on the bulb:", _e)
cfg.scene.bulb_any = _CSC(prim_path=cfg.scene.fresh_bulb.prim_path, history_length=1)  # everything touching the bulb
if not a.body_scan:
    cfg.scene.whole_body_contact = _CSC(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1)
cfg.scene.bulb_vs_ladder = _CSC(
    prim_path=cfg.scene.fresh_bulb.prim_path, filter_prim_paths_expr=[cfg.scene.ladder.prim_path], history_length=1
)
if a.pin_root:
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
if a.follow_root:
    cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    cfg.scene.ladder.spawn.rigid_props.kinematic_enabled = True
env = gym.make(a.task, cfg=cfg).unwrapped
env.reset()
base, dev = env, env.device
robot, ladder = base.scene["robot"], base.scene["ladder"]
ob = LadderManObs(base, a.motion)
DT, DEC = float(base.physics_dt), int(base.cfg.decimation)
print(f"[sim] physics dt {DT:.5f} s x decimation {DEC} = {DT * DEC:.3f} s per clip step")
if a.lm_gains:
    robot.write_joint_stiffness_to_sim(ob.lm_kp[None], joint_ids=ob.j_fx)
    robot.write_joint_damping_to_sim(ob.lm_kd[None], joint_ids=ob.j_fx)

MASS = float(robot.root_physx_view.get_masses().sum())
WEIGHT = MASS * 9.81
print(f"[robot] {a.robot}  mass {MASS:.2f} kg  weight {WEIGHT:.0f} N")
lad0 = ladder.data.root_state_w.clone()
q_all0 = robot.data.joint_pos.clone()  # spawn pose: source of the finger angles
jn = list(robot.joint_names)
fx29 = set(ob.j_fx.tolist())
fing = [i for i in range(len(jn)) if i not in fx29]
fL = [i for i in fing if "left" in jn[i].lower() or jn[i].startswith("L_")]
fR = [i for i in fing if i not in fL]
print("[fingers] spawn angles (deg): " + ", ".join(f"{jn[i]} {float(torch.rad2deg(q_all0[0, i])):.0f}" for i in fing))
if a.fingers == "open":
    q_all0[:, fL] = 0.0  # left hand open; the RIGHT keeps S10's cupped grip on the bulb
print(f"[fingers] {len(fL)} left + {len(fR)} right joints outside the clip, held at {a.fingers} angles")
bn = list(robot.body_names)
B = {
    n: bn.index(n)
    for n in ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link")
}
M = {n: ob.lm_body_names.index(n) for n in B}
sens = {"L": base.scene.sensors.get("left_hand_contact"), "R": base.scene.sensors.get("hand_contact")}
eff = (
    robot.data.joint_effort_limits[0]
    if hasattr(robot.data, "joint_effort_limits")
    else robot.root_physx_view.get_dof_max_forces()[0].to(dev)
)


from isaaclab.utils.math import quat_apply as _qa  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse as _qai
from isaaclab.utils.math import quat_mul as _qm

bulb = base.scene["fresh_bulb"]
_WR = ob.lm_body_names.index("right_wrist_yaw_link")
_BP = torch.tensor([[0.1192, 0.0540, -0.1310]], device=dev)
_BQ = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]], device=dev)


_CURL = None
if a.right_curl is not None:
    from fiatlux_task.poses import hand_cup_dex3

    _cup = hand_cup_dex3(a.right_curl)
    _CI = [jn.index(k) for k in _cup]
    _CV = torch.tensor([[_cup[k] for k in _cup]], device=dev)
    _CURL = (_CI, _CV)
    print(f"[fingers] right hand closes from the cup to curl {a.right_curl:.3f} over {a.curl_ramp} steps after seating")
_WJ = [jn.index(f"right_wrist_{k}_joint") for k in ("roll", "pitch", "yaw")]
_WOFF = torch.tensor([a.wrist_offset], device=dev)


def stage(t, lift=0.0):
    tt = torch.tensor([t], device=dev, dtype=torch.long)
    ladder.write_root_state_to_sim(lad0)
    rp = ob.ref_body_pos[tt][:, ob.root_mo].clone()
    rp[:, 2] += lift / 1000.0
    robot.write_root_link_pose_to_sim(torch.cat([rp, ob.ref_body_quat[tt][:, ob.root_mo]], -1))
    robot.write_root_com_velocity_to_sim(torch.zeros(1, 6, device=dev))
    q = q_all0.clone()
    q[:, ob.j_fx] = ob.ref_joint_pos[tt]
    q[:, _WJ] += _WOFF
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    wp = ob.ref_body_pos[tt][:, _WR]
    wq = ob.ref_body_quat[tt][:, _WR]
    wp = wp.clone()
    wp[:, 2] += lift / 1000.0
    if any(a.wrist_offset):  # the reference pose no longer describes the hand: read it back
        base.scene.write_data_to_sim()
        base.sim.forward()
        base.scene.update(0.0)
        _i = robot.body_names.index("right_wrist_yaw_link")
        wp, wq = robot.data.body_link_pos_w[0:1, _i].clone(), robot.data.body_link_quat_w[0:1, _i].clone()
    bulb.write_root_pose_to_sim(torch.cat([wp + _qa(wq, _BP), _qm(wq, _BQ)], dim=-1))
    bulb.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
    base.scene.write_data_to_sim()
    base.sim.forward()
    base.scene.update(0.0)


def bulb_forces():
    """(bulb against the ladder, hardest squeeze by the right hand's links), N."""
    try:
        fl_ = float(base.scene.sensors["bulb_vs_ladder"].data.force_matrix_w[0].norm(dim=-1).max())
    except Exception:
        fl_ = float("nan")
    try:
        fg_ = float(base.scene.sensors["grip_contact"].data.force_matrix_w[0].norm(dim=-1).max())
    except Exception:
        fg_ = float("nan")
    return fl_, fg_


def bulb_off_seat():
    """How far the bulb's glass centre is from where the hand seats it, m."""
    i = robot.body_names.index("right_wrist_yaw_link")
    wp, wq = robot.data.body_link_pos_w[0:1, i], robot.data.body_link_quat_w[0:1, i]
    want = wp + _qa(wq, _BP)
    return float((bulb.data.root_link_pos_w[0:1] - want).norm())


def run(name, mode, t0, t1, lift=0.0):
    stage(t0, lift)
    rows, fell = [], None
    n = (t1 - t0) if mode == "track" else a.hold_steps
    for k in range(n):
        t = t0 + k if mode == "track" else t0
        tt = torch.tensor([t], device=dev, dtype=torch.long)
        tgt = q_all0.clone()
        tgt[:, ob.j_fx] = ob.ref_joint_pos[tt]
        tgt[:, _WJ] += _WOFF
        if _CURL is not None:
            _w = min(1.0, (k + 1) / max(1, a.curl_ramp))
            tgt[:, _CURL[0]] = (1 - _w) * q_all0[:, _CURL[0]] + _w * _CURL[1]
        if a.follow_root and mode == "track":
            t1_ = torch.tensor([min(t + 1, int(ob.motion_len) - 1)], device=dev, dtype=torch.long)
            h_ = DT * DEC
            p0_, p1_ = ob.ref_body_pos[tt][:, ob.root_mo].clone(), ob.ref_body_pos[t1_][:, ob.root_mo]
            q0_, q1_ = ob.ref_body_quat[tt][:, ob.root_mo], ob.ref_body_quat[t1_][:, ob.root_mo]
            dq_ = _qm(q1_, q0_ * torch.tensor([[1.0, -1.0, -1.0, -1.0]], device=dev))
            dq_ = dq_ * torch.sign(dq_[:, :1] + 1e-12)
            j0_, j1_ = ob.ref_joint_pos[tt], ob.ref_joint_pos[t1_]
            tgt[:, ob.j_fx] = j1_
            _rv = torch.cat([(p1_ - p0_) / h_, 2.0 * dq_[:, 1:] / h_], -1)
            _jv = (j1_ - j0_) / h_
        robot.set_joint_position_target(tgt)
        for _s in range(DEC):
            if (
                a.follow_root and mode == "track"
            ):  # every physics step, so a contact cannot carry the puppet off the clip
                _u = _s / DEC
                _qq = q0_ * (1 - _u) + (q1_ if float((q0_ * q1_).sum()) >= 0 else -q1_) * _u
                robot.write_root_link_pose_to_sim(
                    torch.cat([p0_ + (p1_ - p0_) * _u, _qq / _qq.norm(dim=-1, keepdim=True)], -1)
                )
                robot.write_root_com_velocity_to_sim(_rv)
                robot.write_joint_state_to_sim(j0_ + (j1_ - j0_) * _u, _jv, joint_ids=ob.j_fx)
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(DT)
        d = robot.data
        r = {"t": t}
        for nme in B:
            r[nme] = (d.body_pos_w[0, B[nme]] - ob.ref_body_pos[tt][0, M[nme]]).cpu().numpy()
        g = d.projected_gravity_b[0]
        r["tilt"] = float(torch.atan2(g[:2].norm(), -g[2]))
        for s in ("L", "R"):
            f = sens[s].data.net_forces_w[0].sum(0).cpu().numpy() if sens[s] is not None else np.zeros(3)
            r["F" + s] = f
        r["tau"] = float((d.applied_torque[0, ob.j_fx].abs() / eff[ob.j_fx]).max())
        r["tau_j"] = jn[int(ob.j_fx[(d.applied_torque[0, ob.j_fx].abs() / eff[ob.j_fx]).argmax()])]
        r["fingL"] = float((d.joint_pos[0, fL] - q_all0[0, fL]).abs().max()) if fL else 0.0
        r["fingR"] = float((d.joint_pos[0, fR] - q_all0[0, fR]).abs().max()) if fR else 0.0
        r["lad"] = float((ladder.data.root_pos_w[0] - lad0[0, :3]).norm())
        r["bulb_lad"], r["bulb_grip"] = bulb_forces()
        r["bulb_off"] = bulb_off_seat()
        _iw = robot.body_names.index("right_wrist_yaw_link")
        r["bulb_w"] = (
            _qai(
                robot.data.body_link_quat_w[0:1, _iw],
                bulb.data.root_link_pos_w[0:1] - robot.data.body_link_pos_w[0:1, _iw],
            )[0]
            .cpu()
            .numpy()
        )
        _i = robot.body_names.index("right_wrist_yaw_link")
        _n = _qa(robot.data.body_link_quat_w[0:1, _i], torch.tensor([[0.0, 1.0, 0.0]], device=dev))[0]
        _f = _qa(robot.data.body_link_quat_w[0:1, _i], torch.tensor([[1.0, 0.0, 0.0]], device=dev))[0]
        r["palm_tilt"] = float(torch.rad2deg(torch.acos(_n[2].clamp(-1, 1))))
        r["finger_pitch"] = float(torch.rad2deg(torch.asin(_f[2].clamp(-1, 1))))
        r["fingR_each"] = (d.joint_pos[0, fR] - q_all0[0, fR]).cpu().numpy()
        if k < 4:  # what is touching what in the first moments
            _wb = base.scene.sensors["whole_body_contact"]
            _fb = _wb.data.net_forces_w[0].norm(dim=-1)
            _top = [
                f"{_wb.body_names[j]} {float(_fb[j]):.0f}"
                for j in torch.argsort(_fb, descending=True)[:7].tolist()
                if _fb[j] > 2.0 and "ankle" not in _wb.body_names[j]
            ]
            _ba = float(base.scene.sensors["bulb_any"].data.net_forces_w[0].norm(dim=-1).max())
            print(
                f"    step {k}: bulb net contact {_ba:.1f} N, off seat {bulb_off_seat() * 1000:.0f} mm; robot bodies pushed: "
                + (", ".join(_top) or "none over 2 N"),
                flush=True,
            )
        if k == 0:
            for sd in ("L", "R"):
                if sens[sd] is not None:
                    fb = sens[sd].data.net_forces_w[0].norm(dim=-1)
                    top = torch.argsort(fb, descending=True)[:3].tolist()
                    print(
                        f"    first step, {sd} hand bodies in contact: "
                        + ", ".join(f"{sens[sd].body_names[j]} {float(fb[j]):.0f} N" for j in top if fb[j] > 1)
                    )
        rows.append(r)
        if fell is None and (r["pelvis"][2] < -0.25 or r["tilt"] > 1.0):
            fell = k
            break
    return rows, fell


def report(name, mode, rows, fell, t0):
    A = lambda k: np.array([r[k] for r in rows])
    pe = np.linalg.norm(A("pelvis"), axis=1) * 1000
    wl = np.linalg.norm(A("left_wrist_yaw_link"), axis=1) * 1000
    wr = np.linalg.norm(A("right_wrist_yaw_link"), axis=1) * 1000
    fl = np.linalg.norm(A("left_ankle_roll_link"), axis=1) * 1000
    fr = np.linalg.norm(A("right_ankle_roll_link"), axis=1) * 1000
    FL, FR = A("FL"), A("FR")
    up = (FL[:, 2] + FR[:, 2]) / WEIGHT * 100
    i = int(np.argmax(A("tau")))
    out = (
        "FELL at +%.2f s (clip step %d)" % (fell * DT * DEC, rows[fell]["t"])
        if fell is not None
        else "stayed up for %.2f s" % (len(rows) * DT * DEC)
    )
    print(f"[{name:>14s} {mode:5s}] {out}")
    print(
        f"    pelvis off reference: median {np.median(pe):.0f} mm, max {pe.max():.0f} mm; tilt max {np.degrees(A('tilt').max()):.0f} deg"
    )
    print(
        f"    wrist off reference:  L median {np.median(wl):.0f} max {wl.max():.0f} mm;  R median {np.median(wr):.0f} max {wr.max():.0f} mm"
    )
    print(
        f"    foot off reference:   L median {np.median(fl):.0f} max {fl.max():.0f} mm;  R median {np.median(fr):.0f} max {fr.max():.0f} mm"
    )
    print(
        f"    hand contact force:   L median {np.median(np.linalg.norm(FL, axis=1)):.0f} max {np.linalg.norm(FL, axis=1).max():.0f} N;  R median {np.median(np.linalg.norm(FR, axis=1)):.0f} max {np.linalg.norm(FR, axis=1).max():.0f} N"
    )
    print(
        f"    hands carry (vertical): median {np.median(up):.0f}% of body weight, range {up.min():.0f}..{up.max():.0f}%"
    )
    print(
        f"    finger joints pushed off their open pose: L max {np.degrees(A('fingL').max()):.0f} deg, R max {np.degrees(A('fingR').max()):.0f} deg"
    )
    print(
        f"    peak joint torque {A('tau')[i] * 100:.0f}% of limit ({rows[i]['tau_j']} at step {rows[i]['t']}); ladder moved {A('lad').max() * 1000:.0f} mm"
    )
    _k = int(np.argmax(A("bulb_off") > 0.15)) if (A("bulb_off") > 0.15).any() else -1
    print(
        f"    right palm off level: start {A('palm_tilt')[0]:.0f} deg, max {A('palm_tilt').max():.0f} deg; fingers pitched (+up) start {A('finger_pitch')[0]:+.0f}, min {A('finger_pitch').min():+.0f} deg; "
        f"right finger joints pushed off the cup by up to {np.degrees(np.abs(A('fingR_each')).max()):.0f} deg"
        + (f"; bulb left the hand at +{_k * DT * DEC:.2f} s" if _k >= 0 else "")
    )
    _bw = A("bulb_w")
    _m = min(len(_bw) - 1, 25)
    print(
        f"    bulb root in the wrist frame (m): after the grip closes {np.round(_bw[_m], 3).tolist()}, median {np.round(np.median(_bw[_m:], 0), 3).tolist()}, "
        f"range {np.round(_bw[_m:].min(0), 3).tolist()} .. {np.round(_bw[_m:].max(0), 3).tolist()}"
    )
    print(
        f"    BULB: against the ladder max {np.nanmax(A('bulb_lad')):.0f} N; hardest squeeze by a hand link max {np.nanmax(A('bulb_grip')):.0f} N (glass limit 50 N); "
        f"off its seat in the hand: end {A('bulb_off')[-1] * 1000:.0f} mm, max {A('bulb_off').max() * 1000:.0f} mm"
    )
    return {
        f"{name}_{mode}_{k}": A(k)
        for k in (
            "t",
            "pelvis",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "FL",
            "FR",
            "tilt",
            "tau",
            "fingL",
            "fingR",
            "lad",
            "bulb_lad",
            "bulb_grip",
            "bulb_off",
            "bulb_w",
        )
    }


print("[sensors] " + ", ".join(base.scene.sensors.keys()))
if a.yaw_sweep:
    with torch.inference_mode():
        jy = {sd: jn.index(f"{sd}_wrist_yaw_joint") for sd in ("left", "right")}
        lim = robot.data.soft_joint_pos_limits[0] if hasattr(robot.data, "soft_joint_pos_limits") else None
        print(
            "[yaw] limits (deg): "
            + ", ".join(
                f"{sd} {float(torch.rad2deg(robot.data.joint_pos_limits[0, jy[sd], 0])):.1f}..{float(torch.rad2deg(robot.data.joint_pos_limits[0, jy[sd], 1])):.1f}"
                for sd in jy
            )
        )
        for deg in range(-92, 93, 4):
            stage(0, 0.0)
            q = robot.data.joint_pos.clone()
            for sd in jy:
                q[:, jy[sd]] = float(np.radians(deg))
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(DT)
            out = []
            for key in ("left_hand_contact", "hand_contact"):
                sn = base.scene.sensors[key]
                fb = sn.data.net_forces_w[0].norm(dim=-1)
                top = [
                    f"{sn.body_names[j]} {float(fb[j]):.0f}"
                    for j in torch.argsort(fb, descending=True)[:3].tolist()
                    if fb[j] > 5
                ]
                out.append(("L: " if key.startswith("left") else "R: ") + (", ".join(top) or "none"))
            print(f"[yaw {deg:+4d} deg] " + " | ".join(out), flush=True)
    import os

    os._exit(0)
if a.body_scan:
    with torch.inference_mode():
        sn = base.scene.sensors["whole_body_contact"]
        T_ = int(ob.motion_len)
        tally = {}
        bmax = gmax = 0.0
        btouch = []
        for t in range(0, T_, a.body_scan):
            stage(t, 0.0)
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(DT)
            fb = sn.data.net_forces_w[0].norm(dim=-1)
            hit = [
                (sn.body_names[j], float(fb[j]))
                for j in torch.argsort(fb, descending=True).tolist()
                if fb[j] > a.scan_min and a.scan_only in sn.body_names[j]
            ]
            v = float(robot.data.root_lin_vel_w[0].norm())
            for n_, f_ in hit:
                tally.setdefault(n_, []).append((t, f_))
            bl_, bg_ = bulb_forces()
            bmax = max(bmax, 0.0 if bl_ != bl_ else bl_)
            gmax = max(gmax, 0.0 if bg_ != bg_ else bg_)
            if bl_ > 1.0:
                btouch.append((t, bl_))
            print(
                f"[scan {t:5d}] pelvis speed {v:.2f} m/s | bulb-ladder {bl_:.0f} N, grip {bg_:.0f} N | "
                + ("; ".join(f"{n_} {f_:.0f}" for n_, f_ in hit[:6]) or "nothing over 400 N"),
                flush=True,
            )
        print(
            f"[scan] BULB against the ladder: max {bmax:.0f} N; touching (>1 N) at {len(btouch)} poses: {[(t_, round(f_)) for t_, f_ in btouch][:20]}"
        )
        print(f"[scan] BULB squeezed by the hand: max {gmax:.0f} N on one link (glass limit 50 N)", flush=True)
        print(
            "[scan] ---- bodies pushed with more than 400 N on the first step (a foot settling reads under ~350 N) ----"
        )
        for n_, lst in sorted(tally.items(), key=lambda kv: -len(kv[1])):
            import sys as _s

            _s.stdout.flush()
            print(
                f"[scan] {n_:28s} at {len(lst)} of {len(range(0, T_, a.body_scan))} poses, up to {max(f for _, f in lst):.0f} N; steps {[t for t, _ in lst][:24]}"
            )
    import os
    import sys as _s2

    _s2.stdout.flush()
    os._exit(0)
if a.probe:
    with torch.inference_mode():
        for spec in a.segment:
            name, t0, _ = spec.split(":")
            for lift in range(0, 85, 5):
                stage(int(t0), float(lift))
                base.scene.write_data_to_sim()
                base.sim.step(render=False)
                base.scene.update(DT)
                out = []
                for key, sn in base.scene.sensors.items():
                    if not hasattr(sn.data, "net_forces_w") or sn.data.net_forces_w is None:
                        continue
                    fb = sn.data.net_forces_w[0].norm(dim=-1)
                    for j in torch.argsort(fb, descending=True)[:4].tolist():
                        if fb[j] > 5:
                            out.append(f"{sn.body_names[j]} {float(fb[j]):.0f}")
                v = float(robot.data.root_lin_vel_w[0].norm())
                print(
                    f"[probe {name} +{lift:2d} mm] pelvis speed after one step {v:.2f} m/s | "
                    + ("; ".join(dict.fromkeys(out)) or "no contact force"),
                    flush=True,
                )
    import os

    os._exit(0)
save = {}
with torch.inference_mode():
    for spec in a.segment:
        name, t0, t1 = spec.split(":")
        t0, t1 = int(t0), int(t1)
        for lift in a.lift or [0.0]:
            for mode in a.modes.split(","):
                rows, fell = run(name, mode, t0, t1, lift)
                save.update(report(f"{name}+{lift:.0f}mm", mode, rows, fell, t0))
                sys.stdout.flush()
o = pathlib.Path(a.out)
o.parent.mkdir(parents=True, exist_ok=True)
np.savez(o, **save)
print(f"[done] -> {o}", flush=True)
import os  # noqa: E402

os._exit(0)  # Kit hangs on shutdown after the results are written
