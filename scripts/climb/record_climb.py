"""Drive a LadderMan teacher through a fiatlux climb subtask and record the bag.

A sibling of ``scripts/record_run.py``, not a modification of it: the teacher is not a
``make_policy`` spec (it needs four observation groups the benchmark does not build, its own
reference clip and its own phase clock), and wiring it in would mean editing both ``policy.py``
and the spec list that has to stay in sync with it. Everything else -- the env, the recorder, the
bag format -- is the benchmark's own.

Step order matters and mirrors LadderMan's: refresh the history buffers from the CURRENT state,
compute the groups, act, then record the action and advance the reference clock. That ordering is
what makes ``prev_actions`` and ``applied_action`` hold the PREVIOUS action at observation time,
which is what they held in training.

Examples
--------
    python scripts/climb/record_climb.py --headless --checkpoint .../checkpoint_2700.pt \
        --motion /home/yujin-chen/psi0_ft/climb/topout --out /tmp/climb-run --max_steps 1200
"""

import argparse
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description="Record a LadderMan teacher on a climb subtask.")
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--checkpoint", required=True, help="a LadderMan teacher .pt")
ap.add_argument("--motion", required=True, help="the motion dir the teacher was trained on")
ap.add_argument("--out", default=None, help="write a bag here (omit to only print the gate trace)")
ap.add_argument("--episodes", type=int, default=1)
ap.add_argument("--max_steps", type=int, default=6000, help="S02 is 120 s at 50 Hz = 6000")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--format", default="hdf5")
ap.add_argument("--print_every", type=int, default=100)
ap.add_argument("--match_torque", action="store_true",
                help="convert targets so OUR actuators reproduce LadderMan's torque")
ap.add_argument("--residual_scale", type=float, default=1.0,
                help="scale the NETWORK's 23 outputs. 0 = pure open-loop replay of the reference "
                     "through the same action path, which isolates the conversion from the policy.")
ap.add_argument("--layout_seed", type=int, default=0,
                help="fix the Replace layout draw (ladder pose, fixture mount). MUST be set "
                     "before the cfg is built or the layout comes from OS entropy and two runs "
                     "are different scenes, not a repeat -- see scene_cfg.set_layout_seed. "
                     "Default 0 = reproducible; pass -1 for OS entropy (varied bags).")
ap.add_argument("--lm_gains", action="store_true",
                help="run the robot on LADDERMAN's PD gains instead of fiatlux's. Not the same "
                     "test as --match_torque: that solves for a target so OUR gains produce "
                     "LadderMan's torque, whereas ImplicitActuator hands kp/kd to PhysX, which "
                     "solves the PD inside the solver -- so a softer joint behaves differently "
                     "from a stiff joint fed a converted target. Gains are a control-stack "
                     "choice, not hardware; armature (the other mismatch) is hardware and is "
                     "deliberately NOT touched here.")
ap.add_argument("--rsi", action="store_true",
                help="DIAGNOSTIC: stage the robot in the REFERENCE's frame-0 state at every "
                     "reset -- root pose, root velocity and the 29 joint positions -- which is "
                     "what LadderMan's `sample_init` does on every training episode. Their "
                     "teacher only ever sees initial states drawn from the reference pose "
                     "+/-0.1 rad per joint and +/-0.02 m of root, whereas the benchmark spawn "
                     "is 0.600 rad off at the elbows and 0.252 m behind. A bag recorded with "
                     "this flag is NOT a valid S02 demo -- the benchmark's spawn pose is part "
                     "of what S02 measures. It answers one question: does the teacher fail "
                     "because it cannot climb, or because it wakes up out of distribution?")
ap.add_argument("--standoff_delta", type=float, default=0.0,
                help="metres to move the spawn TOWARD the ladder along the step-facing ray "
                     "(positive = closer). The reference is anchored to the LADDER, not to the "
                     "robot, so the spawn standoff sets the frame-0 tracking error the policy "
                     "wakes up holding -- see the [ALIGN ] line, which measures it directly. "
                     "Only meaningful for a floor stance; refused on a platform stance.")
ap.add_argument("--step_ladder", action="store_true",
                help="the teacher was trained on the fiatlux STEP ladder (ladder_port/ladder_fiatlux.py) with "
                     "the wrists following the clip: wrist targets come from the reference, and the ladder "
                     "observation reads tread 1's box centre as that trainer does")
ap.add_argument("--jpos_offset", default=None, choices=["default", "refwrist"],
                help="what joint_pos_history is measured against (see LadderManObs); default follows --step_ladder")
ap.add_argument("--pin_ladder", action="store_true",
                help="DIAGNOSTIC: make the ladder kinematic (immovable), as it is in the LadderMan trainer. "
                     "Not a valid benchmark run -- it isolates the free-standing ladder from everything else.")
ap.add_argument("--mass_to", type=float, default=0.0,
                help="DIAGNOSTIC: scale every robot link's mass and inertia so the total is this many kg "
                     "(the step-ladder trainer's robot is 34.39). Not a valid benchmark run.")
ap.add_argument("--trace", type=int, default=0,
                help="DIAGNOSTIC: every N steps print the root's error against the reference in the robot's "
                     "yaw frame, its tilt, and the five joints furthest from the reference")
ap.add_argument("--rsi_noise", type=float, default=0.0,
                help="with --rsi: uniform noise (rad) added to the 29 staged joint angles, so repeated episodes "
                     "differ. LadderMan trains with +-0.1.")
ap.add_argument("--stop_at_clip_end", action="store_true",
                help="end an episode when the reference clock reaches the clip's last frame and report where the "
                     "robot is; with --episodes N this gives a success count")
ap.add_argument("--robot", default="inspire", choices=["inspire", "dex3"],
                help="hand variant. The step-ladder teachers were trained on the OPEN Dex3 hand: use dex3 with them.")
ap.add_argument("--dump_spawn", default=None,
                help="write the task's spawn state (root pose in the CLIP's frame, the 29 joint angles) to this "
                     "json and exit; input for make_approach_prefix.py")
ap.add_argument("--video", default=None, help="write an mp4 of the run from a fixed camera (one frame per 50 Hz step)")
ap.add_argument("--cam_eye", type=float, nargs=3, default=[3.25, 2.2, 1.45], help="camera position, world (layout seed 0 default)")
ap.add_argument("--cam_look_z", type=float, default=1.25, help="aim this far above the ladder's root")
ap.add_argument("--past_success", action="store_true",
                help="DIAGNOSTIC: do not end the episode when the task's success gate fires (S02's fires on tread 4, "
                     "before the feet reach the platform), so the run can continue to the clip's end. The step at "
                     "which the real gate would have fired is printed. Not a valid benchmark run.")
ap.add_argument("--ladder_stab_threshold", type=float, default=None,
                help="DIAGNOSTIC: override the ladder's PhysX stabilization_threshold (the benchmark ships 0.001; with "
                     "it the free-standing ladder starts to tip at 10-12 N of sideways push at tread-4 height, with 0 at "
                     "59 N, which is what its weight and foot spacing give). Not a valid benchmark run.")
ap.add_argument("--zscore", type=int, default=0,
                help="DIAGNOSTIC: every N steps print, per observation term, the largest |z| against the "
                     "checkpoint's own running statistics. A phase-averaged oracle: a large value flags a term "
                     "to look at, it does not prove the term is wrong.")
ap.add_argument("--asset_meta", default=None, help="the training run's own asset_meta.json (default: the rung-ladder teacher's)")
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import fiatlux_task.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.recording import TrajectoryRecorder  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from ladderman_policy import LadderManPolicy  # noqa: E402
from ladderman_runtime import LadderManObs  # noqa: E402

from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402

# Before parse_env_cfg, not after: the layout is drawn in __post_init__, which runs INSIDE
# parse_env_cfg, so cfg.seed is far too late. Without this every run is a different ladder pose
# and yaw, and a sweep compares scenes instead of whatever it meant to vary.
set_layout_seed(None if a.layout_seed < 0 else a.layout_seed)
print(f"[LAYOUT] seed {'OS entropy' if a.layout_seed < 0 else a.layout_seed}")

cfg = parse_env_cfg(a.task, num_envs=1)

# Move the spawn along the ladder->robot ray. balance.stand_robot_at_ladder_base already put the
# robot on the ladder's step-facing ray and turned it to face the ladder, so sliding along that
# same ray is exactly a standoff change and leaves the bearing alone. Recomputed from the two
# poses rather than imported, so it stays correct if the layout reseeds.
if a.standoff_delta != 0.0:
    import math  # noqa: E402

    rx, ry, rz = cfg.scene.robot.init_state.pos
    lx, ly = cfg.scene.ladder.init_state.pos[:2]
    d = math.hypot(rx - lx, ry - ly)
    if d < 0.3:
        raise SystemExit(
            f"[STANCE] refusing --standoff_delta: the spawn is {d:.3f} m from the ladder root, "
            "which is a platform stance (descent), not a floor stance. Sliding along this ray "
            "would walk the robot off the top, not change a standoff."
        )
    new_d = d - a.standoff_delta
    cfg.scene.robot.init_state.pos = (lx + (rx - lx) / d * new_d, ly + (ry - ly) / d * new_d, rz)
    print(f"[STANCE] standoff {d:.3f} -> {new_d:.3f} m (delta {a.standoff_delta:+.3f} toward the ladder)")

if a.robot != "inspire":
    from fiatlux_task.robots.g1 import swap_robot_variant  # noqa: E402
    swap_robot_variant(cfg, a.robot)
    print(f"[ROBOT ] hand variant {a.robot}")
_real_gate = None
if a.past_success:
    _real_gate = (cfg.terminations.success.func, dict(cfg.terminations.success.params or {}))
    def _never(env):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    cfg.terminations.success.func = _never
    cfg.terminations.success.params = {}
    print("[GATE  ] DIAGNOSTIC: the success termination is disabled; the real gate is only evaluated and reported")
if a.video:
    from fiatlux_task.viz import make_video_camera_cfg  # noqa: E402
    cfg.scene.video_cam = make_video_camera_cfg(1280, 720)
if a.trace:
    from isaaclab.sensors import ContactSensorCfg as _CSC  # noqa: E402
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.whole_body_contact = _CSC(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1)
    try:                                               # what the LADDER feels, split into robot bodies and everything else
        cfg.scene.ladder.spawn.activate_contact_sensors = True
        cfg.scene.ladder_contact = _CSC(prim_path=cfg.scene.ladder.prim_path, history_length=1,
                                        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"])
    except Exception as _e:
        print("[LADDER] no contact sensor on the ladder:", _e)
if a.ladder_stab_threshold is not None:
    cfg.scene.ladder.spawn.rigid_props.stabilization_threshold = a.ladder_stab_threshold
    print(f"[LADDER] DIAGNOSTIC: ladder stabilization_threshold = {a.ladder_stab_threshold}")
if a.pin_ladder:
    cfg.scene.ladder.spawn.rigid_props.kinematic_enabled = True
    print("[LADDER] DIAGNOSTIC: ladder is kinematic")
env = gym.make(a.task, cfg=cfg)
base = env.unwrapped

# Tread 1's box centre in the ladder's own frame: clip x -0.58825 (ladder_fiatlux.py) -> ladder y +0.1996.
obs_builder = LadderManObs(base, a.motion, asset_meta=a.asset_meta,
                           rung0_obs_local=(0.0, -0.38865, 0.196) if a.step_ladder else None,
                           jpos_offset=a.jpos_offset or ("refwrist" if a.step_ladder else "default"))
net = LadderManPolicy(a.checkpoint, device=base.device)
print(f"[POLICY] {a.checkpoint}")
print(f"[MOTION] {obs_builder.motion_len} steps at 50 Hz = {obs_builder.motion_len / 50:.2f} s")
print(f"[FRAME ] rung0 {[round(v, 3) for v in obs_builder.rung0_w[0].tolist()]}, "
      f"climb yaw {float(obs_builder.climb_yaw[0]) * 57.2958:+.1f} deg")

recorder = (
    TrajectoryRecorder(base, policy_spec=f"ladderman:{pathlib.Path(a.checkpoint).name}",
                       seed=a.seed, checkpoint=a.checkpoint, out_dir=a.out, fmt=a.format)
    if a.out else None
)

obs, _ = env.reset(seed=a.seed)
obs_builder.reset()
_lad0 = base.scene["ladder"].data.root_pos_w[0].clone()
_L_W = float(base.scene["ladder"].root_physx_view.get_masses().sum()) * 9.81
_pv = base.scene["robot"].root_physx_view
_m = _pv.get_masses().clone()
_bn = list(base.scene["robot"].body_names)
print(f"[MASS  ] robot total {float(_m.sum()):.2f} kg; heaviest: " + ", ".join(f"{_bn[i]} {float(_m[0, i]):.2f}" for i in torch.argsort(_m[0], descending=True)[:6].tolist()))
print("[MASS  ] hands: " + ", ".join(f"{_bn[i]} {float(_m[0, i]):.3f}" for i in range(len(_bn)) if "hand" in _bn[i] or "wrist_yaw" in _bn[i]))
if a.mass_to > 0:
    _k = a.mass_to / float(_m.sum())
    _idx = torch.arange(base.num_envs, device="cpu")
    _pv.set_masses((_m * _k).cpu(), _idx)
    _pv.set_inertias((_pv.get_inertias().clone() * _k).cpu(), _idx)
    print(f"[MASS  ] DIAGNOSTIC: every link scaled by {_k:.4f} -> {float(_pv.get_masses().sum()):.2f} kg")

from isaaclab.utils.math import quat_apply_inverse  # noqa: E402


def apply_lm_gains() -> None:
    """Write LadderMan's kp/kd onto the 29 mapped joints. Re-applied after every reset.

    Combining this with --match_torque is self-consistent, not double-applied: the conversion
    reads the LIVE joint_stiffness, so once the gains ARE LadderMan's the conversion collapses
    to the identity.
    """
    rob = base.scene["robot"]
    n = base.num_envs
    rob.write_joint_stiffness_to_sim(obs_builder.lm_kp[None].expand(n, -1), joint_ids=obs_builder.j_fx)
    rob.write_joint_damping_to_sim(obs_builder.lm_kd[None].expand(n, -1), joint_ids=obs_builder.j_fx)


def apply_rsi() -> None:
    """Write the reference's frame-0 state onto the robot (LadderMan's reference state init).

    Mirrors `RobotTracking.sample_init`: root link pose, root com velocity, and the 29 mapped
    joint positions. Joint velocity is the clip's own forward difference at 50 Hz rather than
    zero, so the robot is handed the reference's motion as well as its pose. The hand joints are
    left at their spawn values -- they are not in the clip, and j_fx addresses only the 29.

    Must run AFTER env.reset() (which stages the benchmark spawn) and BEFORE
    obs_builder.reset() (which seeds the history buffers from the live state).
    """
    rob = base.scene["robot"]
    ob = obs_builder
    t0 = ob.t
    root_p = ob.ref_body_pos[t0][:, ob.root_mo]
    root_q = ob.ref_body_quat[t0][:, ob.root_mo]
    lin = ob.ref_body_linvel[t0][:, ob.root_mo]
    ang = ob.ref_body_angvel[t0][:, ob.root_mo]
    rob.write_root_link_pose_to_sim(torch.cat([root_p, root_q], dim=-1))
    rob.write_root_com_velocity_to_sim(torch.cat([lin, ang], dim=-1))
    q = ob.ref_joint_pos[t0]
    if a.rsi_noise > 0:
        q = q + (torch.rand_like(q) * 2 - 1) * a.rsi_noise
    q_next = ob.ref_joint_pos[torch.clamp(t0 + 1, max=ob.motion_len - 1)]
    qd = (q_next - q) * 50.0
    rob.write_joint_state_to_sim(q, qd, joint_ids=ob.j_fx)
    if a.step_ladder:
        _a = torch.arange(len(rob.joint_names), device=base.device)
        _f = _a[~torch.isin(_a, ob.j_fx)]
        if _f.numel():
            _z = torch.zeros(base.num_envs, _f.numel(), device=base.device)
            rob.write_joint_state_to_sim(_z, _z, joint_ids=_f)


if os.environ.get("FX_LADDER_INFO"):
    _L = base.scene["ladder"]; _v = _L.root_physx_view
    _com = _v.get_coms()[0].flatten().tolist()
    print(f"[LADINF] mass {float(_v.get_masses().sum()):.2f} kg; centre of mass in the ladder's own frame (x across, y climb direction, z up): "
          f"{[round(v, 3) for v in _com[:3]]}; inertia diag {[round(float(v), 3) for v in _v.get_inertias()[0].flatten()[[0, 4, 8]].tolist()]}")
    print(f"[LADINF] root at rest {[round(float(v), 3) for v in _L.data.root_pos_w[0].tolist()]}; clip-frame x of the frame origin {-0.1996:+.4f} (ladder y = clip x + 0.1996)")
if a.dump_spawn:
    import json as _json  # noqa: E402
    _rb = base.scene["robot"]
    for _ in range(int(os.environ.get("FX_SPAWN_SETTLE_STEPS", "0"))):
        env.step(torch.zeros(base.num_envs, len(_rb.joint_names), device=base.device))
    _p, _q = obs_builder.to_clip(_rb.data.root_pos_w[:1], _rb.data.root_quat_w[:1])
    _jp = _rb.data.joint_pos[0, obs_builder.j_fx]
    _out = {"task": a.task, "layout_seed": a.layout_seed, "seed": a.seed, "robot": a.robot,
            "root_pos_clip": [round(float(v), 5) for v in _p[0]], "root_quat_clip": [round(float(v), 6) for v in _q[0]],
            "joint_pos": {n: round(float(v), 5) for n, v in zip(obs_builder.lm_joint_names, _jp)}}
    pathlib.Path(a.dump_spawn).write_text(_json.dumps(_out, indent=1))
    print(f"[SPAWN ] root in the clip's frame {_out['root_pos_clip']} quat {_out['root_quat_clip']}")
    print("[SPAWN ] nonzero joints (deg): " + ", ".join(f"{n.replace('_joint', '')} {float(torch.rad2deg(v)):.0f}" for n, v in zip(obs_builder.lm_joint_names, _jp) if abs(float(v)) > 0.01))
    env.close(); app.close(); raise SystemExit(0)
if a.rsi:
    apply_rsi()
    obs_builder.reset()       # the history buffers were seeded from the benchmark spawn pose above
    print("[RSI   ] staged in the reference frame-0 state (root pose+vel, 29 joints). "
          "The [ALIGN ] line below should now read ~0 -- that is the check this flag worked.")

if a.lm_gains:
    apply_lm_gains()
    _names = list(base.scene["robot"].joint_names)
    _show = [n for n in ("left_hip_pitch_joint", "left_knee_joint", "waist_pitch_joint",
                         "left_ankle_pitch_joint", "left_shoulder_pitch_joint") if n in _names]
    print("[GAINS ] running LadderMan PD gains (armature unchanged):")
    for n in _show:
        i = _names.index(n)
        j = (obs_builder.j_fx == i).nonzero()
        if len(j):
            print(f"         {n:28s} kp {float(obs_builder.lm_kp[int(j[0])]):7.2f}  "
                  f"kd {float(obs_builder.lm_kd[int(j[0])]):6.3f}")

# The frame-0 tracking error, measured rather than assumed. The reference is anchored to the
# ladder (rung 0 in its local frame), so wherever the robot is staged, THIS is the offset the
# policy wakes up holding -- and it is the quantity `priv.ref_root_pos_future_b` feeds the net.
# Read without calling update(): update() pushes the current state into the history buffers, and
# doing that twice at t=0 would double the first sample in `joint_pos_history`.
_rob = base.scene["robot"].data
_ref0 = obs_builder.ref_body_pos[obs_builder.t][:, obs_builder.root_mo]        # [N,3] world
_d_b = quat_apply_inverse(_rob.root_quat_w, _ref0 - _rob.root_pos_w)[0]
_lad = base.scene["ladder"].data.root_pos_w[0]
print(f"[ALIGN ] robot root {[round(v, 3) for v in _rob.root_pos_w[0].tolist()]}, "
      f"standoff {float(((_rob.root_pos_w[0, :2] - _lad[:2]) ** 2).sum().sqrt()):.3f} m")
print(f"[ALIGN ] reference root at t=0 is {float(_d_b[0]):+.3f} m forward, "
      f"{float(_d_b[1]):+.3f} m left, {float(_d_b[2]):+.3f} m up of the robot "
      f"(forward > 0 means the robot is staged BEHIND where the clip starts)")

_all = torch.arange(len(base.scene["robot"].joint_names), device=base.device)
_fing = _all[~torch.isin(_all, obs_builder.j_fx)]
print(f"[HANDS ] {_fing.numel()} finger joints outside the clip; spawn angles (deg): "
      + ", ".join(f"{float(torch.rad2deg(v)):.0f}" for v in base.scene["robot"].data.default_joint_pos[0, _fing]))
_vw = None
if a.video:
    import imageio.v2 as _imageio  # noqa: E402
    _vc = base.scene["video_cam"]
    _ll = base.scene["ladder"].data.root_pos_w[0]
    _vc.set_world_poses_from_view(torch.tensor([a.cam_eye], dtype=torch.float32, device=base.device),
                                  torch.tensor([[float(_ll[0]), float(_ll[1]), float(_ll[2]) + a.cam_look_z]], device=base.device))
    _vw = _imageio.get_writer(a.video, fps=50, quality=7)
gate_hits = 0
best = {}
steps = 0
try:
    with torch.inference_mode():
        while steps < a.max_steps:
            obs_builder.update()                       # buffers see the CURRENT state
            groups = obs_builder.groups()
            a_lm = net.act(groups)
            if a.zscore and steps % a.zscore == 0:
                import ladderman_obs as _lo  # noqa: E402
                _bits = []
                for _g in _lo.VECNORM_GROUPS:
                    _z = net._n(_g, groups[_g])[0].abs()
                    for _nm, _s, _e in _lo.offsets(_g):
                        _m = float(_z[_s:_e].max())
                        if _m > 4.0:
                            _bits.append(f"{_g}.{_nm} {_m:.1f}")
                _res = (a_lm - groups["ref_joint_pos_"])[0]
                print(f"[ZSCORE] step {steps:4d}: residual |max| {float(_res.abs().max()):.2f}; terms over 4: " + (", ".join(_bits) or "none"))
            if a.residual_scale != 1.0:
                a_lm = groups["ref_joint_pos_"] + (a_lm - groups["ref_joint_pos_"]) * a.residual_scale
            action = net.to_fiatlux(a_lm, obs_builder, match_torque=a.match_torque, wrist_from_ref=a.step_ladder)
            if a.step_ladder and _fing.numel():
                # the clip's holds are authored for a flat, open hand: finger targets 0, not the spawn curl
                action[:, _fing] = (0.0 - base.scene["robot"].data.default_joint_pos[:, _fing]) / 0.5
            obs_builder.step(a_lm)                     # record action, advance the clock

            obs, reward, terminated, truncated, extras = env.step(action)
            if _vw is not None and not bool((terminated | truncated).any()):
                base.sim.render(); _vc.update(0.0)
                _vw.append_data(_vc.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype("uint8"))
            if recorder is not None:
                recorder.record_step(obs, action, reward, terminated, truncated)
            steps += 1

            # the gate conjuncts, read live so a run that never scores still says how close
            pel = float(base.scene["robot"].data.root_pos_w[0, 2])
            best["pelvis_max"] = max(best.get("pelvis_max", -9e9), pel)
            if steps % a.print_every == 0:
                print(f"[STEP  ] {steps:5d}  phase {float(obs_builder.t[0]) / obs_builder.motion_len:5.2f}"
                      f"  pelvis {pel:6.3f} m  action |max| {float(action.abs().max()):6.2f}"
                      f"  ladder moved {float((base.scene['ladder'].data.root_pos_w[0] - _lad0).norm()) * 1000:5.0f} mm"
                      f"  root off ref {float((base.scene['robot'].data.root_pos_w[0] - obs_builder.ref_body_pos[obs_builder.t][0, obs_builder.root_mo]).norm()) * 1000:5.0f} mm")

            if a.trace and steps % a.trace == 0:
                _r = base.scene["robot"].data; _ob = obs_builder
                from isaaclab.utils.math import yaw_quat as _yq  # noqa: E402
                _e = quat_apply_inverse(_yq(_r.root_quat_w), _r.root_pos_w - _ob.ref_body_pos[_ob.t][:, _ob.root_mo])[0] * 1000
                _g = _r.projected_gravity_b[0]
                _je = (_r.joint_pos[0, _ob.j_fx] - _ob.ref_joint_pos[_ob.t][0])
                _top = torch.argsort(_je.abs(), descending=True)[:5].tolist()
                _tau = (_r.applied_torque[0, _ob.j_fx].abs() / _r.joint_effort_limits[0, _ob.j_fx])
                _tt = torch.argsort(_tau, descending=True)[:3].tolist()
                print(f"[TRACE ] {steps:4d} root err fwd {float(_e[0]):+5.0f} left {float(_e[1]):+5.0f} up {float(_e[2]):+5.0f} mm; "
                      f"tilt fwd {float(torch.rad2deg(torch.atan2(_g[0], -_g[2]))):+5.1f} side {float(torch.rad2deg(torch.atan2(_g[1], -_g[2]))):+5.1f} deg; joints: "
                      + ", ".join(f"{_ob.lm_joint_names[i].replace('_joint','')} {float(torch.rad2deg(_je[i])):+.0f}" for i in _top)
                      + "; torque: " + ", ".join(f"{_ob.lm_joint_names[i].replace('_joint','')} {float(_tau[i])*100:.0f}%" for i in _tt))
                _wb = base.scene.sensors["whole_body_contact"]; _F = _wb.data.net_forces_w[0]
                _W = float(base.scene["robot"].root_physx_view.get_masses().sum()) * 9.81
                _grp = {"L foot": "left_ankle", "R foot": "right_ankle", "L hand": "left_hand|left_wrist", "R hand": "right_hand|right_wrist"}
                import re as _re  # noqa: E402
                _o = []
                for _k, _pat in _grp.items():
                    _ids = [j for j, n in enumerate(_wb.body_names) if _re.search(_pat, n)]
                    _f = _F[_ids].sum(0)
                    _o.append(f"{_k} x {float(_f[0]):+.0f} y {float(_f[1]):+.0f} z {float(_f[2]):+.0f}")
                _oth = [(n, float(_F[j].norm())) for j, n in enumerate(_wb.body_names) if not _re.search("ankle|hand|wrist", n) and float(_F[j].norm()) > 20]
                if os.environ.get("FX_TRACE_HAND_LINKS"):
                    _hl = [(n, _F[j]) for j, n in enumerate(_wb.body_names) if _re.search("hand|wrist", n) and float(_F[j].norm()) > 5]
                    if _hl:
                        print(f"[HANDLK] {steps:4d} " + "; ".join(f"{n.replace('_link', '')} ({float(v[0]):+.0f},{float(v[1]):+.0f},{float(v[2]):+.0f})" for n, v in _hl))
                _ld = base.scene["ladder"].data
                _dl = quat_apply_inverse(_yq(_ld.root_quat_w), _ld.root_pos_w - _lad0[None])[0] * 1000
                _lg = quat_apply_inverse(_ld.root_quat_w, torch.tensor([[0.0, 0.0, -1.0]], device=base.device))[0]
                print(f"[LADDER] {steps:4d} moved in its own frame x {float(_dl[0]):+5.1f} y(climb dir) {float(_dl[1]):+5.1f} z {float(_dl[2]):+5.1f} mm; "
                      f"tilt about x {float(torch.rad2deg(torch.atan2(_lg[1], -_lg[2]))):+.2f} about y {float(torch.rad2deg(torch.atan2(_lg[0], -_lg[2]))):+.2f} deg; "
                      f"speed {float(_ld.root_lin_vel_w[0].norm()) * 1000:.0f} mm/s")
                _lc = base.scene.sensors.get("ladder_contact") if hasattr(base.scene.sensors, "get") else None
                if _lc is not None:
                    _net = _lc.data.net_forces_w[0].sum(0)
                    _fm = _lc.data.force_matrix_w[0].sum(0) if _lc.data.force_matrix_w is not None else None      # [robot bodies, 3]
                    _rob = _fm.sum(0) if _fm is not None else torch.zeros(3, device=base.device)
                    _gnd = _net - _rob
                    print(f"[LADFRC] {steps:4d} on the ladder, world N: from the robot x {float(_rob[0]):+.0f} y {float(_rob[1]):+.0f} z {float(_rob[2]):+.0f}; "
                          f"from everything else (floor, fixture) x {float(_gnd[0]):+.0f} y {float(_gnd[1]):+.0f} z {float(_gnd[2]):+.0f}; ladder weight {float(_L_W):.0f}")
                print(f"[FORCE ] {steps:4d} (N, world; weight {_W:.0f}) " + "; ".join(_o) + ("; other: " + ", ".join(f"{n} {v:.0f}" for n, v in _oth) if _oth else ""))

            if _real_gate is not None and not best.get("gate_step"):
                if bool(_real_gate[0](base, **_real_gate[1])[0]):
                    best["gate_step"] = steps
                    print(f"[GATE  ] the task's success gate is true at step {steps} (pelvis {pel:.3f} m); continuing")
            done = terminated | truncated
            ep_steps = locals().get("ep_steps", 0) + 1
            ep_best = max(locals().get("ep_best", -9e9), pel)
            at_end = a.stop_at_clip_end and int(obs_builder.t[0]) >= obs_builder.motion_len - 1
            if at_end and not bool(done.any()):
                _ref_end = float(obs_builder.ref_body_pos[obs_builder.t][0, obs_builder.root_mo, 2])
                _fz = [float(base.scene["robot"].data.body_pos_w[0, list(base.scene["robot"].body_names).index(n), 2]) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
                _lz = float(base.scene["ladder"].data.root_pos_w[0, 2])
                print(f"[EPISODE] ankles {_fz[0] - _lz:.3f} / {_fz[1] - _lz:.3f} m above the ladder's base (platform top 1.152, an ankle standing on it about 1.19)")
                print(f"[EPISODE] {gate_hits + 1}: reached the clip's end at step {ep_steps}; pelvis {pel:.3f} m "
                      f"(reference {_ref_end:.3f}), highest {ep_best:.3f} -> {'CLIMBED' if pel > _ref_end - 0.10 else 'not on top'}")
                ep_steps, ep_best = 0, -9e9
                obs, _ = env.reset()
                obs_builder.t[:] = 0
                if a.rsi:
                    apply_rsi()
                obs_builder.reset()
                if a.lm_gains:
                    apply_lm_gains()
                gate_hits += 1
                if gate_hits >= a.episodes:
                    break
                continue
            if bool(done.any()):
                _tm = getattr(base, "termination_manager", None)
                _why = [t for t in (_tm.active_terms if _tm is not None else []) if bool(_tm.get_term(t)[0])]
                print(f"[EPISODE] {gate_hits + 1}: ended at step {ep_steps}, highest pelvis {ep_best:.3f} m -> "
                      + ("SUCCESS (the task's own gate)" if "success" in _why else f"FAILED {_why}"))
                ep_steps, ep_best = 0, -9e9
                ids = done.nonzero(as_tuple=False).squeeze(-1)
                why = []
                tm = getattr(base, "termination_manager", None)
                if tm is not None:
                    for term in tm.active_terms:
                        try:
                            if bool(tm.get_term(term)[0]):
                                why.append(term)
                        except Exception:
                            pass
                print(f"[RESET ] step {steps}: {len(ids)} env(s) done "
                      f"(terminated={bool(terminated.any())}, truncated={bool(truncated.any())})"
                      f" reason={why or 'unknown'}")
                if a.rsi:
                    obs_builder.t[ids] = 0    # apply_rsi reads t; reset() would zero it after
                    apply_rsi()               # the env reset restores the benchmark spawn pose
                obs_builder.reset(ids)
                if a.lm_gains:
                    apply_lm_gains()      # the env reset restores the cfg's gains
                gate_hits += 1
                if gate_hits >= a.episodes:
                    break
finally:
    print(f"[RESULT] {steps} steps; highest pelvis {best.get('pelvis_max', float('nan')):.3f} m")
    if _vw is not None:
        _vw.close(); print(f"[RESULT] wrote video {a.video}")
    if recorder is not None:
        info = recorder.write(a.out, fmt=a.format)
        print(f"[RESULT] wrote bag {info['bag']} ({info['episodes']} episodes)")

env.close()
app.close()
