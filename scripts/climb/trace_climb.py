"""Per-step trace of the first seconds of a fiatlux S02 episode.

A variant of record_climb.py (which stays runnable, unchanged). The probe says every
episode ends in `fell_over` -- mdp.bad_orientation at 1.0 rad -- within ~40 steps, with
zero pelvis gain. This logs enough per step to say WHY: whether the policy is commanding
something violent from step 0 (observation mismatch) or whether the state diverges under
sane commands (dynamics/actuator mismatch).
"""
import numpy as _np_trace  # noqa
_ORIG_DOC = """Drive a LadderMan teacher through a fiatlux climb subtask and record the bag.

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

import os
import argparse
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
ap.add_argument("--trace_out", default=None, help="npz to write the per-step trace to")
ap.add_argument("--trace_steps", type=int, default=120)
ap.add_argument("--null_action", action="store_true",
                help="ignore the policy and send a zero action every step (hold the default\n                     pose). Isolates what the SPAWN does from what the POLICY does.")
ap.add_argument("--action_delay", type=int, default=0,
                help="Action delay in PHYSICS steps, reproducing LadderMan's action term "
                     "(min_delay 2, max_delay 6, randomised per episode). We apply actions "
                     "instantly, i.e. delay 0 -- OUTSIDE the range the policy trained on, which "
                     "is the only such condition left after the structural audit. Their term "
                     "picks a buffer slot per physics substep; fiatlux holds one target for the "
                     "whole control step, so this applies the exact time-average over the 4 "
                     "substeps instead.")
ap.add_argument("--mass_scale", type=float, default=None,
                help="Scale EVERY body mass (and its inertia) by this factor. Completes the "
                     "--lm_masses probe: name matching only reaches the hands, so this is the "
                     "way to put the robot on the teacher's 33.34 kg total without needing a "
                     "body-by-body correspondence. Also a diagnostic -- it changes the "
                     "benchmark's robot, so nothing recorded under it is a demo.")
ap.add_argument("--lm_masses", default=None,
                help="JSON of LadderMan's URDF link masses. Overwrites this robot's body masses "
                     "(and scales inertia with them) so the ONLY thing left differing from the "
                     "teacher's training robot is geometry. A DIAGNOSTIC -- it changes the "
                     "benchmark's robot, so nothing recorded under it is a demo.")
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

env = gym.make(a.task, cfg=cfg)
base = env.unwrapped

obs_builder = LadderManObs(base, a.motion)
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
    q_next = ob.ref_joint_pos[torch.clamp(t0 + 1, max=ob.motion_len - 1)]
    qd = (q_next - q) * 50.0
    rob.write_joint_state_to_sim(q, qd, joint_ids=ob.j_fx)


if a.rsi:
    apply_rsi()
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

if os.environ.get("FX_DEBUG_PRIV"):
    _ob = obs_builder
    print("[DBG] motion_len:", int(_ob.motion_len), " t0:", int(_ob.t[0]))
    print("[DBG] root_mo:", int(_ob.root_mo))
    _tb = list(_ob.track_bodies)
    print(f"[DBG] track_bodies ({len(_tb)}):")
    for _i, _n in enumerate(_tb):
        print(f"[DBG]    b{_i:<2d} {_n}")
    _pel = _tb.index("pelvis") if "pelvis" in _tb else -1
    print("[DBG] pelvis is b%d" % _pel)
    _fi = _ob._future_idx()
    print("[DBG] future idx from t=%d: %s" % (int(_ob.t[0]), _fi[0].tolist()))
    _rp = _ob.ref_body_pos[_fi][:, :, _ob.track_b_mo]          # [N,F,B,3]
    _r0 = _ob.ref_body_pos[_ob.t][:, _ob.root_mo]              # [N,3]
    for _j, _k in enumerate(_fi[0].tolist()):
        _d = (_rp[0, _j, _pel] - _r0[0]).tolist()
        print(f"[DBG] ref pelvis world displacement at idx {_k}: "
              f"dx {_d[0]:+.4f} dy {_d[1]:+.4f} dz {_d[2]:+.4f}")

if a.mass_scale:
    _rb = base.scene["robot"]
    _m = _rb.root_physx_view.get_masses().clone()
    _t0 = float(_m[0].sum())
    _m *= a.mass_scale
    _I = _rb.root_physx_view.get_inertias().clone() * a.mass_scale
    _idx = torch.arange(_m.shape[0], dtype=torch.int32)
    _rb.root_physx_view.set_masses(_m, _idx)
    _rb.root_physx_view.set_inertias(_I, _idx)
    print(f"[MASS  ] every body scaled by {a.mass_scale:.4f}: "
          f"total {_t0:.2f} -> {float(_m[0].sum()):.2f} kg")

if a.lm_masses:
    import json as _json
    _lm = _json.load(open(a.lm_masses))
    _rb = base.scene["robot"]
    _bn = list(_rb.body_names)
    _m = _rb.root_physx_view.get_masses().clone()
    _I = _rb.root_physx_view.get_inertias().clone()
    _m0 = _m.clone()
    _hand_tok = ("hand", "thumb", "index", "middle", "ring", "little", "pinky", "palm")
    _hand = [i for i, n in enumerate(_bn) if any(t in n.lower() for t in _hand_tok)]
    _hit = 0
    for i, n in enumerate(_bn):
        if n in _lm:
            _m[:, i] = float(_lm[n]); _hit += 1
    # LadderMan's whole hand is ONE 0.170 kg rubber link. Spread that over whatever our
    # multi-link hand has, in proportion, so each side ends up at the same 0.170 kg.
    for side in ("left", "right"):
        idx = [i for i in _hand if _bn[i].lower().startswith(side)]
        if not idx:
            continue
        tot = float(_m0[0, idx].sum())
        if tot <= 0:
            continue
        for i in idx:
            _m[:, i] = _m0[:, i] * (float(_lm.get(side + "_rubber_hand", 0.170)) / tot)
    _ratio = (_m / _m0.clamp_min(1e-9)).unsqueeze(-1)
    _I = _I * _ratio
    _idx = torch.arange(_m.shape[0], dtype=torch.int32)
    _rb.root_physx_view.set_masses(_m, _idx)
    _rb.root_physx_view.set_inertias(_I, _idx)
    print(f"[MASS  ] {_hit} of {len(_bn)} bodies matched by name; hands lumped to "
          f"{float(_lm.get('left_rubber_hand', 0.170)):.3f} kg a side. "
          f"total {float(_m0[0].sum()):.2f} -> {float(_m[0].sum()):.2f} kg "
          f"(LadderMan URDF {sum(_lm.values()):.2f} kg)")

if a.trace_out:
    import json as _json
    _o = obs_builder
    _fx23 = _o.j_fx[_o.act_mo]
    _lim = base.scene["robot"].data
    _json.dump({
        "act_names": list(_o.act_names),
        "act_scaling": _o.act_scaling.tolist(),
        "lm_default_act": _o.lm_default[_o.act_mo].tolist(),
        "fx_joint_names_23": [base.scene["robot"].joint_names[int(i)] for i in _fx23],
        "fx_effort_limit_23": _lim.joint_effort_limits[0][_fx23].tolist()
            if hasattr(_lim, "joint_effort_limits") else [],
        "fx_stiffness_23": _lim.joint_stiffness[0][_fx23].tolist(),
        "fx_damping_23": _lim.joint_damping[0][_fx23].tolist(),
        "lm_kp_act": _o.lm_kp[_o.act_mo].tolist(),
        "lm_kd_act": _o.lm_kd[_o.act_mo].tolist(),
        "track_bodies": list(_o.track_bodies),
        # every body's mass, so the "our robot is 38.61 kg where the teacher trained on 33.34 kg"
        # claim can be checked per link instead of as one total.
        "fx_body_masses": {n: float(m) for n, m in zip(
            base.scene["robot"].body_names,
            base.scene["robot"].root_physx_view.get_masses()[0].tolist())},
    }, open(str(a.trace_out) + ".meta.json", "w"), indent=1)
    print("[TRACE ] meta -> " + str(a.trace_out) + ".meta.json")

DECIMATION = 4
_delay_w = {}
if a.action_delay > 0:
    for _s in range(DECIMATION):
        _k = (a.action_delay - _s + DECIMATION - 1) // DECIMATION
        _delay_w[_k] = _delay_w.get(_k, 0.0) + 1.0 / DECIMATION
    print(f"[DELAY ] {a.action_delay} physics steps -> per-control-step weights "
          + ", ".join(f"t-{k}: {v:.2f}" for k, v in sorted(_delay_w.items())))
_abuf = None

gate_hits = 0
best = {}
steps = 0
TR = {k: [] for k in (
    'root_pos', 'root_quat', 'root_lin', 'root_ang', 'tilt_rad', 'joint_pos', 'joint_vel',
    'a_lm', 'ref_q', 'resid', 'action', 'torque', 'foot_force', 'ref_root_pos')}
try:
    with torch.inference_mode():
        while steps < a.max_steps:
            obs_builder.update()                       # buffers see the CURRENT state
            groups = obs_builder.groups()
            a_lm = net.act(groups)
            if a.residual_scale != 1.0:
                a_lm = groups["ref_joint_pos_"] + (a_lm - groups["ref_joint_pos_"]) * a.residual_scale
            if _delay_w:
                if _abuf is None:
                    # LadderMan zeroes action_buf on reset, so the delayed slots really do hold
                    # zero on the first steps -- a zero action commands their DEFAULT pose, not
                    # the reference. Reproduce that rather than softening it: the point of this
                    # flag is fidelity to what the policy trained under.
                    _abuf = [torch.zeros_like(a_lm) for _ in range(3)]
                _abuf.insert(0, a_lm.clone())
                _abuf.pop()
                a_applied = sum(w * _abuf[k] for k, w in _delay_w.items())
            else:
                a_applied = a_lm
            action = net.to_fiatlux(a_applied, obs_builder, match_torque=a.match_torque)
            if a.null_action:
                action = torch.zeros_like(action)
            obs_builder.step(a_lm)                     # record action, advance the clock

            obs, reward, terminated, truncated, extras = env.step(action)

            if a.trace_out and steps < a.trace_steps:
                rd = base.scene["robot"].data
                q = rd.root_quat_w[0]
                # angle between the base's own +z and world +z == mdp.bad_orientation's measure
                upz = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
                TR['tilt_rad'].append(float(torch.acos(upz.clamp(-1.0, 1.0))))
                TR['root_pos'].append(rd.root_pos_w[0].tolist())
                TR['root_quat'].append(q.tolist())
                TR['root_lin'].append(rd.root_lin_vel_w[0].tolist())
                TR['root_ang'].append(rd.root_ang_vel_w[0].tolist())
                TR['joint_pos'].append(rd.joint_pos[0].tolist())
                TR['joint_vel'].append(rd.joint_vel[0].tolist())
                # the 23 ACTUATED joints: what the reference asks, what the robot does,
                # and what the actuator manages -- the three numbers that separate "the arm was
                # never told to reach" from "it was told and could not".
                _o2 = obs_builder
                _fx23 = _o2.j_fx[_o2.act_mo]
                TR.setdefault('ref_rad', []).append(
                    _o2.ref_joint_pos[_o2.t][0, _o2.act_mo].tolist())
                TR.setdefault('q_track', []).append(rd.joint_pos[0][_fx23].tolist())
                TR.setdefault('qd_track', []).append(rd.joint_vel[0][_fx23].tolist())
                try:
                    TR.setdefault('tau_track', []).append(rd.applied_torque[0][_fx23].tolist())
                except Exception:
                    TR.setdefault('tau_track', []).append([])
                TR['a_lm'].append(a_lm[0].detach().cpu().tolist())
                _rq = groups.get("ref_joint_pos_")
                TR['ref_q'].append(_rq[0].detach().cpu().tolist() if _rq is not None else [])
                TR['resid'].append((a_lm[0] - _rq[0]).detach().cpu().tolist()
                                   if _rq is not None else [])
                TR['action'].append(action[0].detach().cpu().tolist())
                try:
                    TR['torque'].append(rd.applied_torque[0].tolist())
                except Exception:
                    TR['torque'].append([])
                # This env has no sensor called "contact_forces" -- it has hand_contact,
                # left_hand_contact and ladder_contact. Asking for the wrong name returned an
                # empty array for the whole of the first trace, which reads as "no contact"
                # when it actually means "no sensor". Capture whatever is actually there.
                for _sn, _sv in base.scene.sensors.items():
                    try:
                        TR.setdefault('c_' + _sn, []).append(
                            _sv.data.net_forces_w[0].norm(dim=-1).tolist())
                    except Exception:
                        pass
                TR['foot_force'].append([])
                for _gn, _gv in groups.items():
                    try:
                        TR.setdefault('g_' + _gn, []).append(_gv[0].detach().cpu().tolist())
                    except Exception:
                        pass
                try:
                    ob = obs_builder
                    TR['ref_root_pos'].append(
                        ob.ref_body_pos[ob.t][0, ob.root_mo].tolist())
                except Exception:
                    TR['ref_root_pos'].append([])

            if recorder is not None:
                recorder.record_step(obs, action, reward, terminated, truncated)
            steps += 1

            # the gate conjuncts, read live so a run that never scores still says how close
            pel = float(base.scene["robot"].data.root_pos_w[0, 2])
            best["pelvis_max"] = max(best.get("pelvis_max", -9e9), pel)
            if steps % a.print_every == 0:
                print(f"[STEP  ] {steps:5d}  phase {float(obs_builder.t[0]) / obs_builder.motion_len:5.2f}"
                      f"  pelvis {pel:6.3f} m  action |max| {float(action.abs().max()):6.2f}")

            done = terminated | truncated
            if bool(done.any()):
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
    if a.trace_out and TR['root_pos']:
        import numpy as _np
        _np.savez(a.trace_out, **{k: _np.asarray(v, dtype=object if (
            v and isinstance(v[0], list) and len({len(x) for x in v}) > 1) else _np.float32)
            for k, v in TR.items() if v})
        print(f"[TRACE ] {len(TR['root_pos'])} steps -> {a.trace_out}")
    if recorder is not None:
        info = recorder.write(a.out, fmt=a.format)
        print(f"[RESULT] wrote bag {info['bag']} ({info['episodes']} episodes)")

env.close()
app.close()
