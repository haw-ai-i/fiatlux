# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the bulb attach/detach state machine (issue #54) end-to-end, policy-free.

``verify_scene.py`` proves the Replace scene is solid and that the attach mechanic
*holds* the old bulb seated without exploding. It does NOT exercise the make/break
*transitions* -- the screw/unscrew gates (``GRASP_RADIUS``, ``SCREW_ANGLE``, ratcheting)
in ``mdp/attach.py``. This script does, with the robot root fixed and the wrist-roll
joint driven by a scripted ratcheted sawtooth (no trained policy):

  1. HOLD    -- the old bulb starts screwed in; confirm it is attached and held at the
                fixture seat, within grasp.
  2. UNSCREW -- ratchet the wrist roll in the unscrew direction past ``SCREW_ANGLE``;
                confirm ``old_attached`` flips False.
  3. RELEASE -- kick the detached bulb; confirm it moves freely (a kinematic-locked mock
                would ignore the impulse) -- i.e. it is genuinely removable now.
  4. ALIGN   -- present the fresh bulb at the seat pose (as a hand would).
  5. SCREW   -- ratchet the wrist roll in the screw direction past ``SCREW_ANGLE`` while
                the fresh bulb is aligned + gripped + the old bulb detached; confirm
                ``fresh_attached`` flips True.
  6. HOLD    -- shove the freshly-attached bulb; confirm the mechanic re-seats it.
  7. SUCCESS -- drop the old bulb in the crate; confirm ``attached_replacement_success``.

The seat is elevated (~2.2 m) and unreachable by a fixed-root default pose, so instead of
solving the manipulation we move the (kinematic) socket every step so its seat pose rides
at the right palm -- keeping palm, seat, and the held bulb inside ``GRASP_RADIUS``. That
isolates the FSM logic, which is what this check is about. The wrist joint is driven
kinematically (``write_joint_state_to_sim``) so the exact scripted roll signal reaches the
gate integrator, and the fresh bulb floats (demo-only ``disable_gravity``) so re-seating it
each step -- emulating a hand grip -- holds it inside the tight screw-in alignment window.

Examples
--------
    uv run python scripts/verify_attach.py --headless
    uv run python scripts/verify_attach.py --headless --video /tmp/attach/attach.mp4
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive the bulb attach/detach FSM end-to-end.")
parser.add_argument("--seed", type=int, default=0, help="Env seed (deterministic).")
parser.add_argument(
    "--video",
    type=str,
    default=None,
    help="Write an MP4 of the run to this path (implies camera rendering).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else follows."""

import importlib
import math

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import torch
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import rewards as task_rewards
from prettytable import PrettyTable

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_tasks.utils import parse_env_cfg

GRASP_RADIUS = 0.12  # m  (mdp.attach default)
SCREW_ANGLE = math.pi  # rad (mdp.attach default)

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}{' -- ' + detail if detail else ''}", flush=True)


def info(msg: str) -> None:
    print(f"  [INFO] {msg}", flush=True)


def build_replace_cfg(num_envs: int = 1):
    """Replace-task cfg stripped for a deterministic, single-env scripted run."""
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=num_envs)
    cfg.seed = args_cli.seed
    for ev in ("randomize_sky_intensity", "randomize_key_light", "randomize_material_tint"):
        if getattr(cfg.events, ev, None) is not None:
            setattr(cfg.events, ev, None)
    if getattr(cfg.events, "reset_robot_joints", None) is not None:
        cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    # No episode should end mid-demo: we deliberately trip task terminations.
    for term in ("success", "old_bulb_dropped", "fresh_bulb_dropped"):
        if getattr(cfg.terminations, term, None) is not None:
            setattr(cfg.terminations, term, None)
    # Drop camera-based observation terms so a plain --headless run needs neither
    # --enable_cameras nor a feature extractor. Replace mounts a head-mounted RGB
    # camera (``ego_camera`` / ``policy.ego_rgb``); null both, plus the older
    # torso-/wrist-named variants other presets may still carry.
    for cam in ("ego_camera", "torso_camera", "wrist_camera"):
        if getattr(cfg.scene, cam, None) is not None:
            setattr(cfg.scene, cam, None)
    for grp in ("policy", "privileged"):
        g = getattr(cfg.observations, grp, None)
        for term in ("ego_rgb", "torso_rgb", "wrist_rgb"):
            if g is not None and getattr(g, term, None) is not None:
                setattr(g, term, None)
    # Fixed root: no balance controller needed for a scripted-joint demo.
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    # Demo-only: float the fresh bulb and drop its collisions so that re-seating it each
    # step (emulating a firm hand grip) holds it inside the 15 mm screw-in tolerance.
    # Unlike the old bulb it is not FSM-slaved until it attaches, so gravity/contact would
    # otherwise eject it mid-step and stall the screw gate. The pose-based mdp.attach logic
    # under test depends on neither gravity nor collision, so this only removes a rig
    # artifact -- it does not weaken the check.
    if getattr(cfg.scene.bulb.spawn, "rigid_props", None) is not None:
        cfg.scene.bulb.spawn.rigid_props.disable_gravity = True
    cfg.scene.bulb.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    # Old bulb: drop collision too, so the post-detach velocity kick can move it freely
    # (otherwise it stays wedged in the socket cup against the hand colliders). This only
    # tests that it is now a free dynamic body -- a kinematic-locked mock ignores forces.
    cfg.scene.old_bulb.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    if args_cli.video:
        from fiatlux_task.viz import make_video_camera_cfg

        cfg.scene.video_cam = make_video_camera_cfg()
    return cfg


def make_env(cfg):
    spec = gym.spec("FIATLUX-Replace-v0")
    module_name, class_name = spec.entry_point.split(":")
    env_class = getattr(importlib.import_module(module_name), class_name)
    env = env_class(cfg=cfg)
    env.reset(seed=args_cli.seed)
    return env


VIDEO = None


def _build_rig(
    env,
    robot,
    socket,
    fresh_bulb,
    palm_id,
    wrist_id,
    n_act,
    all_ids,
    all_names,
    default_q,
    zeros6,
    seat,
    plug,
    seat_axis,
    roll0,
):
    """The scripted-demo rig: pose helpers, the step wrapper, and the ratchet driver.

    Lives at module scope purely so ``main`` stays under ruff's C901 limit -- these are
    one closure and depend on each other, so they move together. Returned in a fixed
    order that ``main`` unpacks into the same names it used before.
    """

    # -- helpers ------------------------------------------------------------------
    def act_from_roll(roll_target: float) -> torch.Tensor:
        """Action holding every joint at default except the wrist roll (scale=0.5)."""
        act = torch.zeros((env.num_envs, n_act), device=env.device)
        for k, (jid, name) in enumerate(zip(all_ids, all_names)):
            if name == "right_wrist_roll_joint":
                act[:, k] = 2.0 * (roll_target - default_q[:, jid])
        return act

    def palm_pos() -> torch.Tensor:
        return robot.data.body_link_pos_w[0, palm_id]

    def seat_pose():
        q = socket.data.root_quat_w[0]
        pos = (
            socket.data.root_pos_w[0]
            + quat_apply(q.unsqueeze(0), seat.unsqueeze(0))[0]
            - quat_apply(q.unsqueeze(0), plug.unsqueeze(0))[0]
        )
        return pos, q

    def track_socket_to_palm(clearance_z: float = 0.04) -> None:
        """Move the kinematic socket so its seat pose sits at the palm (+clearance),
        keeping the FSM-held bulb inside the grasp radius as the hand turns."""
        q = socket.data.root_quat_w[0]
        tgt = palm_pos() + torch.tensor([0.0, 0.0, clearance_z], device=env.device)
        spos = (
            tgt
            - quat_apply(q.unsqueeze(0), seat.unsqueeze(0))[0]
            + quat_apply(q.unsqueeze(0), plug.unsqueeze(0))[0]
        )
        socket.write_root_pose_to_sim(torch.cat([spos, q]).unsqueeze(0))
        socket.write_root_velocity_to_sim(zeros6)

    def drive_wrist(roll: float) -> None:
        """Kinematically pin the wrist-roll joint so the exact roll signal reaches the
        gate integrator (PD alone lags the target)."""
        pos = torch.full((1, 1), roll, device=env.device)
        robot.write_joint_state_to_sim(pos, torch.zeros((1, 1), device=env.device), joint_ids=[wrist_id])

    def hold_fresh_at_seat(roll: float) -> None:
        """Hold the fresh bulb at the seat, TURNED about the mating axis by ``roll``.

        Turning it is the point: that is what a bulb gripped in a turning hand does.
        Writing the socket's own quaternion instead would pin full-frame orientation
        error at zero and hide a gate that wrongly scores rotation about the mating
        axis as misalignment -- the bug this exercises (#54 review). ``roll=0`` gives
        an identity spin, so the seated pose is unchanged.
        """
        p, q = seat_pose()
        q = _spin_about(q, quat_apply(q.unsqueeze(0), seat_axis.unsqueeze(0))[0], roll)
        fresh_bulb.write_root_pose_to_sim(torch.cat([p, q]).unsqueeze(0))
        fresh_bulb.write_root_velocity_to_sim(zeros6)

    def aim_camera() -> None:
        """Frame the work point (seat) up close so the make/break is legible."""
        if VIDEO is None:
            return
        sp, _ = seat_pose()
        eye = (sp[0].item() + 0.48, sp[1].item() - 0.58, sp[2].item() + 0.20)
        VIDEO.set_pose(eye, tuple(sp.tolist()))

    def animate_bulb(bulb, start, end, quat, n: int) -> None:
        """Slide a (detached / not-yet-attached) bulb from start->end over n steps,
        writing its pose each frame so the motion is visible on camera."""
        for i in range(n):
            t = (i + 1) / n
            p = start * (1.0 - t) + end * t
            bulb.write_root_pose_to_sim(torch.cat([p, quat]).unsqueeze(0))
            bulb.write_root_velocity_to_sim(zeros6)
            step(roll0, track=False)

    def step(roll: float, track: bool = True, hold_fresh: bool = False, drive: bool = False) -> None:
        if track:
            track_socket_to_palm()
        if hold_fresh and not bool(task_attach.fresh_bulb_attached(env)[0].item()):
            # Turn the bulb WITH the wrist, as a gripped bulb does. This is what makes
            # the screw phase a real test of the gate's alignment check rather than a
            # tautology (#54 review): a full-frame check would read this as misaligned
            # within a fraction of a turn and the gate could never fire.
            hold_fresh_at_seat(roll=roll - roll0)
        if drive:  # only pin the joint while actively ratcheting (constant-hold pins
            drive_wrist(roll)  # would feed the gate integrator phantom drift deltas)
        env.step(act_from_roll(roll))
        if VIDEO is not None:
            VIDEO.capture()

    def ratchet(direction: float, n_cycles: int, amp: float, half_steps: int, hold_fresh: bool = False):
        """Sawtooth the wrist roll. direction -1 = unscrew (decreasing roll counts),
        +1 = screw (increasing). Working stroke ratchets; return stroke is free."""
        for _ in range(n_cycles):
            for going_out in (True, False):
                for s in range(half_steps):
                    f = (s + 1) / half_steps
                    frac = f if going_out else (1.0 - f)
                    step(roll0 + direction * amp * frac, track=True, hold_fresh=hold_fresh, drive=True)

    return (
        act_from_roll,
        palm_pos,
        seat_pose,
        track_socket_to_palm,
        drive_wrist,
        hold_fresh_at_seat,
        aim_camera,
        animate_bulb,
        step,
        ratchet,
    )


def _spin_about(q: torch.Tensor, axis_w: torch.Tensor, angle: float) -> torch.Tensor:
    """``q`` rotated by ``angle`` rad about the world-frame ``axis_w``. Identity at 0."""
    axis = axis_w / axis_w.norm().clamp(min=1e-9)
    half = torch.tensor(angle / 2.0, device=q.device)
    spin = torch.cat([torch.cos(half).unsqueeze(0), torch.sin(half) * axis])
    return quat_mul(spin.unsqueeze(0), q.unsqueeze(0))[0]


def main() -> int:
    global VIDEO
    cfg = build_replace_cfg()
    env = make_env(cfg)
    try:
        robot = env.scene["robot"]
        socket = env.scene["socket"]
        old_bulb = env.scene["old_bulb"]
        fresh_bulb = env.scene["bulb"]

        mgr = getattr(env, task_attach._ENV_ATTR, None)
        if mgr is None:
            record("attach:manager_present", False, "no bulb_attachment term wired on FIATLUX-Replace-v0")
            return _summary()
        record("attach:manager_present", True, "mdp.bulb_attachment is wired every step")

        wrist_id = robot.find_joints("right_wrist_roll_joint")[0][0]
        palm_id = robot.find_bodies("right_hand_base_link")[0][0]
        all_ids, all_names = robot.find_joints(".*")  # ascending id == action-vector order
        n_act = env.action_manager.total_action_dim
        default_q = robot.data.default_joint_pos
        zeros6 = torch.zeros((env.num_envs, 6), device=env.device)
        seat = torch.tensor(SOCKET_SEAT_OFFSET, device=env.device)
        seat_axis = torch.tensor(SOCKET_SEAT_AXIS, device=env.device)
        plug = torch.tensor(BULB_PLUG_OFFSET, device=env.device)

        if args_cli.video:
            from fiatlux_task.viz import VideoRecorder

            VIDEO = VideoRecorder(env, env.scene["video_cam"], args_cli.video, fps=20)

        roll0 = robot.data.joint_pos[0, wrist_id].item()
        # -- helpers (module-level factory; see _build_rig) ---------------------------
        (
            act_from_roll,
            palm_pos,
            seat_pose,
            track_socket_to_palm,
            drive_wrist,
            hold_fresh_at_seat,
            aim_camera,
            animate_bulb,
            step,
            ratchet,
        ) = _build_rig(
            env,
            robot,
            socket,
            fresh_bulb,
            palm_id,
            wrist_id,
            n_act,
            all_ids,
            all_names,
            default_q,
            zeros6,
            seat,
            plug,
            seat_axis,
            roll0,
        )

        # -- settle: bulb rests at its real elevated fixture (NOT tracked to the palm) --
        # so the ratcheted unscrew gate does not farm settling jitter before we mean to
        # unscrew (the gate counts any grip-direction wiggle -- see the module docstring).
        for _ in range(40):
            step(roll0, track=False)

        if VIDEO is not None:
            for _ in range(10):
                step(roll0, track=False)

        # -- phase 1: HOLD -- old bulb attached and held at the fixture seat -----------
        old_att0 = bool(task_attach.old_bulb_attached(env)[0].item())
        sp0, _ = seat_pose()
        seat_err0 = torch.norm(old_bulb.data.root_pos_w[0] - sp0).item()
        record("attach:old_starts_attached", old_att0, f"old_bulb_attached={old_att0}")
        record(
            "attach:old_held_at_fixture",
            old_att0 and seat_err0 < 0.03,
            f"old bulb {seat_err0 * 1000:.1f} mm from its fixture seat while held",
        )

        # -- engage: bring the fixture seat to the palm (the hand grips the bulb) ------
        for _ in range(12):
            step(roll0, track=True)
        aim_camera()
        d_old = torch.norm(palm_pos() - old_bulb.data.root_pos_w[0]).item()
        record(
            "attach:old_within_grasp",
            d_old < GRASP_RADIUS,
            f"palm->old_bulb {d_old * 100:.1f} cm (grasp radius {GRASP_RADIUS * 100:.0f} cm)",
        )

        # -- phase 2: UNSCREW ---------------------------------------------------------
        # integrate the deliberate unscrew gesture from zero (isolate it from any grip
        # jitter counted while engaging -- the ratcheted gate is jitter-sensitive by design)
        mgr._unscrew_accum[:] = 0.0
        ratchet(direction=-1.0, n_cycles=5, amp=0.8, half_steps=15)
        unscrew_accum = mgr._unscrew_accum[0].item()
        old_att1 = bool(task_attach.old_bulb_attached(env)[0].item())
        record(
            "attach:unscrew_detaches",
            not old_att1,
            f"unscrew_accum {unscrew_accum:.2f} rad (gate {SCREW_ANGLE:.2f}) -> old_attached={old_att1}",
        )

        # -- phase 3: REMOVE ----------------------------------------------------------
        pos_at_detach = old_bulb.data.root_pos_w[0].clone()
        q_old = old_bulb.data.root_quat_w[0]
        # (a) ASSERTION -- physics-based freedom: a pure velocity impulse (no pose writes)
        # must move the freed bulb. A kinematic-locked body (the old mock) ignores this;
        # only a genuinely dynamic, detached body responds.
        old_bulb.write_root_velocity_to_sim(
            torch.tensor([[0.20, 0.0, -0.30, 0.0, 0.0, 0.0]], device=env.device)
        )
        for _ in range(12):
            step(roll0, track=False)  # pure physics response -- do NOT script the pose here
        moved = torch.norm(old_bulb.data.root_pos_w[0] - pos_at_detach).item()
        record(
            "attach:detached_bulb_moves_freely",
            (not old_att1) and moved > 0.02,
            f"under a velocity impulse the freed bulb moved {moved * 100:.1f} cm (physics-driven)",
        )
        # (b) PRESENTATION -- reset and lift it visibly out of the socket for the video.
        old_bulb.write_root_pose_to_sim(torch.cat([pos_at_detach, q_old]).unsqueeze(0))
        old_bulb.write_root_velocity_to_sim(zeros6)
        out = pos_at_detach + torch.tensor([0.0, 0.0, 0.18], device=env.device)  # straight out of the seat
        aside = out + torch.tensor([0.28, -0.10, 0.0], device=env.device)  # set aside, staying in frame
        animate_bulb(old_bulb, pos_at_detach, out, q_old, 18)
        animate_bulb(old_bulb, out, aside, q_old, 24)

        # -- phase 4: INSERT -- bring a fresh bulb into the seat (visible approach) -----
        for _ in range(4):
            step(roll0, track=True)  # socket held at the palm
        aim_camera()
        sp, sq = seat_pose()
        approach = sp + torch.tensor([0.0, -0.05, 0.22], device=env.device)  # come in from front/above
        animate_bulb(fresh_bulb, approach, sp, sq, 24)
        pos_err = task_rewards._bulb_socket_pos_error(env)[0].item()
        record(
            "attach:fresh_aligned",
            pos_err < 0.015,
            f"fresh bulb plug-vs-seat error {pos_err * 1000:.1f} mm (tol 15 mm)",
        )

        # -- phase 5: SCREW -----------------------------------------------------------
        ratchet(direction=1.0, n_cycles=5, amp=0.8, half_steps=15, hold_fresh=True)
        screw_accum = mgr._screw_accum[0].item()
        fresh_att = bool(task_attach.fresh_bulb_attached(env)[0].item())
        record(
            "attach:screw_attaches_fresh",
            fresh_att,
            f"screw_accum {screw_accum:.2f} rad (gate {SCREW_ANGLE:.2f}) -> fresh_attached={fresh_att}",
        )

        # -- phase 6: HOLD -- shove the attached fresh bulb; it must be re-seated -------
        p, _ = seat_pose()
        shove = p + torch.tensor([0.12, 0.10, 0.05], device=env.device)
        fresh_bulb.write_root_pose_to_sim(torch.cat([shove, socket.data.root_quat_w[0]]).unsqueeze(0))
        fresh_bulb.write_root_velocity_to_sim(zeros6)
        for _ in range(20):
            step(roll0, track=True)  # attached -> the FSM re-seats it, not hold_fresh
        pos_err_after = task_rewards._bulb_socket_pos_error(env)[0].item()
        still_att = bool(task_attach.fresh_bulb_attached(env)[0].item())
        record(
            "attach:fresh_held_after_shove",
            still_att and pos_err_after < 0.015,
            f"after a 16 cm shove: re-seated to {pos_err_after * 1000:.1f} mm, attached={still_att}",
        )

        # -- phase 7: SUCCESS -- old bulb in the crate + fresh screwed in --------------
        crate = env.scene["bin"]
        cpos = crate.data.root_pos_w[0].clone()
        cpos[2] += 0.05
        old_bulb.write_root_pose_to_sim(torch.cat([cpos, old_bulb.data.root_quat_w[0]]).unsqueeze(0))
        old_bulb.write_root_velocity_to_sim(zeros6)
        for _ in range(10):
            step(roll0, track=False)
        success = bool(task_attach.attached_replacement_success(env)[0].item())
        record("attach:replacement_success", success, f"attached_replacement_success={success}")

        # -- sanity: no NaN / explosion anywhere --------------------------------------
        nan = bool(
            torch.isnan(robot.data.root_pos_w).any()
            or torch.isnan(old_bulb.data.root_pos_w).any()
            or torch.isnan(fresh_bulb.data.root_pos_w).any()
        )
        record("attach:no_nan", not nan, "states finite throughout" if not nan else "NaN in states")

        if VIDEO is not None:
            info(f"wrote video {VIDEO.write()} ({len(VIDEO)} frames)")
    finally:
        env.close()
    return _summary()


def _summary() -> int:
    table = PrettyTable()
    table.field_names = ["#", "check", "result", "detail"]
    table.align["check"] = table.align["detail"] = "l"
    passed = 0
    for i, (name, ok, detail) in enumerate(RESULTS, 1):
        table.add_row([i, name, "PASS" if ok else "FAIL", detail])
        passed += int(ok)
    print("\n" + table.get_string(), flush=True)
    fails = [n for n, ok, _ in RESULTS if not ok]
    verdict = "PASS" if not fails else "FAIL"
    print(f"\n[verify] {passed}/{len(RESULTS)} checks passed -- OVERALL: {verdict}", flush=True)
    if fails:
        print("[verify] Failing checks:\n   - " + "\n   - ".join(fails), flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    import os
    import sys

    code = 1
    try:
        code = main()
    except Exception:  # noqa: BLE001 -- print the traceback before the process exits
        import traceback

        traceback.print_exc()
    finally:
        # Do NOT call simulation_app.close() here: it ends in a native framework shutdown
        # that terminates the process with exit code 0, so nothing after it (sys.exit or
        # os._exit included) ever runs, and a FAIL would report success. main() already
        # closed the env; os._exit skips Kit's graceful shutdown on purpose -- process
        # teardown releases the GPU, and CI must see a non-zero code on FAIL. Same
        # reasoning and same shape as verify_scene.py.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
