# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the bayonet bulb/socket state machine (issue #54) end-to-end.

The scripted run manipulates bulb poses while robot actions remain zero. It verifies the
reviewed architecture directly:

1. The old bulb starts fully inserted and rotationally locked.
2. Axial motion is rejected until the bulb rotates to the release angle; reversing
   mid-unlock re-locks it.
3. Rotation is rejected while the released bulb travels along the insertion axis.
4. The bulb becomes free only after it passes the configured insertion depth.
5. A fresh bulb never engages an occupied socket or a misaligned entry; it enters
   axially, may start rotation only at full depth, and becomes attached only after the
   bulb itself reaches the configured rotation angle.
6. The completed replacement still satisfies the task success predicate.
7. `(low, high)` parameter ranges sample per env and re-sample on reset.
8. The lock state (`_phase`, `_theta`) reaches the privileged observation group, at a
   known offset, carrying the manager's own state (issue #77).

The socket is moved to the fixed-root robot's palm for a compact, visible test rig. Both
bulbs have gravity and collisions disabled only in this harness, isolating the tensorized
state machine from contact artifacts while retaining a physics-driven free-body check.

Examples
--------
    uv run python scripts/verify_attach.py --headless
    uv run python scripts/verify_attach.py --headless --video /tmp/attach/attach.mp4
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive the bayonet bulb/socket FSM end-to-end.")
parser.add_argument("--seed", type=int, default=0, help="Env seed (deterministic).")
parser.add_argument(
    "--video",
    type=str,
    default=None,
    help="Write an MP4 of the run to this path (implies camera rendering).",
)
parser.add_argument(
    "--check-ranges",
    action="store_true",
    help="Run only the (low, high) parameter-range sampling check (spec item 7). A second "
    "env build in one Isaac Sim process hangs, so this check needs its own invocation.",
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

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import torch
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach
from fiatlux_task.tasks.manager_based.fiatlux_task.replace_env_cfg import (
    BAYONET_INSERTION_DEPTH,
    BAYONET_ROTATION_ANGLE,
)
from prettytable import PrettyTable

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_tasks.utils import parse_env_cfg

RESULTS: list[tuple[str, bool, str]] = []
VIDEO = None


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}{' -- ' + detail if detail else ''}", flush=True)


def info(message: str) -> None:
    print(f"  [INFO] {message}", flush=True)


def build_replace_cfg(num_envs: int = 1):
    """Return a deterministic Replace config stripped to the bayonet test rig."""
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=num_envs)
    cfg.seed = args_cli.seed
    for event in ("randomize_sky_intensity", "randomize_key_light", "randomize_material_tint"):
        if getattr(cfg.events, event, None) is not None:
            setattr(cfg.events, event, None)
    if getattr(cfg.events, "reset_robot_joints", None) is not None:
        cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    for term in ("success", "old_bulb_dropped", "fresh_bulb_dropped"):
        if getattr(cfg.terminations, term, None) is not None:
            setattr(cfg.terminations, term, None)
    for camera in ("ego_camera", "torso_camera", "wrist_camera"):
        if getattr(cfg.scene, camera, None) is not None:
            setattr(cfg.scene, camera, None)
    for group_name in ("policy", "privileged"):
        group = getattr(cfg.observations, group_name, None)
        for term in ("ego_rgb", "torso_rgb", "wrist_rgb"):
            if group is not None and getattr(group, term, None) is not None:
                setattr(group, term, None)
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    for bulb_cfg in (cfg.scene.bulb, cfg.scene.old_bulb):
        if getattr(bulb_cfg.spawn, "rigid_props", None) is not None:
            bulb_cfg.spawn.rigid_props.disable_gravity = True
        bulb_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
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


def _spin_about(quat: torch.Tensor, world_axis: torch.Tensor, angle: float) -> torch.Tensor:
    half = 0.5 * angle
    spin = torch.cat(
        [
            torch.cos(torch.tensor([half], device=quat.device)),
            world_axis * torch.sin(torch.tensor(half, device=quat.device)),
        ]
    )
    return quat_mul(spin.unsqueeze(0), quat.unsqueeze(0))[0]


def _build_rig(env, robot, socket, old_bulb, fresh_bulb, palm_id, zero_action, zeros6):
    seat_offset = torch.tensor(SOCKET_SEAT_OFFSET, device=env.device)
    plug_offset = torch.tensor(BULB_PLUG_OFFSET, device=env.device)
    seat_axis = torch.tensor(SOCKET_SEAT_AXIS, device=env.device)

    def palm_pos() -> torch.Tensor:
        return robot.data.body_link_pos_w[0, palm_id]

    def seat_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        socket_quat = socket.data.root_quat_w[0]
        seat_point = socket.data.root_pos_w[0] + quat_apply(socket_quat.unsqueeze(0), seat_offset.unsqueeze(0))[0]
        world_axis = quat_apply(socket_quat.unsqueeze(0), seat_axis.unsqueeze(0))[0]
        return seat_point, socket_quat, world_axis

    def track_socket_to_palm(clearance: float = 0.04) -> None:
        socket_quat = socket.data.root_quat_w[0]
        world_axis = quat_apply(socket_quat.unsqueeze(0), seat_axis.unsqueeze(0))[0]
        target_seat = palm_pos() + clearance * world_axis
        socket_pos = target_seat - quat_apply(socket_quat.unsqueeze(0), seat_offset.unsqueeze(0))[0]
        socket.write_root_pose_to_sim(torch.cat([socket_pos, socket_quat]).unsqueeze(0))
        socket.write_root_velocity_to_sim(zeros6)

    def lateral_axis(world_axis: torch.Tensor) -> torch.Tensor:
        reference = torch.tensor([1.0, 0.0, 0.0], device=env.device)
        if abs(torch.dot(world_axis, reference).item()) > 0.9:
            reference = torch.tensor([0.0, 1.0, 0.0], device=env.device)
        lateral = torch.linalg.cross(world_axis, reference)
        return lateral / torch.norm(lateral)

    def bulb_pose(axial_distance: float, rotation: float, lateral_distance: float = 0.0) -> torch.Tensor:
        seat_point, socket_quat, world_axis = seat_geometry()
        bulb_quat = _spin_about(socket_quat, world_axis, rotation)
        plug_point = seat_point + axial_distance * world_axis + lateral_distance * lateral_axis(world_axis)
        bulb_pos = plug_point - quat_apply(bulb_quat.unsqueeze(0), plug_offset.unsqueeze(0))[0]
        return torch.cat([bulb_pos, bulb_quat])

    def place_bulb(bulb, axial_distance: float, rotation: float, lateral_distance: float = 0.0) -> None:
        bulb.write_root_pose_to_sim(bulb_pose(axial_distance, rotation, lateral_distance).unsqueeze(0))
        bulb.write_root_velocity_to_sim(zeros6)

    def axial_distance(bulb) -> float:
        seat_point, _, world_axis = seat_geometry()
        plug_point = (
            bulb.data.root_pos_w[0] + quat_apply(bulb.data.root_quat_w[0].unsqueeze(0), plug_offset.unsqueeze(0))[0]
        )
        return torch.dot(plug_point - seat_point, world_axis).item()

    def bulb_twist(bulb) -> float:
        _, socket_quat, _ = seat_geometry()
        return task_attach._signed_twist(
            socket_quat.unsqueeze(0),
            bulb.data.root_quat_w[0].unsqueeze(0),
            seat_axis.unsqueeze(0),
        )[0].item()

    def bulb_lateral_distance(bulb) -> float:
        seat_point, _, world_axis = seat_geometry()
        plug_point = (
            bulb.data.root_pos_w[0] + quat_apply(bulb.data.root_quat_w[0].unsqueeze(0), plug_offset.unsqueeze(0))[0]
        )
        displacement = plug_point - seat_point
        axial = torch.dot(displacement, world_axis)
        return torch.norm(displacement - axial * world_axis).item()

    def step(track: bool = True) -> None:
        if track:
            track_socket_to_palm()
        env.step(zero_action)
        if VIDEO is not None:
            VIDEO.capture()

    def drive_pose(bulb, axial_start: float, axial_end: float, rotation_start: float, rotation_end: float, steps: int):
        for index in range(steps):
            fraction = (index + 1) / steps
            axial = axial_start + fraction * (axial_end - axial_start)
            rotation = rotation_start + fraction * (rotation_end - rotation_start)
            track_socket_to_palm()
            place_bulb(bulb, axial, rotation)
            step(track=False)

    def animate_free_bulb(bulb, start: torch.Tensor, end: torch.Tensor, quat: torch.Tensor, steps: int) -> None:
        for index in range(steps):
            fraction = (index + 1) / steps
            pos = start * (1.0 - fraction) + end * fraction
            bulb.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))
            bulb.write_root_velocity_to_sim(zeros6)
            step()

    def aim_camera() -> None:
        if VIDEO is None:
            return
        seat_point, _, _ = seat_geometry()
        eye = (
            seat_point[0].item() + 0.48,
            seat_point[1].item() - 0.58,
            seat_point[2].item() + 0.20,
        )
        VIDEO.set_pose(eye, tuple(seat_point.tolist()))

    return (
        seat_geometry,
        track_socket_to_palm,
        place_bulb,
        axial_distance,
        bulb_twist,
        bulb_lateral_distance,
        step,
        drive_pose,
        animate_free_bulb,
        aim_camera,
    )


def _check_parameter_ranges() -> None:
    """Spec §9.7: ``(low, high)`` ranges sample per env and re-sample on reset."""
    cfg = build_replace_cfg(num_envs=2)
    cfg.events.bulb_attachment.params["insertion_depth"] = (0.020, 0.045)
    cfg.events.bulb_attachment.params["rotation_angle"] = (0.6, 2.4)
    env = make_env(cfg)
    try:
        manager = getattr(env, task_attach._ENV_ATTR)
        first_depth, first_angle = manager._depth.clone(), manager._angle.clone()
        env.reset(seed=args_cli.seed + 1)
        depth, angle = manager._depth.clone(), manager._angle.clone()
        per_env = bool((depth[0] != depth[1]).item() or (angle[0] != angle[1]).item())
        re_sampled = bool((depth != first_depth).any().item() or (angle != first_angle).any().item())
        in_range = bool(
            ((depth >= 0.020) & (depth <= 0.045)).all().item() and ((angle >= 0.6) & (angle <= 2.4)).all().item()
        )
        record(
            "bayonet:parameter_ranges_randomize",
            per_env and re_sampled and in_range,
            f"depths={[round(v, 4) for v in depth.tolist()]} m, angles={[round(v, 3) for v in angle.tolist()]} rad",
        )
    finally:
        env.close()


def _find_palm(robot) -> int:
    for name in ("right_hand_base_link", "right_hand_palm_link"):
        body_ids, _ = robot.find_bodies(name)
        if body_ids:
            return body_ids[0]
    raise ValueError(f"no right palm found; available bodies: {robot.body_names}")


def _check_lock_state_observable(env, manager) -> None:
    """Assert the bayonet lock state reaches the privileged observation group (issue #77).

    Placement matters, not only presence: the group concatenates, so a consumer reads the
    lock columns by offset. This pins them to the tail and checks they carry the manager's
    own reset state -- the old bulb locked at the rotation angle, the fresh bulb free.

    Without this term ``_phase`` and ``_theta`` reach no observation, telemetry or recording
    path, and an operator cannot tell a twist that does not register from one the lock
    clamps away.
    """
    terms = env.observation_manager.active_terms.get("privileged", [])
    if "bulb_lock_state" not in terms:
        record("bayonet:lock_state_observable", False, "no bulb_lock_state term in the privileged group")
        return
    priv = env.observation_manager.compute()["privileged"]
    lock = task_attach.bulb_lock_state(env)
    tail_ok = bool(torch.allclose(priv[:, -4:], lock))
    old_locked = bool(torch.all(lock[:, 0] == 2.0)) and bool(torch.allclose(lock[:, 1], manager._angle))
    fresh_free = bool(torch.all(lock[:, 2] == 0.0)) and bool(torch.all(lock[:, 3] == 0.0))
    record(
        "bayonet:lock_state_observable",
        tail_ok and old_locked and fresh_free,
        f"privileged[{priv.shape[-1]}] tail 4 = {[round(v, 4) for v in lock[0].tolist()]} "
        f"(old ROTATING at theta=angle, fresh FREE)",
    )


def main() -> int:
    global VIDEO
    torch.manual_seed(args_cli.seed)
    if args_cli.check_ranges:
        _check_parameter_ranges()
        return _summary()
    env = make_env(build_replace_cfg())
    try:
        robot = env.scene["robot"]
        socket = env.scene["socket"]
        old_bulb = env.scene["old_bulb"]
        fresh_bulb = env.scene["bulb"]
        manager = getattr(env, task_attach._ENV_ATTR, None)
        if manager is None:
            record("bayonet:manager_present", False, "no bulb_attachment term wired on FIATLUX-Replace-v0")
            return _summary()
        record("bayonet:manager_present", True, "mdp.bulb_attachment is enforced every step")
        _check_lock_state_observable(env, manager)

        zero_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        zeros6 = torch.zeros((env.num_envs, 6), device=env.device)
        if args_cli.video:
            from fiatlux_task.viz import VideoRecorder

            VIDEO = VideoRecorder(env, env.scene["video_cam"], args_cli.video, fps=20)

        (
            seat_geometry,
            track_socket_to_palm,
            place_bulb,
            axial_distance,
            bulb_twist,
            bulb_lateral_distance,
            step,
            drive_pose,
            animate_free_bulb,
            aim_camera,
        ) = _build_rig(
            env,
            robot,
            socket,
            old_bulb,
            fresh_bulb,
            _find_palm(robot),
            zero_action,
            zeros6,
        )

        def old_phase() -> int:
            return int(manager._phase[task_attach._OLD, 0].item())

        def fresh_phase() -> int:
            return int(manager._phase[task_attach._FRESH, 0].item())

        def fresh_theta() -> float:
            return manager._theta[task_attach._FRESH, 0].item()

        def old_theta() -> float:
            return manager._theta[task_attach._OLD, 0].item()

        for _ in range(12):
            step(track=False)
        for _ in range(8):
            step()
        aim_camera()

        depth = manager._depth[0].item()
        angle = manager._angle[0].item()
        params_ok = abs(depth - BAYONET_INSERTION_DEPTH) < 1e-5 and abs(angle - BAYONET_ROTATION_ANGLE) < 1e-5
        record(
            "bayonet:parametric_geometry",
            params_ok,
            f"depth={depth:.3f} m, rotation={angle:.3f} rad",
        )

        old_attached = bool(task_attach.old_bulb_attached(env)[0].item())
        old_seat_error = abs(axial_distance(old_bulb))
        record(
            "bayonet:old_starts_locked",
            old_attached and old_seat_error < 0.003,
            f"attached={old_attached}, axial error={old_seat_error * 1000:.1f} mm",
        )

        # A second bulb must never engage an occupied socket, however well aligned.
        fresh_park = fresh_bulb.data.root_pos_w[0].clone()
        fresh_park_quat = fresh_bulb.data.root_quat_w[0].clone()
        track_socket_to_palm()
        place_bulb(fresh_bulb, 0.5 * depth, 0.0)
        step(track=False)
        record(
            "bayonet:occupied_socket_rejects_fresh",
            fresh_phase() == task_attach._FREE,
            f"aligned fresh bulb in occupied channel stays FREE (phase={fresh_phase()})",
        )
        fresh_bulb.write_root_pose_to_sim(torch.cat([fresh_park, fresh_park_quat]).unsqueeze(0))
        fresh_bulb.write_root_velocity_to_sim(zeros6)
        step()

        track_socket_to_palm()
        place_bulb(old_bulb, 0.75 * depth, angle)
        step(track=False)
        locked_axial = abs(axial_distance(old_bulb))
        record(
            "bayonet:rotation_stage_blocks_translation",
            locked_axial < 0.003 and bool(task_attach.old_bulb_attached(env)[0].item()),
            f"attempted {0.75 * depth * 1000:.1f} mm, retained {locked_axial * 1000:.1f} mm",
        )

        # Reversing a partial unlock must re-lock (theta clamps at the lock angle).
        drive_pose(old_bulb, 0.0, 0.0, angle, 0.4 * angle, 12)
        drive_pose(old_bulb, 0.0, 0.0, 0.4 * angle, angle, 12)
        relocked = bool(task_attach.old_bulb_attached(env)[0].item())
        record(
            "bayonet:reversed_unlock_relocks",
            relocked and abs(old_theta() - angle) < 0.02,
            f"theta={old_theta():.3f} rad after reversal (lock angle {angle:.3f})",
        )

        # A shove landing on the very step the bulb unlocks (twist -> 0 and a large axial
        # offset in one step) must stay pinned at the seat: the unlock transition step
        # begins AXIAL travel only on the *next* step.
        drive_pose(old_bulb, 0.0, 0.0, angle, 0.1, 20)
        track_socket_to_palm()
        place_bulb(old_bulb, 0.9 * depth, 0.0)
        step(track=False)
        unlock_shove_axial = abs(axial_distance(old_bulb))
        record(
            "bayonet:unlock_step_pins_axial",
            unlock_shove_axial < 0.003 and old_phase() == task_attach._AXIAL,
            f"unlocked with 0.9*depth shove, retained {unlock_shove_axial * 1000:.1f} mm axial",
        )
        # Restore the locked state so the removal sequence below starts as it expects.
        place_bulb(old_bulb, 0.0, 0.05)
        step(track=False)
        drive_pose(old_bulb, 0.0, 0.0, 0.05, angle, 12)

        drive_pose(old_bulb, 0.0, 0.0, angle, 0.0, 24)
        old_released = not bool(task_attach.old_bulb_attached(env)[0].item())
        record(
            "bayonet:bulb_rotation_unlocks_old",
            old_released and old_phase() == task_attach._AXIAL,
            f"bulb rotation={old_theta():.3f} rad, released={old_released}",
        )

        track_socket_to_palm()
        place_bulb(old_bulb, 0.5 * depth, 0.5 * angle, lateral_distance=0.025)
        step(track=False)
        axial_twist = abs(bulb_twist(old_bulb))
        axial_position = axial_distance(old_bulb)
        lateral_error = bulb_lateral_distance(old_bulb)
        record(
            "bayonet:axial_stage_allows_only_axis_travel",
            axial_twist < 0.02 and lateral_error < 0.003 and abs(axial_position - 0.5 * depth) < 0.003,
            f"twist={axial_twist:.3f} rad, lateral error={lateral_error * 1000:.1f} mm",
        )

        drive_pose(old_bulb, 0.5 * depth, 1.15 * depth, 0.0, 0.0, 18)
        old_free = old_phase() == task_attach._FREE
        record(
            "bayonet:old_ejects_after_full_depth",
            old_free,
            f"phase={old_phase()}, travel={axial_distance(old_bulb) * 1000:.1f} mm",
        )

        free_start = old_bulb.data.root_pos_w[0].clone()
        _, _, world_axis = seat_geometry()
        free_velocity = torch.cat([0.5 * world_axis, torch.zeros(3, device=env.device)])
        old_bulb.write_root_velocity_to_sim(free_velocity.unsqueeze(0))
        for _ in range(10):
            step()
        free_motion = torch.norm(old_bulb.data.root_pos_w[0] - free_start).item()
        record(
            "bayonet:ejected_bulb_moves_freely",
            old_free and free_motion > 0.02,
            f"physics-driven displacement={free_motion * 100:.1f} cm",
        )

        old_quat = old_bulb.data.root_quat_w[0].clone()
        old_start = old_bulb.data.root_pos_w[0].clone()
        old_aside = old_start + torch.tensor([0.20, -0.12, 0.06], device=env.device)
        animate_free_bulb(old_bulb, old_start, old_aside, old_quat, 18)

        # A misaligned bulb must never engage the (now empty) channel.
        track_socket_to_palm()
        place_bulb(fresh_bulb, 0.5 * depth, 0.0, lateral_distance=0.03)
        step(track=False)
        record(
            "bayonet:misaligned_entry_rejected",
            fresh_phase() == task_attach._FREE,
            f"3 cm lateral offset stays FREE (phase={fresh_phase()})",
        )

        drive_pose(fresh_bulb, 1.15 * depth, 0.75 * depth, 0.0, 0.0, 18)
        fresh_axial = fresh_phase() == task_attach._AXIAL
        record(
            "bayonet:fresh_enters_axial_channel",
            fresh_axial,
            f"phase={fresh_phase()}, depth={axial_distance(fresh_bulb) * 1000:.1f} mm",
        )

        track_socket_to_palm()
        place_bulb(fresh_bulb, 0.5 * depth, 0.5 * angle)
        step(track=False)
        fresh_axial_twist = abs(bulb_twist(fresh_bulb))
        record(
            "bayonet:fresh_cannot_rotate_during_insertion",
            fresh_axial_twist < 0.02 and fresh_phase() == task_attach._AXIAL,
            f"attempted {0.5 * angle:.3f} rad, retained {fresh_axial_twist:.3f} rad",
        )

        drive_pose(fresh_bulb, 0.5 * depth, -0.002, 0.0, 0.0, 18)
        fresh_at_turning_point = fresh_phase() == task_attach._AXIAL
        record(
            "bayonet:full_insertion_reaches_turning_point",
            fresh_at_turning_point
            and abs(axial_distance(fresh_bulb)) < 0.003
            and not bool(task_attach.fresh_bulb_attached(env)[0].item()),
            f"phase={fresh_phase()}, rotation={fresh_theta():.3f} rad",
        )

        track_socket_to_palm()
        place_bulb(fresh_bulb, 0.0, 0.15 * angle)
        step(track=False)
        rotation_started = fresh_phase() == task_attach._ROTATING
        record(
            "bayonet:bulb_twist_selects_rotation_stage",
            rotation_started and fresh_theta() > 0.0,
            f"phase={fresh_phase()}, rotation={fresh_theta():.3f} rad",
        )

        track_socket_to_palm()
        place_bulb(fresh_bulb, 0.75 * depth, fresh_theta())
        step(track=False)
        fresh_locked_axial = abs(axial_distance(fresh_bulb))
        record(
            "bayonet:fresh_cannot_translate_while_rotating",
            fresh_locked_axial < 0.003 and fresh_phase() == task_attach._ROTATING,
            f"attempted {0.75 * depth * 1000:.1f} mm, retained {fresh_locked_axial * 1000:.1f} mm",
        )

        rotation_start = fresh_theta()
        drive_pose(fresh_bulb, 0.0, 0.0, rotation_start, angle, 24)
        fresh_attached = bool(task_attach.fresh_bulb_attached(env)[0].item())
        record(
            "bayonet:bulb_rotation_attaches_fresh",
            fresh_attached,
            f"bulb rotation={fresh_theta():.3f} rad, attached={fresh_attached}",
        )

        seat_point, _, _ = seat_geometry()
        shove = seat_point + torch.tensor([0.12, 0.10, 0.05], device=env.device)
        fresh_bulb.write_root_pose_to_sim(torch.cat([shove, fresh_bulb.data.root_quat_w[0]]).unsqueeze(0))
        fresh_bulb.write_root_velocity_to_sim(zeros6)
        step()
        shove_error = abs(axial_distance(fresh_bulb))
        record(
            "bayonet:locked_fresh_rejects_shove",
            shove_error < 0.003 and bool(task_attach.fresh_bulb_attached(env)[0].item()),
            f"re-seated to {shove_error * 1000:.1f} mm axial error",
        )

        crate = env.scene["bin"]
        crate_pos = crate.data.root_pos_w[0].clone()
        crate_pos[2] += 0.05
        old_bulb.write_root_pose_to_sim(torch.cat([crate_pos, old_bulb.data.root_quat_w[0]]).unsqueeze(0))
        old_bulb.write_root_velocity_to_sim(zeros6)
        for _ in range(10):
            step(track=False)
        success = bool(task_attach.attached_replacement_success(env)[0].item())
        record("bayonet:replacement_success", success, f"attached_replacement_success={success}")

        finite = not bool(
            torch.isnan(robot.data.root_pos_w).any()
            or torch.isnan(old_bulb.data.root_pos_w).any()
            or torch.isnan(fresh_bulb.data.root_pos_w).any()
        )
        record("bayonet:no_nan", finite, "states finite throughout" if finite else "NaN in states")

        if VIDEO is not None:
            info(f"wrote video {VIDEO.write()} ({len(VIDEO)} frames)")
    finally:
        env.close()
    info("run again with --check-ranges for the parameter-range sampling check")
    return _summary()


def _summary() -> int:
    table = PrettyTable()
    table.field_names = ["#", "check", "result", "detail"]
    table.align["check"] = table.align["detail"] = "l"
    passed = 0
    for index, (name, ok, detail) in enumerate(RESULTS, 1):
        table.add_row([index, name, "PASS" if ok else "FAIL", detail])
        passed += int(ok)
    print("\n" + table.get_string(), flush=True)
    failures = [name for name, ok, _ in RESULTS if not ok]
    verdict = "PASS" if not failures else "FAIL"
    print(f"\n[verify] {passed}/{len(RESULTS)} checks passed -- OVERALL: {verdict}", flush=True)
    if failures:
        print("[verify] Failing checks:\n   - " + "\n   - ".join(failures), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    import os
    import sys

    exit_code = 1
    try:
        exit_code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code:
            os._exit(exit_code)
        simulation_app.close()
