# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Can contact drive the bayonet twist at all? (issue #77 task 3)

The projection rewrites the old bulb's pose and angular velocity every env step. ``theta``
advances only if physics rotates the bulb between the write at step N and the read at N+1.
`verify_attach.py` never tests that: it drives the bulb with direct pose writes, so it *is*
the write. It also disables both bulbs' colliders, which turns out to be the whole question.

This drives the bulb with an external torque instead -- the faithful proxy for finger
friction -- and reads theta. It tries both directions at four magnitudes, so no sign
convention can hide a working mechanism, and it ends with a scripted pose-drive control that
must unlock (otherwise the harness itself is broken and the run says nothing).

What it found, on `FIATLUX-Replace-v0` at `rotation_sign=-1`:

    bulb collider   gravity   torque unlocks?   pose-drive control unlocks?
    on              on        NO                NO      <- the real scene
    off             on        yes               yes
    off             off       yes               yes     <- verify_attach's rig

Two results, and the second is the important one.

#77's cause 2 as originally framed is REFUTED. Whether a contact twist survives the pose
writes was never the problem: with the collider off, a torque of 0.001 N.m advances theta
and unlocks the bulb, and injected angular velocity does too.

With the collider on, NOTHING unlocks -- not torque, not injected velocity, and not even
direct pose writes, which is the method `verify_attach` proves works. The control run drives
theta to exactly 0.0000 and the bulb still does not release. `unlock` needs
``theta <= _EPS`` AND ``delta < 0`` on the SAME step, and contact keeps breaking that
conjunction: held at the commanded release angle, theta jitters 0.0025 -> 0.0125 -> 0.0054
-> 0.0122 instead of settling. The bulb's collider, pinned into the socket's triangle mesh
by the projection every step, is what supplies the jitter. Gravity is not a factor; the
middle row isolates it.

So the mechanic is not operable in the real scene by ANY drive method, which is a stronger
statement than "teleoperation cannot turn it".

That is why `verify_attach` cannot see this: it disables the colliders that are the whole
question. Its 23/23 remains a statement about the state machine alone.

Examples
--------
    python scripts/diagnose_contact_twist.py                       # real scene: colliders on
    DISABLE_COLLISION=1 python scripts/diagnose_contact_twist.py   # isolate the projection
    NO_COLLIDE=1 python scripts/diagnose_contact_twist.py          # verify_attach's rig
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive the bayonet twist with contact-like torque.")
parser.add_argument("--seed", type=int, default=0, help="Env seed (deterministic).")
parser.add_argument("--steps", type=int, default=120, help="Steps per trial.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything else follows."""

import importlib
import os

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import torch
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_tasks.utils import parse_env_cfg

# Two independent switches, so a run can tell the bulb's collider apart from its weight.
# NO_COLLIDE sets both, reproducing verify_attach's rig.
NO_COLLIDE = os.environ.get("NO_COLLIDE") == "1"
DISABLE_COLLISION = NO_COLLIDE or os.environ.get("DISABLE_COLLISION") == "1"
DISABLE_GRAVITY = NO_COLLIDE or os.environ.get("DISABLE_GRAVITY") == "1"


def build_cfg():
    cfg = parse_env_cfg("FIATLUX-Replace-v0", num_envs=1)
    cfg.seed = args_cli.seed
    for camera in ("ego_camera", "torso_camera", "wrist_camera"):
        if getattr(cfg.scene, camera, None) is not None:
            setattr(cfg.scene, camera, None)
    for group_name in ("policy", "privileged"):
        group = getattr(cfg.observations, group_name, None)
        for term in ("ego_rgb", "torso_rgb", "wrist_rgb"):
            if group is not None and getattr(group, term, None) is not None:
                setattr(group, term, None)
    for term in ("success", "old_bulb_dropped", "fresh_bulb_dropped"):
        if getattr(cfg.terminations, term, None) is not None:
            setattr(cfg.terminations, term, None)
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    for bulb_cfg in (cfg.scene.bulb, cfg.scene.old_bulb):
        if DISABLE_GRAVITY and getattr(bulb_cfg.spawn, "rigid_props", None) is not None:
            bulb_cfg.spawn.rigid_props.disable_gravity = True
        if DISABLE_COLLISION:
            bulb_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    return cfg


def main() -> int:
    cfg = build_cfg()
    spec = gym.spec("FIATLUX-Replace-v0")
    module_name, class_name = spec.entry_point.split(":")
    env = getattr(importlib.import_module(module_name), class_name)(cfg=cfg)
    env.reset(seed=args_cli.seed)

    socket = env.scene["socket"]
    old_bulb = env.scene["old_bulb"]
    manager = task_attach.attachment_manager(env)
    if manager is None:
        print("FATAL no bulb_attachment term on FIATLUX-Replace-v0", flush=True)
        return 1

    device = env.device
    zero_action = torch.zeros((1, env.action_manager.total_action_dim), device=device)
    seat_axis = torch.tensor(SOCKET_SEAT_AXIS, device=device)
    seat_offset = torch.tensor(SOCKET_SEAT_OFFSET, device=device)
    plug_offset = torch.tensor(BULB_PLUG_OFFSET, device=device)
    angle = float(manager._angle[0].item())
    sign = manager.rotation_sign
    # Release drives the physical twist toward zero, so it opposes the locked twist's sign.
    release_dir = -1.0 if sign > 0 else 1.0

    print(f"SETUP collider={'off' if DISABLE_COLLISION else 'ON'} gravity={'off' if DISABLE_GRAVITY else 'ON'}")
    print(f"SETUP rotation_sign={sign:+.0f} lock_angle={angle:.4f} release_dir={release_dir:+.0f}", flush=True)

    def theta() -> float:
        return float(manager._theta[task_attach._OLD, 0].item())

    def phase() -> int:
        return int(manager._phase[task_attach._OLD, 0].item())

    def axial_omega() -> float:
        axis_w = quat_apply(socket.data.root_quat_w, seat_axis.unsqueeze(0))
        return float((old_bulb.data.root_ang_vel_w * axis_w).sum().item())

    def settle(steps: int = 3) -> None:
        env.reset(seed=args_cli.seed)
        for _ in range(steps):
            env.step(zero_action)

    def run_torque(magnitude: float, direction: float) -> bool:
        """Constant body-frame twist torque. Bulb local +Z is the seat axis while seated."""
        settle()
        start = theta()
        forces = torch.zeros((1, 1, 3), device=device)
        torque = torch.zeros((1, 1, 3), device=device)
        torque[0, 0, 2] = direction * magnitude
        trace = []
        for i in range(args_cli.steps):
            old_bulb.set_external_force_and_torque(forces, torque)
            env.step(zero_action)
            if i % 40 == 0 or i == args_cli.steps - 1:
                trace.append((i, round(theta(), 4), phase(), round(axial_omega(), 2)))
        old_bulb.set_external_force_and_torque(forces, torch.zeros((1, 1, 3), device=device))
        unlocked = phase() == task_attach._AXIAL
        print(
            f"TORQUE mag={magnitude:g} dir={direction:+.0f} {start:.4f} -> {theta():.4f} "
            f"unlocked={unlocked} trace={trace}",
            flush=True,
        )
        return unlocked

    def run_velocity(omega: float) -> bool:
        """Inject an axial angular velocity each step, never writing the pose."""
        settle()
        start = theta()
        for _ in range(args_cli.steps):
            axis_w = quat_apply(socket.data.root_quat_w, seat_axis.unsqueeze(0))
            velocity = torch.zeros((1, 6), device=device)
            velocity[:, 3:] = omega * axis_w
            old_bulb.write_root_velocity_to_sim(velocity)
            env.step(zero_action)
        unlocked = phase() == task_attach._AXIAL
        print(f"VELOCITY omega={omega:+g} {start:.4f} -> {theta():.4f} unlocked={unlocked}", flush=True)
        return unlocked

    def run_pose_drive() -> bool:
        """Control: the way verify_attach drives it. Must unlock, or the run says nothing."""
        settle()
        start = theta()
        locked_twist = sign * angle
        steps = 40
        for i in range(steps):
            rotation = locked_twist * (1.0 - (i + 1) / steps)
            socket_quat = socket.data.root_quat_w[0]
            seat = socket.data.root_pos_w[0] + quat_apply(socket_quat.unsqueeze(0), seat_offset.unsqueeze(0))[0]
            axis_w = quat_apply(socket_quat.unsqueeze(0), seat_axis.unsqueeze(0))[0]
            half = torch.tensor(0.5 * rotation, device=device)
            spin = torch.cat([torch.cos(half).unsqueeze(0), axis_w * torch.sin(half)])
            bulb_quat = quat_mul(spin.unsqueeze(0), socket_quat.unsqueeze(0))[0]
            bulb_pos = seat - quat_apply(bulb_quat.unsqueeze(0), plug_offset.unsqueeze(0))[0]
            old_bulb.write_root_pose_to_sim(torch.cat([bulb_pos, bulb_quat]).unsqueeze(0))
            old_bulb.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
            env.step(zero_action)
        driven = theta()
        # Hold at the commanded release angle. If theta converges to 0 the shortfall above
        # was lag; if it plateaus, contact is holding a steady-state twist error that the
        # unlock threshold (theta <= _EPS) can never clear.
        hold_trace = []
        for i in range(60):
            socket_quat = socket.data.root_quat_w[0]
            seat = socket.data.root_pos_w[0] + quat_apply(socket_quat.unsqueeze(0), seat_offset.unsqueeze(0))[0]
            bulb_pos = seat - quat_apply(socket_quat.unsqueeze(0), plug_offset.unsqueeze(0))[0]
            old_bulb.write_root_pose_to_sim(torch.cat([bulb_pos, socket_quat]).unsqueeze(0))
            old_bulb.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
            env.step(zero_action)
            if i % 20 == 0 or i == 59:
                hold_trace.append((i, round(theta(), 4), phase()))
        unlocked = phase() == task_attach._AXIAL
        print(
            f"CONTROL_POSE_DRIVE {start:.4f} -> driven={driven:.4f} -> held={theta():.4f} "
            f"unlocked={unlocked} hold={hold_trace}",
            flush=True,
        )
        return unlocked

    contact_unlocked = False
    for magnitude in (1e-4, 1e-3, 1e-2, 1e-1):
        for direction in (release_dir, -release_dir):
            contact_unlocked |= run_torque(magnitude, direction)
    for omega in (2.0, 10.0):
        contact_unlocked |= run_velocity(omega * release_dir)
    control_unlocked = run_pose_drive()

    print(
        f"VERDICT collider={'off' if DISABLE_COLLISION else 'ON'} "
        f"contact_unlocked={contact_unlocked} control_unlocked={control_unlocked}",
        flush=True,
    )
    if not control_unlocked:
        print("INVALID the pose-drive control failed; this run proves nothing", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
