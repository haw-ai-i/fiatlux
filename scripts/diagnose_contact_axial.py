# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Can contact drive the bayonet AXIAL insertion at all? (companion to diagnose_contact_twist.py)

``diagnose_contact_twist.py`` (issue #77 task 3) found that with BOTH the bulb's and the
socket's colliders on -- the real scene, before ``_spawn_bulb_socket_filtered`` filters the
pair -- driving the old bulb's ROTATING-phase twist with an external torque never releases
it: theta hovers at 0.004-0.02 rad instead of converging, at every magnitude tried, because
the bulb is pinned into the socket's collision mesh. That result was entirely about the
RELEASE/UNSCREW direction of an already-seated bulb.

This is the same question for the OTHER direction: can a fresh bulb travel axially INTO the
channel at all, under real bulb-socket contact, given a plausible entry pose? Reports here
are what `plans/bayonet-force-based-attachment.md` needs before any production code changes
its enforcement from a pose-overwrite to an external-wrench scheme: if axial travel is
ALSO geometrically blocked by the socket's collider (like twist was), a wrench-based fix
does not help either -- the collider geometry itself would need fixing first.

Method, deliberately parallel to diagnose_contact_twist.py:

* A pose-drive CONTROL that walks the fresh bulb's plug point from just outside the channel
  to the seat via direct ``write_root_pose_to_sim`` calls. This must succeed (axial reaches
  within ``seat_tolerance``) or the run proves nothing -- same non-negotiable control as the
  twist script.
* A FORCE-driven trial: apply a constant external force along the socket's seat axis (the
  faithful proxy for a hand pushing the bulb in) at four magnitudes, both toward-seat and
  away, and read whether real axial displacement tracks it under contact.
* A VELOCITY-driven trial: inject axial linear velocity every step without writing pose,
  mirroring the twist script's ``run_velocity``.

Each trial runs twice: real geometry (both colliders on, the default) and with the
bulb-socket pair collision force-enabled past the shipped ``_spawn_bulb_socket_filtered``
filter is irrelevant here since that filter is what we are trying to determine we can safely
remove -- so BOTH trial passes here run with that filter intact-or-not controlled by
``FORCE_SOCKET_COLLISION``, which monkeypatches the spawner to skip the ``FilteredPairsAPI``
call (Isaac Lab has no runtime per-pair toggle, so this has to happen before the scene is
built, like ``DISABLE_SOCKET_COLLISION`` did in the twist script, just inverted: that one
removed contact where the shipped scene has it enabled; this one adds contact where the
shipped scene has it filtered out).

``bulb_attachment`` keeps running every step throughout (never bypassed): a FREE-phase bulb
is untouched by its projection (only AXIAL/ROTATING bodies get pose/velocity overwritten;
see ``attach.py:_advance``), so nothing here fights the state machine unless/until ``engage``
actually fires -- which is itself part of what a trial is checking.

Examples
--------
    python scripts/diagnose_contact_axial.py                            # real scene (filter on, shipped default)
    FORCE_SOCKET_COLLISION=1 python scripts/diagnose_contact_axial.py   # remove the filter: real bulb-socket contact
    DISABLE_COLLISION=1 python scripts/diagnose_contact_axial.py        # isolate the projection (bulb's own collider off)
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive bayonet AXIAL insertion with contact-like force.")
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

from isaaclab.sim.utils import clone as _clone
import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import torch
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task import scene_cfg as scene_cfg_mod
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg

# Same two independent switches as diagnose_contact_twist.py, for the bulb's own collider.
NO_COLLIDE = os.environ.get("NO_COLLIDE") == "1"
DISABLE_COLLISION = NO_COLLIDE or os.environ.get("DISABLE_COLLISION") == "1"
DISABLE_GRAVITY = NO_COLLIDE or os.environ.get("DISABLE_GRAVITY") == "1"
# The inverse of diagnose_contact_twist.py's DISABLE_SOCKET_COLLISION: the SHIPPED scene
# already has the bulb-socket pair filtered OUT (_spawn_bulb_socket_filtered). This flag
# monkeypatches that spawner to skip the FilteredPairsAPI call, so the pair collides for
# real -- the condition this plan needs measured.
FORCE_SOCKET_COLLISION = os.environ.get("FORCE_SOCKET_COLLISION") == "1"


@_clone
def _unfiltered_bulb_spawn(prim_path, cfg, translation=None, orientation=None):
    """``_spawn_bulb_socket_filtered`` without the ``FilteredPairsAPI`` call.

    MUST carry @clone like the original: the spawner is handed a regex prim path
    (/World/envs/env_.*/Bulb) and @clone is what resolves it to the source env and
    replicates. Without it USD gets the regex verbatim and the run dies on an
    ill-formed SdfPath.
    """
    return scene_cfg_mod._spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)


def build_cfg():
    # Same reasoning as diagnose_contact_twist.py: Replace draws its fixture mount inside
    # __post_init__, before cfg.seed exists, so the layout seed must be set first.
    set_layout_seed(args_cli.seed)
    if FORCE_SOCKET_COLLISION:
        scene_cfg_mod._spawn_bulb_socket_filtered = _unfiltered_bulb_spawn
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=1)
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
    for bulb_cfg in (cfg.scene.fresh_bulb, cfg.scene.old_bulb):
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
    fresh_bulb = env.scene["fresh_bulb"]
    manager = task_attach.attachment_manager(env)
    if manager is None:
        print("FATAL no bulb_attachment term on FIATLUX-Replace-v0", flush=True)
        return 1

    device = env.device
    zero_action = torch.zeros((1, env.action_manager.total_action_dim), device=device)
    seat_axis = torch.tensor(SOCKET_SEAT_AXIS, device=device)
    seat_offset = torch.tensor(SOCKET_SEAT_OFFSET, device=device)
    plug_offset = torch.tensor(BULB_PLUG_OFFSET, device=device)
    depth = float(manager._depth[0].item())
    seat_tolerance = 0.004  # matches attach.py's default; not exposed on the manager

    print(
        f"SETUP bulb_collider={'off' if DISABLE_COLLISION else 'ON'} "
        f"gravity={'off' if DISABLE_GRAVITY else 'ON'} "
        f"socket_pair_collision={'FORCED ON' if FORCE_SOCKET_COLLISION else 'filtered off (shipped default)'}"
    )
    print(f"SETUP insertion_depth={depth:.4f} seat_tolerance={seat_tolerance:.4f}", flush=True)

    def geometry() -> tuple[float, float, float]:
        """(axial, lateral, tilt) of the fresh bulb relative to the socket seat, same math as
        ``attach.py:_advance`` -- axial is signed distance along the seat axis from the seat
        plane (0 = bottomed out), lateral is perpendicular offset, tilt is the plug-vs-seat
        axis angle (issue #90's entry gate, ignoring twist)."""
        socket_quat = socket.data.root_quat_w
        axis_w = quat_apply(socket_quat, seat_axis.unsqueeze(0))
        seat = socket.data.root_pos_w + quat_apply(socket_quat, seat_offset.unsqueeze(0))
        plug = fresh_bulb.data.root_pos_w + quat_apply(fresh_bulb.data.root_quat_w, plug_offset.unsqueeze(0))
        displacement = plug - seat
        axial = (displacement * axis_w).sum(dim=1)
        lateral = torch.norm(displacement - axial.unsqueeze(1) * axis_w, dim=1)
        tilt = task_attach._tilt_error(socket_quat, fresh_bulb.data.root_quat_w, seat_axis)
        return float(axial.item()), float(lateral.item()), float(tilt.item())

    def phase() -> int:
        return int(manager._phase[task_attach._FRESH, 0].item())

    def entry_pose(offset_along_axis: float) -> tuple[torch.Tensor, torch.Tensor]:
        """A pose ``offset_along_axis`` m outside the seat plane, aligned with the socket
        (zero tilt, zero lateral, arbitrary clock angle) -- a plausible, generous entry pose.
        Real teleop entries are rarely this clean; the point is to isolate whether contact
        blocks travel even from the easiest starting pose, before blaming anything else."""
        socket_quat = socket.data.root_quat_w[0]
        seat = socket.data.root_pos_w[0] + quat_apply(socket_quat.unsqueeze(0), seat_offset.unsqueeze(0))[0]
        axis_w = quat_apply(socket_quat.unsqueeze(0), seat_axis.unsqueeze(0))[0]
        pos = seat + offset_along_axis * axis_w - quat_apply(socket_quat.unsqueeze(0), plug_offset.unsqueeze(0))[0]
        return pos, socket_quat

    def settle(start_offset: float = 0.05) -> None:
        env.reset(seed=args_cli.seed)
        pos, quat = entry_pose(start_offset)
        fresh_bulb.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))
        fresh_bulb.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        for _ in range(3):
            env.step(zero_action)

    def run_force(magnitude: float, direction: float) -> bool:
        """Constant world-frame force along the seat axis. Positive direction pushes toward
        the seat (into the channel); negative pulls out -- a sign-asymmetry check, the way
        diagnose_contact_twist.py tries both twist directions."""
        settle()
        axial0, _, _ = geometry()
        socket_quat = socket.data.root_quat_w
        axis_w = quat_apply(socket_quat, seat_axis.unsqueeze(0))
        forces = (direction * magnitude * axis_w).unsqueeze(1)
        torque = torch.zeros((1, 1, 3), device=device)
        trace = []
        for i in range(args_cli.steps):
            fresh_bulb.set_external_force_and_torque(forces, torque)
            env.step(zero_action)
            if i % 20 == 0 or i == args_cli.steps - 1:
                axial, lateral, tilt = geometry()
                trace.append((i, round(axial, 4), round(lateral, 4), round(tilt, 3), phase()))
        fresh_bulb.set_external_force_and_torque(torch.zeros_like(forces), torque)
        axial, lateral, tilt = geometry()
        seated = phase() != task_attach._FREE or abs(axial) <= seat_tolerance
        print(
            f"FORCE mag={magnitude:g} dir={direction:+.0f} axial {axial0:.4f} -> {axial:.4f} "
            f"lateral={lateral:.4f} tilt={tilt:.3f} phase={phase()} seated={seated} trace={trace}",
            flush=True,
        )
        return seated

    def run_velocity(speed: float) -> bool:
        """Inject axial linear velocity every step, never writing pose -- mirrors
        diagnose_contact_twist.py's ``run_velocity``."""
        settle()
        axial0, _, _ = geometry()
        for _ in range(args_cli.steps):
            socket_quat = socket.data.root_quat_w
            axis_w = quat_apply(socket_quat, seat_axis.unsqueeze(0))
            velocity = torch.zeros((1, 6), device=device)
            velocity[:, :3] = speed * axis_w
            fresh_bulb.write_root_velocity_to_sim(velocity)
            env.step(zero_action)
        axial, lateral, tilt = geometry()
        seated = phase() != task_attach._FREE or abs(axial) <= seat_tolerance
        print(
            f"VELOCITY speed={speed:+g} axial {axial0:.4f} -> {axial:.4f} lateral={lateral:.4f} "
            f"tilt={tilt:.3f} phase={phase()} seated={seated}",
            flush=True,
        )
        return seated

    def run_pose_drive() -> bool:
        """Control: the way the shipped code drives it today. Must reach the seat, or the run
        proves nothing about contact -- same non-negotiable control as the twist script."""
        settle()
        axial0, _, _ = geometry()
        steps = 60
        for i in range(steps):
            offset = 0.05 * (1.0 - (i + 1) / steps)
            pos, quat = entry_pose(offset)
            fresh_bulb.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))
            fresh_bulb.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
            env.step(zero_action)
        driven_axial, _, _ = geometry()
        hold_trace = []
        for i in range(40):
            pos, quat = entry_pose(0.0)
            fresh_bulb.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))
            fresh_bulb.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
            env.step(zero_action)
            if i % 10 == 0 or i == 39:
                axial, lateral, tilt = geometry()
                hold_trace.append((i, round(axial, 4), round(lateral, 4), phase()))
        axial, lateral, tilt = geometry()
        seated = phase() != task_attach._FREE or abs(axial) <= seat_tolerance
        print(
            f"CONTROL_POSE_DRIVE axial {axial0:.4f} -> driven={driven_axial:.4f} -> held={axial:.4f} "
            f"seated={seated} hold={hold_trace}",
            flush=True,
        )
        return seated

    contact_seated = False
    for magnitude in (0.5, 2.0, 10.0, 50.0):
        for direction in (1.0, -1.0):
            contact_seated |= run_force(magnitude, direction)
    for speed in (0.05, 0.5):
        contact_seated |= run_velocity(speed)
    control_seated = run_pose_drive()

    print(
        f"VERDICT socket_pair_collision={'FORCED ON' if FORCE_SOCKET_COLLISION else 'filtered off'} "
        f"contact_seated={contact_seated} control_seated={control_seated}",
        flush=True,
    )
    env.close()
    if not control_seated:
        print("INVALID the pose-drive control failed; this run proves nothing", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    # Same shutdown shape as diagnose_contact_twist.py: closing the env does not tear down
    # Kit, and a blanket finally:-close would hang on failure, so hard-exit past it instead.
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
