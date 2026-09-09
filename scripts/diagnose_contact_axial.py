# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Can contact drive fresh-bulb AXIAL insertion, at the current plug/bore clearance?

Adapted from the original ``diagnose_contact_axial.py`` (deleted in the bayonet -> axial-detent
rewrite, issue #167, since the bayonet API it drove no longer exists) to validate a *follow-up*
size change: issue #171 found that after the 2026-09-07 radial shrink (plug scaled to 0.84,
leaving +2.69mm clearance vs. the 20.2mm bore), seated bulbs sit visibly tilted/swinging in real
teleop -- too much clearance, not too little. This script re-runs the same axial-insertion
question (can contact let a fresh bulb travel in at all, from a plausible entry pose) against
whatever plug radius ``assets/omniverse_bulb/LightBulb_bulb_z_rigid.usda`` currently ships, so a
size increase can be checked for new interference before trusting it.

Real bulb-socket collision is now ALWAYS on (issue #167 removed ``_spawn_bulb_socket_filtered`
globally) -- there is no filter left to toggle, unlike the original script's
``FORCE_SOCKET_COLLISION`` monkeypatch, which is gone. ``bulb_attachment`` (the current
FREE/SEATED axial detent, not the old bayonet) keeps running every step throughout, same as
before: a FREE bulb gets zero force (see ``attach.py:_advance``), so nothing here fights it
unless/until the engage gate actually fires.

Method (unchanged from the original):

* A pose-drive CONTROL that walks the fresh bulb's plug point from just outside the channel to
  the seat via direct ``write_root_pose_to_sim`` calls. Must succeed or the run proves nothing.
* A FORCE-driven trial: constant external force along the seat axis, four magnitudes, both
  directions.
* A VELOCITY-driven trial: inject axial linear velocity every step without writing pose.
* A FORCE_HELD trial: axial push + a small spring-damper torque holding tilt near zero (a rough
  hand-orientation stand-in).
* A FORCE_HARD_HOLD trial: axial push while lateral and tilt are corrected EXACTLY every step.
  Unlike the original (which reused the old bayonet's twist-preserving projection math,
  ``_signed_twist``/``_axis_angle_quat`` -- both gone, since the new mechanism tracks no twist
  at all), this just snaps the bulb's orientation to the socket's exactly (tilt=0) and its
  position onto the seat axis (lateral=0) each step: there is no twist left to preserve, so the
  hold is simpler than before, not a workaround.

Run it via ``./pyrun`` (repo root), not a bare ``.venv/bin/python``: on a machine where
``.venv`` is shared with a different checkout (this one's own dev setup), a bare invocation can
silently resolve ``fiatlux_task``/``fiatlux_teleop`` to the WRONG checkout's source -- ``pyrun``
fixes that by pointing ``PYTHONPATH`` at this repo's own ``source/`` first. (pytest gets this
right on its own, via its rootdir sys.path insertion; only direct script invocation needs this.)

Examples
--------
    ./pyrun scripts/diagnose_contact_axial.py                 # real scene, real contact (shipped default, issue #167)
    DISABLE_COLLISION=1 ./pyrun scripts/diagnose_contact_axial.py   # isolate the projection (bulb's own collider off)
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive fresh-bulb AXIAL insertion with contact-like force.")
parser.add_argument("--seed", type=int, default=0, help="Env seed (deterministic).")
parser.add_argument("--steps", type=int, default=120, help="Steps per trial.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything else follows."""

import os
import sys

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import importlib
import torch
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg

# The bulb's own collider/gravity -- independent of the (now-removed) bulb-socket filter.
NO_COLLIDE = os.environ.get("NO_COLLIDE") == "1"
DISABLE_COLLISION = NO_COLLIDE or os.environ.get("DISABLE_COLLISION") == "1"
DISABLE_GRAVITY = NO_COLLIDE or os.environ.get("DISABLE_GRAVITY") == "1"


def build_cfg():
    # Replace draws its fixture mount inside __post_init__, before cfg.seed exists, so the
    # layout seed must be set first (same reasoning as the original script).
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=1)
    assert hasattr(cfg.scene, "fresh_bulb"), (
        f"cfg.scene ({type(cfg.scene)} from {sys.modules[type(cfg.scene).__module__].__file__}) has no "
        "fresh_bulb -- fiatlux_task likely resolved to the wrong checkout again; check sys.path"
    )
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

    def _measure_live_plug_radius() -> float | None:
        """Query the plug mesh's ACTUAL bounding radius from the running stage -- confirms
        which asset Isaac Sim really resolved at runtime, independent of any offline USD
        inspection or assumption about what ``download_assets.sh`` last synced."""
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.endswith("Geom/BulbGrp/Base") and "env_0" in path and prim.IsA(UsdGeom.Mesh):
                cache = UsdGeom.BBoxCache(0, ["default", "render"], useExtentsHint=False)
                rng = cache.ComputeWorldBound(prim).ComputeAlignedBox()
                lo, hi = rng.GetMin(), rng.GetMax()
                return 0.5 * max(hi[0] - lo[0], hi[1] - lo[1])
        return None

    live_radius = _measure_live_plug_radius()
    print(
        f"SETUP live_plug_radius={live_radius if live_radius is None else round(live_radius, 5)} "
        "(bore is a constant 0.0202m; positive clearance means this must read below that)",
        flush=True,
    )

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
    seat_tolerance = 0.004  # matches attach.py's default; not exposed on the manager
    radial_tolerance = 0.015  # matches attach.py's engage-gate default
    tilt_tolerance = 0.2  # matches attach.py's engage-gate default

    print(
        f"SETUP bulb_collider={'off' if DISABLE_COLLISION else 'ON'} "
        f"gravity={'off' if DISABLE_GRAVITY else 'ON'} "
        "socket_pair_collision=ALWAYS ON (issue #167 -- no filter left to toggle)",
        flush=True,
    )
    print(f"SETUP seat_tolerance={seat_tolerance:.4f}", flush=True)

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
        (zero tilt, zero lateral, arbitrary clock angle) -- a plausible, generous entry pose."""
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
        """Constant world-frame force along the seat axis (``is_global=True``). Positive
        direction pushes along +seat_axis, which is OUTWARD (``replace_env_cfg.py``: "positive
        axial travel leaves the socket, which is what `eject` tests"); negative is inward."""
        settle()
        axial0, _, _ = geometry()
        socket_quat = socket.data.root_quat_w
        axis_w = quat_apply(socket_quat, seat_axis.unsqueeze(0))
        forces = (direction * magnitude * axis_w).unsqueeze(1)
        torque = torch.zeros((1, 1, 3), device=device)
        trace = []
        for i in range(args_cli.steps):
            fresh_bulb.set_external_force_and_torque(forces, torque, is_global=True)
            env.step(zero_action)
            if i % 20 == 0 or i == args_cli.steps - 1:
                axial, lateral, tilt = geometry()
                trace.append((i, round(axial, 4), round(lateral, 4), round(tilt, 3), phase()))
        fresh_bulb.set_external_force_and_torque(torch.zeros_like(forces), torque, is_global=True)
        axial, lateral, tilt = geometry()
        seated = phase() != task_attach._FREE or (abs(axial) <= seat_tolerance and lateral < radial_tolerance and tilt < tilt_tolerance)
        print(
            f"FORCE mag={magnitude:g} dir={direction:+.0f} axial {axial0:.4f} -> {axial:.4f} "
            f"lateral={lateral:.4f} tilt={tilt:.3f} phase={phase()} seated={seated} trace={trace}",
            flush=True,
        )
        return seated

    def run_velocity(speed: float) -> bool:
        """Inject axial linear velocity every step, never writing pose."""
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
        seated = phase() != task_attach._FREE or (abs(axial) <= seat_tolerance and lateral < radial_tolerance and tilt < tilt_tolerance)
        print(
            f"VELOCITY speed={speed:+g} axial {axial0:.4f} -> {axial:.4f} lateral={lateral:.4f} "
            f"tilt={tilt:.3f} phase={phase()} seated={seated}",
            flush=True,
        )
        return seated

    def run_force_held_orientation(magnitude: float, direction: float, k_p: float = 0.02, k_d: float = 0.002) -> bool:
        """Axial push force plus a spring-damper torque holding tilt near zero -- a rough
        stand-in for a hand's orientation control. Only corrects SWING (tilt), never TWIST
        about the seat axis (the new mechanism, like the old bayonet's entry gate, treats
        twist as free -- there is no lock to turn into)."""
        settle()
        axial0, _, tilt0 = geometry()
        socket_quat_ref = socket.data.root_quat_w
        seat_axis_w = quat_apply(socket_quat_ref, seat_axis.unsqueeze(0))
        forces = (direction * magnitude * seat_axis_w).unsqueeze(1)
        trace = []
        for i in range(args_cli.steps):
            bulb_quat = fresh_bulb.data.root_quat_w
            plug_axis_w = quat_apply(bulb_quat, seat_axis.unsqueeze(0))
            correction = torch.cross(plug_axis_w, seat_axis_w, dim=-1)
            ang_vel = fresh_bulb.data.root_ang_vel_w
            torque = (k_p * correction - k_d * ang_vel).unsqueeze(1)
            fresh_bulb.set_external_force_and_torque(forces, torque, is_global=True)
            env.step(zero_action)
            if i % 20 == 0 or i == args_cli.steps - 1:
                axial, lateral, tilt = geometry()
                trace.append((i, round(axial, 4), round(lateral, 4), round(tilt, 3), phase()))
        fresh_bulb.set_external_force_and_torque(
            torch.zeros_like(forces), torch.zeros((1, 1, 3), device=device), is_global=True
        )
        axial, lateral, tilt = geometry()
        seated = phase() != task_attach._FREE or (
            abs(axial) <= seat_tolerance and lateral < radial_tolerance and tilt < tilt_tolerance
        )
        print(
            f"FORCE_HELD mag={magnitude:g} dir={direction:+.0f} axial {axial0:.4f} -> {axial:.4f} "
            f"lateral={lateral:.4f} tilt0={tilt0:.3f}->tilt={tilt:.3f} phase={phase()} seated={seated} "
            f"trace={trace}",
            flush=True,
        )
        return seated

    def run_force_hard_hold(magnitude: float, direction: float) -> bool:
        """Axial push while lateral and tilt are corrected EXACTLY every step (an exact
        re-alignment, not a spring): snap orientation to the socket's (tilt=0) and position
        onto the seat axis (lateral=0) at the currently-integrated axial distance.

        Unlike the original bayonet-era version of this trial, there is no twist to preserve
        (the new mechanism tracks none), so this hold is a plain project-onto-axis -- simpler
        than before, not a workaround. This isolates whether the geometry permits axial travel
        independent of any orientation controller: if insertion STILL fails here, that is
        direct evidence of a geometry/contact-margin problem, not "the controller was bad".

        This deliberately reproduces the "authority overwrites contact" pattern production code
        (``attach.py``) moved away from -- fine for an isolating diagnostic, never to be copied
        into production as-is.
        """
        settle()
        axial0, _, _ = geometry()
        trace = []
        for i in range(args_cli.steps):
            socket_quat = socket.data.root_quat_w
            axis_w = quat_apply(socket_quat, seat_axis.unsqueeze(0))
            forces = (direction * magnitude * axis_w).unsqueeze(1)
            fresh_bulb.set_external_force_and_torque(
                forces, torch.zeros((1, 1, 3), device=device), is_global=True
            )
            env.step(zero_action)

            socket_quat = socket.data.root_quat_w
            axis_w = quat_apply(socket_quat, seat_axis.unsqueeze(0))
            seat = socket.data.root_pos_w + quat_apply(socket_quat, seat_offset.unsqueeze(0))
            bulb_quat = fresh_bulb.data.root_quat_w
            plug = fresh_bulb.data.root_pos_w + quat_apply(bulb_quat, plug_offset.unsqueeze(0))
            axial_raw = ((plug - seat) * axis_w).sum(dim=1)
            # Sanity bound only (catch a blowup), NOT a gate -- a genuine crossing of any
            # physically meaningful depth must never be masked by this clamp.
            axial_now = axial_raw.clamp(min=-0.1, max=0.1)
            proj_quat = socket_quat  # tilt=0, twist irrelevant (the new mechanism tracks none)
            proj_pos = seat + axial_now.unsqueeze(1) * axis_w - quat_apply(proj_quat, plug_offset.unsqueeze(0))
            fresh_bulb.write_root_pose_to_sim(torch.cat([proj_pos, proj_quat], dim=-1))
            fresh_bulb.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

            if i % 20 == 0 or i == args_cli.steps - 1:
                axial, lateral, tilt = geometry()
                trace.append(
                    (i, round(axial, 4), round(float(axial_raw.item()), 4), round(lateral, 4), round(tilt, 3), phase())
                )
        axial, lateral, tilt = geometry()
        seated = phase() != task_attach._FREE or (
            abs(axial) <= seat_tolerance and lateral < radial_tolerance and tilt < tilt_tolerance
        )
        print(
            f"FORCE_HARD_HOLD mag={magnitude:g} dir={direction:+.0f} axial {axial0:.4f} -> {axial:.4f} "
            f"lateral={lateral:.4f} tilt={tilt:.3f} phase={phase()} seated={seated} trace={trace}",
            flush=True,
        )
        return seated

    def run_pose_drive() -> bool:
        """Control: the way a hand actually inserts it. Must reach the seat, or the run proves
        nothing about contact."""
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
        seated = phase() != task_attach._FREE or (abs(axial) <= seat_tolerance and lateral < radial_tolerance and tilt < tilt_tolerance)
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
    held_seated = False
    for magnitude in (0.5, 2.0, 10.0):
        for direction in (1.0, -1.0):
            held_seated |= run_force_held_orientation(magnitude, direction)
    hard_hold_seated = False
    for magnitude in (0.5, 2.0, 10.0):
        for direction in (1.0, -1.0):
            hard_hold_seated |= run_force_hard_hold(magnitude, direction)
    control_seated = run_pose_drive()

    print(
        f"VERDICT socket_pair_collision=ALWAYS_ON "
        f"contact_seated={contact_seated} held_seated={held_seated} "
        f"hard_hold_seated={hard_hold_seated} control_seated={control_seated}",
        flush=True,
    )
    env.close()
    if not control_seated:
        print("INVALID the pose-drive control failed; this run proves nothing", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    # Same shutdown shape as the original: closing the env does not tear down Kit, and a
    # blanket finally:-close would hang on failure, so hard-exit past it instead. UNLIKE the
    # original, an uncaught exception is printed before that hard exit -- the original's bare
    # try/finally called os._exit(1) (exit_code's untouched default) from inside the finally
    # clause during exception unwinding, which preempts Python's normal unhandled-exception
    # traceback entirely. That silently swallowed the actual error (found while running this
    # for issue #171/#167): the process would exit 1 with zero diagnostic output.
    import os
    import sys
    import traceback

    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code:
            os._exit(exit_code)
        simulation_app.close()
