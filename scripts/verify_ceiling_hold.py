# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Does a ceiling-seated bulb hold against gravity, no operator, no contact? (issue #171)

Directly reproduces the real-teleop finding that started this investigation: a ceiling-mounted
fixture's seat axis points straight down, so a seated bulb's own weight acts entirely along the
release direction. Earlier versions of ``mdp.bulb_attachment`` let a ceiling-seated bulb sag
past ``release_threshold`` and fall out unassisted within under a second.

This forces a ceiling mount (regardless of what the layout seed would otherwise draw), spawns
the old bulb already seated at that fixture with zero velocity, and steps forward under a
constant zero action for several real seconds -- no hand, no disturbance, nothing overwriting
its pose or velocity -- reading the manager's own state (phase, axial displacement) every step
straight off the real, contact-resolved simulation. Real hold: phase stays SEATED and axial
stays bounded the whole run. Real drop (the original bug): phase flips to FREE partway through
and axial runs away.

Run via ./pyrun (repo root), not a bare .venv/bin/python: on a machine where .venv is shared
with a different checkout, a bare invocation can silently resolve fiatlux_task to the WRONG
checkout's source -- pyrun fixes that by pointing PYTHONPATH at this repo's own source/ first.

Example
-------
    ./pyrun scripts/verify_ceiling_hold.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify a ceiling-seated bulb holds under gravity alone.")
parser.add_argument("--seed", type=int, default=0, help="Env seed (deterministic).")
parser.add_argument("--seconds", type=float, default=5.0, help="Real simulated seconds to hold and watch.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything else follows."""

import importlib
import sys

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import torch
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import CEILING_FIXTURE_Z, _quat_y_deg, set_layout_seed

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg


def build_cfg():
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=1)
    assert hasattr(cfg.scene, "fresh_bulb"), (
        f"cfg.scene ({type(cfg.scene)} from {sys.modules[type(cfg.scene).__module__].__file__}) has no "
        "fresh_bulb -- fiatlux_task likely resolved to the wrong checkout again; check sys.path/pyrun"
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

    # Force a CEILING mount regardless of what this seed's random draw gave: 180 deg about Y
    # flips the socket's local +z (the seat axis) to point straight down in world frame, the
    # same orientation _sample_fixture_mount uses for its ceiling branch. Position doesn't
    # matter for this test beyond height -- only orientation determines whether gravity acts
    # along the release direction.
    ceiling_quat = _quat_y_deg(180.0)
    x, y, _ = cfg.scene.socket.init_state.pos
    cfg.scene.socket.init_state.pos = (x, y, CEILING_FIXTURE_Z)
    cfg.scene.socket.init_state.rot = ceiling_quat
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
    release_threshold = 0.02  # matches attach.py's default; not exposed on the manager

    def seated_pose() -> tuple[torch.Tensor, torch.Tensor]:
        socket_quat = socket.data.root_quat_w[0]
        seat = socket.data.root_pos_w[0] + quat_apply(socket_quat.unsqueeze(0), seat_offset.unsqueeze(0))[0]
        pos = seat - quat_apply(socket_quat.unsqueeze(0), plug_offset.unsqueeze(0))[0]
        return pos, socket_quat

    def axial() -> float:
        socket_quat = socket.data.root_quat_w
        axis_w = quat_apply(socket_quat, seat_axis.unsqueeze(0))
        seat = socket.data.root_pos_w + quat_apply(socket_quat, seat_offset.unsqueeze(0))
        plug = old_bulb.data.root_pos_w + quat_apply(old_bulb.data.root_quat_w, plug_offset.unsqueeze(0))
        return float(((plug - seat) * axis_w).sum(dim=1).item())

    def phase() -> int:
        return int(manager._phase[task_attach._OLD, 0].item())

    # Spawn the old bulb EXACTLY at the seat, aligned with the (now ceiling) socket, at rest --
    # matching how a real ceiling-mounted Replace scene starts (old_bulb pre-seated). Written
    # once, before the loop; nothing overwrites it again after this.
    pos, quat = seated_pose()
    old_bulb.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))
    old_bulb.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    env.step(zero_action)  # resolves spawn phase -> SEATED, since it's within tolerance of the seat
    print(f"SETUP mount=ceiling axial0={axial():.4f} phase0={phase()} (expect phase0=1/SEATED)", flush=True)
    if phase() != task_attach._SEATED:
        print("FATAL did not seat at spawn -- the setup itself is wrong, not testing retention", flush=True)
        return 1

    steps = int(round(args_cli.seconds / env.step_dt))
    trace = []
    max_abs_axial = 0.0
    dropped_at = None
    for i in range(steps):
        env.step(zero_action)  # no hand, no disturbance -- gravity + real contact + retention only
        a, p = axial(), phase()
        max_abs_axial = max(max_abs_axial, abs(a))
        if p != task_attach._SEATED and dropped_at is None:
            dropped_at = i * env.step_dt
        if i % max(1, steps // 20) == 0 or i == steps - 1:
            trace.append((round(i * env.step_dt, 3), round(a, 4), p))

    held = dropped_at is None and max_abs_axial <= release_threshold
    print(
        f"RESULT held={held} max_abs_axial={max_abs_axial:.4f} release_threshold={release_threshold:.4f} "
        f"dropped_at={dropped_at} final_phase={phase()} trace={trace}",
        flush=True,
    )
    env.close()
    return 0 if held else 1


if __name__ == "__main__":
    import os
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
