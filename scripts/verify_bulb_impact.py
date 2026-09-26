# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate the bulb's impact bound by dropping it into the disposal crate (#138).

Releases the old bulb from a range of heights above the crate with zero velocity and nothing
touching it, so the only contact is the landing. Reports, per drop, the landing speed, the largest
velocity change one control step took (what ``mdp.impact_terms.payload_struck`` measures, free
fall removed), and whether the leg's own impact termination fired.

The robot's root is pinned and the leg's re-seat event is off, so the drop owns the bulb's motion.

Examples
--------
    uv run python scripts/verify_bulb_impact.py --headless --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure landing forces on the bulb in the disposal crate.")
parser.add_argument("--seed", type=int, default=0, help="Layout seed.")
parser.add_argument("--settle-steps", type=int, default=120, help="Control steps to watch after each release.")
parser.add_argument("--release-clearance", type=float, default=0.03, help="Release height above the crate origin.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import fiatlux_task.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.grasp_poses import BULB_IMPACT_SPEED_LIMIT  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "FIATLUX-S06-DisposeBulb-v0"
DROP_HEIGHTS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]

set_layout_seed(args_cli.seed)
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
for name in list(vars(env_cfg.terminations)):
    if not name.startswith("_") and not name.endswith("_bulb_struck"):
        setattr(env_cfg.terminations, name, None)
env_cfg.events.settle_bulb = None
env = gym.make(TASK, cfg=env_cfg).unwrapped

bulb = env.scene["old_bulb"]
crate = env.scene["bin"]
robot = env.scene["robot"]
zero_action = torch.zeros((1, env.action_space.shape[1]), device=env.device)
# The term's own `last_dv`, not a recomputation here: on the step that fires the termination,
# `env.step()` has already reset the bulb to the next episode's state by the time it returns, so a
# `bulb.data.root_lin_vel_w` read afterward is the wrong episode. The term computed `last_dv` from
# the correct pre-reset velocity, during `termination_manager.compute()`, before that reset ran.
old_bulb_struck = env.termination_manager.get_term_cfg("old_bulb_struck").func


def drop(height: float) -> tuple[float, float, float, bool]:
    """Release the bulb ``height`` above the crate; return fall, landing speed, peak dv, gate."""
    env.reset()
    state = bulb.data.root_state_w.clone()
    state[0, :2] = crate.data.root_pos_w[0, :2]
    state[0, 2] = crate.data.root_pos_w[0, 2] + args_cli.release_clearance + height
    release_z = float(state[0, 2])
    bulb.write_root_pose_to_sim(state[:, :7])
    bulb.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))

    pinned_root = robot.data.root_state_w[:, :7].clone()
    previous = bulb.data.root_lin_vel_w[0].clone()
    landing_speed, peak_dv, fired = 0.0, 0.0, False
    resting_z = release_z
    for _ in range(args_cli.settle_steps):
        speed = float(previous.norm())
        resting_z = float(bulb.data.root_pos_w[0, 2])
        robot.write_root_pose_to_sim(pinned_root)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
        env.step(zero_action)
        dv = float(old_bulb_struck.last_dv[0])
        if dv > peak_dv:
            peak_dv, landing_speed = dv, speed
        # env.step auto-resets on termination, so a later read is the next episode's velocity --
        # fine here, since `previous` only feeds next iteration's `speed`, and a termination breaks
        # the loop before that iteration runs.
        previous = bulb.data.root_lin_vel_w[0].clone()
        if bool(env.termination_manager.get_term("old_bulb_struck")[0]):
            fired = True
            break
    return release_z - resting_z, landing_speed, peak_dv, fired


print(f"\n[verify] {TASK}, layout seed {args_cli.seed}, bound {BULB_IMPACT_SPEED_LIMIT} m/s")
print(f"{'fall':>8} {'landing speed':>14} {'peak dv':>9} {'gate':>6}")
for height in DROP_HEIGHTS:
    fall, speed, dv, fired = drop(height)
    print(f"{fall:>7.3f}m {speed:>11.2f} m/s {dv:>6.2f} m/s {'FIRED' if fired else 'ok':>6}")

env.close()
simulation_app.close()
