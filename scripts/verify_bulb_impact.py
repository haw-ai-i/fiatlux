# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate the bulb's impact bound by dropping it into the disposal crate (#138).

Releases the old bulb from a range of heights above the crate with zero velocity and nothing
touching it, so the only contact is the landing. Reports, per drop, the landing speed, the largest
velocity change one control step took (what ``mdp.impact_terms.payload_struck`` measures, free
fall removed), and the leg's own impact termination status: ``FIRED``, a clean ``ok``, or
``interrupted`` where the scene's other bulb's termination cut the drop short first.

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
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp.pre_reset_snapshot import (  # noqa: E402
    snapshot_before_reset,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402

from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "FIATLUX-S06-DisposeBulb-v0"
DROP_HEIGHTS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]

set_layout_seed(args_cli.seed)
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
for name in list(vars(env_cfg.terminations)):
    if not name.startswith("_") and not name.endswith("_bulb_struck"):
        setattr(env_cfg.terminations, name, None)
env_cfg.events.settle_bulb = None
# Watches the bulb's height across the same pre-reset boundary `old_bulb_struck` already watches
# its velocity across (#255): both read whatever a script needs from the exact step a termination
# fires, before that step's own auto-reset overwrites it with the next episode's state.
env_cfg.terminations.bulb_snapshot = DoneTerm(
    func=snapshot_before_reset, params={"fields": (("old_bulb", "root_pos_w"),)}
)
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
bulb_snapshot = env.termination_manager.get_term_cfg("bulb_snapshot").func


def drop(height: float) -> tuple[float, float, float, str]:
    """Release the bulb ``height`` above the crate; return fall, landing speed, peak dv, status.

    Status is ``"FIRED"`` (old_bulb_struck tripped), ``"ok"`` (the full settle window ran clean),
    or ``"interrupted"`` (some other termination -- in practice fresh_bulb_struck -- ended the
    episode first, so old_bulb was never necessarily observed long enough to judge).
    """
    env.reset()
    state = bulb.data.root_state_w.clone()
    state[0, :2] = crate.data.root_pos_w[0, :2]
    state[0, 2] = crate.data.root_pos_w[0, 2] + args_cli.release_clearance + height
    release_z = float(state[0, 2])
    bulb.write_root_pose_to_sim(state[:, :7])
    bulb.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))

    pinned_root = robot.data.root_state_w[:, :7].clone()
    landing_speed, peak_dv, status = 0.0, 0.0, "ok"
    resting_z = release_z
    for _ in range(args_cli.settle_steps):
        # Live read, not a carried-over clone: nothing writes to the bulb between one iteration's
        # env.step() and the next iteration's read, so this always matches what a stored value from
        # last iteration would have held.
        speed = float(bulb.data.root_lin_vel_w[0].norm())
        robot.write_root_pose_to_sim(pinned_root)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
        env.step(zero_action)
        # `bulb_snapshot`, not a live `bulb.data.root_pos_w` read: on the step that fires a
        # termination, env.step()'s internal auto-reset has already run by the time it returns, so
        # a read here would land on the next episode's spawn height, not the impact height.
        # `bulb_snapshot` cached the height during `termination_manager.compute()`, before that
        # reset ran -- same trick `old_bulb_struck.last_dv` uses for velocity, generalized (#255).
        resting_z = float(bulb_snapshot.snapshot["old_bulb.root_pos_w"][0, 2])
        dv = float(old_bulb_struck.last_dv[0])
        if dv > peak_dv:
            peak_dv, landing_speed = dv, speed
        # `.dones`, not a name-specific `get_term("old_bulb_struck")`: the Replace preset this task
        # uses spawns a `fresh_bulb` too, so `fresh_bulb_struck` is a second live termination that
        # can also auto-reset this env (e.g. solver jitter on the parked bulb). Missing that would
        # keep looping past the reset, reading the next episode's state as if nothing happened.
        if bool(env.termination_manager.dones[0]):
            # Distinguish which termination actually broke the loop (issue #256): old_bulb_struck
            # means we have a real answer; anything else (fresh_bulb_struck) means old_bulb was cut
            # off before it was necessarily observed long enough to judge, so it's not a genuine ok.
            if bool(env.termination_manager.get_term("old_bulb_struck")[0]):
                status = "FIRED"
            else:
                status = "interrupted"
            break
    return release_z - resting_z, landing_speed, peak_dv, status


print(f"\n[verify] {TASK}, layout seed {args_cli.seed}, bound {BULB_IMPACT_SPEED_LIMIT} m/s")
print(f"{'fall':>8} {'landing speed':>14} {'peak dv':>9} {'gate':>11}")
for height in DROP_HEIGHTS:
    fall, speed, dv, status = drop(height)
    print(f"{fall:>7.3f}m {speed:>11.2f} m/s {dv:>6.2f} m/s {status:>11}")

env.close()
simulation_app.close()
