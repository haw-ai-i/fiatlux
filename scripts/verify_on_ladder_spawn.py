# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that an on-ladder subtask starts the robot STANDING on the platform, not hanging on the frame.

The step ladder's top frame rises to hip height and narrows toward the front of the platform.
``stand_robot_on_ladder_top`` places the pelvis over the platform and the reset then jitters the
root. A draw that lands far enough forward starts the robot with its hip links inside the frame:
the first physics step answers with kilonewtons on the hips, the feet carry nothing, and the
episode is decided before any action.

One environment, many resets. After each reset the robot is stepped a few times with zero action
and the script records

* where the reset put the root relative to the nominal stance (forward = the way the robot faces,
  toward the fixture; left; yaw),
* the hardest contact on any hip or pelvis link, and
* the vertical load on the feet at the last step.

A reset is ``wedged`` when a hip or pelvis link reads more than ``--wedge_force``. The script
prints every reset sorted by forward offset, the count, and exits non-zero if any reset is wedged.

One layout per run (the layout is drawn when the cfg is parsed): pass ``--layout_seed`` to cover a
wall and a ceiling draw.

Examples
--------
    uv run python scripts/verify_on_ladder_spawn.py --headless
    uv run python scripts/verify_on_ladder_spawn.py --headless --task FIATLUX-S03-RemoveOldBulb-v0 --resets 60
    uv run python scripts/verify_on_ladder_spawn.py --headless --layout_seed 3 --robot inspire
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Verify on-ladder subtasks start the robot clear of the ladder's top frame."
)
parser.add_argument("--task", default="FIATLUX-S12-ClimbDown-v0", help="An on-ladder subtask (S03, S04, S11, S12).")
parser.add_argument("--layout_seed", type=int, default=0, help="Layout seed; one layout per run.")
parser.add_argument("--resets", type=int, default=40, help="Resets to sample.")
parser.add_argument("--steps", type=int, default=3, help="Zero-action steps after each reset.")
parser.add_argument("--robot", default="dex3", choices=["inspire", "dex3"], help="Hand variant.")
parser.add_argument(
    "--wedge_force", type=float, default=1000.0, help="Hip or pelvis contact (N) that counts as wedged."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# The frozen policy observation group carries an RGB term, so the env needs camera rendering.
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import math  # noqa: E402
import re  # noqa: E402

import fiatlux_task.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402

from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# Before parse_env_cfg: the layout is drawn inside it.
set_layout_seed(args_cli.layout_seed)
cfg = parse_env_cfg(args_cli.task, num_envs=1)
if args_cli.robot != "inspire":
    from fiatlux_task.robots.g1 import swap_robot_variant  # noqa: E402

    swap_robot_variant(cfg, args_cli.robot)
cfg.scene.robot.spawn.activate_contact_sensors = True
cfg.scene.whole_body_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1)

nominal_xy = torch.tensor(cfg.scene.robot.init_state.pos[:2])
nominal_quat = torch.tensor([cfg.scene.robot.init_state.rot])

env = gym.make(args_cli.task, cfg=cfg)
base = env.unwrapped
robot = base.scene["robot"]
sensor = base.scene.sensors["whole_body_contact"]
hips = [i for i, n in enumerate(sensor.body_names) if re.search(r"hip_|pelvis", n)]
feet = [i for i, n in enumerate(sensor.body_names) if "ankle_roll" in n]
nominal_xy = nominal_xy.to(base.device)
nominal_quat = nominal_quat.to(base.device)
nominal_yaw = math.atan2(
    2 * float(nominal_quat[0, 0] * nominal_quat[0, 3] + nominal_quat[0, 1] * nominal_quat[0, 2]),
    1 - 2 * float(nominal_quat[0, 2] ** 2 + nominal_quat[0, 3] ** 2),
)
zero = torch.zeros(1, base.action_manager.total_action_dim, device=base.device)

rows = []
for k in range(args_cli.resets):
    env.reset()
    origin = base.scene.env_origins[0]
    root = robot.data.root_pos_w[0] - origin
    d = torch.zeros(1, 3, device=base.device)
    d[0, :2] = root[:2] - nominal_xy
    off = quat_apply_inverse(yaw_quat(nominal_quat), d)[0]  # x = the way the nominal stance faces
    q = robot.data.root_quat_w[0]
    yaw = math.atan2(2 * float(q[0] * q[3] + q[1] * q[2]), 1 - 2 * float(q[2] ** 2 + q[3] ** 2))
    dyaw = math.degrees((yaw - nominal_yaw + math.pi) % (2 * math.pi) - math.pi)
    hip_peak, hip_name = 0.0, ""
    for _ in range(args_cli.steps):
        env.step(zero)
        f = sensor.data.net_forces_w[0, hips].norm(dim=-1)
        if float(f.max()) > hip_peak:
            hip_peak, hip_name = float(f.max()), sensor.body_names[hips[int(f.argmax())]]
    foot_load = float(sensor.data.net_forces_w[0, feet, 2].sum())
    rows.append((float(off[0]) * 1000, float(off[1]) * 1000, dyaw, hip_peak, hip_name, foot_load))

rows.sort(key=lambda r: r[0])
weight = float(robot.root_physx_view.get_masses().sum()) * 9.81
print(
    f"[spawn] {args_cli.task}, {args_cli.robot}, layout seed {args_cli.layout_seed}: {len(rows)} resets, "
    f"{args_cli.steps} zero-action steps each; robot weight {weight:.0f} N"
)
print("[spawn] forward mm | left mm | yaw deg | hardest hip/pelvis contact N | feet load N")
for fwd, left, dyaw, peak, name, load in rows:
    flag = "  WEDGED" if peak > args_cli.wedge_force else ""
    print(f"[spawn] {fwd:+8.1f} | {left:+7.1f} | {dyaw:+6.2f} | {peak:9.0f} {name:22s} | {load:7.0f}{flag}")
wedged = [r[0] for r in rows if r[3] > args_cli.wedge_force]
clear = [r[0] for r in rows if r[3] <= args_cli.wedge_force]
summary = f"[spawn] wedged: {len(wedged)} of {len(rows)}"
if wedged:
    summary += f"; forward offsets of the wedged resets {min(wedged):+.1f} .. {max(wedged):+.1f} mm"
if clear:
    summary += f"; of the clear ones {min(clear):+.1f} .. {max(clear):+.1f} mm"
print(summary)
print("[spawn] " + ("FAIL" if wedged else "PASS"))
env.close()
simulation_app.close()
raise SystemExit(1 if wedged else 0)
