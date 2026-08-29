# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Does a released bulb fall out of the hand? (issue #92)

Yujin reported on 2026-08-21 that the old bulb stays wedged in the hand after it detaches, and
opening the fingers does not drop it. The bags confirm the behaviour -- in the fail take the bulb
never comes to rest, is still moving 24 s later, and rises 11 cm while the fingers open -- but they
cannot give a cause. The bulb is ``FREE`` by then, so the state machine is not holding it. This is
hand-versus-bulb physics, and the bags carry no left-hand contact force (they predate #89).

So this reproduces the release in isolation, with no state machine, no operator and no ladder:

1. Park the bulb at the palm and close the fingers on it.
2. Hold, so contact settles.
3. Open the fingers.
4. Watch what the bulb does for a second afterwards.

A bulb that leaves reports a falling z and a growing distance from the palm. A bulb that sticks
holds its distance while the fingers are demonstrably open -- which is the report.

``--variant`` matters: the session ran Dex3, and the two hands differ in geometry, not only in
joint names. ``--hand`` matters too, because the session drove the bulb left-handed and the grasp
presets are mirrored rather than identical.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reproduce the stuck-bulb release (issue #92).")
parser.add_argument("--task", type=str, default="FIATLUX-Remove-v0")
parser.add_argument("--variant", type=str, default="dex3", choices=["dex3", "inspire"])
parser.add_argument("--hand", type=str, default="left", choices=["left", "right"])
parser.add_argument("--settle-steps", type=int, default=60, help="Steps to hold the closed grasp.")
parser.add_argument("--release-steps", type=int, default=60, help="Steps to watch after opening.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else follows."""

import fiatlux_task.tasks  # noqa: F401, E402  -- registers the FIATLUX Gym environments
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.robots.g1 import (  # noqa: E402
    G1_DEX3_HAND_GRASP,
    G1_DEX3_LEFT_HAND_GRASP,
    G1_DEX3_PALM_BODIES,
    G1_HAND_GRASP,
    G1_PALM_BODIES,
    swap_robot_variant,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import observations as _obs  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_LEFT, _RIGHT = 0, 1


def grasp_targets(variant: str, hand: str) -> dict[str, float]:
    """The closed-hand joint preset for this variant and side.

    The Dex3 presets are MIRRORED, not shared: right fingers curl toward + and left toward -, so
    applying the right-hand preset to the left hand opens it instead of closing it.
    """
    if variant == "dex3":
        return dict(G1_DEX3_LEFT_HAND_GRASP if hand == "left" else G1_DEX3_HAND_GRASP)
    if hand == "left":
        return {j.replace("R_", "L_", 1): v for j, v in G1_HAND_GRASP.items()}
    return dict(G1_HAND_GRASP)


def main() -> int:
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # The scene is authored Inspire, and swapping a variant onto itself trips the remap guard on
    # the task's own hand-scoped action term. Only swap when the target differs.
    if args_cli.variant != "inspire":
        swap_robot_variant(cfg, args_cli.variant)
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    bulb = env.scene["old_bulb" if "old_bulb" in env.scene.rigid_objects else "fresh_bulb"]
    palms = G1_DEX3_PALM_BODIES if args_cli.variant == "dex3" else G1_PALM_BODIES
    palm_idx = robot.find_bodies(palms[_LEFT if args_cli.hand == "left" else _RIGHT])[0][0]

    targets = grasp_targets(args_cli.variant, args_cli.hand)
    joint_ids = [robot.find_joints(name)[0][0] for name in targets]
    closed = torch.tensor([[v for v in targets.values()]], device=env.device)
    opened = torch.zeros_like(closed)
    zeros6 = torch.zeros((env.num_envs, 6), device=env.device)
    action = torch.zeros(env.action_space.shape, device=env.device)

    sensors = [n for n in ("hand_contact", "left_hand_contact") if n in env.scene.sensors]

    def hold(joint_target: torch.Tensor, park_bulb: bool) -> None:
        robot.set_joint_position_target(joint_target, joint_ids=joint_ids)
        if park_bulb:
            palm = robot.data.body_pos_w[:, palm_idx, :]
            bulb.write_root_pose_to_sim(torch.cat([palm, bulb.data.root_quat_w], dim=-1))
            bulb.write_root_velocity_to_sim(zeros6)
        env.step(action)

    def report() -> tuple[float, float, float, float, float]:
        palm = robot.data.body_pos_w[0, palm_idx, :]
        pos = bulb.data.root_pos_w[0]
        dist = float(torch.norm(pos - palm)) * 1000.0
        speed = float(torch.norm(bulb.data.root_lin_vel_w[0])) * 1000.0
        force = max(
            (float(torch.norm(_obs.object_contact_forces(env.scene.sensors[s]), dim=-1).max())) for s in sensors
        )
        # The ACTUAL finger angles, not the commanded target. "The bulb did not fall" has a
        # trivial explanation -- the fingers never opened -- and nothing else here rules it out.
        curl = float(robot.data.joint_pos[0, joint_ids].abs().max())
        return dist, float(pos[2]), speed, force, curl

    print(f"SETUP variant={args_cli.variant} hand={args_cli.hand} task={args_cli.task}", flush=True)

    # 1-2. close on the bulb and let contact settle, holding the bulb at the palm meanwhile
    for _ in range(args_cli.settle_steps):
        hold(closed, park_bulb=True)
    grasped = report()
    print(
        f"GRASPED   dist={grasped[0]:6.1f} mm  z={grasped[1]:.4f} m  "
        f"|v|={grasped[2]:6.1f} mm/s  force={grasped[3]:8.2f} N  max|curl|={grasped[4]:.3f} rad",
        flush=True,
    )

    # 3. release: the bulb is now physics-owned, nothing writes its pose again
    for step_index in range(args_cli.release_steps):
        hold(opened, park_bulb=False)
        if step_index in (0, 4, 9, 19, 39, args_cli.release_steps - 1):
            dist, z, speed, force, curl = report()
            print(
                f"  +{step_index + 1:3d} steps  dist={dist:6.1f} mm  z={z:.4f} m  "
                f"|v|={speed:6.1f} mm/s  force={force:8.2f} N  max|curl|={curl:.3f} rad",
                flush=True,
            )

    final = report()
    dropped_mm = (grasped[1] - final[1]) * 1000.0
    moved_mm = final[0] - grasped[0]
    # Three outcomes, not two. An earlier version scored "moved far from the palm" as RELEASED,
    # which called a bulb flung upward across the room a successful drop.
    if dropped_mm > 50.0:
        verdict = "RELEASED (fell)"
    elif moved_mm > 50.0:
        verdict = "EJECTED (left the hand without falling -- flung, not dropped)"
    else:
        verdict = "STUCK"
    print(
        f"\nVERDICT {verdict} -- dropped {dropped_mm:+.1f} mm, moved {moved_mm:+.1f} mm from the "
        f"palm, fingers at max|curl|={final[4]:.3f} rad (0 = fully open)",
        flush=True,
    )
    env.close()
    return 0


if __name__ == "__main__":
    import os
    import sys

    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        # os._exit below skips the interpreter's traceback printing.
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code:
            os._exit(exit_code)
        simulation_app.close()
