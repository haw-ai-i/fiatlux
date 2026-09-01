# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact force against bulb-to-palm distance (issue #92).

``diagnose_stuck_bulb.py`` reports 181-223x the bulb's weight in a hand whose fingers are open. It
also parks the bulb at the palm body's ORIGIN, and the bulb's radius is about 39 mm, so the bulb
probably encloses the palm geometry: those forces may be the solver depenetrating an overlap the
script created rather than anything about the hand.

This decides it. Hold the bulb at a series of distances from the palm and read the hand-bulb
contact force at each.

* Force decays sharply as the offset grows -> the high readings are interpenetration. The scripted
  stuck-bulb result is an artifact of its own placement, and the recording is the only evidence.
* Force stays at tens of newtons with the bulb well clear of the hand -> the contact pathology is
  real, and it belongs in the bulb or hand collision setup.

The bulb is held at each offset rather than dropped. The question is how hard the solver pushes at
a given separation, which is what holding it measures.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep hand-bulb contact force against distance (issue #92).")
parser.add_argument("--task", type=str, default="FIATLUX-Remove-v0")
parser.add_argument("--variant", type=str, default="dex3", choices=["dex3", "inspire"])
parser.add_argument("--hand", type=str, default="right", choices=["left", "right"])
parser.add_argument(
    "--offsets",
    type=float,
    nargs="+",
    default=[0.0, 0.01, 0.02, 0.04, 0.06, 0.10],
    help="Metres from the palm body origin. 0 is where diagnose_stuck_bulb.py parks it.",
)
parser.add_argument("--settle", type=int, default=20, help="Steps to hold at each offset before reading.")
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
from fiatlux_task.robots.g1 import G1_DEX3_PALM_BODIES, G1_PALM_BODIES, swap_robot_variant  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import observations as _obs  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import BULB_MASS_KG, set_layout_seed  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_LEFT, _RIGHT = 0, 1


def main() -> int:
    set_layout_seed(0)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if args_cli.variant != "inspire":
        swap_robot_variant(cfg, args_cli.variant)
    # Move the furniture aside rather than deleting it -- event and observation terms reference
    # these entities by name. Pin the root: a free base under zero actions sags, and a moving palm
    # would change the offset being swept.
    for entity in ("table", "bin", "ladder", "socket", "pendant", "fixture"):
        item = getattr(cfg.scene, entity, None)
        if item is not None and getattr(item, "init_state", None) is not None:
            pos = item.init_state.pos
            item.init_state.pos = (pos[0] + 20.0, pos[1] + 20.0, pos[2])
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    bulb = env.scene["old_bulb" if "old_bulb" in env.scene.rigid_objects else "fresh_bulb"]
    palms = G1_DEX3_PALM_BODIES if args_cli.variant == "dex3" else G1_PALM_BODIES
    palm_idx = robot.find_bodies(palms[_LEFT if args_cli.hand == "left" else _RIGHT])[0][0]
    sensors = [n for n in ("hand_contact", "left_hand_contact") if n in env.scene.sensors]

    action = torch.zeros(env.action_space.shape, device=env.device)
    zeros6 = torch.zeros((1, 6), device=env.device)
    weight = BULB_MASS_KG * 9.81
    up = torch.tensor([0.0, 0.0, 1.0], device=env.device)

    print(f"SETUP variant={args_cli.variant} hand={args_cli.hand} bulb weight={weight:.3f} N", flush=True)
    print("  offset   peak force      x weight   note", flush=True)

    results = []
    for offset in args_cli.offsets:
        peak = 0.0
        for _ in range(args_cli.settle):
            # Hold the bulb at the offset every step. Measuring how hard the solver pushes at a
            # fixed separation is the whole question, and holding is what exposes it.
            target = robot.data.body_pos_w[:, palm_idx, :] + offset * up
            bulb.write_root_pose_to_sim(torch.cat([target, bulb.data.root_quat_w], dim=-1))
            bulb.write_root_velocity_to_sim(zeros6)
            env.step(action)
            peak = max(
                peak,
                max(float(torch.norm(_obs.object_contact_forces(env.scene.sensors[s]), dim=-1).max()) for s in sensors),
            )
        results.append((offset, peak))
        note = "inside the palm" if offset < 0.039 else "bulb radius clears the origin"
        print(f"  {offset * 1000:5.0f} mm  {peak:9.2f} N  {peak / weight:9.0f}x   {note}", flush=True)

    # Report the SHAPE of the curve, not its endpoints. An earlier version compared the first and
    # last readings and printed "DECAYS" whenever the last one was small -- but the largest offset
    # is where the bulb is out of reach of the hand entirely, so its reading is zero for a reason
    # that has nothing to do with how the solver behaves in contact. That comparison called a
    # curve holding 100-330 N through 60 mm a decay.
    touching = [(off, f) for off, f in results if f > 1.0]
    print("\nVERDICT", flush=True)
    if not touching:
        print("  NO CONTACT at any offset. Nothing was measured; move the bulb closer.", flush=True)
    else:
        widest, force_at_widest = touching[-1]
        print(
            f"  contact at {len(touching)}/{len(results)} offsets, out to {widest * 1000:.0f} mm, "
            f"where it reads {force_at_widest:.2f} N = {force_at_widest / weight:.0f}x the bulb's weight",
            flush=True,
        )
        gentlest = min(f for _, f in touching)
        if gentlest < 5.0 * weight:
            print(
                f"  There IS a gentle regime: the lightest contact reads {gentlest:.2f} N, within a "
                "few times the bulb's weight. Contact force tracks separation, as it should.",
                flush=True,
            )
        else:
            print(
                f"  There is NO gentle regime: even the lightest contact reads {gentlest:.2f} N = "
                f"{gentlest / weight:.0f}x the bulb's weight. Every touch is violent.",
                flush=True,
            )
    print(
        "\n  CAVEAT this sweep HOLDS the bulb in place every step, and pose-writing a body that is\n"
        "  already in contact is itself the pathology #77 measured at 1067 N on the socket. So a\n"
        "  large reading here cannot be blamed on the hand alone. To separate the two, place the\n"
        "  bulb at an offset that reads contact, then stop writing its pose and watch what happens.",
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
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code:
            os._exit(exit_code)
        simulation_app.close()
