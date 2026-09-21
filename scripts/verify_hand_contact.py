"""Does each hand's contact channel actually report force? (issue #89)

A zero-action rollout records all-zero contact for both hands, which is exactly what a dead
sensor records. That is the failure mode #89 exists to remove -- zero force reading as "not
touching" -- so verifying the fix with a no-contact run proves nothing.

This presses the bulb into each palm in turn and reads both channels. The right hand is the
control: it worked before this change, so if the left reports force where the right does, the
plumbing is real.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Prove each hand's contact channel reports force.")
parser.add_argument("--task", type=str, default="FIATLUX-Remove-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import fiatlux_task.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.robots.g1 import G1_PALM_BODIES  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import observations as _obs  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def peak(sensor) -> float:
    return float(torch.norm(_obs.object_contact_forces(sensor), dim=-1).max())


def main() -> int:
    set_layout_seed(0)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    bulb = env.scene["old_bulb" if "old_bulb" in env.scene.rigid_objects else "fresh_bulb"]
    right_sensor = env.scene.sensors["hand_contact"]
    left_sensor = env.scene.sensors.get("left_hand_contact")
    if left_sensor is None:
        print("FAIL no left_hand_contact sensor on this scene", flush=True)
        return 1

    results = {}
    for label, body in (("left", G1_PALM_BODIES[0]), ("right", G1_PALM_BODIES[1])):
        env.reset()
        palm_idx = robot.find_bodies(body)[0][0]
        best_left = best_right = 0.0
        # Walk the bulb into the palm and hold it there. Contact needs a few steps to build.
        for _ in range(40):
            palm = robot.data.body_pos_w[:, palm_idx, :]
            pose = torch.cat([palm, bulb.data.root_quat_w], dim=-1)
            bulb.write_root_pose_to_sim(pose)
            bulb.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
            env.step(torch.zeros(env.action_space.shape, device=env.device))
            best_left = max(best_left, peak(left_sensor))
            best_right = max(best_right, peak(right_sensor))
        results[label] = (best_left, best_right)
        print(
            f"bulb in {label:5s} palm -> left channel {best_left:8.3f} N | right channel {best_right:8.3f} N",
            flush=True,
        )

    left_in_left = results["left"][0]
    right_in_right = results["right"][1]
    # The left channel must see the bulb in the left palm. The right is the control: it worked
    # before this change, so it says whether the rig itself produces contact at all.
    ok = left_in_left > 0.01 and right_in_right > 0.01
    print(
        f"\nVERDICT {'PASS' if ok else 'FAIL'} -- left channel {left_in_left:.3f} N with the bulb "
        f"in the left palm, right channel {right_in_right:.3f} N with it in the right",
        flush=True,
    )
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    import os
    import sys

    code = 1
    try:
        code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if code:
            os._exit(code)
        simulation_app.close()
