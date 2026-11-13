# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop check of encoded SONIC tokens: replay a teleop take's tokens through SonicDecoder in sim.

``encode_sonic_tokens.py`` turns a take's executed motion into GEAR-SONIC tokens; this drives the
Dex3 G1 with exactly those tokens (50 Hz, one per recorded step) plus the take's commanded finger
targets, from the same task and layout seed, and reports how closely the robot re-traces the demo
(mean absolute joint error, whole body and arms) and which termination ended the replay. If the
encoding is right, a replay of a successful dispose take should put the bulb in the crate too.

    python scripts/psi0/replay_tokens.py --npz ~/psi0_ft/encoded/success/<take>.npz \
        --meta ~/psi0_ft/raw/success/<take>/meta.json --out ~/psi0_ft/replay/<take> --headless --enable_cameras
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="FIATLUX-S06-DisposeBulb-v0")
parser.add_argument("--npz", required=True)
parser.add_argument("--meta", required=True, help="the take's meta.json (layout seed)")
parser.add_argument("--out", required=True)
parser.add_argument("--video", action="store_true", help="also write the ego-camera video")
parser.add_argument("--extra_steps", type=int, default=100, help="steps to hold the last token after the take ends")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import os  # noqa: E402

import fiatlux_task.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.groot import SONIC_JOINT_NAMES, SonicDecoder, _StartupBlend  # noqa: E402
from fiatlux_task.policy import prepare_env_cfg  # noqa: E402
from fiatlux_task.psi0 import PSI0_HAND_JOINT_NAMES, PSI0_STATE_JOINT_NAMES  # noqa: E402
from fiatlux_task.recording import term_flag  # noqa: E402
from fiatlux_task.robots.g1 import swap_robot_variant  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    with open(args_cli.meta) as f:
        seed = int(json.load(f)["seed"])
    data = np.load(args_cli.npz)
    tokens, hands, demo_state = data["token"], data["hand_target"], data["state"]
    set_layout_seed(seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = seed
    swap_robot_variant(env_cfg, "dex3")
    prepare_env_cfg("psi0", env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset(seed=seed)
    robot = base.scene["robot"]
    decoder = SonicDecoder(base)
    blend = _StartupBlend(base, robot, decoder.joint_ids, decoder.sonic_default, decoder.action_scale)
    hand_ids = torch.tensor(robot.find_joints(PSI0_HAND_JOINT_NAMES, preserve_order=True)[0], device=base.device)
    body_ids = torch.tensor(robot.find_joints(SONIC_JOINT_NAMES, preserve_order=True)[0], device=base.device)
    state_ids = torch.tensor(robot.find_joints(PSI0_STATE_JOINT_NAMES, preserve_order=True)[0], device=base.device)
    # demo_state is in psi0 order; bring its 29 body joints into SONIC (IsaacLab) order to compare.
    to_sonic = [PSI0_STATE_JOINT_NAMES.index(n) for n in SONIC_JOINT_NAMES]
    arm = [i for i, n in enumerate(SONIC_JOINT_NAMES) if any(k in n for k in ("shoulder", "elbow", "wrist"))]

    video = None
    if args_cli.video:
        from fiatlux_task.viz import VideoRecorder

        video = VideoRecorder(base, base.scene["ego_camera"], os.path.join(args_cli.out, "ego.mp4"), fps=50)
    start = robot.data.joint_pos[0, state_ids].cpu().numpy()
    start_err = float(np.abs(start[:29] - demo_state[0, :29]).mean())
    n_total = len(tokens) + args_cli.extra_steps
    q_log, ended_by = [], None
    with torch.inference_mode():
        for t in range(n_total):
            k = min(t, len(tokens) - 1)
            mask = blend.mask(base)
            action = decoder.step(base, torch.from_numpy(tokens[k][None]).to(base.device), hold_mask=mask)
            target = torch.from_numpy(hands[k]).to(base.device)
            action[:, hand_ids] = (target - robot.data.default_joint_pos[0, hand_ids]) / decoder.action_scale
            action = blend.override(base, action, mask)
            _, _, terminated, truncated, _ = env.step(action)
            if video is not None:
                video.capture()
            if bool((terminated | truncated)[0]):
                terms = base.termination_manager.active_terms
                fired = [n for n in terms if bool(term_flag(base, n, 1, base.device)[0])]
                ended_by = (t, fired)
                break
            q_log.append(robot.data.joint_pos[0, body_ids].cpu().numpy())
    q = np.array(q_log)
    n = min(len(q), len(tokens) - 1)
    demo = demo_state[1 : n + 1][:, to_sonic]
    err = np.abs(q[:n] - demo)
    report = {
        "take": os.path.basename(args_cli.npz),
        "seed": seed,
        "demo_steps": len(tokens),
        "replayed_steps": len(q),
        "start_pose_err_rad": start_err,
        "mean_abs_joint_err_rad": float(err.mean()),
        "mean_abs_arm_err_rad": float(err[:, arm].mean()),
        "demo_arm_motion_rad": float(np.abs(demo[:, arm] - demo[:1, arm]).mean()),
        "ended_by": ended_by,
    }
    os.makedirs(args_cli.out, exist_ok=True)
    with open(os.path.join(args_cli.out, "replay.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("REPLAY", json.dumps(report))
    if video is not None and len(video):
        video.write()
    env.close()


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
