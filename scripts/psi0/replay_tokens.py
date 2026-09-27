# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay recorded SONIC motion tokens through ``SonicDecoder`` and score the joint tracking.

Checks the ``psi0`` chain below the VLA -- SONIC token space, the in-process release decoder, this
robot's joint order and the 30 Hz -> 50 Hz hold -- against ground truth: an episode of a Psi-0
SONIC LeRobot pack (e.g. ``USC-PSI-Lab/psi-data:sonic/unifolm_sonic_lerobot_val.zip``, which holds
``action.body_token`` and the robot's recorded ``observation.state``), pre-converted to an ``.npz``
with ``tokens`` (N, 64), ``action`` (N, 36; ``[:14]`` = Dex3 targets) and ``state`` (N, 43; legs,
waist, arms, hands). Rows are fed exactly as ``fiatlux_task.psi0`` feeds a chunk; the simulated
joints are compared with the recorded ones per joint group, at the best lag within 0.5 s (a
tracking controller trails its reference).

    python scripts/psi0/replay_tokens.py --npz episode_000008.npz --out replay.json --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay SONIC tokens through SonicDecoder.")
parser.add_argument("--task", type=str, default="FIATLUX-S07-ApproachNewBulb-v0", help="Env to replay in.")
parser.add_argument("--npz", type=str, required=True, nargs="+", help="Episode .npz file(s).")
parser.add_argument("--seconds", type=float, default=None, help="Cap per episode (default: whole episode).")
parser.add_argument("--settle_s", type=float, default=1.0, help="Transient excluded from the error.")
parser.add_argument("--out", type=str, default=None, help="Write the per-episode results as JSON here.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # every FIATLUX scene attaches the ego camera
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

import fiatlux_task.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.groot import SonicDecoder, _StartupBlend  # noqa: E402
from fiatlux_task.psi0 import PSI0_HAND_JOINT_NAMES, PSI0_STATE_JOINT_NAMES, chunk_row, fsq_quantize  # noqa: E402
from fiatlux_task.robots.g1 import swap_robot_variant  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

GROUPS = {"legs": slice(0, 12), "waist": slice(12, 15), "arms": slice(15, 29), "hands": slice(29, 43)}
MAX_LAG_ROWS = 15  # 0.5 s at 30 Hz


def replay(env, npz_path: str) -> dict:
    base = env.unwrapped
    ep = np.load(npz_path)
    tokens, targets, state = ep["tokens"], ep["action"][:, :14], ep["state"]
    robot = base.scene["robot"]
    decoder = SonicDecoder(base)
    blend = _StartupBlend(base, robot, decoder.joint_ids, decoder.sonic_default, decoder.action_scale)
    state_ids = torch.tensor(robot.find_joints(PSI0_STATE_JOINT_NAMES, preserve_order=True)[0], device=base.device)
    hand_ids = torch.tensor(robot.find_joints(PSI0_HAND_JOINT_NAMES, preserve_order=True)[0], device=base.device)

    env.reset(seed=0)
    ticks = -(-len(tokens) * 50 // 30)
    if args_cli.seconds is not None:
        ticks = min(ticks, int(args_cli.seconds * 50))
    sim_q, rows, root_xy, ended = [], [], [], None
    with torch.inference_mode():
        for n in range(ticks):
            row = min(chunk_row(n), len(tokens) - 1)
            mask = blend.mask(base)
            tok = torch.from_numpy(fsq_quantize(tokens[row][None])).to(base.device)
            action = decoder.step(base, tok, hold_mask=mask)
            hand = torch.from_numpy(targets[row]).to(base.device)
            action[:, hand_ids] = (hand - robot.data.default_joint_pos[0, hand_ids]) / decoder.action_scale
            action = blend.override(base, action, mask)
            _, _, terminated, truncated, _ = env.step(action)
            if bool((terminated | truncated).any()):
                ended = n  # the env auto-resets; nothing after this is the same rollout
                break
            sim_q.append(robot.data.joint_pos[0, state_ids].cpu().numpy())
            rows.append(row)
            root_xy.append(robot.data.root_pos_w[0, :2].cpu().numpy())
    sim_q, rows = np.asarray(sim_q), np.asarray(rows)
    keep = np.arange(len(rows)) >= int(args_cli.settle_s * 50)
    result = {
        "npz": os.path.basename(npz_path),
        "rows": int(len(tokens)),
        "ticks_run": int(len(rows)),
        "terminated_at_tick": ended,
    }
    if keep.sum() < 10:
        result["error"] = "too short after the settle window"
        return result
    for name, sl in GROUPS.items():
        best = None
        for lag in range(MAX_LAG_ROWS + 1):
            ref = state[np.clip(rows[keep] - lag, 0, len(state) - 1), sl]
            rmse = float(np.sqrt(np.mean((sim_q[keep, sl] - ref) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, lag)
        motion = float(np.sqrt(np.mean((state[rows[keep], sl] - state[rows[keep], sl].mean(0)) ** 2)))
        result[name] = {"rmse_rad": round(best[0], 4), "lag_rows": best[1], "recorded_motion_rms_rad": round(motion, 4)}
    drift = np.linalg.norm(np.asarray(root_xy)[-1] - np.asarray(root_xy)[0])
    result["root_xy_drift_m"] = round(float(drift), 3)
    return result


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    swap_robot_variant(env_cfg, "dex3")
    env = gym.make(args_cli.task, cfg=env_cfg)
    results = []
    for path in args_cli.npz:
        res = replay(env, path)
        print(json.dumps(res), flush=True)
        results.append(res)
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(results, f, indent=2)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
