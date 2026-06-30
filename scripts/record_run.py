# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a Fiatlux run: a video of the rollout and/or a trajectory "bag".

A single rollout produces both artifacts so they describe the *same* run:

- ``video/`` : an MP4 of the run (gymnasium ``RecordVideo``, third-person viewport).
- ``run.h5`` + ``meta.json`` : the experiment bag -- every per-step signal needed to
  score the run offline (see ``scripts/score.py``). ``--format npz`` for a flat fallback.

The policy is anything ``make_policy`` accepts (``zero`` / ``random`` / a TorchScript
``.pt`` / ``rsl_rl[:<ckpt>]``) -- the recorder is policy-agnostic.

Examples:
    python scripts/record_run.py --task FIATLUX-Insert-v0 --policy random \
        --episodes 2 --record both --out logs/runs/random0
    python scripts/record_run.py --task FIATLUX-Insert-v0 --policy logs/.../policy.pt \
        --record bag --episodes 50 --seed 0 --out logs/runs/policyA
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a Fiatlux run (video and/or bag).")
parser.add_argument("--task", type=str, required=True, help="Task / env id.")
parser.add_argument(
    "--policy",
    type=str,
    default="zero",
    help="Policy spec: zero | random | <path>.pt | rsl_rl[:<ckpt>].",
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Checkpoint path for rsl_rl policies."
)
parser.add_argument(
    "--record",
    type=str,
    default="both",
    choices=["video", "bag", "both"],
    help="Which artifacts to produce.",
)
parser.add_argument("--episodes", type=int, default=1, help="Episodes to record.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (bag covers all).")
parser.add_argument("--seed", type=int, default=0, help="Run seed.")
parser.add_argument("--out", type=str, required=True, help="Output directory.")
parser.add_argument(
    "--format", type=str, default="hdf5", choices=["hdf5", "npz"], help="Bag file format."
)
parser.add_argument(
    "--cam",
    type=str,
    default="third_person",
    choices=["third_person", "closeup"],
    help="Viewport camera preset for the video.",
)
parser.add_argument(
    "--video_length", type=int, default=600, help="Video length (env steps)."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Use USD I/O."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

want_video = args_cli.record in ("video", "both")
want_bag = args_cli.record in ("bag", "both")
# Cameras are required to render video frames.
if want_video:
    args_cli.enable_cameras = True
# Headless by default (video still renders via enable_cameras).
args_cli.headless = True if args_cli.headless is None else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import os

import fiatlux_task.tasks  # noqa: F401
import gymnasium as gym
import torch
from fiatlux_task.policy import make_policy
from fiatlux_task.recording import TrajectoryRecorder

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

_CAM_PRESETS = {
    "third_person": {"eye": (2.0, 2.0, 2.0), "lookat": (0.45, 0.0, 1.1)},
    "closeup": {"eye": (0.9, 0.8, 1.4), "lookat": (0.45, 0.0, 1.15)},
}


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    preset = _CAM_PRESETS[args_cli.cam]
    env_cfg.viewer.eye = preset["eye"]
    env_cfg.viewer.lookat = preset["lookat"]

    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if want_video else None
    )

    if want_video:
        video_dir = os.path.join(args_cli.out, "video")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[INFO] recording video to {video_dir}")

    base_env = env.unwrapped

    policy = make_policy(args_cli.policy, base_env, checkpoint=args_cli.checkpoint)
    recorder = (
        TrajectoryRecorder(
            base_env,
            policy_spec=args_cli.policy,
            seed=args_cli.seed,
            checkpoint=args_cli.checkpoint,
        )
        if want_bag
        else None
    )

    obs, _ = env.reset(seed=args_cli.seed)
    episodes_done = 0
    with torch.inference_mode():
        while episodes_done < args_cli.episodes:
            actions = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(actions)
            if recorder is not None:
                recorder.record_step(obs, actions, reward, terminated, truncated)
            episodes_done += int((terminated | truncated).sum().item())

    if recorder is not None:
        info = recorder.write(args_cli.out, fmt=args_cli.format)
        print(f"[INFO] wrote bag {info['bag']} ({info['episodes']} episodes)")
        print(f"[INFO] wrote metadata {info['meta']}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
