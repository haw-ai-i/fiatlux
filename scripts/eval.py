# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standardized Fiatlux benchmark evaluation.

Runs a policy for a fixed number of episodes from a fixed seed and reports the
benchmark metrics as JSON:

- ``success_rate``        : fraction of episodes ending in the task's ``success``
                            termination (task-agnostic: whatever the env defines).
- ``mean_episode_length`` : average steps per episode.
- ``mean_control_effort`` : average sum-of-squared actions per step.
- ``peak_contact_force``  : max net hand contact force seen (N).
- ``score_breakdown``     : per-term episode means from the env's own reward /
                            termination managers (``Episode_Reward/<term>`` is the
                            episodic sum averaged per second; ``Episode_Termination/
                            <term>`` is the fraction of episodes that term ended).
                            For ``FIATLUX-Replace-v0`` this is the benchmark's score
                            breakdown: dense normalized progress, sparse completions,
                            penalties, and full success as separate named channels.

Determinism: same ``--task``, ``--seed`` and ``--policy`` give the same numbers.

Examples:
    python scripts/eval.py --task FIATLUX-S08-GrabNewBulb-v0 --policy zero --episodes 20
    python scripts/eval.py --task FIATLUX-Replace-v0 --policy basic_standard \
        --episodes 2 --enable_cameras
    python scripts/eval.py --task FIATLUX-Replace-v0 --policy basic_cheatcode \
        --episodes 2 --enable_cameras
    python scripts/eval.py --task FIATLUX-Replace-v0 --policy rsl_rl \
        --checkpoint logs/rsl_rl/fiatlux_replace/<run>/model_*.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# NOT fiatlux_task.policy: importing anything under the fiatlux_task package runs
# fiatlux_task/__init__.py's `from .tasks import *`, which -- even no-oping pre-Kit -- still
# leaves torch imported and cached along the way. See policy_cli_help.py's own docstring.
from policy_cli_help import add_policy_cli_args

parser = argparse.ArgumentParser(description="Fiatlux benchmark evaluation.")
parser.add_argument("--task", type=str, required=True, help="Task / env id.")
add_policy_cli_args(parser)
parser.add_argument(
    "--no_randomize",
    action="store_true",
    default=False,
    help="Deterministic canonical spawns: strip the task's reset-time randomization terms.",
)
parser.add_argument("--episodes", type=int, default=20, help="Episodes to evaluate.")
parser.add_argument("--num_envs", type=int, default=None, help="Parallel envs.")
parser.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Use USD I/O.")
# Benchmark telemetry flags (--wandb, --wandb_project, ...); mirrors fiatlux_task.telemetry.
parser.add_argument("--wandb", action="store_true", default=False, help="Stream the score breakdown to wandb.")
parser.add_argument("--wandb_project", type=str, default="fiatlux", help="wandb project name.")
parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity (team/user).")
parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Evaluation is always headless. (AppLauncher's --headless is store_true with default
# False -- never None -- so the old `if args_cli.headless is None` guard was dead code:
# on a machine without a display the app then launched in GUI mode and Kit spun a CPU
# core forever waiting on a window that cannot exist. For interactive viewing use
# --livestream, which works alongside headless.)
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import json
import os
import random

import fiatlux_task.tasks  # noqa: F401
import gymnasium as gym
import torch
from fiatlux_task.policy import make_policy
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed
from fiatlux_task.telemetry import ScoreLogger

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    # The replace preset's room layout is drawn at cfg-build time, so its seed must be
    # declared before parse_env_cfg -- this is what puts the layout under the benchmark's
    # same-task/seed/policy -> same-numbers contract.
    set_layout_seed(args_cli.seed)
    random.seed(args_cli.seed)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    if args_cli.robot != "inspire":
        from fiatlux_task.robots.g1 import swap_robot_variant

        swap_robot_variant(env_cfg, args_cli.robot)
    if args_cli.no_randomize:
        env_cfg.disable_randomization()
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    policy = make_policy(args_cli.policy, env, checkpoint=args_cli.checkpoint, instruction=args_cli.instruction)
    # All metric definitions (success, episode stats, score breakdown) live in
    # fiatlux_task.telemetry; this loop only feeds it raw step artifacts.
    score_logger = ScoreLogger.from_args(args_cli)

    obs, _ = env.reset(seed=args_cli.seed)
    step_in_ep = torch.zeros(env.num_envs, device=env.device)

    target = args_cli.episodes
    with torch.inference_mode():
        while score_logger.episodes_done < target:
            actions = policy(obs)
            obs, _, terminated, truncated, extras = env.step(actions)
            done = terminated | truncated
            step_in_ep += 1
            score_logger.step(
                env,
                extras,
                done,
                episode_lengths=step_in_ep,
                actions=actions,
                max_episodes=target,
                # Optional policy-side diagnostics (a policy may refresh `policy.info`
                # per call, e.g. a critic value estimate); never required.
                policy_info=getattr(policy, "info", None),
            )
            step_in_ep[done] = 0

    results = score_logger.close()
    print(json.dumps(results, indent=2))
    if args_cli.output:
        os.makedirs(os.path.dirname(args_cli.output) or ".", exist_ok=True)
        with open(args_cli.output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[INFO] wrote {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
