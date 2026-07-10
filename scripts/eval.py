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
    python scripts/eval.py --task FIATLUX-Insert-v0 --policy zero --episodes 20
    python scripts/eval.py --task FIATLUX-Replace-v0 --policy basic_standard \
        --episodes 2 --enable_cameras
    python scripts/eval.py --task FIATLUX-Replace-v0 --policy basic_cheatcode \
        --episodes 2 --enable_cameras
    python scripts/eval.py --task FIATLUX-Insert-v0 --policy rsl_rl \
        --checkpoint logs/rsl_rl/fiatlux_task/<run>/model_*.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fiatlux benchmark evaluation.")
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
parser.add_argument("--episodes", type=int, default=20, help="Episodes to evaluate.")
parser.add_argument("--num_envs", type=int, default=None, help="Parallel envs.")
parser.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Use USD I/O."
)
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

import fiatlux_task.tasks  # noqa: F401
import gymnasium as gym
import torch
from fiatlux_task.policy import make_policy
from fiatlux_task.recording import term_flag

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    policy = make_policy(args_cli.policy, env, checkpoint=args_cli.checkpoint)

    successes = 0
    episodes_done = 0
    ep_lengths: list[int] = []
    control_efforts: list[float] = []
    peak_contact = 0.0
    # Weighted sums of the managers' per-episode term stats (each `log` entry is already
    # averaged over the envs that reset that step, so weight by how many we counted).
    breakdown_sums: dict[str, float] = {}
    breakdown_counts: dict[str, int] = {}

    obs, _ = env.reset(seed=args_cli.seed)
    step_in_ep = torch.zeros(env.num_envs, device=env.device)

    target = args_cli.episodes
    with torch.inference_mode():
        while episodes_done < target:
            actions = policy(obs)
            obs, _, terminated, truncated, extras = env.step(actions)
            done = terminated | truncated
            step_in_ep += 1

            control_efforts.append(float(torch.mean(torch.sum(actions**2, dim=-1))))
            if "hand_contact" in env.scene.sensors:
                f = env.scene.sensors["hand_contact"].data.net_forces_w
                peak_contact = max(peak_contact, float(torch.norm(f, dim=-1).max()))

            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            if len(done_ids) > 0:
                # The env's own `success` termination term is the task-agnostic verdict
                # (valid for the terminating step; survives the in-step auto-reset).
                success = term_flag(env, "success", env.num_envs, env.device)
                counted = 0
                for i in done_ids.tolist():
                    if episodes_done >= target:
                        break
                    successes += int(bool(success[i].item()))
                    ep_lengths.append(int(step_in_ep[i].item()))
                    episodes_done += 1
                    counted += 1
                step_in_ep[done_ids] = 0
                # Score breakdown: the managers publish per-term episode stats on reset.
                log = extras.get("log") or {}
                for key, value in log.items():
                    if key.startswith(("Episode_Reward/", "Episode_Termination/")):
                        breakdown_sums[key] = breakdown_sums.get(key, 0.0) + float(value) * counted
                        breakdown_counts[key] = breakdown_counts.get(key, 0) + counted

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    results = {
        "task": args_cli.task,
        "policy": args_cli.policy,
        "seed": args_cli.seed,
        "episodes": episodes_done,
        "success_rate": successes / max(episodes_done, 1),
        "mean_episode_length": _mean(ep_lengths),
        "mean_control_effort": _mean(control_efforts),
        "peak_contact_force": peak_contact,
        "score_breakdown": {
            key: breakdown_sums[key] / breakdown_counts[key] for key in sorted(breakdown_sums)
        },
    }
    print(json.dumps(results, indent=2))
    if args_cli.output:
        with open(args_cli.output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[INFO] wrote {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
