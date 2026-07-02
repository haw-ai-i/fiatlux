# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standardized Fiatlux benchmark evaluation.

Runs a policy for a fixed number of episodes from a fixed seed and reports the
benchmark metrics as JSON:

- ``success_rate``        : fraction of episodes where the bulb ends seated.
- ``mean_episode_length`` : average steps per episode.
- ``mean_final_pos_error``: average bulb->socket distance at episode end (m).
- ``mean_control_effort`` : average sum-of-squared actions per step.
- ``peak_contact_force``  : max net hand contact force seen (N).

Determinism: same ``--task``, ``--seed`` and ``--policy`` give the same numbers.

Examples:
    python scripts/eval.py --task FIATLUX-Insert-v0 --policy zero --episodes 20
    python scripts/eval.py --task FIATLUX-Insert-v0 --policy random --episodes 20
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
# Evaluation is headless by default unless overridden.
args_cli.headless = True if args_cli.headless is None else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import json

import fiatlux_task.tasks  # noqa: F401
import gymnasium as gym
import torch
from fiatlux_task.policy import make_policy
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import rewards as fiatlux_rewards

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
    final_pos_errors: list[float] = []
    control_efforts: list[float] = []
    peak_contact = 0.0

    obs, _ = env.reset(seed=args_cli.seed)
    step_in_ep = torch.zeros(env.num_envs, device=env.device)

    target = args_cli.episodes
    with torch.inference_mode():
        while episodes_done < target:
            actions = policy(obs)
            obs, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            step_in_ep += 1

            control_efforts.append(float(torch.mean(torch.sum(actions**2, dim=-1))))
            if "hand_contact" in env.scene.sensors:
                f = env.scene.sensors["hand_contact"].data.net_forces_w
                peak_contact = max(peak_contact, float(torch.norm(f, dim=-1).max()))

            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            if len(done_ids) > 0:
                seated = fiatlux_rewards.bulb_seated(env)
                pos_err = fiatlux_rewards._bulb_socket_pos_error(env)
                for i in done_ids.tolist():
                    if episodes_done >= target:
                        break
                    successes += int(bool(seated[i].item()))
                    final_pos_errors.append(float(pos_err[i].item()))
                    ep_lengths.append(int(step_in_ep[i].item()))
                    episodes_done += 1
                step_in_ep[done_ids] = 0

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    results = {
        "task": args_cli.task,
        "policy": args_cli.policy,
        "seed": args_cli.seed,
        "episodes": episodes_done,
        "success_rate": successes / max(episodes_done, 1),
        "mean_episode_length": _mean(ep_lengths),
        "mean_final_pos_error": _mean(final_pos_errors),
        "mean_control_effort": _mean(control_efforts),
        "peak_contact_force": peak_contact,
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
