# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Framework-agnostic policy factory for the Fiatlux benchmark.

A *policy* is just a callable ``policy(obs) -> actions`` returning a
``(num_envs, action_dim)`` tensor. The benchmark never inspects how a policy was
produced -- RL, imitation, scripted, or a hand-written baseline all plug in here.

Supported specs (``make_policy(spec, env)``):

- ``"zero"``            -- no action (sanity floor).
- ``"random"``          -- uniform actions in ``[-1, 1]`` (sanity floor).
- ``"<path>.pt"``       -- a TorchScript module taking the policy observation
                           tensor and returning actions. This is the portable,
                           framework-agnostic artifact ``scripts/rsl_rl/play.py``
                           already exports to ``.../exported/policy.pt``.
- ``"rsl_rl[:<ckpt>]"`` -- convenience loader for an RSL-RL ``OnPolicyRunner``
                           checkpoint (one example training backend, not required).

The ``rsl_rl`` and TorchScript loaders are imported lazily so the baselines work
without those dependencies installed.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _policy_obs(obs):
    """Extract the sensor-realizable policy observation tensor from a raw env obs."""
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def make_policy(
    spec: str,
    env,
    *,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Callable:
    """Return a callable ``policy(obs) -> actions`` for ``spec``.

    Args:
        spec: One of ``"zero"``, ``"random"``, a TorchScript ``.pt`` path, or
            ``"rsl_rl"`` / ``"rsl_rl:<checkpoint>"``.
        env: The (unwrapped) environment; used for ``num_envs``/``action_space``/``device``.
        checkpoint: Checkpoint path for ``"rsl_rl"`` (alternative to the ``rsl_rl:`` suffix).
        device: Override device; defaults to ``env.device``.
    """
    device = device or env.device
    action_shape = (env.num_envs, env.action_space.shape[-1])

    if spec == "zero":
        return lambda obs: torch.zeros(action_shape, device=device)
    if spec == "random":
        return lambda obs: torch.rand(action_shape, device=device) * 2.0 - 1.0

    # RSL-RL convenience loader: "rsl_rl" (+ checkpoint) or "rsl_rl:<path>".
    if spec == "rsl_rl" or spec.startswith("rsl_rl:"):
        from rsl_rl.runners import OnPolicyRunner

        ckpt = spec.split(":", 1)[1] if ":" in spec else None
        ckpt = ckpt or checkpoint
        assert ckpt, "rsl_rl policy requires a checkpoint (--checkpoint or rsl_rl:<path>)"
        runner = OnPolicyRunner(env, {}, log_dir=None, device=device)
        runner.load(ckpt)
        return runner.get_inference_policy(device=device)

    # Otherwise treat the spec as a TorchScript artifact (portable, no framework dep).
    path = spec[4:] if spec.startswith("jit:") else spec
    module = torch.jit.load(path, map_location=device)
    module.eval()

    def policy(obs):
        return module(_policy_obs(obs))

    return policy
