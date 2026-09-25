# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Framework-agnostic policy factory for the Fiatlux benchmark.

A *policy* is just a callable ``policy(obs) -> actions`` returning a
``(num_envs, action_dim)`` tensor. The benchmark never inspects how a policy was
produced -- RL, imitation, scripted, or a hand-written baseline all plug in here.

Optional telemetry hook: a policy MAY carry an ``info`` attribute -- a
``dict[str, float]`` it refreshes on each call (e.g. a critic value estimate,
action log-prob). The evaluation scripts forward it to the benchmark telemetry
(``fiatlux_task.telemetry``), which streams running means under the ``policy/``
namespace, kept apart from the score channels. Policies without it pay nothing.

Supported specs (``make_policy(spec, env)``):

- ``"zero"``            -- no action (sanity floor).
- ``"random"``          -- uniform actions in ``[-1, 1]`` (sanity floor).
- ``"basic_standard"``  -- smoke-test policy for the *standard* observation mode: consumes
                           only the sensor-realizable ``policy`` group (asserts it exists,
                           never reads privileged state) and holds posture (zero action).
                           Its purpose is to prove the env / observations / action
                           interface / resets / episode loop / recording / scoring run
                           end-to-end -- acceptance is valid episode execution and score
                           artifact generation, not task success.
- ``"basic_cheatcode"`` -- smoke-test policy for the *privileged* observation mode (the
                           spec value keeps its original name): additionally asserts
                           the ``privileged`` group exists and reads it every step, still
                           acting through the normal action space (zero action). Not a
                           solver; same acceptance bar as ``basic_standard``.
- ``"<path>.pt"``       -- a TorchScript module taking the policy observation
                           tensor and returning actions. This is the portable,
                           framework-agnostic artifact ``scripts/rsl_rl/play.py``
                           already exports to ``.../exported/policy.pt``.
- ``"rsl_rl[:<ckpt>]"`` -- convenience loader for an RSL-RL ``OnPolicyRunner``
                           checkpoint (one example training backend, not required).
- ``"sonic_stand"``     -- the GEAR-SONIC whole-body controller holding its standing
                           latent (no VLA); the SONIC stack's sim2sim stand gate
                           (see ``fiatlux_task.groot``).
- ``"wbc_stand"``       -- the decoupled GEAR WBC holding zero commands (no VLA);
                           the stand gate for the ``groot`` baseline's lower body.
- ``"groot[:<host:port>]"`` -- the GR00T N1.7 baseline: queries a running
                           Isaac-GR00T PolicyServer (external process, ``REAL_G1``
                           embodiment) and decodes its navigation/height/arm
                           chunks through the decoupled WBC in-process.
                           ``instruction`` sets the language prompt.

The ``rsl_rl``, TorchScript, and GR00T/SONIC loaders are imported lazily so the
baselines work without those dependencies installed.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# Shared with scripts/eval.py and scripts/record_run.py's --policy argparse help, so the two
# CLIs can't drift apart on what make_policy() actually accepts. Safe to import before
# AppLauncher runs: this module has no Isaac-Sim-dependent top-level imports.
POLICY_SPEC_HELP = (
    "Policy spec: zero | random | basic_standard | basic_cheatcode | wbc_stand | "
    "sonic_stand | groot[:<host:port>] | rsl_rl[:<ckpt>] | <path>.pt (or jit:<path>). "
    "See fiatlux_task/policy.py."
)
CHECKPOINT_HELP = "Checkpoint path for rsl_rl policies."


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
    instruction: str | None = None,
) -> Callable:
    """Return a callable ``policy(obs) -> actions`` for ``spec``.

    Args:
        spec: One of ``"zero"``, ``"random"``, ``"basic_standard"``, ``"basic_cheatcode"``,
            ``"wbc_stand"``, ``"sonic_stand"``, ``"rsl_rl"`` / ``"rsl_rl:<checkpoint>"``, or
            ``"groot"`` / ``"groot:<host:port>"``. Anything else is treated as a TorchScript
            path (optionally prefixed ``jit:``).
        env: The (unwrapped) environment; used for ``num_envs``/``action_space``/``device``.
        checkpoint: Checkpoint path for ``"rsl_rl"`` (alternative to the ``rsl_rl:`` suffix).
        device: Override device; defaults to ``env.device``.
        instruction: Language prompt for ``"groot"`` (the task description the VLA
            conditions on); ignored by every other spec.
    """
    device = device or env.device
    action_shape = (env.num_envs, env.action_space.shape[-1])

    if spec == "zero":
        return lambda obs: torch.zeros(action_shape, device=device)
    if spec == "random":
        return lambda obs: torch.rand(action_shape, device=device) * 2.0 - 1.0

    if spec == "basic_standard":

        def basic_standard(obs):
            # The standard contract: the sensor-realizable group must exist and is the
            # ONLY thing consumed -- privileged state is never touched.
            assert isinstance(obs, dict) and "policy" in obs, (
                "basic_standard requires a 'policy' observation group (the standard, sensor-realizable mode)"
            )
            _ = obs["policy"]
            return torch.zeros(action_shape, device=device)

        return basic_standard

    if spec == "basic_cheatcode":

        def basic_cheatcode(obs):
            # Contract for this (privileged-mode) smoke test: privileged simulator state
            # must be present and readable; actions still go through the normal action space.
            assert isinstance(obs, dict) and "privileged" in obs, (
                "basic_cheatcode requires a 'privileged' observation group (the privileged "
                "mode); this env exposes only sensor-realizable observations"
            )
            privileged = obs["privileged"]
            assert torch.isfinite(privileged).all(), "privileged observations must be finite"
            return torch.zeros(action_shape, device=device)

        return basic_cheatcode

    # GR00T N1.7 whole-body baseline (fiatlux_task.groot; lazy heavy deps).
    if spec == "sonic_stand":
        from .groot import make_sonic_stand_policy

        return make_sonic_stand_policy(env)
    if spec == "wbc_stand":
        from .groot import make_wbc_stand_policy

        return make_wbc_stand_policy(env)
    if spec == "groot" or spec.startswith("groot:"):
        from .groot import make_groot_policy

        endpoint = spec.split(":", 1)[1] if ":" in spec else None
        return make_groot_policy(env, endpoint=endpoint, instruction=instruction)

    # RSL-RL convenience loader: "rsl_rl" (+ checkpoint) or "rsl_rl:<path>".
    if spec == "rsl_rl" or spec.startswith("rsl_rl:"):
        import os

        import yaml
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        ckpt = spec.split(":", 1)[1] if ":" in spec else None
        ckpt = ckpt or checkpoint
        assert ckpt, "rsl_rl policy requires a checkpoint (--checkpoint or rsl_rl:<path>)"
        # The runner needs the training config (rsl_rl >= 3 reads policy/algorithm/
        # obs_groups from it) and a VecEnv-interfaced env. Isaac Lab's train.py dumps
        # the config next to every checkpoint (params/agent.yaml), so the loaded
        # policy always matches what the checkpoint was trained with.
        agent_yaml = os.path.join(os.path.dirname(ckpt), "params", "agent.yaml")
        assert os.path.isfile(agent_yaml), (
            f"expected the run's training config at {agent_yaml} (written by "
            "scripts/rsl_rl/train.py next to its checkpoints)"
        )
        with open(agent_yaml) as f:
            train_cfg = yaml.safe_load(f)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=train_cfg.get("clip_actions"))
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=device)
        runner.load(ckpt)
        return runner.get_inference_policy(device=device)

    # Otherwise treat the spec as a TorchScript artifact (portable, no framework dep).
    path = spec[4:] if spec.startswith("jit:") else spec
    module = torch.jit.load(path, map_location=device)
    module.eval()

    def policy(obs):
        return module(_policy_obs(obs))

    return policy
