# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "fiatlux_task"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ClimbPPORunnerCfg(PPORunnerCfg):
    """PPO runner for the ladder-climb task (house hyperparameters, longer horizon).

    ``obs_groups`` must be explicit: rsl_rl's ``resolve_obs_groups`` only auto-routes
    an env obs group literally named ``critic`` -- a group named ``privileged`` would
    otherwise silently never reach the critic (it falls back to the policy set).
    ``PPORunnerCfg`` above predates this finding and is left unchanged by review
    decision; revisit when Insert's privileged group should feed its critic.
    """

    max_iterations = 3000  # locomotion-scale training budget
    save_interval = 100
    experiment_name = "fiatlux_climb"
    obs_groups = {"policy": ["policy"], "critic": ["policy", "privileged"]}


@configclass
class ReplacePPORunnerCfg(PPORunnerCfg):
    """PPO runner for the full replacement task (same explicit obs-group routing as Climb:
    the standard/``policy`` group feeds the actor, the cheatcode/``privileged`` group is
    critic-only). The budget is aspirational -- the flat full task is far past what PPO
    from scratch solves; this config exists so the train/play/eval tooling runs end-to-end.
    """

    max_iterations = 5000
    save_interval = 100
    experiment_name = "fiatlux_replace"


@configclass
class CarryPPORunnerCfg(PPORunnerCfg):
    """PPO runner for the ladder-handling / positioning task (FIATLUX-Carry-v0).

    Explicit ``obs_groups`` so the privileged ladder-pose group reaches the critic (a group
    named ``privileged`` is otherwise silently dropped -- see :class:`ClimbPPORunnerCfg`).

    Orphaned like ``ClimbPPORunnerCfg`` above: ``FIATLUX-Carry-v0`` is no longer registered
    (this cleanup removed it along with ``carry_env_cfg.py``'s standalone RL registration), so
    this config currently has no env to train against.
    """

    max_iterations = 2000
    save_interval = 100
    experiment_name = "fiatlux_carry"
    obs_groups = {"policy": ["policy"], "critic": ["policy", "privileged"]}
