# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How hard the bulb was struck, by anything at all (#138).

The family's other contact channels are mounted on the robot and filtered to the bulb, so they
report only what the hand does. The measure here is the bulb's own velocity change across one
control step with free fall taken out: no sensor, every source, and the whole control step rather
than its last physics sub-step.

``scripts/verify_bulb_impact.py`` calibrates the bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Steps a leg spends staging its payload rather than acting on it (#105).
STAGING_STEPS = 2


class payload_struck(ManagerTermBase):
    """True where the payload's speed changed by more than ``limit`` (m/s) in one control step.

    Use ONLY in the termination manager; pay the penalty from the reward side with
    ``gates.success_term_fired``, which reads this term's result rather than keeping a second
    velocity buffer that could disagree.

    Gravity is subtracted, so a bulb in free fall reads zero however far it falls. What is left is
    the velocity contact took away or added. The episode's first ``STAGING_STEPS`` steps are exempt.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._previous[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        limit: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("old_bulb"),
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        velocity = asset.data.root_lin_vel_w
        gravity = torch.tensor(env.sim.cfg.gravity, device=env.device)
        struck = (velocity - self._previous - gravity * env.step_dt).norm(dim=-1) > limit
        self._previous = velocity.clone()
        return struck & (env.episode_length_buf > STAGING_STEPS)
