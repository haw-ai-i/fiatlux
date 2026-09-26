# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read any asset field as it stood the instant before an auto-reset, from outside the env.

``env.step()`` runs every termination term, then -- for whichever envs just terminated -- resets
the scene to the next episode's state, all before returning. A script watching from outside can
read live scene tensors between two calls to ``env.step()``, but never *during* the one call where
a reset happens: by the time that call returns, the asset it cares about already holds the next
episode's pose and velocity, not the one that ended the episode.

``mdp.impact_terms.payload_struck`` solved this once, narrowly, for its own gravity-compensated
velocity-change norm (``last_dv``): compute it every step, then have the term's own ``reset()``
carefully avoid clearing it, so it survives past the same-step reset that follows. That trick is
the only way to get a value from inside the moment terminations are computed, since class-based
termination terms are the one thing IsaacLab calls before that reset and lets survive after it --
but it only works because ``payload_struck`` already needed a class instance for its own job.

``snapshot_before_reset`` is that same trick generalized into a standalone term: instead of one
more one-off cached attribute bolted onto an unrelated termination, any script can add this term
to ``env_cfg.terminations``, tell it which ``(asset name, data field)`` pairs to watch, and read
them back afterward -- for the bulb's position (issue #255), or anything else a future script
needs the true pre-reset value of.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class snapshot_before_reset(ManagerTermBase):
    """Never terminates anything; caches asset fields for a caller to read after ``env.step()``.

    ``snapshot`` maps ``"{asset_name}.{attr_name}"`` to that field's tensor, as it stood during
    this step's ``termination_manager.compute()`` -- before any reset that this or another
    termination firing this step may trigger. It is overwritten every step, for every env, whether
    or not that step resets anything, so a caller always finds the value for the step that just
    ran.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.snapshot: dict[str, torch.Tensor] = {}

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Deliberately a no-op: this runs later in the same env.step() call that populated
        # `snapshot`, for whichever env_ids just terminated. Clearing it here would erase the one
        # value external readers need, the same reason `payload_struck.reset()` spares `last_dv`.
        pass

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        fields: tuple[tuple[str, str], ...],
    ) -> torch.Tensor:
        for asset_name, attr_name in fields:
            self.snapshot[f"{asset_name}.{attr_name}"] = getattr(env.scene[asset_name].data, attr_name).clone()
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
