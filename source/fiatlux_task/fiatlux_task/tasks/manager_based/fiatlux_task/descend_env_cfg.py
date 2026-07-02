# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Descend-v0`` -- bipedal ladder descent.

Scaffolding only: the env shares the common scene and managers; no rewards/terminations/policy.
"""

from isaaclab.utils import configclass

from .base_env_cfg import FamilyBaseEnvCfg


@configclass
class DescendEnvCfg(FamilyBaseEnvCfg):
    """Ladder-descent environment (foundation: identical to the base env)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # TODO(task phase): often the reverse curriculum of climbing -- start the robot high on the
        #   ladder and reward a controlled descent to the floor (switch base to ManagerBasedRLEnvCfg).
