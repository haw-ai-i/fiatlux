# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Climb-v0`` -- bipedal ladder ascent (the roadmap climb slot, as a scaffold).

Scaffolding only: the env shares the common scene and managers; no rewards/terminations/policy.
"""

from isaaclab.utils import configclass

from .g1_ladder_env_cfg import G1LadderEnvCfg


@configclass
class ClimbEnvCfg(G1LadderEnvCfg):
    """Ladder-climbing environment (foundation: identical to the base env)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # TODO(task phase): contact sensors on hands/feet, hand-to-rung observations, and an ascent
        #   reward (switch base to ManagerBasedRLEnvCfg). Keep the ladder kinematic for ascent.
