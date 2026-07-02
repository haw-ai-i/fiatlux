# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Carry-v0`` -- grab and install/position the ladder.

Scaffolding only: the env shares the common scene and managers; no rewards/terminations/policy.
"""

from isaaclab.utils import configclass

from .g1_ladder_env_cfg import G1LadderEnvCfg


@configclass
class CarryEnvCfg(G1LadderEnvCfg):
    """Ladder-handling environment (foundation: identical to the base env)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # TODO(task phase): make the ladder a dynamic body so it can be grabbed and repositioned:
        #   self.scene.ladder.spawn.rigid_props.kinematic_enabled = False
        # TODO(task phase): add rewards/terminations (switch base to ManagerBasedRLEnvCfg) for grasping
        #   and placing the ladder upright at a target.
