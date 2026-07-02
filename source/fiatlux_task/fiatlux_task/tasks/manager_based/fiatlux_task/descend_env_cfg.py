# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Descend-v0`` -- bipedal ladder descent.

Scaffolding only: the env shares the common scene and managers; no rewards/terminations/policy.
Start layout: elevated chandelier fixture above the ladder, robot already at the upper steps.
"""

from isaaclab.utils import configclass

from .base_env_cfg import FamilyBaseEnvCfg
from .scene_cfg import apply_at_height_preset


@configclass
class DescendEnvCfg(FamilyBaseEnvCfg):
    """Ladder-descent environment (at-height preset, robot at the top)."""

    scene_preset: str = "descend"
    orbit_center: tuple[float, float, float] = (1.4, 0.0, 1.6)
    orbit_radius: float = 3.4
    orbit_height: float = 2.6

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_at_height_preset(self.scene, robot_at="top")
        # TODO(task phase): often the reverse curriculum of climbing -- start the robot high on the
        #   ladder and reward a controlled descent to the floor (switch base to ManagerBasedRLEnvCfg).
