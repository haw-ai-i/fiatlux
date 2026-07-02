# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Climb-v0`` -- bipedal ladder ascent (the roadmap climb slot, as a scaffold).

Scaffolding only: the env shares the common scene and managers; no rewards/terminations/policy.
Start layout: elevated chandelier fixture above the ladder, robot at the ladder's base.
"""

from isaaclab.utils import configclass

from .base_env_cfg import FamilyBaseEnvCfg
from .scene_cfg import apply_at_height_preset


@configclass
class ClimbEnvCfg(FamilyBaseEnvCfg):
    """Ladder-climbing environment (at-height preset, robot at the base)."""

    scene_preset: str = "climb"
    # frame the ladder from its feet up to the chandelier
    orbit_center: tuple[float, float, float] = (1.4, 0.0, 1.5)
    orbit_radius: float = 3.4
    orbit_height: float = 2.6

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_at_height_preset(self.scene, robot_at="base")
        # TODO(task phase): contact sensors on hands/feet, hand-to-rung observations, and an ascent
        #   reward (switch base to ManagerBasedRLEnvCfg). Keep the ladder kinematic for ascent.
