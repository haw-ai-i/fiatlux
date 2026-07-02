# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Carry-v0`` -- grab and install/position the ladder.

Scaffolding only: the env shares the common scene and managers; no rewards/terminations/policy.
Start layout: the ladder is *stored* by the room wall with the robot beside it; the work
area (the floor socket-lamp) is across the room.
"""

from isaaclab.utils import configclass

from .base_env_cfg import FamilyBaseEnvCfg
from .scene_cfg import apply_carry_preset


@configclass
class CarryEnvCfg(FamilyBaseEnvCfg):
    """Ladder-handling environment (carry preset: stored ladder + robot at the wall)."""

    scene_preset: str = "carry"
    # orbit framing between the stored ladder (-3.2, 1.8) and the work area (origin)
    orbit_center: tuple[float, float, float] = (-1.5, 1.0, 1.0)
    orbit_radius: float = 2.8
    orbit_height: float = 2.4

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_carry_preset(self.scene)
        # TODO(task phase): make the ladder a dynamic body so it can be grabbed and repositioned:
        #   self.scene.ladder.spawn.rigid_props.kinematic_enabled = False
        # TODO(task phase): add rewards/terminations (switch base to ManagerBasedRLEnvCfg) for grasping
        #   and placing the ladder upright at a target.
