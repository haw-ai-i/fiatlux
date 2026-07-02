# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Remove-v0`` -- unscrew/remove the existing bulb.

Scaffolding only: the env shares the common scene and managers; no rewards/terminations/policy.
Start layout: the OLD BULB sits seated in the elevated fixture (kinematic -- a stand-in for
"screwed in" until the attach joint exists), robot at the upper steps. The unscrew motion
will later be modelled as a revolute/screw joint or a make/break fixed-joint attach gated by
rotation -- NOT threaded geometry -- anchored at the fixture's socket seat pose.
"""

from isaaclab.utils import configclass

from .base_env_cfg import FamilyBaseEnvCfg
from .scene_cfg import apply_remove_preset


@configclass
class RemoveEnvCfg(FamilyBaseEnvCfg):
    """Bulb-removal environment (at-height preset, old bulb seated in the fixture)."""

    scene_preset: str = "remove"
    # frame from the floor up past the seated bulb below the fixture cage
    orbit_center: tuple[float, float, float] = (1.5, 0.0, 1.5)
    orbit_radius: float = 3.6
    orbit_height: float = 2.8

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_remove_preset(self.scene)
        # TODO(task phase): create a bulb<->socket joint at the seat pose (revolute/screw or a
        #   fixed joint broken by rotation), add a grasp/attach action, and a removal reward.
