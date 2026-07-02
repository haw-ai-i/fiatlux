# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Remove-v0`` -- unscrew/remove the existing bulb.

Scaffolding only: the env shares the common scene and managers; no rewards/terminations/policy.
Start layout: Insert's bench world with the OLD BULB seated in the table lamp (kinematic --
a stand-in for "screwed in" until the attach joint exists) and an empty parts crate beside
the bench as its destination. The unscrew motion will later be modelled as a revolute/screw
joint or a make/break fixed-joint attach gated by rotation -- NOT threaded geometry --
anchored at the lamp's socket seat pose.
"""

from isaaclab.utils import configclass

from .base_env_cfg import FamilyBaseEnvCfg
from .scene_cfg import apply_remove_preset


@configclass
class RemoveEnvCfg(FamilyBaseEnvCfg):
    """Bulb-removal environment (bench preset, old bulb seated in the table lamp)."""

    scene_preset: str = "remove"
    # bench framing, wide enough to include the floor crate
    orbit_center: tuple[float, float, float] = (0.4, -0.2, 1.0)
    orbit_radius: float = 3.4
    orbit_height: float = 2.2

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_remove_preset(self.scene)
        # TODO(task phase): create a bulb<->socket joint at the seat pose (revolute/screw or a
        #   fixed joint broken by rotation), add a grasp/attach action, and a removal reward.
