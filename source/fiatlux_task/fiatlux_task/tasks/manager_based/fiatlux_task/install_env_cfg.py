# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Install-v0`` -- insert/screw in a new bulb.

The screw-in counterpart of ``FIATLUX-Insert-v0`` on the same bench world.
Scaffolding only: the env shares the common scene and managers; no rewards/terminations/policy.
Start layout: the table lamp's socket is EMPTY and the fresh bulb rests in a parts crate on
the floor beside the bench. Like removal, the screw-in motion will be a joint or make/break
attach gated by rotation, anchored at the lamp's socket seat pose -- NOT threaded geometry.
"""

from isaaclab.utils import configclass

from .base_env_cfg import FamilyBaseEnvCfg
from .scene_cfg import apply_install_preset


@configclass
class InstallEnvCfg(FamilyBaseEnvCfg):
    """Bulb-installation environment (bench preset, fresh bulb in the crate)."""

    scene_preset: str = "install"
    # bench framing, wide enough to include the floor crate
    orbit_center: tuple[float, float, float] = (0.4, -0.2, 1.0)
    orbit_radius: float = 3.4
    orbit_height: float = 2.2

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_install_preset(self.scene)
        # Tabletop preset has no ladder; the top-level SceneEntityCfg would fail to resolve.
        self.events.randomize_ladder_scale = None
        # TODO(task phase): start the bulb in the robot's hand (or nearby), reward seating it into
        #   the socket and forming the attach joint at the seat pose.
