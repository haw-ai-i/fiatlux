# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Install-v0`` -- insert/screw in a new bulb at the fixture.

The at-fixture counterpart of the tabletop ``FIATLUX-Insert-v0`` manipulation task.
Scaffolding only: the env shares the common scene and managers; no rewards/terminations/policy.
Start layout: the elevated fixture is EMPTY, the fresh bulb rests in a parts crate at the
ladder base, robot at the upper steps. Like removal, the screw-in motion will be a joint or
make/break attach gated by rotation, anchored at the fixture's socket seat pose -- NOT
threaded geometry.
"""

from isaaclab.utils import configclass

from .base_env_cfg import FamilyBaseEnvCfg
from .scene_cfg import apply_install_preset


@configclass
class InstallEnvCfg(FamilyBaseEnvCfg):
    """Bulb-installation environment (at-height preset, fresh bulb in the crate)."""

    scene_preset: str = "install"
    # frame wide/low enough to include the parts crate at the ladder base
    orbit_center: tuple[float, float, float] = (1.3, -0.2, 1.3)
    orbit_radius: float = 3.6
    orbit_height: float = 2.8

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_install_preset(self.scene)
        # TODO(task phase): start the bulb in the robot's hand (or nearby), reward seating it into
        #   the socket and forming the attach joint at the seat pose.
