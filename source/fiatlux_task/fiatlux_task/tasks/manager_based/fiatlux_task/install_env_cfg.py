# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Install-v0`` -- insert/screw in a new bulb at the fixture.

The at-fixture counterpart of the tabletop ``FIATLUX-Insert-v0`` manipulation task.
Scaffolding only: the env shares the common scene and managers; no rewards/terminations/policy.
Like removal, the screw-in motion will be a joint or make/break attach gated by rotation,
anchored at the lamp's socket seat pose -- NOT threaded geometry.
"""

from isaaclab.utils import configclass

from .g1_ladder_env_cfg import G1LadderEnvCfg


@configclass
class InstallEnvCfg(G1LadderEnvCfg):
    """Bulb-installation environment (foundation: identical to the base env)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # TODO(task phase): start the bulb in the robot's hand (or nearby), reward seating it into
        #   the socket and forming the attach joint at the seat pose.
