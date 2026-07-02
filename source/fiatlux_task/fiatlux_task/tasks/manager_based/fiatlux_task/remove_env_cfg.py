# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Remove-v0`` -- unscrew/remove the existing bulb.

Scaffolding only: the env shares the common scene and managers; no rewards/terminations/policy.
The unscrew motion will later be modelled as a revolute/screw joint or a make/break fixed-joint
attach gated by rotation -- NOT threaded geometry -- anchored at the lamp's socket seat pose.
"""

from isaaclab.utils import configclass

from .g1_ladder_env_cfg import G1LadderEnvCfg


@configclass
class RemoveEnvCfg(G1LadderEnvCfg):
    """Bulb-removal environment (foundation: identical to the base env)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # TODO(task phase): create a bulb<->lamp joint at the socket seat pose (revolute/screw or a
        #   fixed joint broken by rotation), add a grasp/attach action, and a removal reward.
