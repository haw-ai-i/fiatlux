# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Replace-v0`` -- the whole family world at once (issue #20).

Scaffolding only: the env shares the common scene and managers; no rewards/terminations/
policy-chaining logic here (that's descoped -- see the issue's own comment: "I will remove
the policy stitching -- this is again structurally not part of the benchmark, but of
solution. I will only focus on building the full scene and testing it."). This scaffold is
exactly that: robot, ladder, table+bulb, and the elevated socket/lamp ("fixture") all spawn
together, each floor occupant randomized into its own non-overlapping "safe zone" within the
room, and the fixture randomly ceiling- or wall-mounted (:func:`apply_replace_preset`).
"""

from typing import Literal

from isaaclab.utils import configclass

from .base_env_cfg import FamilyBaseEnvCfg
from .scene_cfg import apply_replace_preset


@configclass
class ReplaceEnvCfg(FamilyBaseEnvCfg):
    """Combined-family environment: the whole world, randomized zone layout per build."""

    scene_preset: str = "replace"
    # wide framing: the room-scale layout, not a fixed bench corner. The Simple Room is NOT
    # centered on the world origin (measured wall bbox: x=[-4.52,4.52], y=[-3.4,4.86] -- see
    # scene_cfg.py's ROOM_FLOOR_MIN/MAX comment), so the orbit center is offset to the room's
    # actual y-midpoint and the radius stays well inside the nearer (south) wall.
    orbit_center: tuple[float, float, float] = (0.0, 0.7, 2.2)
    orbit_radius: float = 3.0
    orbit_height: float = 3.0

    # apply_replace_preset knobs, exposed the same way FamilyBaseEnvCfg exposes
    # enable_dressing_randomization -- plain fields a subclass/instantiation can override.
    couple_ladder_to_fixture: bool = True
    """Place the ladder's zone reachably relative to wherever the fixture mounted (beneath a
    ceiling point, or standing off from a mounted wall) instead of sampling it independently."""
    bulb_state: Literal["install", "remove"] = "install"
    """"install": fixture starts empty, fresh bulb on the table. "remove": fixture starts with
    a bulb already seated (kinematic), table is the empty destination."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_replace_preset(
            self.scene,
            couple_ladder_to_fixture=self.couple_ladder_to_fixture,
            bulb_state=self.bulb_state,
        )
        # TODO(task phase): reward/termination logic for the combined climb+insert episode
        #   stays on the roadmap -- this scaffold only builds and verifies the scene.
