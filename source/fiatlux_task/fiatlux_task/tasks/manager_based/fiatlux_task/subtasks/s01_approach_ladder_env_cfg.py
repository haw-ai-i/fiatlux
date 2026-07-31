# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S01-ApproachLadder-v0`` -- walk to the ladder and stand at it, ready to grasp.

First subtask of the chain, and the only one whose start state comes from the randomized layout
rather than from a predecessor's end state. Success is arriving within grasping range of a rail,
facing it, standing, with the ladder still upright.

The gate needs no sustain: the speed cap already rejects a fly-through, and falls are separate
terminations. That matters because a stateful gate would be instantiated twice -- the termination
manager and the reward manager each build their own copy, and the two counters can disagree.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp
from ..scene_cfg import LADDER_APPROACH_RADIUS, add_ego_camera, add_mid360_lidar, apply_replace_preset
from ..subtask_env_cfg import (
    ARRIVAL_FACING_TOLERANCE,
    ARRIVAL_MAX_SPEED,
    NavigateRewardsCfg,
    NavigateSubtaskCfg,
    SubtaskTerminationsCfg,
)


@configclass
class S01RewardsCfg(NavigateRewardsCfg):
    ladder_tipped = RewTerm(func=mdp.ladder_tipped, weight=-200.0, params={"tilt_limit": mdp.LADDER_TILT_LIMIT})


@configclass
class S01TerminationsCfg(SubtaskTerminationsCfg):
    ladder_tipped = DoneTerm(func=mdp.ladder_tipped, params={"tilt_limit": mdp.LADDER_TILT_LIMIT})


@configclass
class S01ApproachLadderEnvCfg(NavigateSubtaskCfg):
    """Walk to the ladder (randomized Replace layout, ladder dynamic)."""

    scene_preset: str = "replace"
    orbit_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orbit_radius: float = 5.0
    orbit_height: float = 2.4

    success_predicate = mdp.arrived_at_ladder
    success_params: dict | None = {
        "xy_radius": LADDER_APPROACH_RADIUS,
        "facing_tolerance": ARRIVAL_FACING_TOLERANCE,
        "max_speed": ARRIVAL_MAX_SPEED,
    }
    progress_distance_fn = mdp.base_ladder_distance

    rewards: S01RewardsCfg = S01RewardsCfg()
    terminations: S01TerminationsCfg = S01TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_replace_preset(self.scene)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        # ~2x the 2.43 m robot->ladder traverse measured at ~0.5 m/s.
        self.episode_length_s = 20.0
        self.viewer.eye = (4.0, 4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
