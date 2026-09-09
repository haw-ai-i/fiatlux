# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared tier for S01 (move-the-ladder) and S06 (dispose-of-the-bulb).

Both subtasks end with an object left where it belongs: a placement-progress channel toward the
target, a single-target contact sensor for release detection, and the thresholds the leaves'
success gates share (the family's usual ``sustained`` window is too short here -- a placement
that only holds while a hand steadies it has not been made).

Release detection is offered, not mandated. S06 gates on it, because a bulb still in the hand
over the crate has not been disposed of. S01 does not: whether the ladder was carried, dragged or
walked upright is not a property of the ladder standing at the fixture, so it adds no sensor.
"""

from collections.abc import Callable

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .. import mdp
from ..scene_cfg import G1ReplaceSceneCfg
from ..subtask_env_cfg import SubtaskEnvCfg, SubtaskRewardsCfg, SubtaskShapingRewardsCfg

# Longer than a momentary dip: a release that is only stable while a hand steadies it is not placed.
PLACE_SUSTAIN_SECONDS = 1.0

# At-rest / release thresholds shared by both leaves' success gates.
AT_REST_LIN_VEL_LIMIT = 0.05  # m/s
AT_REST_ANG_VEL_LIMIT = 0.10  # rad/s
RELEASE_FORCE_THRESHOLD_N = 1.0  # N


def add_release_contact_sensor(scene: G1ReplaceSceneCfg, target_prim_path: str) -> None:
    """Hand-only contact sensor filtered to ONE target prim, for release detection.

    ``mdp.place_terms.object_released`` needs an unambiguous per-target column; the scene's
    shared ``hand_contact`` filters two prims in the replace preset (Bulb + OldBulb), so its
    columns cannot be attributed to a single one of them.
    """
    scene.release_contact = ContactSensorCfg(
        prim_path=scene.hand_contact.prim_path,
        filter_prim_paths_expr=[target_prim_path],
        history_length=1,
        track_air_time=False,
    )
    # Left mirror: released means BOTH hands are off it, not just the right (issue #151).
    scene.release_contact_left = ContactSensorCfg(
        prim_path=scene.left_hand_contact.prim_path,
        filter_prim_paths_expr=[target_prim_path],
        history_length=1,
        track_air_time=False,
    )


@configclass
class PlaceRewardsCfg(SubtaskShapingRewardsCfg):
    """Placement-progress channel; ``distance_fn`` comes from the leaf."""

    placement_progress = RewTerm(
        func=mdp.distance_progress, weight=500.0, params={"distance_fn": mdp.ladder_fixture_distance}
    )


@configclass
class PlaceSubtaskCfg(SubtaskEnvCfg):
    """A subtask whose job is to release a held object at a target and leave it there."""

    progress_distance_fn: Callable | None = None
    rewards: SubtaskRewardsCfg = SubtaskRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.progress_distance_fn is None:
            raise ValueError(f"{type(self).__name__} must set progress_distance_fn")
        if hasattr(self.rewards, "placement_progress"):
            self.rewards.placement_progress.params["distance_fn"] = self.progress_distance_fn
