# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S04-PlaceLadder-v0`` -- set the carried ladder down at the fixture and let go.

Success is the ladder upright, on its feet, at rest, released, close enough to the fixture for
the climb subtask to reach it (``LADDER_READY_XY_RADIUS``), and the robot left on the ladder's
step-facing side within mounting range -- the successor climbs from wherever this leaves it, not
from ``LADDER_POSITION``.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from fiatlux_task.grasp_poses import LADDER_CARRY_ARM_JOINT_POS, LADDER_IN_ROOT_CARRIED

from .. import mdp
from ..mdp import place_terms
from ..mdp.nav_terms import compose_carried_pose
from ..scene_cfg import (
    LADDER_NEAR_RAIL_OFFSET,
    LADDER_READY_XY_RADIUS,
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    face_robot_at,
    frame_viewer_on,
)
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT, SubtaskTerminationsCfg
from ..subtask_tiers.place import (
    AT_REST_ANG_VEL_LIMIT,
    AT_REST_LIN_VEL_LIMIT,
    PLACE_SUSTAIN_SECONDS,
    RELEASE_FORCE_THRESHOLD_N,
    PlaceRewardsCfg,
    PlaceSubtaskCfg,
    add_release_contact_sensor,
)

# The ladder's root reads "on the floor" within this of the floor plane.
LADDER_FEET_TOLERANCE = 0.02  # m

# Rectangle the robot must land in, on the ladder's step-facing side: forward along the live
# step face, lateral across it. Not LADDER_APPROACH_RADIUS -- that is an arm's-reach bound for
# grasping a rail, an unrelated foot-placement question (CRITIQUE A6).
LADDER_MOUNT_FORWARD_LIMIT = 1.0  # m
LADDER_MOUNT_LATERAL_LIMIT = LADDER_NEAR_RAIL_OFFSET  # m, the A-frame's own half-width

# The success gate, as reviewable data (mdp.all_of) rather than a hand-written conjunction --
# an omitted conjunct here is a gate that passes vacuously.
LADDER_PLACED_CONJUNCTS = [
    (mdp.ladder_ready, {"xy_radius": LADDER_READY_XY_RADIUS, "tilt_limit": mdp.LADDER_TILT_LIMIT}),
    (place_terms.ladder_feet_down, {"tolerance": LADDER_FEET_TOLERANCE}),
    (
        place_terms.object_at_rest,
        {
            "asset_cfg": SceneEntityCfg("ladder"),
            "lin_vel_limit": AT_REST_LIN_VEL_LIMIT,
            "ang_vel_limit": AT_REST_ANG_VEL_LIMIT,
        },
    ),
    (
        place_terms.object_released,
        {"sensor_cfg": SceneEntityCfg("release_contact"), "force_threshold": RELEASE_FORCE_THRESHOLD_N},
    ),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
    (
        place_terms.robot_at_ladder_base,
        {"forward_limit": LADDER_MOUNT_FORWARD_LIMIT, "lateral_limit": LADDER_MOUNT_LATERAL_LIMIT},
    ),
]


@configclass
class S04RewardsCfg(PlaceRewardsCfg):
    ladder_tipped = RewTerm(func=mdp.ladder_tipped, weight=-200.0, params={"tilt_limit": mdp.LADDER_TILT_LIMIT})


@configclass
class S04TerminationsCfg(SubtaskTerminationsCfg):
    ladder_tipped = DoneTerm(func=mdp.ladder_tipped, params={"tilt_limit": mdp.LADDER_TILT_LIMIT})
    # No ladder_dropped: letting go is the goal here, not a failure mode.


@configclass
class S04PlaceLadderEnvCfg(PlaceSubtaskCfg):
    """Set the ladder down at the fixture (randomized Replace layout, ladder dynamic)."""

    scene_preset: str = "replace"
    orbit_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orbit_radius: float = 5.0
    orbit_height: float = 2.4

    success_predicate = mdp.sustained
    success_params: dict | None = {
        "predicate_fn": mdp.all_of,
        "seconds": PLACE_SUSTAIN_SECONDS,
        "predicate_params": {"predicates": LADDER_PLACED_CONJUNCTS},
    }
    progress_distance_fn = mdp.ladder_fixture_distance

    rewards: S04RewardsCfg = S04RewardsCfg()
    terminations: S04TerminationsCfg = S04TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_replace_preset(self.scene)
        # S03's end state: the ladder starts already held, at the carry offset from the robot's
        # own (randomized) root pose -- not at the preset's independently-sampled ladder zone --
        # with the robot facing the fixture it is about to set the ladder down at.
        face_robot_at(self.scene, self.scene.socket.init_state.pos[:2])
        self.scene.ladder.init_state.pos, self.scene.ladder.init_state.rot = compose_carried_pose(
            self.scene.robot.init_state.pos, self.scene.robot.init_state.rot, LADDER_IN_ROOT_CARRIED
        )
        # The arm that's carrying it, matching the pose LADDER_IN_ROOT_CARRIED was measured
        # against -- merge, don't assign: this dict only names right-arm/right-hand joints.
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            **LADDER_CARRY_ARM_JOINT_POS,
        }
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        add_release_contact_sensor(self.scene, self.scene.ladder.prim_path)
        self.episode_length_s = 20.0
        frame_viewer_on(self.viewer, self.scene.robot.init_state.pos)
