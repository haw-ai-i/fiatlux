# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S11-GrabNewBulb-v0`` -- pick the fresh bulb off the bench.

Success is the bulb lifted clear of the tabletop, held by at least two hand bodies, filtered
grip force within the glass fragility limit, and the robot standing -- held for
``GRASP_SUSTAIN_SECONDS``. Produces ``grasp_poses.BULB_IN_ROOT_STANDING`` as its end state
(consumed by S08 and S12).

The glass shell is fragile (``grasp_poses.GLASS_CONTACT_LIMIT_N`` = 50 N, also
``scripts/score.py``'s fragility threshold): unlike the ladder leaf, this gate's force conjunct
is an UPPER bound, not a lower one -- a grip that lifts the bulb by crushing it does not count.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from fiatlux_task.assets import BULB_STAND_Z_OFFSET
from fiatlux_task.grasp_poses import GLASS_CONTACT_LIMIT_N

from .. import mdp
from ..mdp import grasp_terms, place_terms
from ..mdp.nav_terms import BULB_APPROACH_OFFSET
from ..scene_cfg import (
    TABLETOP_SURFACE_Z,
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    frame_viewer_on,
    park_old_bulb_in_crate,
    stand_robot_at_offset,
)
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT, SubtaskTerminationsCfg
from ..subtask_tiers.grasp import GRASP_SUSTAIN_SECONDS, GraspRewardsCfg, GraspSubtaskCfg, add_grasp_contact_sensor

# Resting root height on the (always z=0) table, standing on its cap.
BULB_TABLETOP_REST_Z = TABLETOP_SURFACE_Z - BULB_STAND_Z_OFFSET  # m

# Root rise counted as "lifted". PROVISIONAL.
BULB_LIFT_CLEARANCE_M = 0.03  # m
BULB_LIFTED_HEIGHT_M = BULB_TABLETOP_REST_Z + BULB_LIFT_CLEARANCE_M

# Below the tabletop rest height, above the floor (~-0.036 m): catches "knocked off the table"
# before it lands. PROVISIONAL.
BULB_DROPPED_HEIGHT_M = 0.5  # m

# Any contact above sensor noise counts as "a body is on the bulb"; the fragility bound is a
# separate conjunct below.
BULB_HAND_CONTACT_THRESHOLD_N = 1.0
BULB_HELD_MIN_HAND_BODIES = 2

# The success gate, as reviewable data (mdp.all_of) rather than a hand-written conjunction --
# an omitted conjunct here is a gate that passes vacuously.
BULB_GRASPED_CONJUNCTS = [
    (grasp_terms.object_lifted, {"asset_cfg": SceneEntityCfg("bulb"), "min_height": BULB_LIFTED_HEIGHT_M}),
    (
        grasp_terms.hand_bodies_in_contact,
        {
            "sensor_cfg": SceneEntityCfg("grasp_contact"),
            "min_bodies": BULB_HELD_MIN_HAND_BODIES,
            "force_threshold": BULB_HAND_CONTACT_THRESHOLD_N,
        },
    ),
    (
        grasp_terms.grasp_force_within,
        {"sensor_cfg": SceneEntityCfg("grasp_contact"), "limit": GLASS_CONTACT_LIMIT_N},
    ),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
]


@configclass
class S11RewardsCfg(GraspRewardsCfg):
    bulb_dropped = RewTerm(
        func=mdp.object_dropped,
        weight=-200.0,
        params={"asset_cfg": SceneEntityCfg("bulb"), "min_height": BULB_DROPPED_HEIGHT_M},
    )


@configclass
class S11TerminationsCfg(SubtaskTerminationsCfg):
    bulb_dropped = DoneTerm(
        func=mdp.object_dropped, params={"asset_cfg": SceneEntityCfg("bulb"), "min_height": BULB_DROPPED_HEIGHT_M}
    )


@configclass
class S11GrabNewBulbEnvCfg(GraspSubtaskCfg):
    """Pick the fresh bulb off the bench (randomized Replace layout)."""

    scene_preset: str = "replace"
    orbit_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orbit_radius: float = 5.0
    orbit_height: float = 2.4

    success_predicate = mdp.sustained
    success_params: dict | None = {
        "predicate_fn": mdp.all_of,
        "seconds": GRASP_SUSTAIN_SECONDS,
        "predicate_params": {"predicates": BULB_GRASPED_CONJUNCTS},
    }
    progress_distance_fn = grasp_terms.hand_bulb_distance

    rewards: S11RewardsCfg = S11RewardsCfg()
    terminations: S11TerminationsCfg = S11TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # S04 already stood the ladder at the fixture; every leg after it inherits that.
        apply_replace_preset(self.scene, couple_ladder_to_fixture=True)
        # The old bulb was disposed of back in S09; the fixture is empty from here on.
        park_old_bulb_in_crate(self.scene)
        # A radius alone can land the robot under the table, so use the table's own approach
        # vector from the bulb.
        stand_robot_at_offset(self.scene, self.scene.bulb.init_state.pos[:2], BULB_APPROACH_OFFSET)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        add_grasp_contact_sensor(self.scene, self.scene.bulb.prim_path)
        self.episode_length_s = 20.0
        frame_viewer_on(self.viewer, self.scene.robot.init_state.pos)
