# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S08-GrabNewBulb-v0`` -- pick the fresh bulb off the bench.

Success is the bulb lifted clear of the tabletop, held by at least two hand bodies, filtered
grip force within the glass fragility limit, and the robot standing -- held for
``GRASP_SUSTAIN_SECONDS``. Produces ``grasp_poses.BULB_IN_ROOT_STANDING`` as its end state
(consumed by S05 and S09).

The glass shell is fragile (``grasp_poses.GLASS_CONTACT_LIMIT_N`` = 50 N, also
``scripts/score.py``'s fragility threshold): unlike the ladder leaf, this gate's force conjunct
is an UPPER bound, not a lower one -- a grip that lifts the bulb by crushing it does not count.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from fiatlux_task.grasp_poses import GLASS_CONTACT_LIMIT_N

from .. import mdp
from ..mdp import grasp_terms, place_terms
from ..mdp.nav_terms import BULB_APPROACH_OFFSET
from ..scene_cfg import (
    TABLE_COLLISION_HALF_EXTENT,
    TABLE_POSITION,
    TABLETOP_SURFACE_Z,
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    frame_viewer_on,
    park_old_bulb_in_crate,
    stand_robot_at_offset,
)
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT, SubtaskRewardsCfg, SubtaskTerminationsCfg
from ..subtask_tiers.grasp import GRASP_SUSTAIN_SECONDS, GraspRewardsCfg, GraspSubtaskCfg, add_grasp_contact_sensor

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
    (
        grasp_terms.object_clear_of_surface,
        {
            "asset_cfg": SceneEntityCfg("fresh_bulb"),
            "surface_z": TABLETOP_SURFACE_Z,
            "surface_centre": TABLE_POSITION[:2],  # rebound to this layout's bench in __post_init__
            "surface_half_extent": TABLE_COLLISION_HALF_EXTENT,
        },
    ),
    (
        grasp_terms.hand_bodies_in_contact,
        {
            "sensor_cfg": SceneEntityCfg("grasp_contact"),
            "other_sensor_cfg": SceneEntityCfg("grasp_contact_left"),
            "min_bodies": BULB_HELD_MIN_HAND_BODIES,
            "force_threshold": BULB_HAND_CONTACT_THRESHOLD_N,
        },
    ),
    (
        grasp_terms.grasp_force_within,
        {
            "sensor_cfg": SceneEntityCfg("grasp_contact"),
            "other_sensor_cfg": SceneEntityCfg("grasp_contact_left"),
            "limit": GLASS_CONTACT_LIMIT_N,
        },
    ),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
]


@configclass
class S08RewardsCfg(GraspRewardsCfg):
    bulb_dropped = RewTerm(
        func=mdp.object_dropped,
        weight=-200.0,
        params={"asset_cfg": SceneEntityCfg("fresh_bulb"), "min_height": BULB_DROPPED_HEIGHT_M},
    )


@configclass
class S08TerminationsCfg(SubtaskTerminationsCfg):
    bulb_dropped = DoneTerm(
        func=mdp.object_dropped, params={"asset_cfg": SceneEntityCfg("fresh_bulb"), "min_height": BULB_DROPPED_HEIGHT_M}
    )


@configclass
class S08GrabNewBulbEnvCfg(GraspSubtaskCfg):
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

    rewards: SubtaskRewardsCfg = SubtaskRewardsCfg()
    terminations: S08TerminationsCfg = S08TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # S01 already stood the ladder at the fixture; every leg after it inherits that.
        apply_replace_preset(self.scene, couple_ladder_to_fixture=True)
        # The old bulb was disposed of back in S06; the fixture is empty from here on.
        park_old_bulb_in_crate(self.scene)
        # A radius alone can land the robot under the table, so use the table's own approach
        # vector from the bulb.
        stand_robot_at_offset(self.scene, self.scene.fresh_bulb.init_state.pos[:2], BULB_APPROACH_OFFSET)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        add_grasp_contact_sensor(self.scene, self.scene.fresh_bulb.prim_path)
        # The bench moves with the layout draw, so the footprint the gate tests is this
        # layout's bench, not the tabletop preset's constant.
        for _fn, _params in BULB_GRASPED_CONJUNCTS:
            if _fn is grasp_terms.object_clear_of_surface:
                _params["surface_centre"] = tuple(self.scene.table.init_state.pos[:2])
        frame_viewer_on(self.viewer, self.scene.robot.init_state.pos)
