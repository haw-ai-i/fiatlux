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

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from fiatlux_task.assets import BULB_STAND_Z_OFFSET, G1_HORIZONTAL_REACH
from fiatlux_task.grasp_poses import GLASS_CONTACT_LIMIT_N

from .. import mdp
from ..mdp import grasp_terms, place_terms
from ..scene_cfg import (
    TABLETOP_BULB_POSITION,
    TABLETOP_ROBOT_POSITION,
    TABLETOP_SURFACE_Z,
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    frame_viewer_on,
    stand_robot_at_offset,
)
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT, SubtaskTerminationsCfg
from ..subtask_tiers.grasp import GRASP_SUSTAIN_SECONDS, GraspRewardsCfg, GraspSubtaskCfg, add_grasp_contact_sensor

# The fresh bulb's resting root height on the (always z=0) table, standing on its cap -- the same
# derivation apply_replace_preset itself uses for TABLETOP_BULB_POSITION.
BULB_TABLETOP_REST_Z = TABLETOP_SURFACE_Z - BULB_STAND_Z_OFFSET  # m

# Which side of the table to stand on, from the non-subtask tabletop task's own layout (a
# direction, not a distance -- standing distance alone can put the robot under the table).
_TABLETOP_APPROACH_DX = TABLETOP_ROBOT_POSITION[0] - TABLETOP_BULB_POSITION[0]
_TABLETOP_APPROACH_DY = TABLETOP_ROBOT_POSITION[1] - TABLETOP_BULB_POSITION[1]
_TABLETOP_APPROACH_DIST = math.hypot(_TABLETOP_APPROACH_DX, _TABLETOP_APPROACH_DY)

# Same direction, but standing close enough to actually reach the bulb: G1_HORIZONTAL_REACH is
# the measured forward-arm reach (same constant LADDER_APPROACH_RADIUS is built from), less a
# margin. The un-scaled tabletop offset (0.626 m) is a walking-approach distance from a scene
# where the robot closes the gap over the episode, not a grasp-ready standing point -- at that
# distance the bulb is 12 cm outside reach even fully extended.
BULB_APPROACH_OFFSET = (
    _TABLETOP_APPROACH_DX / _TABLETOP_APPROACH_DIST * (G1_HORIZONTAL_REACH - 0.05),
    _TABLETOP_APPROACH_DY / _TABLETOP_APPROACH_DIST * (G1_HORIZONTAL_REACH - 0.05),
)

# Root rise counted as "lifted": more than resting-contact/settling jitter. PROVISIONAL -- not
# measured against the settled bulb's own jitter.
BULB_LIFT_CLEARANCE_M = 0.03  # m
BULB_LIFTED_HEIGHT_M = BULB_TABLETOP_REST_Z + BULB_LIFT_CLEARANCE_M

# Comfortably below the tabletop rest height, comfortably above the floor (~-0.036 m): catches
# "knocked off the table" before it lands. PROVISIONAL.
BULB_DROPPED_HEIGHT_M = 0.5  # m

# Any contact above sensor noise counts toward "a body is on the bulb"; the fragility bound is a
# separate, upper-bound conjunct below. Same order as the place tier's release threshold.
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
        apply_replace_preset(self.scene)
        # The preset's own robot zone is independent of the table's; place the robot at the
        # table's own validated approach vector from the bulb, not just some radius from it
        # (a radius alone can land the robot under the table).
        stand_robot_at_offset(self.scene, self.scene.bulb.init_state.pos[:2], BULB_APPROACH_OFFSET)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        add_grasp_contact_sensor(self.scene, self.scene.bulb.prim_path)
        self.episode_length_s = 20.0
        frame_viewer_on(self.viewer, self.scene.robot.init_state.pos)
