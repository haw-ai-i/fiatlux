# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S02-GrabLadder-v0`` -- grasp a rail and take the ladder's weight.

Success is filtered hand-to-ladder contact force past a threshold, held for
``GRASP_SUSTAIN_SECONDS``, while the ladder stands clear of the floor, stays upright, and the
robot stays standing.

The ladder's root is the A-frame's base centre, so tilting the ladder raises the root exactly
the way lifting it does: a 3 cm root rise costs only ~5.1 deg of lean, well under the 34 deg
(``mdp.LADDER_TILT_LIMIT`` = 0.6 rad) tipping termination. The gate below never reads the root
height for that reason -- ``grasp_terms.ladder_feet_clear`` reads it only as a floor-clearance
check, paired with ``grasp_terms.ladder_near_vertical`` at a much tighter 0.15 rad, so "raised by
leaning" fails the tilt conjunct long before it could pass the clearance one.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp
from ..mdp import grasp_terms, place_terms
from ..scene_cfg import (
    LADDER_APPROACH_RADIUS,
    LADDER_MASS_KG,
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    frame_viewer_on,
    stand_robot_near,
)
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT, SubtaskTerminationsCfg
from ..subtask_tiers.grasp import GRASP_SUSTAIN_SECONDS, GraspRewardsCfg, GraspSubtaskCfg, add_grasp_contact_sensor

# 0.8 x mg (mg = LADDER_MASS_KG x 9.81 m/s^2): enough to show the HAND is carrying the ladder,
# not the floor -- comfortably above noise, comfortably below the 500 N solver-wedge tripwire
# (``grasp_poses.LADDER_GRIP_TRIPWIRE_N``).
LADDER_GRASP_FORCE_N = 0.8 * LADDER_MASS_KG * 9.81  # ~56.9 N

# Root rise counted as "feet clear of the floor": more than resting-contact noise, well short of
# a real step. PROVISIONAL -- not measured against the settled ladder's contact jitter.
LADDER_LIFT_CLEARANCE_M = 0.02  # m

# Tight enough that leaning to fake the clearance conjunct fails this one first: 5.1 deg of lean
# clears LADDER_LIFT_CLEARANCE_M, well under this bound, and far under the 34 deg tipping
# termination. Never gate success on the ladder's root height alone.
LADDER_GRASP_TILT_LIMIT = 0.15  # rad

# The success gate, as reviewable data (mdp.all_of) rather than a hand-written conjunction --
# an omitted conjunct here is a gate that passes vacuously.
LADDER_GRASPED_CONJUNCTS = [
    (
        grasp_terms.grasp_force_above,
        {"sensor_cfg": SceneEntityCfg("grasp_contact"), "force_threshold": LADDER_GRASP_FORCE_N},
    ),
    (grasp_terms.ladder_feet_clear, {"tolerance": LADDER_LIFT_CLEARANCE_M}),
    (grasp_terms.ladder_near_vertical, {"tilt_limit": LADDER_GRASP_TILT_LIMIT}),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
]


@configclass
class S02RewardsCfg(GraspRewardsCfg):
    ladder_tipped = RewTerm(func=mdp.ladder_tipped, weight=-200.0, params={"tilt_limit": mdp.LADDER_TILT_LIMIT})


@configclass
class S02TerminationsCfg(SubtaskTerminationsCfg):
    ladder_tipped = DoneTerm(func=mdp.ladder_tipped, params={"tilt_limit": mdp.LADDER_TILT_LIMIT})


@configclass
class S02GrabLadderEnvCfg(GraspSubtaskCfg):
    """Grasp a rail and take the ladder's weight (randomized Replace layout, ladder dynamic)."""

    scene_preset: str = "replace"
    orbit_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orbit_radius: float = 5.0
    orbit_height: float = 2.4

    success_predicate = mdp.sustained
    success_params: dict | None = {
        "predicate_fn": mdp.all_of,
        "seconds": GRASP_SUSTAIN_SECONDS,
        "predicate_params": {"predicates": LADDER_GRASPED_CONJUNCTS},
    }
    progress_distance_fn = grasp_terms.hand_ladder_distance

    rewards: S02RewardsCfg = S02RewardsCfg()
    terminations: S02TerminationsCfg = S02TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_replace_preset(self.scene)
        # The preset's own robot zone is independent of the ladder's; pull the robot into grasp
        # range and re-aim it there instead of just at the table.
        stand_robot_near(self.scene, self.scene.ladder.init_state.pos[:2], LADDER_APPROACH_RADIUS)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        add_grasp_contact_sensor(self.scene, self.scene.ladder.prim_path)
        self.episode_length_s = 20.0
        frame_viewer_on(self.viewer, self.scene.robot.init_state.pos)
