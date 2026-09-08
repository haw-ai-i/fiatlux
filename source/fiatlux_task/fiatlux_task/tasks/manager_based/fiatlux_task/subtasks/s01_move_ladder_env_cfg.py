# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S01-MoveLadder-v0`` -- get the ladder from wherever it stands to the fixture.

One episode covering what the retired ``S01-ApproachLadder`` / ``S02-GrabLadder`` /
``S03-CarryLadder`` / ``S04-PlaceLadder`` split scored as four: walk to the ladder, take it,
move it, stand it up under the fixture. Head of the chain, so its start state is drawn from the
randomized layout rather than composed from a predecessor's end state -- the robot spawns in its
own zone with both hands free, and the ladder in its own, independently-sampled one
(``couple_ladder_to_fixture`` stays off: positioning the ladder is the task).

**Success is the ladder standing at the fixture, and nothing about how it got there.** Carrying,
dragging, shouldering, pushing along the floor and nudging it a foot at a time all count the same:
no gate conjunct reads contact, so none of them can tell those apart. The gate is the deliverable
itself, held for ``PLACE_SUSTAIN_SECONDS``:

* ``ladder_ready`` -- the ladder's top is horizontally within ``LADDER_READY_XY_RADIUS`` of the
  fixture and the ladder is upright;
* ``ladder_feet_down`` -- its root is on the floor, so a ladder merely *held* in the right place
  at the right angle does not read as standing;
* ``object_at_rest`` -- it has settled, not swung through;
* ``robot_standing`` -- the robot has not collapsed on the step the gate would otherwise fire.

The successor climbs from a stance it spawns for itself
(``subtask_tiers.balance.stand_robot_at_ladder_base``), so the retired ``robot_at_ladder_base``
conjunct constrained a handoff that does not exist at run time.

The one remaining constraint on how the ladder travels is the ``ladder_tipped`` termination
(0.6 rad from vertical), the family's shared constant and what "standing" means here: a 34-degree
lean while shoving the ladder along passes, laying it flat and walking it end over end does not.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp
from ..mdp import place_terms
from ..mdp.nav_terms import add_grip_contact_sensor
from ..scene_cfg import (
    LADDER_READY_XY_RADIUS,
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    face_robot_at,
    frame_viewer_between,
)
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT, SubtaskTerminationsCfg
from ..subtask_tiers.balance import BulbAttachmentEventCfg
from ..subtask_tiers.place import (
    AT_REST_ANG_VEL_LIMIT,
    AT_REST_LIN_VEL_LIMIT,
    PLACE_SUSTAIN_SECONDS,
    PlaceRewardsCfg,
    PlaceSubtaskCfg,
)

# The ladder's root reads "on the floor" within this of the floor plane.
LADDER_FEET_TOLERANCE = 0.02  # m

# The success gate as data (mdp.all_of): an omitted conjunct in a hand-written conjunction is a
# gate that passes vacuously.
LADDER_MOVED_CONJUNCTS = [
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
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
]


@configclass
class S01RewardsCfg(PlaceRewardsCfg):
    """The place tier's ``placement_progress`` (ladder -> fixture) plus the navigate tier's
    ``approach_progress`` (robot -> ladder), under both canonical names: this leaf spans the two
    legs those tiers score separately, and without the second channel the breakdown is silent for
    the walk out to the ladder."""

    approach_progress = RewTerm(
        func=mdp.distance_progress, weight=500.0, params={"distance_fn": mdp.base_ladder_distance}
    )
    ladder_tipped = RewTerm(func=mdp.ladder_tipped, weight=-200.0, params={"tilt_limit": mdp.LADDER_TILT_LIMIT})


@configclass
class S01TerminationsCfg(SubtaskTerminationsCfg):
    ladder_tipped = DoneTerm(func=mdp.ladder_tipped, params={"tilt_limit": mdp.LADDER_TILT_LIMIT})
    # No ladder_dropped: setting the ladder down is the goal here, not a failure mode.


@configclass
class S01MoveLadderEnvCfg(PlaceSubtaskCfg):
    """Move the ladder to the fixture (randomized Replace layout, ladder dynamic, hands free)."""

    scene_preset: str = "replace"
    orbit_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orbit_radius: float = 5.0
    orbit_height: float = 2.4

    success_predicate = mdp.sustained
    success_params: dict | None = {
        "predicate_fn": mdp.all_of,
        "seconds": PLACE_SUSTAIN_SECONDS,
        "predicate_params": {"predicates": LADDER_MOVED_CONJUNCTS},
    }
    progress_distance_fn = mdp.ladder_fixture_distance

    rewards: S01RewardsCfg = S01RewardsCfg()
    terminations: S01TerminationsCfg = S01TerminationsCfg()
    # Lock the socketed old bulb into the overhead fixture with the bayonet FSM (#108). The Place
    # tier wires no events, so a dynamic bulb in the inverted socket just falls out at spawn; this
    # is the same attach term the Balance/Descend tiers already use to hold their seated bulbs.
    events: BulbAttachmentEventCfg = BulbAttachmentEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Uncoupled draw: the ladder's zone is sampled independently of the fixture, because
        # closing that gap is the whole task.
        apply_replace_preset(self.scene)
        # The preset aims the robot at the table; this leg's first target is the ladder.
        face_robot_at(self.scene, self.scene.ladder.init_state.pos[:2])
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        # Ladder grip force, MEASUREMENT ONLY -- no gate conjunct reads it, so success still says
        # nothing about how the ladder travelled (see the module docstring). Issue #106: the
        # shared ``hand_contact`` filters whichever bulbs the preset built, so on this leg -- where
        # the ladder is the manipuland and the bulbs are scenery -- it reads a flat 0.0 N, which
        # is indistinguishable from a dead sensor. Filtered to the ladder alone, this column is
        # attributable, and ``recording.py`` writes it into the bag as ``grip_force``.
        add_grip_contact_sensor(self.scene, self.scene.ladder.prim_path)
        self.episode_length_s = 120.0
        frame_viewer_between(self.viewer, self.scene.robot.init_state.pos, self.scene.ladder.init_state.pos)
