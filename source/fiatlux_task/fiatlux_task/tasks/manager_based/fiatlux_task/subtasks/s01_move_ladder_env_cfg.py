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

**Success is the ladder standing at the fixture, and nothing about how it got there.** The four
retired gates between them also required a rail grasped, the ladder's feet lifted clear of the
floor, the grip retained through the traverse, the hand released at the end, and the robot left on
the ladder's step-facing side. None of those are conditions on the deliverable; each is one more
way for an operator who has already stood the ladder where it belongs to score zero. Carrying it,
dragging it, shouldering it, pushing it across the floor, or nudging it along a foot at a time all
count the same. Nothing here reads a contact sensor -- this leaf adds none -- so no gate can tell
those apart. What remains is the deliverable itself, held for ``PLACE_SUSTAIN_SECONDS``:

* ``ladder_ready`` -- the ladder's top is horizontally within ``LADDER_READY_XY_RADIUS`` of the
  fixture and the ladder is upright;
* ``ladder_feet_down`` -- its root is on the floor, so a ladder merely *held* in the right place
  at the right angle does not read as standing;
* ``object_at_rest`` -- it has settled, not swung through;
* ``robot_standing`` -- the robot has not collapsed on the step the gate would otherwise fire.

The successor climbs the ladder from a stance it spawns for itself
(``subtask_tiers.balance.stand_robot_at_ladder_base``), so dropping the retired
``robot_at_ladder_base`` conjunct costs the chain nothing: where this leg leaves the robot was
never what the next leg started from.

The ONE remaining constraint on how the ladder travels is ``ladder_tipped`` (0.6 rad from
vertical), which terminates the episode. It is the family's shared constant, carried by nine other
leaves and by ``FIATLUX-Replace-v0``, and it is what "standing" means here -- so laying the ladder
flat and walking it end over end is out, while a 34-degree lean while shoving it along is not. If
end-over-end walking is wanted, that is a change to this leaf's tilt bound, not to the gate.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp
from ..mdp import place_terms
from ..scene_cfg import (
    LADDER_READY_XY_RADIUS,
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    face_robot_at,
    frame_viewer_between,
)
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT, SubtaskTerminationsCfg
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
    ``approach_progress`` (robot -> ladder), under both canonical names and both canonical
    functions: this leaf spans the two legs those tiers score separately, and without the second
    channel the breakdown is silent for the whole walk out to the ladder."""

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

    def __post_init__(self) -> None:
        super().__post_init__()
        # Uncoupled draw: the ladder's zone is sampled independently of the fixture, because
        # closing that gap is the whole task.
        apply_replace_preset(self.scene)
        # The preset aims the robot at the table; this leg's first target is the ladder.
        face_robot_at(self.scene, self.scene.ladder.init_state.pos[:2])
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        # The three retired horizons summed (20 s approach + 45 s loaded traverse + 20 s
        # placement), rounded up. PROVISIONAL, like every horizon in the family that no rollout
        # has set.
        self.episode_length_s = 90.0
        frame_viewer_between(self.viewer, self.scene.robot.init_state.pos, self.scene.ladder.init_state.pos)
