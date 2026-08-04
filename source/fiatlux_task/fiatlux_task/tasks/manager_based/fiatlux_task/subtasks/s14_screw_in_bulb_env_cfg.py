# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S14-ScrewInBulb-v0`` -- seat the fresh bulb in the fixture. Terminal manipulation.

Starts from S13's end state: the robot balanced on the upper steps with the fresh bulb in hand
(``BULB_IN_ROOT_ON_LADDER``), the fixture inverted and empty, the old bulb in the disposal crate.

The bayonet attach/detach state machine (issue #54, ``mdp.bulb_attachment``, wired once for the
tier in ``subtask_tiers.mate.MateEventCfg``) governs the fresh bulb from here: FREE at reset
(``BULB_IN_ROOT_ON_LADDER`` puts it in the hand, not the insertion channel, so it starts
unconstrained same as before), through AXIAL once it enters the channel, to ROTATING once
bottomed and turned through the full lock angle. ``mdp.fresh_bulb_attached`` is the success
conjunct that actually requires that whole sequence, not just transiting the old geometric seating
thresholds -- the attach-aware replacement for ``bulb_seated``.

Rotation about the mating axis is free by construction. The dense alignment term scores the angle
BETWEEN the plug and seat axes (``mate_terms.bulb_axis_alignment_tanh``), not full-quaternion
error, which would grow as the bulb is screwed home and fight the motion the task is named for.

What makes the gate "screwed in" rather than "held in the socket" is that the hand must be OFF and
the bulb still locked a second later -- the conjunction is debounced as a whole, so every part of
it has to survive the release.

Still open, NOT fixed by the attach FSM: the start state itself. The bayonet mechanic governs
bulb-vs-socket, not bulb-vs-hand -- ``compose_carried_pose`` positions the bulb correctly but
nothing holds it there, so it is subject to the same free-fall-from-a-carried-pose gap as
S03/S04/S07/S08/S12/S13 for the first part of the episode, until it either gets carried into the
channel (which starts constraining it) or hits the floor first.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from fiatlux_task.grasp_poses import BULB_IN_ROOT_ON_LADDER

from .. import mdp
from ..mdp import grasp_terms, mate_terms, place_terms
from ..mdp.nav_terms import add_grip_contact_sensor, compose_carried_pose
from ..replace_env_cfg import FRESH_BULB_DROP_HEIGHT
from ..scene_cfg import park_old_bulb_in_crate
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT
from ..subtask_tiers.mate import (
    MATE_ALIGNMENT_STD,
    MATE_ALIGNMENT_WEIGHT,
    MATE_RELEASE_SUSTAIN_SECONDS,
    MateRewardsCfg,
    MateSubtaskCfg,
    MateTerminationsCfg,
)
from ..subtask_tiers.place import AT_REST_ANG_VEL_LIMIT, AT_REST_LIN_VEL_LIMIT, RELEASE_FORCE_THRESHOLD_N

_BULB = SceneEntityCfg("bulb")
_GRIP = SceneEntityCfg("grip_contact")

# The success gate, as reviewable data (mdp.all_of) rather than a hand-written conjunction --
# an omitted conjunct here is a gate that passes vacuously.
BULB_SCREWED_IN_CONJUNCTS = [
    (mdp.fresh_bulb_attached, {}),
    (
        place_terms.object_at_rest,
        {"asset_cfg": _BULB, "lin_vel_limit": AT_REST_LIN_VEL_LIMIT, "ang_vel_limit": AT_REST_ANG_VEL_LIMIT},
    ),
    (place_terms.object_released, {"sensor_cfg": _GRIP, "force_threshold": RELEASE_FORCE_THRESHOLD_N}),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
    (grasp_terms.ladder_near_vertical, {"tilt_limit": mdp.LADDER_TILT_LIMIT}),
]


@configclass
class S14RewardsCfg(MateRewardsCfg):
    approach_progress = RewTerm(
        func=mdp.distance_progress, weight=500.0, params={"distance_fn": mdp.bulb_fixture_distance}
    )
    alignment = RewTerm(
        func=mate_terms.bulb_axis_alignment_tanh, weight=MATE_ALIGNMENT_WEIGHT, params={"std": MATE_ALIGNMENT_STD}
    )
    bulb_dropped = RewTerm(
        func=mdp.object_dropped, weight=-200.0, params={"asset_cfg": _BULB, "min_height": FRESH_BULB_DROP_HEIGHT}
    )


@configclass
class S14TerminationsCfg(MateTerminationsCfg):
    bulb_dropped = DoneTerm(func=mdp.object_dropped, params={"asset_cfg": _BULB, "min_height": FRESH_BULB_DROP_HEIGHT})


@configclass
class S14ScrewInBulbEnvCfg(MateSubtaskCfg):
    """Seat the fresh bulb in the fixture from the ladder (randomized Replace layout)."""

    success_predicate = mdp.sustained
    success_params: dict | None = {
        "predicate_fn": mdp.all_of,
        "seconds": MATE_RELEASE_SUSTAIN_SECONDS,
        "predicate_params": {"predicates": BULB_SCREWED_IN_CONJUNCTS},
    }

    rewards: S14RewardsCfg = S14RewardsCfg()
    terminations: S14TerminationsCfg = S14TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        park_old_bulb_in_crate(self.scene)
        # S13's end state: the fresh bulb starts already held, at the carry offset from the
        # robot's own (now-final) on-ladder root pose.
        self.scene.bulb.init_state.pos, self.scene.bulb.init_state.rot = compose_carried_pose(
            self.scene.robot.init_state.pos, self.scene.robot.init_state.rot, BULB_IN_ROOT_ON_LADDER
        )
        add_grip_contact_sensor(self.scene, self.scene.bulb.prim_path)
        # The longest of the chain: fine insertion under balance.
        self.episode_length_s = 40.0
