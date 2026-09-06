# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S03-RemoveOldBulb-v0`` -- free the old bulb from the fixture while on the ladder.

Starts from S02's end state: the robot balanced on the upper steps with hands free, the old bulb
locked in the fixture's bayonet channel (``mdp.bulb_attachment`` resets it ``ROTATING`` at full
lock angle -- issue #54, wired once for the tier in ``subtask_tiers.mate.MateEventCfg``).

**The held conjunct is the entire subtask.** A gate that only checked geometric clearance from the
fixture could be satisfied by the bulb sitting anywhere the bayonet projection allows without ever
being taken -- ``old_bulb_removed_after_release`` requires the attach state machine to have
actually left the channel (``_phase != ROTATING``), which only happens if something rotates it
through the unlock angle and pulls it clear. A zero-action rollout scores 0.

``removal_progress`` uses the away-from formulation: the bulb starts AT the fixture, so the
normalized ``(d0 - d) / d0`` form would divide by ~zero. It reads the attach-aware
``old_bulb_release_clearance``, pinned to 0 while the bulb is still constrained, per
``mdp.attach``'s own ordering caveat (rewards run before the interval projection step, so raw
geometry can transiently read clear a step before the bulb has actually exited the channel).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp
from ..mdp import grasp_terms, place_terms
from ..mdp.nav_terms import GRIP_FORCE_THRESHOLD_N, add_grip_contact_sensor, payload_held
from ..replace_env_cfg import OLD_BULB_DROP_HEIGHT, REMOVAL_CLEARANCE
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT
from ..subtask_tiers.mate import MATE_GRASP_SUSTAIN_SECONDS, MateRewardsCfg, MateSubtaskCfg, MateTerminationsCfg

_OLD_BULB = SceneEntityCfg("old_bulb")

# The success gate, as reviewable data (mdp.all_of) rather than a hand-written conjunction --
# an omitted conjunct here is a gate that passes vacuously.
OLD_BULB_TAKEN_CONJUNCTS = [
    (mdp.old_bulb_removed_after_release, {"clearance_threshold": REMOVAL_CLEARANCE}),
    (payload_held, {"sensor_cfg": SceneEntityCfg("grip_contact"), "force_threshold": GRIP_FORCE_THRESHOLD_N}),
    (grasp_terms.object_lifted, {"asset_cfg": _OLD_BULB, "min_height": OLD_BULB_DROP_HEIGHT}),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
    (grasp_terms.ladder_near_vertical, {"tilt_limit": mdp.LADDER_TILT_LIMIT}),
]


@configclass
class S03RewardsCfg(MateRewardsCfg):
    reach_progress = RewTerm(
        func=mdp.distance_progress, weight=500.0, params={"distance_fn": grasp_terms.hand_old_bulb_distance}
    )
    removal_progress = RewTerm(
        func=mdp.distance_progress,
        weight=500.0,
        params={"distance_fn": mdp.old_bulb_release_clearance, "away_threshold": REMOVAL_CLEARANCE},
    )
    # Not attach-aware: the bayonet projection pins the bulb near fixture height for the
    # entire ROTATING/AXIAL phase, well above OLD_BULB_DROP_HEIGHT, so there is no transient
    # mid-step value near this threshold for the interval projection to correct.
    bulb_dropped = RewTerm(
        func=mdp.object_dropped, weight=-200.0, params={"asset_cfg": _OLD_BULB, "min_height": OLD_BULB_DROP_HEIGHT}
    )


@configclass
class S03TerminationsCfg(MateTerminationsCfg):
    bulb_dropped = DoneTerm(
        func=mdp.object_dropped, params={"asset_cfg": _OLD_BULB, "min_height": OLD_BULB_DROP_HEIGHT}
    )


@configclass
class S03RemoveOldBulbEnvCfg(MateSubtaskCfg):
    """Take the old bulb out of the fixture from the ladder (randomized Replace layout)."""

    success_predicate = mdp.sustained
    success_params: dict | None = {
        "predicate_fn": mdp.all_of,
        "seconds": MATE_GRASP_SUSTAIN_SECONDS,
        "predicate_params": {"predicates": OLD_BULB_TAKEN_CONJUNCTS},
    }

    rewards: S03RewardsCfg = S03RewardsCfg()
    terminations: S03TerminationsCfg = S03TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # The old bulb stays where apply_replace_preset put it: seated in the inverted fixture.
        add_grip_contact_sensor(self.scene, self.scene.old_bulb.prim_path)
        self.episode_length_s = 120.0
