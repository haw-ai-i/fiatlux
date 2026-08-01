# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S06-RemoveOldBulb-v0`` -- free the old bulb from the fixture while on the ladder.

Starts from S05's end state: the robot balanced on the upper steps with hands free, the old bulb
seated in the inverted fixture above, at the fixture's own pose (both halves authored assembled at
identity).

**The held conjunct is the entire subtask.** Clearance alone is satisfied by the bulb falling out
of an inverted socket under gravity -- which, with #54 unimplemented, is exactly what happens at
t=0 with no action at all, so a clearance-only gate would score 100% for a policy that does
nothing. A zero-action rollout must score 0.

``removal_progress`` uses the away-from formulation: the bulb starts AT the fixture, so the
normalized ``(d0 - d) / d0`` form would divide by ~zero.

BLOCKED on #54 (bulb attach/detach). The success gate is PROVISIONAL until it lands.
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
    (mdp.old_bulb_removed, {"clearance_threshold": REMOVAL_CLEARANCE, "asset_cfg": _OLD_BULB}),
    (payload_held, {"sensor_cfg": SceneEntityCfg("grip_contact"), "force_threshold": GRIP_FORCE_THRESHOLD_N}),
    (grasp_terms.object_lifted, {"asset_cfg": _OLD_BULB, "min_height": OLD_BULB_DROP_HEIGHT}),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
    (grasp_terms.ladder_near_vertical, {"tilt_limit": mdp.LADDER_TILT_LIMIT}),
]


@configclass
class S06RewardsCfg(MateRewardsCfg):
    reach_progress = RewTerm(
        func=mdp.distance_progress, weight=500.0, params={"distance_fn": grasp_terms.hand_old_bulb_distance}
    )
    removal_progress = RewTerm(
        func=mdp.distance_progress,
        weight=500.0,
        params={"distance_fn": mdp.old_bulb_fixture_clearance, "away_threshold": REMOVAL_CLEARANCE},
    )
    bulb_dropped = RewTerm(
        func=mdp.object_dropped, weight=-200.0, params={"asset_cfg": _OLD_BULB, "min_height": OLD_BULB_DROP_HEIGHT}
    )


@configclass
class S06TerminationsCfg(MateTerminationsCfg):
    bulb_dropped = DoneTerm(
        func=mdp.object_dropped, params={"asset_cfg": _OLD_BULB, "min_height": OLD_BULB_DROP_HEIGHT}
    )


@configclass
class S06RemoveOldBulbEnvCfg(MateSubtaskCfg):
    """Take the old bulb out of the fixture from the ladder (randomized Replace layout)."""

    success_predicate = mdp.sustained
    success_params: dict | None = {
        "predicate_fn": mdp.all_of,
        "seconds": MATE_GRASP_SUSTAIN_SECONDS,
        "predicate_params": {"predicates": OLD_BULB_TAKEN_CONJUNCTS},
    }

    rewards: S06RewardsCfg = S06RewardsCfg()
    terminations: S06TerminationsCfg = S06TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # The old bulb stays where apply_replace_preset put it: seated in the inverted fixture.
        add_grip_contact_sensor(self.scene, self.scene.old_bulb.prim_path)
        self.episode_length_s = 30.0
