# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S12-ClimbDown-v0`` -- come back down with the new bulb installed. Terminal subtask.

Starts from S11's end state: the robot balanced on the upper steps with hands free, the fresh bulb
seated in the fixture above, the old bulb at rest in the disposal crate.

What makes this more than a descent is the last conjunct: the bulb must still be seated on
arrival. A robot that shakes the ladder hard enough to unseat the freshly installed bulb on the way
down has undone the task, so leaving the socket also ends the episode.

Its success region is the chain's declared terminal state -- there is no successor to hand off to.

The axial retention spring (issue #167, ``mdp.bulb_attachment``) is wired at the balance tier
(``subtask_tiers.balance.BalanceEventCfg``), retaining the bulb in the inverted fixture while
the robot descends.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp
from ..mdp import balance_terms, grasp_terms, place_terms
from ..replace_env_cfg import SEAT_ORI_THRESHOLD, SEAT_POS_THRESHOLD
from ..scene_cfg import park_old_bulb_in_crate, seat_bulb_in_fixture
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT, SubtaskRewardsCfg
from ..subtask_tiers.balance import (
    LADDER_FLOOR_STANCE_HEIGHT,
    LADDER_SUCCESS_MAX_SPEED,
    LADDER_SUCCESS_XY_RADIUS,
    BalanceTerminationsCfg,
    DescendRewardsCfg,
    DescendSubtaskCfg,
)

_SEATING = {"pos_threshold": SEAT_POS_THRESHOLD, "ori_threshold": SEAT_ORI_THRESHOLD}

# The success gate, as reviewable data (mdp.all_of) rather than a hand-written conjunction --
# an omitted conjunct here is a gate that passes vacuously.
DESCENDED_INTACT_CONJUNCTS = [
    (
        balance_terms.descended_from_ladder,
        {
            "maximum_height": LADDER_FLOOR_STANCE_HEIGHT,
            "xy_radius": LADDER_SUCCESS_XY_RADIUS,
            "max_speed": LADDER_SUCCESS_MAX_SPEED,
        },
    ),
    (mdp.bulb_seated, _SEATING),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
    (grasp_terms.ladder_near_vertical, {"tilt_limit": mdp.LADDER_TILT_LIMIT}),
]


@configclass
class S12RewardsCfg(DescendRewardsCfg):
    # Undoing the task's goal costs what destroying its equipment costs (ladder_tipped).
    bulb_unseated = RewTerm(func=mdp.bulb_unseated, weight=-200.0, params=_SEATING)


@configclass
class S12TerminationsCfg(BalanceTerminationsCfg):
    bulb_unseated = DoneTerm(func=mdp.bulb_unseated, params=_SEATING)


@configclass
class S12ClimbDownEnvCfg(DescendSubtaskCfg):
    """Descend the placed ladder leaving the new bulb installed (randomized Replace layout)."""

    # Both hands are free, so the contact bootstrap keeps its full feet-and-palms denominator.
    ladder_contact_bodies: list[str] | None = None

    success_predicate = mdp.all_of
    success_params: dict | None = {"predicates": DESCENDED_INTACT_CONJUNCTS}

    rewards: SubtaskRewardsCfg = SubtaskRewardsCfg()
    terminations: S12TerminationsCfg = S12TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # S11's end state: the job is done -- fresh bulb in the fixture, old one thrown away.
        park_old_bulb_in_crate(self.scene)
        seat_bulb_in_fixture(self.scene)
