# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared tier for S06 (remove the old bulb) and S14 (screw the fresh one in).

Both are manipulation performed *while balancing*, so the tier is the balance tier plus the
manipulation channels: it inherits the placed dynamic ladder, the tipping gates and the on-ladder
stance, and adds a filtered hand-force channel with a compliance penalty and a crush gate.

No limb-on-ladder bootstrap: ``ladder_contact_fraction`` pays for limbs ON the ladder, and the
whole job here is to get a hand off it and onto the fixture.

The bayonet attach/detach state machine (issue #54, ``mdp.bulb_attachment``) is wired here so
every mate-tier leaf gets it once: without it the old bulb has nothing retaining it in the
inverted fixture and falls out under gravity from t=0, and a merely-resting fresh bulb falls out
the instant the hand releases it. Both leaves' success gates read the attach-aware predicates
(``mdp.old_bulb_removed_after_release``, ``mdp.fresh_bulb_attached``, ...) rather than the raw
geometric ones, per the module's own ordering caveat: rewards/terminations run before the
``mode="interval"`` projection step, so raw geometry can transiently read "success" a step before
the bulb has actually left (or locked into) the channel.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from fiatlux_task.grasp_poses import GLASS_CONTACT_LIMIT_N

from .. import mdp
from ..mdp import mate_terms
from ..replace_env_cfg import BAYONET_INSERTION_DEPTH, BAYONET_ROTATION_ANGLE, SEAT_ORI_THRESHOLD, SEAT_POS_THRESHOLD
from ..subtask_env_cfg import SubtaskEventCfg
from .balance import BalanceSubtaskCfg, BalanceTerminationsCfg, OnLadderRewardsCfg, stand_robot_on_ladder_top


@configclass
class MateEventCfg(SubtaskEventCfg):
    """Adds the bayonet channel enforcement, same params ``FIATLUX-Replace-v0`` validated."""

    bulb_attachment = EventTerm(
        func=mdp.bulb_attachment,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "insertion_depth": BAYONET_INSERTION_DEPTH,
            "rotation_angle": BAYONET_ROTATION_ANGLE,
            "rotation_sign": 1.0,
            "radial_tolerance": SEAT_POS_THRESHOLD,
            "orientation_tolerance": SEAT_ORI_THRESHOLD,
        },
    )

# Dense alignment kernel width, in radians of mating-axis error: the family's existing value for
# the orientation kernel this one replaces (install/g1_bulb), which is well outside the 0.2 rad
# seating gate, so the term stays informative on approach.
MATE_ALIGNMENT_STD = 0.3
MATE_ALIGNMENT_WEIGHT = 0.3

# Taking hold only has to prove the grasp is not a glancing contact (the grasp tier's window).
# Leaving something installed has to prove it stays there once the hand is off, which is a
# statement about a second of unaided physics (the place tier's window).
MATE_GRASP_SUSTAIN_SECONDS = 0.5
MATE_RELEASE_SUSTAIN_SECONDS = 1.0


@configclass
class MateRewardsCfg(OnLadderRewardsCfg):
    """Compliance and the crush penalty; the task channels come from the leaf.

    ``contact_penalty`` reads force applied to the OBJECT, not the hand's net force, so an arm
    braced against a rail while working does not read as crushing the bulb.
    """

    contact_penalty = RewTerm(
        func=mdp.hand_contact_force_l2, weight=-1.0e-4, params={"sensor_cfg": SceneEntityCfg("grip_contact")}
    )
    bulb_crushed = RewTerm(
        func=mate_terms.grip_force_exceeded,
        weight=-200.0,
        params={"sensor_cfg": SceneEntityCfg("grip_contact"), "limit": GLASS_CONTACT_LIMIT_N},
    )


@configclass
class MateTerminationsCfg(BalanceTerminationsCfg):
    """Adds the crush gate: past the glass bound the bulb is broken and the episode is over.

    A success conjunct could not express this -- it would only be read at the scoring step, so a
    bulb crushed on the way and then seated anyway would still score.
    """

    bulb_crushed = DoneTerm(
        func=mate_terms.grip_force_exceeded,
        params={"sensor_cfg": SceneEntityCfg("grip_contact"), "limit": GLASS_CONTACT_LIMIT_N},
    )


@configclass
class MateSubtaskCfg(BalanceSubtaskCfg):
    """Manipulation at the fixture from the ladder's upper steps."""

    events: MateEventCfg = MateEventCfg()
    rewards: MateRewardsCfg = MateRewardsCfg()
    terminations: MateTerminationsCfg = MateTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        stand_robot_on_ladder_top(self.scene)
