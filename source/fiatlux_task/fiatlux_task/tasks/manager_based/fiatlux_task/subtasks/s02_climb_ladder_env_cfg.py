# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S02-ClimbLadder-v0`` -- climb the placed ladder to working height, hands free.

Starts from S01's end state: the ladder standing at rest at the fixture, the robot on the floor at
its steps, the old bulb still seated above. Success is a controlled stance on the upper steps, with
the height and the horizontal centre read from the ladder's LIVE pose -- an older, fixed-layout
approach hardcoded both from ``TOP_ROBOT_POSITION``, which describes the default workshop layout
rather than wherever a placement subtask left this ladder.
"""

from isaaclab.utils import configclass

from .. import mdp
from ..mdp import balance_terms, grasp_terms, place_terms
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT
from ..subtask_tiers.balance import (
    LADDER_SUCCESS_MAX_SPEED,
    LADDER_SUCCESS_XY_RADIUS,
    LADDER_TOP_HEIGHT_SLACK,
    ClimbSubtaskCfg,
)

# The success gate, as reviewable data (mdp.all_of) rather than a hand-written conjunction --
# an omitted conjunct here is a gate that passes vacuously.
CLIMBED_CONJUNCTS = [
    (
        balance_terms.climbed_to_ladder_top,
        {
            "height_slack": LADDER_TOP_HEIGHT_SLACK,
            "xy_radius": LADDER_SUCCESS_XY_RADIUS,
            "max_speed": LADDER_SUCCESS_MAX_SPEED,
        },
    ),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
    (grasp_terms.ladder_near_vertical, {"tilt_limit": mdp.LADDER_TILT_LIMIT}),
]


@configclass
class S02ClimbLadderEnvCfg(ClimbSubtaskCfg):
    """Climb the placed ladder (randomized Replace layout, ladder dynamic, hands free)."""

    success_predicate = mdp.all_of
    success_params: dict | None = {"predicates": CLIMBED_CONJUNCTS}

    # Both hands are free, so the contact bootstrap keeps its full feet-and-palms denominator.
    ladder_contact_bodies: list[str] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
