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
    LADDER_TOP_FOOT_MAX_SPEED,
    LADDER_TOP_FOOT_TOLERANCE,
    LADDER_TOP_HEIGHT_SLACK,
    LADDER_TOP_SUSTAIN_SECONDS,
    ClimbSubtaskCfg,
)
from fiatlux_task.robots.g1 import add_foot_sole, fix_sensor_frame_mass

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
    (
        balance_terms.feet_on_ladder_top,
        {"height_tolerance": LADDER_TOP_FOOT_TOLERANCE, "max_foot_speed": LADDER_TOP_FOOT_MAX_SPEED},
    ),
    (balance_terms.soles_within_ladder_top, {}),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
    (grasp_terms.ladder_near_vertical, {"tilt_limit": mdp.LADDER_TILT_LIMIT}),
]


@configclass
class S02ClimbLadderEnvCfg(ClimbSubtaskCfg):
    """Climb the placed ladder (randomized Replace layout, ladder dynamic, hands free)."""

    success_predicate = mdp.sustained
    success_params: dict | None = {
        "predicate_fn": mdp.all_of,
        "seconds": LADDER_TOP_SUSTAIN_SECONDS,
        "predicate_params": {"predicates": CLIMBED_CONJUNCTS},
    }

    # Both hands are free, so the contact bootstrap keeps its full feet-and-palms denominator.
    ladder_contact_bodies: list[str] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        # A tread is 76 mm deep and the stock foot only collides at its four corners: without the
        # sole the robot has nothing to stand on here (see robots/g1.py).
        add_foot_sole(self)
        # ... and without the 4 kg PhysX adds for four massless sensor frames (climb tasks only
        # for now; see robots/g1.py).
        fix_sensor_frame_mass(self)
