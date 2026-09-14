# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S04-DescendWithBulb-v0`` -- carry the removed old bulb down the ladder.

Starts from S03's end state: the robot on the upper steps with the old bulb in hand
(``BULB_IN_ROOT_ON_LADDER``), the fixture above now empty. Success is a controlled arrival at
floor stance beside the ladder with the bulb still held and unbroken -- without the held conjunct,
dropping the bulb off the top and walking down after it would score.

``fell_below`` (0.35 m) sits well under the success height, so a controlled arrival fires
``success``; a landing hard enough to dip the pelvis under 0.35 m reads as a fall, which is
correct rather than something to tune away.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from fiatlux_task.grasp_poses import BULB_IN_ROOT_ON_LADDER
from fiatlux_task.poses import ARM_CRADLE, HAND_CUP

from .. import mdp
from ..mdp import balance_terms, grasp_terms, place_terms
from ..mdp.nav_terms import (
    GRIP_FORCE_THRESHOLD_N,
    add_grip_contact_sensor,
    compose_carried_pose,
    payload_held,
    settle_carried_payload_live,
)
from ..replace_env_cfg import OLD_BULB_DROP_HEIGHT
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT, SubtaskRewardsCfg
from ..subtask_tiers.balance import (
    LADDER_FLOOR_STANCE_HEIGHT,
    LADDER_SUCCESS_MAX_SPEED,
    LADDER_SUCCESS_XY_RADIUS,
    LOADED_LADDER_CONTACT_BODIES,
    BalanceEventCfg,
    BalanceTerminationsCfg,
    DescendRewardsCfg,
    DescendSubtaskCfg,
)
from ..subtask_tiers.carrying import add_bulb_crush_gate

_OLD_BULB = SceneEntityCfg("old_bulb")
_GRIP = SceneEntityCfg("grip_contact")
_GRIP_LEFT = SceneEntityCfg("grip_contact_left")

# The success gate, as reviewable data (mdp.all_of) rather than a hand-written conjunction --
# an omitted conjunct here is a gate that passes vacuously.
DESCENDED_WITH_BULB_CONJUNCTS = [
    (
        balance_terms.descended_from_ladder,
        {
            "maximum_height": LADDER_FLOOR_STANCE_HEIGHT,
            "xy_radius": LADDER_SUCCESS_XY_RADIUS,
            "max_speed": LADDER_SUCCESS_MAX_SPEED,
        },
    ),
    (payload_held, {"sensor_cfg": _GRIP, "other_sensor_cfg": _GRIP_LEFT, "force_threshold": GRIP_FORCE_THRESHOLD_N}),
    (grasp_terms.object_lifted, {"asset_cfg": _OLD_BULB, "min_height": OLD_BULB_DROP_HEIGHT}),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
    (grasp_terms.ladder_near_vertical, {"tilt_limit": mdp.LADDER_TILT_LIMIT}),
]


@configclass
class S04EventCfg(BalanceEventCfg):
    """Re-seats the bulb against the hand's live, actually-simulated pose -- see
    ``nav_terms.settle_carried_payload_live``. Root/joint randomization zeroed for now while
    the open-palm rest calibration is being worked out (adds noise we don't need yet)."""

    settle_bulb = EventTerm(
        func=settle_carried_payload_live,
        mode="reset",
        params={"payload_cfg": SceneEntityCfg("old_bulb"), "reset_mode": True},
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    reset_robot_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {},
        },
    )


@configclass
class S04RewardsCfg(DescendRewardsCfg):
    # Force on the OBJECT, not the hand's net force: a panicked grip while balancing is exactly
    # how a real bulb gets crushed, but an arm braced on a rail is not.
    contact_penalty = RewTerm(func=mdp.hand_contact_force_l2, weight=-1.0e-4, params={"sensor_cfg": _GRIP})
    bulb_dropped = RewTerm(
        func=mdp.object_dropped, weight=-200.0, params={"asset_cfg": _OLD_BULB, "min_height": OLD_BULB_DROP_HEIGHT}
    )


@configclass
class S04TerminationsCfg(BalanceTerminationsCfg):
    bulb_dropped = DoneTerm(
        func=mdp.object_dropped, params={"asset_cfg": _OLD_BULB, "min_height": OLD_BULB_DROP_HEIGHT}
    )


@configclass
class S04DescendWithBulbEnvCfg(DescendSubtaskCfg):
    """Descend the placed ladder holding the old bulb (randomized Replace layout, ladder dynamic)."""

    # The right hand is occupied for the whole episode, so the contact bootstrap must not divide
    # by a palm that can never touch the ladder.
    ladder_contact_bodies: list[str] | None = LOADED_LADDER_CONTACT_BODIES

    success_predicate = mdp.all_of
    success_params: dict | None = {"predicates": DESCENDED_WITH_BULB_CONJUNCTS}

    events: S04EventCfg = S04EventCfg()
    rewards: SubtaskRewardsCfg = SubtaskRewardsCfg()
    terminations: S04TerminationsCfg = S04TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # S03's end state: the old bulb is out of the fixture and in the hand, at the carry offset
        # from the robot's own (now-final) on-ladder root pose.
        self.scene.old_bulb.init_state.pos, self.scene.old_bulb.init_state.rot = compose_carried_pose(
            self.scene.robot.init_state.pos, self.scene.robot.init_state.rot, BULB_IN_ROOT_ON_LADDER
        )
        # Merge, don't assign: these dicts only name right-arm/right-hand joints.
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            **ARM_CRADLE,
            **HAND_CUP,
        }
        add_grip_contact_sensor(self.scene, self.scene.old_bulb.prim_path)
        add_bulb_crush_gate(self)
