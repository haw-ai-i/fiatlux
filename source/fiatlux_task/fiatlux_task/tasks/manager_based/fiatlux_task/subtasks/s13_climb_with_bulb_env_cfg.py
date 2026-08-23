# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S13-ClimbWithBulb-v0`` -- climb to working height holding the fresh bulb.

Starts from S12's end state: the robot at the ladder's steps with the fresh bulb in hand
(``BULB_IN_ROOT_STANDING``), the old bulb already in the disposal crate, the fixture empty.

The physically hardest subtask of the chain: one hand holds a 35 g bulb it must not crush, so the
climb is done with one hand and two feet. Two consequences are wired below -- the contact
bootstrap divides by feet plus the FREE palm only, or it would charge the policy for holding the
bulb, and the glass bound is the live constraint on the way up, since gripping harder is the
obvious way to keep the bulb while climbing.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from fiatlux_task.grasp_poses import BULB_IN_ROOT_STANDING
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
from ..replace_env_cfg import FRESH_BULB_DROP_HEIGHT
from ..scene_cfg import park_old_bulb_in_crate
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT
from ..subtask_tiers.balance import (
    LADDER_SUCCESS_MAX_SPEED,
    LADDER_SUCCESS_XY_RADIUS,
    LADDER_TOP_HEIGHT_SLACK,
    LOADED_LADDER_CONTACT_BODIES,
    BalanceEventCfg,
    BalanceTerminationsCfg,
    ClimbRewardsCfg,
    ClimbSubtaskCfg,
)

_BULB = SceneEntityCfg("bulb")
_GRIP = SceneEntityCfg("grip_contact")

# The success gate, as reviewable data (mdp.all_of) rather than a hand-written conjunction --
# an omitted conjunct here is a gate that passes vacuously.
CLIMBED_WITH_BULB_CONJUNCTS = [
    (
        balance_terms.climbed_to_ladder_top,
        {
            "height_slack": LADDER_TOP_HEIGHT_SLACK,
            "xy_radius": LADDER_SUCCESS_XY_RADIUS,
            "max_speed": LADDER_SUCCESS_MAX_SPEED,
        },
    ),
    (payload_held, {"sensor_cfg": _GRIP, "force_threshold": GRIP_FORCE_THRESHOLD_N}),
    (grasp_terms.object_lifted, {"asset_cfg": _BULB, "min_height": FRESH_BULB_DROP_HEIGHT}),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
    (grasp_terms.ladder_near_vertical, {"tilt_limit": mdp.LADDER_TILT_LIMIT}),
]


@configclass
class S13EventCfg(BalanceEventCfg):
    """Re-seats the bulb against the hand's live, actually-simulated pose -- see
    ``nav_terms.settle_carried_payload_live``. Root/joint randomization zeroed for now while
    the open-palm rest calibration is being worked out (adds noise we don't need yet)."""

    settle_bulb = EventTerm(
        func=settle_carried_payload_live,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"payload_cfg": SceneEntityCfg("bulb")},
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
class S13RewardsCfg(ClimbRewardsCfg):
    contact_penalty = RewTerm(func=mdp.hand_contact_force_l2, weight=-1.0e-4, params={"sensor_cfg": _GRIP})
    bulb_dropped = RewTerm(
        func=mdp.object_dropped, weight=-200.0, params={"asset_cfg": _BULB, "min_height": FRESH_BULB_DROP_HEIGHT}
    )


@configclass
class S13TerminationsCfg(BalanceTerminationsCfg):
    bulb_dropped = DoneTerm(func=mdp.object_dropped, params={"asset_cfg": _BULB, "min_height": FRESH_BULB_DROP_HEIGHT})


@configclass
class S13ClimbWithBulbEnvCfg(ClimbSubtaskCfg):
    """Climb the placed ladder holding the fresh bulb (randomized Replace layout, ladder dynamic)."""

    ladder_contact_bodies: list[str] | None = LOADED_LADDER_CONTACT_BODIES

    success_predicate = mdp.all_of
    success_params: dict | None = {"predicates": CLIMBED_WITH_BULB_CONJUNCTS}

    events: S13EventCfg = S13EventCfg()
    rewards: S13RewardsCfg = S13RewardsCfg()
    terminations: S13TerminationsCfg = S13TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        park_old_bulb_in_crate(self.scene)
        # S12's end state: the fresh bulb starts already held, at the carry offset from the
        # robot's own (now-final) root pose -- not on the table.
        self.scene.bulb.init_state.pos, self.scene.bulb.init_state.rot = compose_carried_pose(
            self.scene.robot.init_state.pos, self.scene.robot.init_state.rot, BULB_IN_ROOT_STANDING
        )
        # Merge, don't assign: these dicts only name right-arm/right-hand joints.
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            **ARM_CRADLE,
            **HAND_CUP,
        }
        add_grip_contact_sensor(self.scene, self.scene.bulb.prim_path)
        # Longer than S05's 20 s: one-handed is slower.
        self.episode_length_s = 30.0
