# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared tier for the on-the-ladder subtasks (S02, S04, S10, S12, and via ``mate`` S03 and S11).

The preset is drawn with the ladder's zone coupled to the fixture's anchor.

The ladder is dynamic here (S01 moves it into place), so it can tip -- tipping is a
termination and a penalty in every subtask in this file, and absent from
``FIATLUX-Climb-v0``/``FIATLUX-Descend-v0``, which climb a kinematic ladder -- and it can stand
anywhere, so the stances and success gates read its live pose, not ``LADDER_POSITION``.

``mdp.bulb_attachment`` is wired here rather than on the mate tier: the old bulb starts locked in
the inverted fixture on every subtask in this file, and nothing else keeps it from falling out
under gravity. Only S03 and S11 score against it.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from fiatlux_task.assets import STEP_LADDER_TOP_OFFSET
from fiatlux_task.robots.g1 import G1_DEX3_PALM_BODIES, G1_FOOT_BODIES, G1_PALM_BODIES

from .. import mdp
from ..mdp.place_terms import step_face_dir_from_yaw, yaw_from_quat
from ..replace_env_cfg import (
    BAYONET_ENTRY_TILT,
    BAYONET_INSERTION_DEPTH,
    BAYONET_ROTATION_ANGLE,
    SEAT_POS_THRESHOLD,
)
from ..scene_cfg import (
    CLIMB_ROBOT_POSITION,
    LADDER_POSITION,
    TOP_STANCE_PELVIS_OFFSET,
    TOP_STANCE_YAW_OFFSET_DEG,
    G1ReplaceSceneCfg,
    _quat_mul,
    _quat_z_deg,
    add_ego_camera,
    add_ladder_contact_sensor,
    add_mid360_lidar,
    apply_replace_preset,
    face_robot_at,
    frame_viewer_on,
)
from ..subtask_env_cfg import SubtaskEnvCfg, SubtaskEventCfg, SubtaskRewardsCfg, SubtaskTerminationsCfg


@configclass
class BalanceEventCfg(SubtaskEventCfg):
    """Bayonet channel enforcement, with ``FIATLUX-Replace-v0``'s parameters."""

    bulb_attachment = EventTerm(
        func=mdp.bulb_attachment,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "insertion_depth": BAYONET_INSERTION_DEPTH,
            "rotation_angle": BAYONET_ROTATION_ANGLE,
            "rotation_sign": -1.0,
            "radial_tolerance": SEAT_POS_THRESHOLD,
            "tilt_tolerance": BAYONET_ENTRY_TILT,
        },
    )


# Shared by every climb/descend gate in the family.
LADDER_SUCCESS_XY_RADIUS = 0.6  # m
LADDER_SUCCESS_MAX_SPEED = 1.5  # m/s

# How far below the tread a stance may leave the pelvis, and the pelvis height that reads as
# back on the floor.
LADDER_TOP_HEIGHT_SLACK = 0.15  # m
LADDER_FLOOR_STANCE_HEIGHT = CLIMB_ROBOT_POSITION[2] + 0.15  # m

# Offset along the ladder's step-facing direction from its root, rotated onto the sampled yaw.
MOUNT_STANCE_STANDOFF = math.dist(CLIMB_ROBOT_POSITION[:2], LADDER_POSITION[:2])


# Feet plus the left palm, for a subtask whose right hand is occupied for the whole episode.
# Both hand variants' left palm are named: the sensor is built before the variant may be
# swapped, and a name belonging to the absent variant never resolves.
LOADED_LADDER_CONTACT_BODIES = [*G1_FOOT_BODIES, G1_PALM_BODIES[0], G1_DEX3_PALM_BODIES[0]]


def _stance_on_step_side(scene: G1ReplaceSceneCfg, standoff: float, pelvis_z: float) -> None:
    """Put the robot ``standoff`` metres out from the ladder's root on its step-facing side."""
    ladder_x, ladder_y = scene.ladder.init_state.pos[:2]
    face_x, face_y = step_face_dir_from_yaw(yaw_from_quat(scene.ladder.init_state.rot))
    scene.robot.init_state.pos = (ladder_x + face_x * standoff, ladder_y + face_y * standoff, pelvis_z)
    face_robot_at(scene, (ladder_x, ladder_y))


def stand_robot_at_ladder_base(scene: G1ReplaceSceneCfg) -> None:
    """Start state for a climb: on the floor at the ladder's steps, facing them."""
    _stance_on_step_side(scene, MOUNT_STANCE_STANDOFF, CLIMB_ROBOT_POSITION[2])


# Reset jitter for the on-ladder tier, against the family's +/-0.05 m / +/-0.1 rad / +/-0.05 rad.
ON_LADDER_RESET_JITTER_M = 0.02
ON_LADDER_RESET_JITTER_RAD = 0.05
ON_LADDER_JOINT_JITTER_RAD = 0.02


def stand_robot_on_ladder_top(scene: G1ReplaceSceneCfg) -> None:
    """Pelvis over the tread's centre, turned a quarter turn from the ladder's own frame.

    Square to the ladder the feet straddle the tread's shallow axis and the right one hung off the
    front edge entirely (issue #103); turned, they straddle the wide axis and both sit on. The
    layout turns the ladder back the same amount, so the robot still faces the fixture.
    """
    ladder_x, ladder_y = scene.ladder.init_state.pos[:2]
    yaw = yaw_from_quat(scene.ladder.init_state.rot)
    local_x, local_y, platform_z = STEP_LADDER_TOP_OFFSET
    dx = local_x * math.cos(yaw) - local_y * math.sin(yaw)
    dy = local_x * math.sin(yaw) + local_y * math.cos(yaw)
    scene.robot.init_state.pos = (ladder_x + dx, ladder_y + dy, platform_z + TOP_STANCE_PELVIS_OFFSET)
    scene.robot.init_state.rot = _quat_mul(scene.ladder.init_state.rot, _quat_z_deg(TOP_STANCE_YAW_OFFSET_DEG))


@configclass
class OnLadderRewardsCfg(SubtaskRewardsCfg):
    """No ``flat_orientation_l2``: working on the ladder requires a sustained forward lean, so
    an upright-torso term fights the task."""

    ladder_tipped = RewTerm(func=mdp.ladder_tipped, weight=-200.0, params={"tilt_limit": mdp.LADDER_TILT_LIMIT})


@configclass
class BalanceRewardsCfg(OnLadderRewardsCfg):
    """Adds the limb-on-ladder bootstrap; the height channel comes from the climb/descend tier."""

    ladder_contact = RewTerm(
        func=mdp.ladder_contact_fraction,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("ladder_contact"), "threshold": 1.0},
    )


@configclass
class ClimbRewardsCfg(BalanceRewardsCfg):
    climb_progress = RewTerm(func=mdp.climb_height_progress, weight=500.0)


@configclass
class DescendRewardsCfg(BalanceRewardsCfg):
    descend_progress = RewTerm(func=mdp.descend_height_progress, weight=500.0)


@configclass
class BalanceTerminationsCfg(SubtaskTerminationsCfg):
    ladder_tipped = DoneTerm(func=mdp.ladder_tipped, params={"tilt_limit": mdp.LADDER_TILT_LIMIT})


@configclass
class BalanceSubtaskCfg(SubtaskEnvCfg):
    """A subtask performed on the placed, dynamic ladder. Leaves place the robot and any payload."""

    scene_preset: str = "replace"
    orbit_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orbit_radius: float = 5.0
    orbit_height: float = 2.4

    events: BalanceEventCfg = BalanceEventCfg()
    rewards: OnLadderRewardsCfg = OnLadderRewardsCfg()
    terminations: BalanceTerminationsCfg = BalanceTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_replace_preset(self.scene, couple_ladder_to_fixture=True)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        frame_viewer_on(self.viewer, self.scene.ladder.init_state.pos)
        # The family's jitter is sized for a robot on an open floor, where it is small against the
        # room. On the tread it is the same order as the clearance to the edge, so it alone could
        # put a foot back off -- randomizing a start state should not decide whether the robot
        # starts supported.
        self.events.reset_robot_root.params["pose_range"] = {
            "x": (-ON_LADDER_RESET_JITTER_M, ON_LADDER_RESET_JITTER_M),
            "y": (-ON_LADDER_RESET_JITTER_M, ON_LADDER_RESET_JITTER_M),
            "yaw": (-ON_LADDER_RESET_JITTER_RAD, ON_LADDER_RESET_JITTER_RAD),
        }
        # Same for the joint offsets, which reach the feet through the hips and moved them further
        # than the root jitter did.
        self.events.reset_robot_joints.params["position_range"] = (
            -ON_LADDER_JOINT_JITTER_RAD,
            ON_LADDER_JOINT_JITTER_RAD,
        )


@configclass
class ClimbSubtaskCfg(BalanceSubtaskCfg):
    """Starts on the floor at the ladder's steps."""

    ladder_contact_bodies: list[str] | None = None
    rewards: ClimbRewardsCfg = ClimbRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        stand_robot_at_ladder_base(self.scene)
        add_ladder_contact_sensor(self.scene, self.ladder_contact_bodies)


@configclass
class DescendSubtaskCfg(BalanceSubtaskCfg):
    """Starts on the tread."""

    ladder_contact_bodies: list[str] | None = None
    rewards: DescendRewardsCfg = DescendRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        stand_robot_on_ladder_top(self.scene)
        add_ladder_contact_sensor(self.scene, self.ladder_contact_bodies)
