# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared tier for the on-the-ladder subtasks (S05, S07, S13, S15, and via ``mate`` S06 and S14).

All of them start from a ladder that a placement subtask already stood at the fixture, so the
preset is drawn with the ladder's zone coupled to the fixture's anchor -- for these subtasks that
coupling is the chain's state, not the curriculum aid the preset's own docstring describes.

Two consequences of the ladder being DYNAMIC here (it must be: S02-S04 carry and place it) shape
everything below. It can tip, so tipping is a termination and a penalty in every one of these
subtasks and is absent from ``FIATLUX-Climb-v0``/``FIATLUX-Descend-v0``, which climb a kinematic
ladder. And it can stand anywhere, so the stances and the success gates are built from its live
pose rather than from ``LADDER_POSITION``.

The A-frame's steps face one way, so both stances are placed along the ladder's own step-facing
direction at the sampled yaw; a stance behind the ladder cannot be climbed from.

The bayonet attach/detach state machine (issue #54, ``mdp.bulb_attachment``) is wired here, not
just on the mate tier: the old bulb starts locked in the fixture on EVERY subtask in this file
(S05's and S07's and S13's start state has it there just as much as S06's does), and nothing but
that event keeps it from falling out of the inverted socket under plain gravity. Only S06 and S14
score against it directly, but S05/S07/S13/S15 would silently show (and physically drop) a
loose old bulb for the whole episode without it -- an unrelated background object failing is
still a failure to notice in a render or a training run.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from fiatlux_task.robots.g1 import G1_DEX3_PALM_BODIES, G1_FOOT_BODIES, G1_PALM_BODIES

from .. import mdp
from ..mdp.place_terms import step_face_dir_from_yaw, yaw_from_quat
from ..replace_env_cfg import BAYONET_INSERTION_DEPTH, BAYONET_ROTATION_ANGLE, SEAT_ORI_THRESHOLD, SEAT_POS_THRESHOLD
from ..scene_cfg import (
    CLIMB_ROBOT_POSITION,
    LADDER_POSITION,
    G1ReplaceSceneCfg,
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

# Success-region shape, shared by every climb/descend gate in the family (Climb's own tolerances).
LADDER_SUCCESS_XY_RADIUS = 0.6  # m
LADDER_SUCCESS_MAX_SPEED = 1.5  # m/s

# How far below the ladder's top step a solid stance may leave the pelvis, and the pelvis height
# that reads as back on the floor. Both are the pre-existing Climb/Descend values.
LADDER_TOP_HEIGHT_SLACK = 0.15  # m
LADDER_FLOOR_STANCE_HEIGHT = CLIMB_ROBOT_POSITION[2] + 0.15  # m

# The two stances, as offsets along the ladder's step-facing direction from its root. Taken from
# the workshop preset's own validated poses relative to its fixed LADDER_POSITION, whose steps
# face world -x; carried to an arbitrarily yawed ladder by rotating rather than by copying the
# world coordinates.
MOUNT_STANCE_STANDOFF = math.dist(CLIMB_ROBOT_POSITION[:2], LADDER_POSITION[:2])

# TOP_STANCE_STANDOFF/PELVIS_Z: NOT TOP_ROBOT_POSITION's own values. That constant (workshop
# preset's fixed-ladder pose) puts the pelvis at 1.85 m -- measured directly against the replace
# preset's live ladder_contact sensor, this stance was airborne: 0.000 N at every one of feet,
# left palm, right palm, from reset through 10+ steps of free fall (pelvis height dropping
# continuously, no rung or platform underneath at all).
#
# Swept standoff x height empirically (teleport, FRESH env.reset() per candidate so an earlier
# candidate's collision/fall can't contaminate a later one's leg state, then settle and read the
# live ladder_contact sensor) against the platform -- a vertex cluster at local z=169-175 cm,
# i.e. world 1.69-1.75 m (matches ``assets.STEP_LADDER_TOP_OFFSET`` = 1.70 m), not 1.85 m.
#
# UNCALIBRATED, still open: standoff=0.22 m, height=1.76-1.80 m is the best candidate found --
# genuine, substantial contact (179-184 N, not a 100s-1000s N collision and not 0 N empty air)
# but with 17-20 cm of settle/sink over a 20-step zero-action window, not a fully stable
# zero-drift stance. A finer (~1 cm) position sweep plus foot orientation, or accepting that a
# static teleport can't perfectly balance a biped at a platform EDGE without any active balance
# correction (zero action gives none -- the same caveat verify_scene.py's own base-env checks
# already documented), are the two most likely next steps. Still a large, measured improvement
# over the prior state: bounded ~0.2 m settle under real support, not unconditional free fall to
# the floor.
TOP_STANCE_STANDOFF = 0.22  # m
TOP_STANCE_PELVIS_Z = 1.78  # m

# Feet plus the LEFT palm, for a subtask whose right hand is occupied for the whole episode
# (``grasp_terms`` picks the right palm as the grasping hand). Both variants' left palm are named
# for the reason ``G1_LADDER_CONTACT_BODIES`` names both: the sensor is built before the hand
# variant may be swapped, and a name belonging to the absent variant simply never resolves.
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


def stand_robot_on_ladder_top(scene: G1ReplaceSceneCfg) -> None:
    """Start state for a descent or for work at the fixture: pelvis at the upper steps."""
    _stance_on_step_side(scene, TOP_STANCE_STANDOFF, TOP_STANCE_PELVIS_Z)


@configclass
class OnLadderRewardsCfg(SubtaskRewardsCfg):
    """What every subtask performed on the dynamic ladder pays, whatever else it is doing.

    No ``flat_orientation_l2``: working on an A-frame requires a sustained forward lean, so an
    upright-torso term fights the task. Do not add one.
    """

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


@configclass
class ClimbSubtaskCfg(BalanceSubtaskCfg):
    """Going up. The robot starts on the floor at the ladder's steps."""

    ladder_contact_bodies: list[str] | None = None
    rewards: ClimbRewardsCfg = ClimbRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        stand_robot_at_ladder_base(self.scene)
        add_ladder_contact_sensor(self.scene, self.ladder_contact_bodies)


@configclass
class DescendSubtaskCfg(BalanceSubtaskCfg):
    """Coming down. The robot starts on the upper steps."""

    ladder_contact_bodies: list[str] | None = None
    rewards: DescendRewardsCfg = DescendRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        stand_robot_on_ladder_top(self.scene)
        add_ladder_contact_sensor(self.scene, self.ladder_contact_bodies)
