# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S12-CarryBulbToLadder-v0`` -- carry the fresh bulb back to the ladder and stop.

Starts from S11's end state: the fresh bulb already held upright in hand
(``BULB_IN_ROOT_STANDING``), composed onto the robot's own (randomized) root pose. The ladder
itself is untouched from ``apply_replace_preset`` -- it is this leg's navigation target, not its
payload, and was placed by S04. Success is the robot at the ladder within foot-placement range to
start climbing (``LADDER_MOUNT_RADIUS``, not the grasp-reach ``LADDER_APPROACH_RADIUS``), facing
it, standing, ladder upright, and the bulb still gripped -- without that last conjunct a thrown
bulb that skids into the radius would score.

Ladder-tipped termination (unlike S08, which never touches the ladder): walking a carried payload
into the now free-standing, dynamic ladder can knock it over, a failure unrelated to the bulb grip.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from fiatlux_task.grasp_poses import BULB_IN_ROOT_STANDING
from fiatlux_task.poses import ARM_CRADLE, HAND_CUP

from .. import mdp
from ..mdp.nav_terms import (
    BULB_APPROACH_OFFSET,
    GRIP_FORCE_THRESHOLD_N,
    LADDER_MOUNT_RADIUS,
    add_grip_contact_sensor,
    arrived_carrying_bulb,
    compose_carried_pose,
    settle_carried_payload_live,
)
from ..scene_cfg import (
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    face_robot_at,
    frame_viewer_between,
    park_old_bulb_in_crate,
    stand_robot_at_offset,
)
from ..subtask_env_cfg import (
    ARRIVAL_FACING_TOLERANCE,
    ARRIVAL_MAX_SPEED,
    NavigateRewardsCfg,
    NavigateSubtaskCfg,
    SubtaskEventCfg,
    SubtaskTerminationsCfg,
)


@configclass
class S12EventCfg(SubtaskEventCfg):
    """Re-seats the bulb against the hand's live, actually-simulated pose -- see
    ``nav_terms.settle_carried_payload_live``. Root/joint randomization zeroed for now while
    the open-palm rest calibration is being worked out (adds noise we don't need yet)."""

    settle_bulb = EventTerm(
        func=settle_carried_payload_live,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"payload_cfg": SceneEntityCfg("fresh_bulb")},
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
class S12RewardsCfg(NavigateRewardsCfg):
    ladder_tipped = RewTerm(func=mdp.ladder_tipped, weight=-200.0, params={"tilt_limit": mdp.LADDER_TILT_LIMIT})


@configclass
class S12TerminationsCfg(SubtaskTerminationsCfg):
    ladder_tipped = DoneTerm(func=mdp.ladder_tipped, params={"tilt_limit": mdp.LADDER_TILT_LIMIT})


@configclass
class S12CarryBulbToLadderEnvCfg(NavigateSubtaskCfg):
    """Carry the fresh bulb back to the ladder (randomized Replace layout, ladder dynamic, held bulb)."""

    scene_preset: str = "replace"
    orbit_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orbit_radius: float = 5.0
    orbit_height: float = 2.4

    success_predicate = arrived_carrying_bulb
    success_params: dict | None = {
        "xy_radius": LADDER_MOUNT_RADIUS,
        "facing_tolerance": ARRIVAL_FACING_TOLERANCE,
        "max_speed": ARRIVAL_MAX_SPEED,
        "grip_force_threshold": GRIP_FORCE_THRESHOLD_N,
    }
    progress_distance_fn = mdp.base_ladder_distance

    events: S12EventCfg = S12EventCfg()
    rewards: S12RewardsCfg = S12RewardsCfg()
    terminations: S12TerminationsCfg = S12TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # S04 already stood the ladder at the fixture; every leg after it inherits that.
        apply_replace_preset(self.scene, couple_ladder_to_fixture=True)
        # The old bulb was disposed of back in S09; the fixture is empty from here on.
        park_old_bulb_in_crate(self.scene)
        # S11's end state: the robot is at the table where it picked the bulb up. Capture that
        # before the carried pose overwrites the bulb's init_state, or the robot spawns in its
        # independently-sampled zone and the held bulb teleports there with it.
        stand_robot_at_offset(self.scene, self.scene.fresh_bulb.init_state.pos[:2], BULB_APPROACH_OFFSET)
        # This leg's target is the ladder S04 placed, not the table apply_replace_preset aims at.
        face_robot_at(self.scene, self.scene.ladder.init_state.pos[:2])
        # S11's end state: the fresh bulb starts already held, at the carry offset from the
        # robot's own (now-final) root pose -- not on the table.
        self.scene.fresh_bulb.init_state.pos, self.scene.fresh_bulb.init_state.rot = compose_carried_pose(
            self.scene.robot.init_state.pos, self.scene.robot.init_state.rot, BULB_IN_ROOT_STANDING
        )
        # Merge, don't assign: these dicts only name right-arm/right-hand joints.
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            **ARM_CRADLE,
            **HAND_CUP,
        }
        add_grip_contact_sensor(self.scene, self.scene.fresh_bulb.prim_path)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        # Worst-case room-diagonal traverse (~10.8 m) at the ~0.5 m/s reference speed, 2x margin;
        # conservative for a carrying leg.
        self.episode_length_s = 45.0
        frame_viewer_between(self.viewer, self.scene.robot.init_state.pos, self.scene.ladder.init_state.pos)
