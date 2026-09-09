# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S05-CarryBulbToDisposal-v0`` -- carry the removed old bulb to the disposal crate.

Starts from S04's end state: the old bulb already held upright in hand
(``BULB_IN_ROOT_STANDING``), composed onto the robot's own (randomized) root pose rather than left
seated at the fixture. Success is the robot at the disposal crate, facing it, standing, and the
bulb still gripped -- without that last conjunct a thrown bulb that skids into the crate's radius
would score.

No ``ladder_tipped`` termination here: this leg never interacts with the ladder (deliberate,
unlike S01/S09 which carry or approach it).
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from fiatlux_task.grasp_poses import BULB_IN_ROOT_STANDING
from fiatlux_task.poses import ARM_CRADLE, HAND_CUP

from .. import mdp
from ..mdp.nav_terms import (
    DISPOSAL_ARRIVAL_CLEARANCE,
    GRIP_FORCE_THRESHOLD_N,
    LADDER_MOUNT_RADIUS,
    add_grip_contact_sensor,
    base_calm,
    base_disposal_distance,
    base_facing,
    base_near,
    compose_carried_pose,
    payload_held,
    settle_carried_payload_live,
)
from ..mdp.place_terms import CRATE_FOOTPRINT_HALF_EXTENT
from ..scene_cfg import (
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    face_robot_at,
    frame_viewer_on,
    stand_robot_near,
)
from ..subtask_env_cfg import ARRIVAL_FACING_TOLERANCE, ARRIVAL_MAX_SPEED, NavigateSubtaskCfg, SubtaskEventCfg
from ..subtask_tiers.carrying import add_bulb_crush_gate

# The success gate as data (mdp.all_of). Without the grip conjunct a thrown bulb that skids into
# the crate's radius would score.
OLD_BULB_AT_CRATE_CONJUNCTS = [
    (
        base_near,
        {
            "asset_cfg": SceneEntityCfg("bin"),
            "xy_radius": DISPOSAL_ARRIVAL_CLEARANCE,
            "half_extent": CRATE_FOOTPRINT_HALF_EXTENT,
        },
    ),
    (base_facing, {"asset_cfg": SceneEntityCfg("bin"), "facing_tolerance": ARRIVAL_FACING_TOLERANCE}),
    (base_calm, {"max_speed": ARRIVAL_MAX_SPEED}),
    (
        payload_held,
        {
            "sensor_cfg": SceneEntityCfg("grip_contact"),
            "other_sensor_cfg": SceneEntityCfg("grip_contact_left"),
            "force_threshold": GRIP_FORCE_THRESHOLD_N,
        },
    ),
]


@configclass
class S05EventCfg(SubtaskEventCfg):
    """Re-seats the bulb against the hand's live, actually-simulated pose -- see
    ``nav_terms.settle_carried_payload_live``. Root/joint randomization zeroed for now while
    the open-palm rest calibration is being worked out (adds noise we don't need yet)."""

    settle_bulb = EventTerm(
        func=settle_carried_payload_live,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"payload_cfg": SceneEntityCfg("old_bulb")},
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
class S05CarryBulbToDisposalEnvCfg(NavigateSubtaskCfg):
    """Carry the old bulb to the disposal crate (randomized Replace layout, bulb held)."""

    scene_preset: str = "replace"
    orbit_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orbit_radius: float = 5.0
    orbit_height: float = 2.4

    success_predicate = mdp.all_of
    success_params: dict | None = {"predicates": OLD_BULB_AT_CRATE_CONJUNCTS}
    progress_distance_fn = base_disposal_distance
    events: S05EventCfg = S05EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # S01 already stood the ladder at the fixture; every leg after it inherits that.
        apply_replace_preset(self.scene, couple_ladder_to_fixture=True)
        # S04's end state: back on the floor at the ladder it descended, holding the old bulb.
        # Without this the robot spawns in its own sampled zone and the held bulb teleports too.
        stand_robot_near(self.scene, self.scene.ladder.init_state.pos[:2], LADDER_MOUNT_RADIUS)
        # This leg's target is the disposal crate, not the table apply_replace_preset aims at.
        face_robot_at(self.scene, self.scene.bin.init_state.pos[:2])
        # S04's end state: the old bulb starts already held, at the carry offset from the
        # robot's own (now-final) root pose -- not seated at the fixture.
        self.scene.old_bulb.init_state.pos, self.scene.old_bulb.init_state.rot = compose_carried_pose(
            self.scene.robot.init_state.pos, self.scene.robot.init_state.rot, BULB_IN_ROOT_STANDING
        )
        # Merge, don't assign: these dicts only name right-arm/right-hand joints.
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            **ARM_CRADLE,
            **HAND_CUP,
        }
        add_grip_contact_sensor(self.scene, self.scene.old_bulb.prim_path)
        add_bulb_crush_gate(self)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        self.episode_length_s = 120.0
        frame_viewer_on(self.viewer, self.scene.robot.init_state.pos)
