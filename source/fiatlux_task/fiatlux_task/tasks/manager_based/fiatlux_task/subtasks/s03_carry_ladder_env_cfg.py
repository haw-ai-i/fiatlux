# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S03-CarryLadder-v0`` -- carry the grasped ladder to the fixture and stop.

Starts from S02's end state: the ladder already held by one rail (``LADDER_IN_ROOT_CARRIED``),
composed onto the robot's own (randomized) root pose rather than spawned at the preset's
independently-sampled ladder zone. Success is the ladder horizontally within placing range of the
fixture (``ladder_ready``, the same check S04's own start condition needs), robot facing it,
standing, and the ladder still gripped -- without that last conjunct a robot that flings the
ladder ahead and merely walks into the facing/speed gate empty-handed would score.

No separate ``ladder_tipped`` termination: the only ladder in this scene IS the carried payload,
and a failed grip already fails the arrival gate's own grip conjunct -- nothing distinct to bump.
"""

from isaaclab.utils import configclass

from fiatlux_task.grasp_poses import LADDER_CARRY_ARM_JOINT_POS, LADDER_IN_ROOT_CARRIED

from .. import mdp
from ..mdp.nav_terms import (
    GRIP_FORCE_THRESHOLD_N,
    add_grip_contact_sensor,
    arrived_carrying_ladder,
    compose_carried_pose,
)
from ..scene_cfg import (
    LADDER_APPROACH_RADIUS,
    LADDER_READY_XY_RADIUS,
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    face_robot_at,
    frame_viewer_on,
    stand_robot_near,
)
from ..subtask_env_cfg import ARRIVAL_FACING_TOLERANCE, ARRIVAL_MAX_SPEED, NavigateSubtaskCfg

# TODO(carry-attach): nothing physically holds the ladder at LADDER_IN_ROOT_CARRIED -- a free
# rigid body at one point of hand contact falls within a few physics steps. A
# UsdPhysics.FixedJoint weld (mdp.attach_terms.weld_to_body) onto a G1 articulation link raises
# PhysX "Body must be non-kinematic" / "disjointed body transforms". Needs either the correct
# PhysX API for an articulation link or a kinematic-object approach.


@configclass
class S03CarryLadderEnvCfg(NavigateSubtaskCfg):
    """Carry the ladder to the fixture (randomized Replace layout, ladder dynamic, held)."""

    scene_preset: str = "replace"
    orbit_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orbit_radius: float = 5.0
    orbit_height: float = 2.4

    success_predicate = arrived_carrying_ladder
    success_params: dict | None = {
        "xy_radius": LADDER_READY_XY_RADIUS,
        "facing_tolerance": ARRIVAL_FACING_TOLERANCE,
        "max_speed": ARRIVAL_MAX_SPEED,
        "grip_force_threshold": GRIP_FORCE_THRESHOLD_N,
    }
    progress_distance_fn = mdp.ladder_fixture_distance

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_replace_preset(self.scene)
        # S02's end state: the robot is where it grasped the ladder, in the ladder's own drawn
        # zone. Capture that before the carried pose overwrites the ladder's init_state, or the
        # robot spawns in its independently-sampled zone and the ladder teleports there with it.
        pickup_xy = self.scene.ladder.init_state.pos[:2]
        stand_robot_near(self.scene, pickup_xy, LADDER_APPROACH_RADIUS)
        # This leg's target is the fixture, not the table apply_replace_preset aims at.
        face_robot_at(self.scene, self.scene.socket.init_state.pos[:2])
        # The ladder starts already held, at the carry offset from the robot's now-final root
        # pose.
        self.scene.ladder.init_state.pos, self.scene.ladder.init_state.rot = compose_carried_pose(
            self.scene.robot.init_state.pos, self.scene.robot.init_state.rot, LADDER_IN_ROOT_CARRIED
        )
        # Merge, don't assign: this dict only names right-arm/right-hand joints.
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            **LADDER_CARRY_ARM_JOINT_POS,
        }
        add_grip_contact_sensor(self.scene, self.scene.ladder.prim_path)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        # Room-diagonal traverse (~10.8 m) at the ~0.5 m/s reference speed, 2x margin.
        self.episode_length_s = 45.0
        frame_viewer_on(self.viewer, self.scene.robot.init_state.pos)
