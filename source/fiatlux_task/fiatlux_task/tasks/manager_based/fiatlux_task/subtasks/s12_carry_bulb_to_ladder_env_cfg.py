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

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from fiatlux_task.grasp_poses import BULB_IN_ROOT_STANDING

from .. import mdp
from ..mdp.nav_terms import (
    GRIP_FORCE_THRESHOLD_N,
    LADDER_MOUNT_RADIUS,
    add_grip_contact_sensor,
    arrived_carrying_bulb,
    compose_carried_pose,
)
from ..scene_cfg import add_ego_camera, add_mid360_lidar, apply_replace_preset, face_robot_at
from ..subtask_env_cfg import (
    ARRIVAL_FACING_TOLERANCE,
    ARRIVAL_MAX_SPEED,
    NavigateRewardsCfg,
    NavigateSubtaskCfg,
    SubtaskTerminationsCfg,
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

    rewards: S12RewardsCfg = S12RewardsCfg()
    terminations: S12TerminationsCfg = S12TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_replace_preset(self.scene)
        # This leg's target is the ladder S04 placed, not the table apply_replace_preset aims at.
        face_robot_at(self.scene, self.scene.ladder.init_state.pos[:2])
        # S11's end state: the fresh bulb starts already held, at the carry offset from the
        # robot's own (now-final) root pose -- not on the table.
        self.scene.bulb.init_state.pos, self.scene.bulb.init_state.rot = compose_carried_pose(
            self.scene.robot.init_state.pos, self.scene.robot.init_state.rot, BULB_IN_ROOT_STANDING
        )
        add_grip_contact_sensor(self.scene, self.scene.bulb.prim_path)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        # Worst-case room-diagonal traverse (~10.8 m) at the ~0.5 m/s reference speed, 2x margin;
        # conservative for a carrying leg.
        self.episode_length_s = 45.0
        self.viewer.eye = (4.0, 4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
