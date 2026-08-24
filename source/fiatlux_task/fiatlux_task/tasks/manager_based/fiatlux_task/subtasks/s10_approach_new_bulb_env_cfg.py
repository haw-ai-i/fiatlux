# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S10-ApproachNewBulb-v0`` -- walk to the fresh bulb on the bench, hands free.

Successor of S03/S04 (ladder placed) and predecessor of S11 (grasp it); a bare walk like S01's,
just pointed at the fresh bulb instead of the ladder. Success is arriving within reach of the
bulb, facing it, standing -- no grip conjunct, since nothing is held yet.
"""

from isaaclab.utils import configclass

from ..mdp.nav_terms import BULB_APPROACH_RADIUS, DISPOSAL_ARRIVAL_RADIUS, arrived_at_bulb, base_bulb_distance
from ..scene_cfg import (
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    face_robot_at,
    frame_viewer_between,
    stand_robot_near,
    park_old_bulb_in_crate,
)
from ..subtask_env_cfg import ARRIVAL_FACING_TOLERANCE, ARRIVAL_MAX_SPEED, NavigateSubtaskCfg


@configclass
class S10ApproachNewBulbEnvCfg(NavigateSubtaskCfg):
    """Walk to the fresh bulb on the table (randomized Replace layout, hands free)."""

    scene_preset: str = "replace"
    orbit_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orbit_radius: float = 5.0
    orbit_height: float = 2.4

    success_predicate = arrived_at_bulb
    success_params: dict | None = {
        "xy_radius": BULB_APPROACH_RADIUS,
        "facing_tolerance": ARRIVAL_FACING_TOLERANCE,
        "max_speed": ARRIVAL_MAX_SPEED,
    }
    progress_distance_fn = base_bulb_distance

    def __post_init__(self) -> None:
        super().__post_init__()
        # S04 already stood the ladder at the fixture; every leg after it inherits that.
        apply_replace_preset(self.scene, couple_ladder_to_fixture=True)
        # S09's end state: the old bulb is in the crate and the fixture is empty. Without this
        # the preset's seated old bulb is still overhead, three subtasks after it was disposed
        # of -- and with no attach FSM on this tier it drops out of the inverted socket at reset.
        park_old_bulb_in_crate(self.scene)
        # S09's end state: standing at the disposal crate, hands free.
        stand_robot_near(self.scene, self.scene.bin.init_state.pos[:2], DISPOSAL_ARRIVAL_RADIUS)
        # This leg's target is the bulb itself, not the table's own origin apply_replace_preset
        # aims at.
        face_robot_at(self.scene, self.scene.bulb.init_state.pos[:2])
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        # Worst-case room-diagonal traverse (~10.8 m) at the ~0.5 m/s reference speed, 2x margin.
        self.episode_length_s = 45.0
        frame_viewer_between(self.viewer, self.scene.robot.init_state.pos, self.scene.bulb.init_state.pos)
