# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S07-ApproachNewBulb-v0`` -- walk to the fresh bulb on the bench, hands free.

Successor of S06 (old bulb binned) and predecessor of S08 (grasp the fresh one); a bare walk,
just pointed at the fresh bulb instead of the ladder. Success is arriving within reach of the
bulb, facing it, standing -- no grip conjunct, since nothing is held yet.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .. import mdp
from ..mdp.nav_terms import (
    BULB_APPROACH_RADIUS,
    DISPOSAL_DEPARTURE_RADIUS,
    base_bulb_distance,
    base_calm,
    base_facing,
    base_near,
)
from ..scene_cfg import (
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    face_robot_at,
    frame_viewer_between,
    park_old_bulb_in_crate,
    stand_robot_near,
)
from ..subtask_env_cfg import ARRIVAL_FACING_TOLERANCE, ARRIVAL_MAX_SPEED, NavigateSubtaskCfg

# The success gate as data (mdp.all_of). No payload conjunct: hands are free here, and nothing
# in this scene can tip over.
AT_FRESH_BULB_CONJUNCTS = [
    (base_near, {"asset_cfg": SceneEntityCfg("fresh_bulb"), "xy_radius": BULB_APPROACH_RADIUS}),
    (base_facing, {"asset_cfg": SceneEntityCfg("fresh_bulb"), "facing_tolerance": ARRIVAL_FACING_TOLERANCE}),
    (base_calm, {"max_speed": ARRIVAL_MAX_SPEED}),
]


@configclass
class S07ApproachNewBulbEnvCfg(NavigateSubtaskCfg):
    """Walk to the fresh bulb on the table (randomized Replace layout, hands free)."""

    scene_preset: str = "replace"
    orbit_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orbit_radius: float = 5.0
    orbit_height: float = 2.4

    success_predicate = mdp.all_of
    success_params: dict | None = {"predicates": AT_FRESH_BULB_CONJUNCTS}
    progress_distance_fn = base_bulb_distance

    def __post_init__(self) -> None:
        super().__post_init__()
        # S01 already stood the ladder at the fixture; every leg after it inherits that.
        apply_replace_preset(self.scene, couple_ladder_to_fixture=True)
        # S06's end state: the old bulb is in the crate and the fixture is empty. Without this
        # the preset's seated old bulb is still overhead, three subtasks after it was disposed
        # of -- and with no attach FSM on this tier it drops out of the inverted socket at reset.
        park_old_bulb_in_crate(self.scene)
        # S06's end state: standing at the disposal crate, hands free -- but turned away from it
        # below, so it has to be clear of the crate on every bearing, not just in front (#205).
        stand_robot_near(
            self.scene,
            self.scene.bin.init_state.pos[:2],
            DISPOSAL_DEPARTURE_RADIUS,
            min_standoff=DISPOSAL_DEPARTURE_RADIUS,
        )
        # This leg's target is the bulb itself, not the table's own origin apply_replace_preset
        # aims at.
        face_robot_at(self.scene, self.scene.fresh_bulb.init_state.pos[:2])
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        frame_viewer_between(self.viewer, self.scene.robot.init_state.pos, self.scene.fresh_bulb.init_state.pos)
