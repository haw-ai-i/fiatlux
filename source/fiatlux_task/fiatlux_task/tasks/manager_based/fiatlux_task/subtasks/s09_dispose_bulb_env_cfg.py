# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S09-DisposeBulb-v0`` -- put the old bulb in the disposal crate and let go.

Success requires the bulb inside the crate's interior footprint (``place_terms.old_bulb_in_bin``,
orientation-agnostic), not merely near the crate's origin -- a bulb balanced on the rim or resting
on the floor beside it would pass the coarser ``mdp.old_bulb_disposed`` radius alone.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .. import mdp
from ..mdp import place_terms
from ..replace_env_cfg import DISPOSAL_THRESHOLD
from ..scene_cfg import BIN_BULB_INTERIOR_Z, add_ego_camera, add_mid360_lidar, apply_replace_preset, face_robot_at
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT
from ..subtask_tiers.place import (
    AT_REST_ANG_VEL_LIMIT,
    AT_REST_LIN_VEL_LIMIT,
    PLACE_SUSTAIN_SECONDS,
    RELEASE_FORCE_THRESHOLD_N,
    PlaceSubtaskCfg,
    add_release_contact_sensor,
)

# The success gate, as reviewable data (mdp.all_of) rather than a hand-written conjunction --
# an omitted conjunct here is a gate that passes vacuously.
OLD_BULB_DISPOSED_CONJUNCTS = [
    (mdp.old_bulb_disposed, {"distance_threshold": DISPOSAL_THRESHOLD}),
    (place_terms.old_bulb_in_bin, {"interior_floor_z": BIN_BULB_INTERIOR_Z}),
    (
        place_terms.object_at_rest,
        {
            "asset_cfg": SceneEntityCfg("old_bulb"),
            "lin_vel_limit": AT_REST_LIN_VEL_LIMIT,
            "ang_vel_limit": AT_REST_ANG_VEL_LIMIT,
        },
    ),
    (
        place_terms.object_released,
        {"sensor_cfg": SceneEntityCfg("release_contact"), "force_threshold": RELEASE_FORCE_THRESHOLD_N},
    ),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
]


@configclass
class S09DisposeBulbEnvCfg(PlaceSubtaskCfg):
    """Put the old bulb in the disposal crate and let go (randomized Replace layout)."""

    scene_preset: str = "replace"
    orbit_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    orbit_radius: float = 5.0
    orbit_height: float = 2.4

    success_predicate = mdp.sustained
    # No old_bulb_dropped termination: letting go is the goal here, not a failure mode.
    success_params: dict | None = {
        "predicate_fn": mdp.all_of,
        "seconds": PLACE_SUSTAIN_SECONDS,
        "predicate_params": {"predicates": OLD_BULB_DISPOSED_CONJUNCTS},
    }
    progress_distance_fn = mdp.old_bulb_disposal_distance

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_replace_preset(self.scene)
        # The preset aims the robot at the table; this subtask's target is the disposal crate.
        face_robot_at(self.scene, self.scene.bin.init_state.pos[:2])
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        add_release_contact_sensor(self.scene, self.scene.old_bulb.prim_path)
        self.episode_length_s = 20.0
        self.viewer.eye = (4.0, 4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
