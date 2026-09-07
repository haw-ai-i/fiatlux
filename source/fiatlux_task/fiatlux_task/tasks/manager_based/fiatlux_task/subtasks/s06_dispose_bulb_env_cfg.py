# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-S06-DisposeBulb-v0`` -- put the old bulb in the disposal crate and let go.

Starts from S05's end state: the robot at the crate with the old bulb already held
(``BULB_IN_ROOT_STANDING``), composed onto its own root pose rather than left seated at the
fixture.

Success requires the bulb inside the crate's interior footprint (``place_terms.old_bulb_in_bin``,
orientation-agnostic): a bulb balanced on the rim, or resting on the floor beside the crate, is
outside it.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from fiatlux_task.grasp_poses import BULB_IN_ROOT_STANDING
from fiatlux_task.poses import ARM_CRADLE, HAND_CUP

from .. import mdp
from ..mdp import place_terms
from ..mdp.nav_terms import DISPOSAL_ARRIVAL_RADIUS, compose_carried_pose, settle_carried_payload_live
from ..scene_cfg import (
    add_ego_camera,
    add_mid360_lidar,
    apply_replace_preset,
    frame_viewer_on,
    stand_robot_near,
)
from ..subtask_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT, SubtaskEventCfg
from ..subtask_tiers.carrying import add_bulb_crush_gate
from ..subtask_tiers.place import (
    AT_REST_LIN_VEL_LIMIT,
    PLACE_SUSTAIN_SECONDS,
    RELEASE_FORCE_THRESHOLD_N,
    PlaceSubtaskCfg,
    add_release_contact_sensor,
)

# The success gate, as reviewable data (mdp.all_of) rather than a hand-written conjunction --
# an omitted conjunct here is a gate that passes vacuously.
OLD_BULB_DISPOSED_CONJUNCTS = [
    (place_terms.old_bulb_in_bin, {}),
    (
        place_terms.object_at_rest,
        {"asset_cfg": SceneEntityCfg("old_bulb"), "lin_vel_limit": AT_REST_LIN_VEL_LIMIT},
    ),
    (
        place_terms.object_released,
        {
            "sensor_cfg": SceneEntityCfg("release_contact"),
            "other_sensor_cfg": SceneEntityCfg("release_contact_left"),
            "force_threshold": RELEASE_FORCE_THRESHOLD_N,
        },
    ),
    (place_terms.robot_standing, {"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT}),
]


@configclass
class S06EventCfg(SubtaskEventCfg):
    """Re-seats the bulb against the hand's live, actually-simulated pose -- see
    ``nav_terms.settle_carried_payload_live``."""

    settle_bulb = EventTerm(
        func=settle_carried_payload_live,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"payload_cfg": SceneEntityCfg("old_bulb")},
    )


@configclass
class S06DisposeBulbEnvCfg(PlaceSubtaskCfg):
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

    events: S06EventCfg = S06EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # S01 already stood the ladder at the fixture; every leg after it inherits that.
        apply_replace_preset(self.scene, couple_ladder_to_fixture=True)
        # The preset's own robot zone is independent of the bin's; pull the robot to where it
        # would be holding the old bulb it starts this subtask already carrying.
        stand_robot_near(self.scene, self.scene.bin.init_state.pos[:2], DISPOSAL_ARRIVAL_RADIUS)
        # S05's end state: the old bulb starts already held, at the carry offset from the robot's
        # own (now-final) root pose -- not seated at the fixture, which is where the preset
        # leaves it and where it stayed until this was added.
        self.scene.old_bulb.init_state.pos, self.scene.old_bulb.init_state.rot = compose_carried_pose(
            self.scene.robot.init_state.pos, self.scene.robot.init_state.rot, BULB_IN_ROOT_STANDING
        )
        # Merge, don't assign: these dicts only name right-arm/right-hand joints.
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            **ARM_CRADLE,
            **HAND_CUP,
        }
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)
        add_release_contact_sensor(self.scene, self.scene.old_bulb.prim_path)
        add_bulb_crush_gate(self, "release_contact")
        self.episode_length_s = 20.0
        frame_viewer_on(self.viewer, self.scene.robot.init_state.pos)
