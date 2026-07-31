# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the Fiatlux G1 tasks.

Re-exports Isaac Lab's built-in MDP terms (actions, observations, events,
terminations) plus the task-specific functions defined in this sub-package.
"""

from isaaclab.envs.mdp import (  # noqa: F401
    JointPositionActionCfg,
    action_rate_l2,
    ang_vel_xy_l2,
    bad_orientation,
    base_ang_vel,
    base_lin_vel,
    base_pos_z,
    body_pose_w,
    image_features,
    is_terminated,
    is_terminated_term,
    joint_deviation_l1,
    joint_pos_rel,
    joint_vel_l2,
    joint_vel_rel,
    last_action,
    projected_gravity,
    randomize_rigid_body_material,
    randomize_rigid_body_scale,
    reset_joints_by_offset,
    reset_root_state_uniform,
    reset_scene_to_default,
    root_ang_vel_w,
    root_height_below_minimum,
    root_lin_vel_w,
    root_pos_w,
    root_quat_w,
    time_out,
)

from .events import (  # noqa: F401
    hand_grip_material_event,
    randomize_light_properties,
    randomize_material_tint,
    randomize_prop_scale,
)
from .observations import (  # noqa: F401
    contact_net_forces,
    lidar_ranges,
    replace_score_distances,
    root_pose_w,
)
from .rewards import (  # noqa: F401
    LADDER_TILT_LIMIT,
    arrived_at_ladder,
    base_facing_error,
    base_ladder_distance,
    bulb_fixture_distance,
    bulb_seated,
    climb_height_progress,
    climbed_to_target,
    com_sway_l2,
    completion_bonus,
    descend_height_progress,
    descended_to_target,
    distance_progress,
    fall_terminated,
    full_replacement_success,
    hand_contact_force_l2,
    joint_acc_l2,
    joint_pos_limits,
    ladder_contact_fraction,
    ladder_fixture_distance,
    ladder_ready,
    ladder_tipped,
    object_dropped,
    object_socket_distance,
    object_socket_distance_exp,
    object_socket_distance_tanh,
    object_socket_orientation_tanh,
    old_bulb_disposal_distance,
    old_bulb_disposed,
    old_bulb_dropped,
    old_bulb_fixture_clearance,
    old_bulb_removed,
    removal_bulb_disposal_distance,
    removal_bulb_fixture_clearance,
)
