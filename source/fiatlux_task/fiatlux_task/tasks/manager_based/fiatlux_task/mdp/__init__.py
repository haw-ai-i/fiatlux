# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the Fiatlux G1 tasks.

Re-exports Isaac Lab's built-in MDP terms (actions, observations, events,
terminations) plus the task-specific functions defined in this sub-package.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .events import randomize_light_properties  # noqa: F401
from .observations import contact_net_forces, root_pose_w  # noqa: F401
from .rewards import (  # noqa: F401
    bulb_seated,
    climb_height_progress,
    climbed_to_target,
    com_sway_l2,
    fall_terminated,
    hand_contact_force_l2,
    joint_acc_l2,
    joint_pos_limits,
    ladder_contact_fraction,
    object_dropped,
    object_socket_distance,
    object_socket_distance_exp,
    object_socket_distance_tanh,
    object_socket_orientation_tanh,
)
