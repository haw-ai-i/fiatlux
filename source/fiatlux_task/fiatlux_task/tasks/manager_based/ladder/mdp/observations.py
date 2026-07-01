# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation terms for the Fiatlux ladder task family.

None are required at the foundation stage -- the base env exposes only generic G1 proprioception,
built from Isaac Lab's stock terms (``joint_pos_rel``, ``joint_vel_rel``, ``root_pos_w``,
``root_quat_w``, ``root_lin_vel_w``, ``root_ang_vel_w``), which are re-exported via the package
``__init__``. Add task-specific observation functions here (e.g. bulb pose, hand-to-rung vectors)
during the task phase.
"""

# TODO(task phase): add custom observation terms, e.g.
#   def rung_in_reach(env, asset_cfg, sensor_cfg) -> torch.Tensor: ...
#   def bulb_pose_in_socket(env, lamp_cfg, bulb_cfg) -> torch.Tensor: ...
