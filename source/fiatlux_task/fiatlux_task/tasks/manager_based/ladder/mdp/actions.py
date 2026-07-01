# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action terms for the Fiatlux ladder task family.

None are required at the foundation stage -- the base env drives the G1 with Isaac Lab's stock
``JointPositionActionCfg`` (re-exported via the package ``__init__``). Add custom action terms here
during the task phase (e.g. a binary "grasp" action, or a make/break attach action for the bulb).
"""

# TODO(task phase): add custom action terms, e.g. a gripper/attach action for bulb install/removal.
