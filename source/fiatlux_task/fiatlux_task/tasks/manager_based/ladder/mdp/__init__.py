# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the Fiatlux ladder task family.

Re-exports every built-in Isaac Lab MDP term (observations, actions, events, ...) so the env cfg
can refer to them as ``mdp.<term>``, and layers the project's custom terms on top.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403  -- built-in MDP terms

from .actions import *  # noqa: F401, F403  -- custom action terms (none yet)
from .events import randomize_light_properties  # noqa: F401  -- custom light-randomization stub
from .observations import *  # noqa: F401, F403  -- custom observation terms (none yet)
