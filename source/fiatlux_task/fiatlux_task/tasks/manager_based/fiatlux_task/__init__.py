# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fiatlux benchmark task registrations.

Task hierarchy (see docs/roadmap.md):
- ``FIATLUX-Insert-v0``  : G1 seats a bulb into a socket (manipulation). FUNCTIONAL.
- ``FIATLUX-Climb-v0``   : G1 climbs a ladder to the fixture. Non-RL scene SCAFFOLD,
  registered with the rest of the ladder task family in ``..ladder``.
- ``FIATLUX-Replace-v0`` : end-to-end climb + insert. ROADMAP (not yet built).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="FIATLUX-Insert-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_bulb_env_cfg:G1BulbInsertEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# ROADMAP: register once the combined climb+insert episode exists. See docs/roadmap.md.
# (``FIATLUX-Climb-v0`` is already registered -- as a non-RL scaffold -- in ``..ladder``.)
# gym.register(id="FIATLUX-Replace-v0", ...)
