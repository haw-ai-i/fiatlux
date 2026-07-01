# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fiatlux ladder task family: G1 + ladder + lamp + bulb scene scaffolds (non-RL).

Six environments share :class:`base_env_cfg.G1LadderEnvCfg` (observation / action / event
managers only -- no rewards, terminations, or training code yet; see ``docs/roadmap.md``):

- ``FIATLUX-Base-v0``    : the shared base scene, no task logic (verify_scene.py's default).
- ``FIATLUX-Carry-v0``   : grab and position the ladder.
- ``FIATLUX-Climb-v0``   : bipedal ladder ascent (the roadmap climb slot, as a scaffold).
- ``FIATLUX-Descend-v0`` : bipedal ladder descent.
- ``FIATLUX-Remove-v0``  : unscrew / remove the seated bulb.
- ``FIATLUX-Install-v0`` : seat a new bulb at the fixture (the at-fixture counterpart of
  the tabletop ``FIATLUX-Insert-v0`` manipulation task).

All six register the **non-RL** ``isaaclab.envs:ManagerBasedEnv`` entry point, so they are
exercised with ``scripts/verify_scene.py`` -- NOT the train / eval / teleop scripts (those
assume RL envs and gym-style 5-tuple stepping).

Registration is deliberately lazy (string entry points only, no eager cfg imports):
``fiatlux_task.tasks`` swallows import errors during its auto-import walk, so an eagerly
imported cfg that fails (e.g. a missing USD) would silently drop every registration here.
"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="FIATLUX-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.base_env_cfg:G1LadderEnvCfg"},
)

gym.register(
    id="FIATLUX-Carry-v0",
    entry_point="isaaclab.envs:ManagerBasedEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.carry_env_cfg:CarryEnvCfg"},
)

gym.register(
    id="FIATLUX-Climb-v0",
    entry_point="isaaclab.envs:ManagerBasedEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.climb_env_cfg:ClimbEnvCfg"},
)

gym.register(
    id="FIATLUX-Descend-v0",
    entry_point="isaaclab.envs:ManagerBasedEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.descend_env_cfg:DescendEnvCfg"},
)

gym.register(
    id="FIATLUX-Remove-v0",
    entry_point="isaaclab.envs:ManagerBasedEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.remove_env_cfg:RemoveEnvCfg"},
)

gym.register(
    id="FIATLUX-Install-v0",
    entry_point="isaaclab.envs:ManagerBasedEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.install_env_cfg:InstallEnvCfg"},
)

# Convenience list for scripts/tests that iterate the family.
TASK_IDS = [
    "FIATLUX-Base-v0",
    "FIATLUX-Carry-v0",
    "FIATLUX-Climb-v0",
    "FIATLUX-Descend-v0",
    "FIATLUX-Remove-v0",
    "FIATLUX-Install-v0",
]
