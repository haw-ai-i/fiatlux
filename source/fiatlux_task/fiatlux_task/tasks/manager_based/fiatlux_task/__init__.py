# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fiatlux benchmark task registrations.

Task hierarchy (see docs/roadmap.md):

One family, one scene (``scene_cfg.G1ReplaceSceneCfg``), preset layouts per task:

- ``FIATLUX-Insert-v0``  : G1 seats a bulb into a socket (manipulation, *tabletop* preset).
  FUNCTIONAL, RL.
- ``FIATLUX-Climb-v0``   : G1 climbs the step ladder to the fixture height (*at-height*
  preset, whole-body RL). FUNCTIONAL, RL.
- Workshop-preset scaffolds: G1 + ladder + socket-lamp + bulb on the floor. The five share
  the **non-RL** :class:`base_env_cfg.FamilyBaseEnvCfg` base (observation / action / event
  managers only -- no rewards, terminations, or training code yet):

  - ``FIATLUX-Base-v0``    : the shared base scene, no task logic (verify_scene.py's default).
  - ``FIATLUX-Carry-v0``   : grab and position the ladder.
  - ``FIATLUX-Descend-v0`` : bipedal ladder descent.
  - ``FIATLUX-Remove-v0``  : unscrew / remove the seated bulb.
  - ``FIATLUX-Install-v0`` : seat a new bulb at the fixture (the at-fixture counterpart of
    the tabletop ``FIATLUX-Insert-v0`` manipulation task).
  - ``FIATLUX-Replace-v0`` : the whole family world at once (issue #20) -- robot, ladder,
    table+bulb, and the elevated fixture, each randomized into its own non-overlapping floor
    "safe zone", fixture randomly ceiling- or wall-mounted. Scene-only: no reward/termination
    or policy-chaining logic (that stays on the roadmap; see the issue's own descoping
    comment).

The scaffolds register the non-RL ``isaaclab.envs:ManagerBasedEnv`` entry point; the
train / eval / teleop scripts assume RL envs and only apply to the RL members.
``scripts/verify_scene.py`` covers every family member (Insert needs ``--enable_cameras``).

Registration is deliberately lazy (string entry points only, no eager cfg imports):
``fiatlux_task.tasks`` swallows import errors during its auto-import walk, so an eagerly
imported cfg that fails (e.g. a missing USD) would silently drop every registration here.
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

gym.register(
    id="FIATLUX-Climb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.climb_env_cfg:ClimbEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ClimbPPORunnerCfg",
    },
)

##
# Workshop-preset scaffolds (non-RL).
##

gym.register(
    id="FIATLUX-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.base_env_cfg:FamilyBaseEnvCfg"},
)

gym.register(
    id="FIATLUX-Carry-v0",
    entry_point="isaaclab.envs:ManagerBasedEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.carry_env_cfg:CarryEnvCfg"},
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

gym.register(
    id="FIATLUX-Replace-v0",
    entry_point="isaaclab.envs:ManagerBasedEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.replace_env_cfg:ReplaceEnvCfg"},
)

# Convenience list for scripts/tests that iterate the ladder family (Climb is the
# family's RL member; the rest are non-RL scaffolds).
TASK_IDS = [
    "FIATLUX-Base-v0",
    "FIATLUX-Carry-v0",
    "FIATLUX-Climb-v0",
    "FIATLUX-Descend-v0",
    "FIATLUX-Remove-v0",
    "FIATLUX-Install-v0",
    "FIATLUX-Replace-v0",
]
