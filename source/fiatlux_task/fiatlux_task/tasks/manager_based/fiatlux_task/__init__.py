# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fiatlux benchmark task registrations.

Task hierarchy (see docs/roadmap.md):

One family, one scene (``scene_cfg.G1ReplaceSceneCfg``), preset layouts per task:

- ``FIATLUX-Replace-v0`` : THE BENCHMARK -- the full light-bulb replacement (randomized
  room layout, normalized-progress scoring, standard/cheatcode observation modes; see
  ``journal/specs/full-task-benchmark-plan.md``). FUNCTIONAL, RL.
- ``FIATLUX-Insert-v0``  : G1 seats a bulb into a socket (manipulation, *tabletop* preset).
  FUNCTIONAL, RL.
- ``FIATLUX-Climb-v0``   : G1 climbs the step ladder to the fixture height (*at-height*
  preset, whole-body RL). FUNCTIONAL, RL.
- ``FIATLUX-Carry-v0``   : G1 grasps a ladder and positions it upright at a target (ladder-
  handling, *carry* preset, arm+hand manipulation RL). FUNCTIONAL, RL.
- ``FIATLUX-Descend-v0`` : bipedal ladder descent -- the mirror image of Climb's ascent
  reward/termination scheme (``descend_env_cfg.py``). FUNCTIONAL, RL.
- ``FIATLUX-Install-v0`` : seat a new bulb from a floor parts crate into the same bench
  lamp socket ``FIATLUX-Insert-v0`` uses (``install_env_cfg.py``, Insert's own reward/
  termination set unchanged -- same entities, larger starting gap). FUNCTIONAL, RL.
- ``FIATLUX-Remove-v0``  : unscrew / remove the seated bulb (``remove_env_cfg.py``,
  Replace's own removal/disposal reward channels, parametrized onto this scene's
  standalone ``bulb`` entity). RL, but **scored-not-yet-achievable**: the bulb is
  kinematic (no attach/detach mechanic yet -- unification spec Phase 4), so no policy
  can move it. Same documented gap as Replace's own removal channel.
- ``FIATLUX-Base-v0``    : the shared scene-only cfg, deliberately **non-RL**
  (:class:`base_env_cfg.FamilyBaseEnvCfg` -- observation/action/event managers only, no
  task to reward). Not a task; ``verify_scene.py``'s default target.

``FIATLUX-Base-v0`` registers the non-RL ``isaaclab.envs:ManagerBasedEnv`` entry point,
so ``gym.make`` (record_run.py, eval.py, zero_agent.py) cannot construct it --
``ManagerBasedEnv.__init__`` has no ``**kwargs`` catch-all for the registration's own
``env_cfg_entry_point``, unlike ``ManagerBasedRLEnv``. ``scripts/verify_scene.py``
bypasses ``gym.make`` for exactly this reason and covers every family member directly
(Insert and Replace need ``--enable_cameras`` for their camera sensors).

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

# Teleoperation variant of Insert (issue #51, Phase 1): arm differential-IK + binary grip,
# driven by scripts/insert_teleop.py. Same scene/obs/rewards as Insert-v0; only the action interface
# differs (see insert_teleop_env_cfg).
gym.register(
    id="FIATLUX-Insert-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.insert_teleop_env_cfg:G1BulbInsertTeleopEnvCfg",
    },
)

# Teleoperation variant of Carry (ladder-positioning): bimanual arm IK + Dex3 grip on the ladder
# scene, driven by scripts/sonic_teleop.py (SONIC legs + whole-body teleop). See carry_teleop_env_cfg.
gym.register(
    id="FIATLUX-Carry-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carry_teleop_env_cfg:CarryTeleopEnvCfg",
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
# The shared scene-only scaffold (non-RL; not a task -- see module docstring).
##

gym.register(
    id="FIATLUX-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.base_env_cfg:FamilyBaseEnvCfg"},
)

##
# Remaining family RL members.
##

gym.register(
    id="FIATLUX-Carry-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carry_env_cfg:CarryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CarryPPORunnerCfg",
    },
)

# NOTE: Descend/Remove/Install intentionally carry no rsl_rl_cfg_entry_point yet -- no
# PPORunnerCfg (network sizes, obs_groups routing) has been designed/tuned for them. They
# work fully with record_run.py / eval.py / any non-rsl_rl policy (including groot); only
# scripts/rsl_rl/{train,play}.py would need one added first.

gym.register(
    id="FIATLUX-Descend-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.descend_env_cfg:DescendEnvCfg"},
)

gym.register(
    id="FIATLUX-Remove-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.remove_env_cfg:RemoveEnvCfg"},
)

gym.register(
    id="FIATLUX-Install-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.install_env_cfg:InstallEnvCfg"},
)

##
# The full-task benchmark (RL).
##

gym.register(
    id="FIATLUX-Replace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.replace_env_cfg:ReplaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ReplacePPORunnerCfg",
    },
)

# Convenience list for scripts/tests that iterate the ladder family. Every member except
# Base is RL now; Base is the shared scene-only cfg (no task, deliberately non-RL).
TASK_IDS = [
    "FIATLUX-Base-v0",
    "FIATLUX-Carry-v0",
    "FIATLUX-Climb-v0",
    "FIATLUX-Descend-v0",
    "FIATLUX-Remove-v0",
    "FIATLUX-Install-v0",
    "FIATLUX-Replace-v0",
]
