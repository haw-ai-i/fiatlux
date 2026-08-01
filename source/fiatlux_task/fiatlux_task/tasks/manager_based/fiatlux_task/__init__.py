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

##
# Subtask family (issue #66): one mode per episode, one module/class/id each.
##

gym.register(
    id="FIATLUX-S01-ApproachLadder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subtasks.s01_approach_ladder_env_cfg:S01ApproachLadderEnvCfg",
    },
)

gym.register(
    id="FIATLUX-S04-PlaceLadder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s04_place_ladder_env_cfg:S04PlaceLadderEnvCfg"},
)

gym.register(
    id="FIATLUX-S09-DisposeBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s09_dispose_bulb_env_cfg:S09DisposeBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S03-CarryLadder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s03_carry_ladder_env_cfg:S03CarryLadderEnvCfg"},
)

gym.register(
    id="FIATLUX-S08-CarryBulbToDisposal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s08_carry_bulb_to_disposal_env_cfg:S08CarryBulbToDisposalEnvCfg"},
)

gym.register(
    id="FIATLUX-S10-ApproachNewBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s10_approach_new_bulb_env_cfg:S10ApproachNewBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S12-CarryBulbToLadder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s12_carry_bulb_to_ladder_env_cfg:S12CarryBulbToLadderEnvCfg"},
)

gym.register(
    id="FIATLUX-S02-GrabLadder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s02_grab_ladder_env_cfg:S02GrabLadderEnvCfg"},
)

gym.register(
    id="FIATLUX-S11-GrabNewBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s11_grab_new_bulb_env_cfg:S11GrabNewBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S05-ClimbLadder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s05_climb_ladder_env_cfg:S05ClimbLadderEnvCfg"},
)

gym.register(
    id="FIATLUX-S07-DescendWithBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s07_descend_with_bulb_env_cfg:S07DescendWithBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S13-ClimbWithBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s13_climb_with_bulb_env_cfg:S13ClimbWithBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S15-ClimbDown-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s15_climb_down_env_cfg:S15ClimbDownEnvCfg"},
)

gym.register(
    id="FIATLUX-S06-RemoveOldBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s06_remove_old_bulb_env_cfg:S06RemoveOldBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S14-ScrewInBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s14_screw_in_bulb_env_cfg:S14ScrewInBulbEnvCfg"},
)

SUBTASK_IDS = [
    "FIATLUX-S01-ApproachLadder-v0",
    "FIATLUX-S02-GrabLadder-v0",
    "FIATLUX-S03-CarryLadder-v0",
    "FIATLUX-S04-PlaceLadder-v0",
    "FIATLUX-S05-ClimbLadder-v0",
    "FIATLUX-S06-RemoveOldBulb-v0",
    "FIATLUX-S07-DescendWithBulb-v0",
    "FIATLUX-S08-CarryBulbToDisposal-v0",
    "FIATLUX-S09-DisposeBulb-v0",
    "FIATLUX-S10-ApproachNewBulb-v0",
    "FIATLUX-S11-GrabNewBulb-v0",
    "FIATLUX-S12-CarryBulbToLadder-v0",
    "FIATLUX-S13-ClimbWithBulb-v0",
    "FIATLUX-S14-ScrewInBulb-v0",
    "FIATLUX-S15-ClimbDown-v0",
]

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
