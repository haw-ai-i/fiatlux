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
  standalone ``bulb`` entity). RL and achievable -- the bulb is dynamic and lifts out of
  the socket's open hole -- but nothing gates unscrewing here, so it scores "pick it up
  and bin it". Replace gates removal on ``mdp.bulb_attachment`` (issue #54); porting that
  term here is what would make this a genuine unscrew task.
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

# NOTE: the teleop task variants (FIATLUX-{Insert,Carry,LadderGallery}-Teleop-v0) live in the separate
# `fiatlux_teleop` extension package (source/fiatlux_teleop) and are registered by importing it -- kept
# out of the benchmark so this package imports/runs without teleop's OpenXR/CloudXR/SONIC deps.

##
# The shared scene-only scaffold (non-RL; not a task -- see module docstring).
##

##
# Remaining family RL members.
##

# NOTE: Descend/Remove/Install intentionally carry no rsl_rl_cfg_entry_point yet -- no
# PPORunnerCfg (network sizes, obs_groups routing) has been designed/tuned for them. They
# work fully with record_run.py / eval.py / any non-rsl_rl policy (including groot); only
# scripts/rsl_rl/{train,play}.py would need one added first.

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
#
# Twelve, not the original fifteen: approach / grab / carry / place the ladder are one
# ``S01-MoveLadder`` episode. Gating an operator on how the ladder travelled -- rail grasped,
# feet lifted, grip retained, hand released -- put four ways to score zero in front of a
# deliverable that is just "the ladder ends up standing at the fixture". The rest of the chain
# renumbered to stay contiguous, which is a `-v0` contract change for every id below (the
# foundation spec's versioning policy: a new id, not a bump, since the old numbering names a
# different chain).
##

gym.register(
    id="FIATLUX-S01-MoveLadder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s01_move_ladder_env_cfg:S01MoveLadderEnvCfg"},
)

gym.register(
    id="FIATLUX-S02-ClimbLadder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s02_climb_ladder_env_cfg:S02ClimbLadderEnvCfg"},
)

gym.register(
    id="FIATLUX-S03-RemoveOldBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s03_remove_old_bulb_env_cfg:S03RemoveOldBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S04-DescendWithBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s04_descend_with_bulb_env_cfg:S04DescendWithBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S05-CarryBulbToDisposal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.subtasks.s05_carry_bulb_to_disposal_env_cfg:S05CarryBulbToDisposalEnvCfg"
        ),
    },
)

gym.register(
    id="FIATLUX-S06-DisposeBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s06_dispose_bulb_env_cfg:S06DisposeBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S07-ApproachNewBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s07_approach_new_bulb_env_cfg:S07ApproachNewBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S08-GrabNewBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s08_grab_new_bulb_env_cfg:S08GrabNewBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S09-CarryBulbToLadder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subtasks.s09_carry_bulb_to_ladder_env_cfg:S09CarryBulbToLadderEnvCfg"
    },
)

gym.register(
    id="FIATLUX-S10-ClimbWithBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s10_climb_with_bulb_env_cfg:S10ClimbWithBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S11-ScrewInBulb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s11_screw_in_bulb_env_cfg:S11ScrewInBulbEnvCfg"},
)

gym.register(
    id="FIATLUX-S12-ClimbDown-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.s12_climb_down_env_cfg:S12ClimbDownEnvCfg"},
)

##
# Training twins: the same subtask plus reward shaping (issue #169). The benchmark ids above
# carry only the channel a score reads, so a shaping weight cannot move a benchmark number.
##

gym.register(
    id="FIATLUX-S01-MoveLadder-Training-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.training_env_cfg:S01MoveLadderTrainingEnvCfg"},
)

gym.register(
    id="FIATLUX-S02-ClimbLadder-Training-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.training_env_cfg:S02ClimbLadderTrainingEnvCfg"},
)

gym.register(
    id="FIATLUX-S03-RemoveOldBulb-Training-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.training_env_cfg:S03RemoveOldBulbTrainingEnvCfg"},
)

gym.register(
    id="FIATLUX-S04-DescendWithBulb-Training-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.training_env_cfg:S04DescendWithBulbTrainingEnvCfg"},
)

gym.register(
    id="FIATLUX-S05-CarryBulbToDisposal-Training-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.training_env_cfg:S05CarryBulbToDisposalTrainingEnvCfg"},
)

gym.register(
    id="FIATLUX-S06-DisposeBulb-Training-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.training_env_cfg:S06DisposeBulbTrainingEnvCfg"},
)

gym.register(
    id="FIATLUX-S07-ApproachNewBulb-Training-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.training_env_cfg:S07ApproachNewBulbTrainingEnvCfg"},
)

gym.register(
    id="FIATLUX-S08-GrabNewBulb-Training-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.training_env_cfg:S08GrabNewBulbTrainingEnvCfg"},
)

gym.register(
    id="FIATLUX-S09-CarryBulbToLadder-Training-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.training_env_cfg:S09CarryBulbToLadderTrainingEnvCfg"},
)

gym.register(
    id="FIATLUX-S10-ClimbWithBulb-Training-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.training_env_cfg:S10ClimbWithBulbTrainingEnvCfg"},
)

gym.register(
    id="FIATLUX-S11-ScrewInBulb-Training-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.training_env_cfg:S11ScrewInBulbTrainingEnvCfg"},
)

gym.register(
    id="FIATLUX-S12-ClimbDown-Training-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.training_env_cfg:S12ClimbDownTrainingEnvCfg"},
)

SUBTASK_IDS = [
    "FIATLUX-S01-MoveLadder-v0",
    "FIATLUX-S02-ClimbLadder-v0",
    "FIATLUX-S03-RemoveOldBulb-v0",
    "FIATLUX-S04-DescendWithBulb-v0",
    "FIATLUX-S05-CarryBulbToDisposal-v0",
    "FIATLUX-S06-DisposeBulb-v0",
    "FIATLUX-S07-ApproachNewBulb-v0",
    "FIATLUX-S08-GrabNewBulb-v0",
    "FIATLUX-S09-CarryBulbToLadder-v0",
    "FIATLUX-S10-ClimbWithBulb-v0",
    "FIATLUX-S11-ScrewInBulb-v0",
    "FIATLUX-S12-ClimbDown-v0",
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

SUBTASK_TRAINING_IDS = [
    "FIATLUX-S01-MoveLadder-Training-v0",
    "FIATLUX-S02-ClimbLadder-Training-v0",
    "FIATLUX-S03-RemoveOldBulb-Training-v0",
    "FIATLUX-S04-DescendWithBulb-Training-v0",
    "FIATLUX-S05-CarryBulbToDisposal-Training-v0",
    "FIATLUX-S06-DisposeBulb-Training-v0",
    "FIATLUX-S07-ApproachNewBulb-Training-v0",
    "FIATLUX-S08-GrabNewBulb-Training-v0",
    "FIATLUX-S09-CarryBulbToLadder-Training-v0",
    "FIATLUX-S10-ClimbWithBulb-Training-v0",
    "FIATLUX-S11-ScrewInBulb-Training-v0",
    "FIATLUX-S12-ClimbDown-Training-v0",
]
