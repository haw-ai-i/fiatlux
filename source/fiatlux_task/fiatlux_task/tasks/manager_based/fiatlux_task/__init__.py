# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fiatlux benchmark task registrations.

One family, one scene (``scene_cfg.G1ReplaceSceneCfg``), preset layouts per task:

- ``FIATLUX-Replace-v0`` : THE BENCHMARK -- the full light-bulb replacement (randomized
  room layout, normalized-progress scoring, standard/privileged observation modes).
  ``verify_scene.py``'s default target.
- ``FIATLUX-S01..S12-*-v0`` : the twelve subtasks Replace decomposes into (issue #66), each
  starting from its predecessor's end state (``subtask_env_cfg.py`` + ``subtasks/``).
- ``FIATLUX-S01..S12-*-Training-v0`` : the same subtasks plus reward shaping (issue #169,
  ``subtasks/training_env_cfg.py``).

Every task spawns an ego camera, so ``verify_scene.py`` / ``record_run.py`` need
``--enable_cameras``.

Registration is deliberately lazy (string entry points only, no eager cfg imports):
``fiatlux_task.tasks`` swallows import errors during its auto-import walk, so an eagerly
imported cfg that fails (e.g. a missing USD) would silently drop every registration here.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# NOTE: the teleop task variants (FIATLUX-S<NN>-*-Teleop-v0, FIATLUX-LadderGallery-Teleop-v0, ...) live
# in the separate `fiatlux_teleop` extension package (source/fiatlux_teleop) and are registered by
# importing it -- kept out of the benchmark so this package imports/runs without teleop's
# OpenXR/CloudXR/SONIC deps.

# NOTE: only Replace carries an rsl_rl_cfg_entry_point. The subtask ids (benchmark and -Training)
# intentionally have none yet -- no PPORunnerCfg (network sizes, obs_groups routing) has been
# designed/tuned for them. They work fully with record_run.py / eval.py / any non-rsl_rl policy
# (including groot); only scripts/rsl_rl/{train,play}.py would need one added first.

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

# Convenience list for scripts/tests that iterate the registered top-level task(s). The other
# ladder-family members (Base/Carry/Climb/Descend/Remove/Install) were retired and are no
# longer registered -- see SUBTASK_IDS above for the 12 subtasks that are.
TASK_IDS = [
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
