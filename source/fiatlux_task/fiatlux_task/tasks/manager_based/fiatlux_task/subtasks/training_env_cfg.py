# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The learner's view of each subtask: the benchmark env plus reward shaping.

The benchmark envs in this package carry only ``gate_progress``, the one reward channel a score
reads. Shaping -- progress channels, discipline penalties, the completion bonus -- lives here, so
a shaping weight can never move a benchmark number, and a subtask can be re-shaped without
touching what it measures.

Each class is its own gym id (``...-Training-v0``); ``subtask_score.base_subtask_id`` maps one
back to the subtask it belongs to, the way it does for the teleop twins.
"""

from isaaclab.utils import configclass

from ..subtask_env_cfg import NavigateRewardsCfg
from ..subtask_tiers.balance import ClimbRewardsCfg
from ..subtask_tiers.place import PlaceRewardsCfg
from .s01_move_ladder_env_cfg import S01MoveLadderEnvCfg, S01RewardsCfg
from .s02_climb_ladder_env_cfg import S02ClimbLadderEnvCfg
from .s03_remove_old_bulb_env_cfg import S03RemoveOldBulbEnvCfg, S03RewardsCfg
from .s04_descend_with_bulb_env_cfg import S04DescendWithBulbEnvCfg, S04RewardsCfg
from .s05_carry_bulb_to_disposal_env_cfg import S05CarryBulbToDisposalEnvCfg
from .s06_dispose_bulb_env_cfg import S06DisposeBulbEnvCfg
from .s07_approach_new_bulb_env_cfg import S07ApproachNewBulbEnvCfg
from .s08_grab_new_bulb_env_cfg import S08GrabNewBulbEnvCfg, S08RewardsCfg
from .s09_carry_bulb_to_ladder_env_cfg import S09CarryBulbToLadderEnvCfg, S09RewardsCfg
from .s10_climb_with_bulb_env_cfg import S10ClimbWithBulbEnvCfg, S10RewardsCfg
from .s11_screw_in_bulb_env_cfg import S11RewardsCfg, S11ScrewInBulbEnvCfg
from .s12_climb_down_env_cfg import S12ClimbDownEnvCfg, S12RewardsCfg


@configclass
class S01MoveLadderTrainingEnvCfg(S01MoveLadderEnvCfg):
    rewards: S01RewardsCfg = S01RewardsCfg()


@configclass
class S02ClimbLadderTrainingEnvCfg(S02ClimbLadderEnvCfg):
    rewards: ClimbRewardsCfg = ClimbRewardsCfg()


@configclass
class S03RemoveOldBulbTrainingEnvCfg(S03RemoveOldBulbEnvCfg):
    rewards: S03RewardsCfg = S03RewardsCfg()


@configclass
class S04DescendWithBulbTrainingEnvCfg(S04DescendWithBulbEnvCfg):
    rewards: S04RewardsCfg = S04RewardsCfg()


@configclass
class S05CarryBulbToDisposalTrainingEnvCfg(S05CarryBulbToDisposalEnvCfg):
    rewards: NavigateRewardsCfg = NavigateRewardsCfg()


@configclass
class S06DisposeBulbTrainingEnvCfg(S06DisposeBulbEnvCfg):
    rewards: PlaceRewardsCfg = PlaceRewardsCfg()


@configclass
class S07ApproachNewBulbTrainingEnvCfg(S07ApproachNewBulbEnvCfg):
    rewards: NavigateRewardsCfg = NavigateRewardsCfg()


@configclass
class S08GrabNewBulbTrainingEnvCfg(S08GrabNewBulbEnvCfg):
    rewards: S08RewardsCfg = S08RewardsCfg()


@configclass
class S09CarryBulbToLadderTrainingEnvCfg(S09CarryBulbToLadderEnvCfg):
    rewards: S09RewardsCfg = S09RewardsCfg()


@configclass
class S10ClimbWithBulbTrainingEnvCfg(S10ClimbWithBulbEnvCfg):
    rewards: S10RewardsCfg = S10RewardsCfg()


@configclass
class S11ScrewInBulbTrainingEnvCfg(S11ScrewInBulbEnvCfg):
    rewards: S11RewardsCfg = S11RewardsCfg()


@configclass
class S12ClimbDownTrainingEnvCfg(S12ClimbDownEnvCfg):
    rewards: S12RewardsCfg = S12RewardsCfg()
