# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The glass bound for legs that carry the bulb (S04, S05, S06, S09, S10).

The five carry legs span four tiers, so the bound cannot be declared on one of them the way
``subtask_tiers.mate`` declares it for S03 and S11. It is wired per leaf instead, against
whichever single-target hand sensor that leaf already builds.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from fiatlux_task.grasp_poses import GLASS_CONTACT_LIMIT_N

from ..mdp import mate_terms
from ..subtask_env_cfg import SubtaskShapingRewardsCfg

CRUSH_PENALTY_WEIGHT = -200.0


def add_bulb_crush_gate(cfg, sensor_name: str = "grip_contact") -> None:
    """Terminate and penalize a leg whose hand presses the carried bulb past the glass bound.

    Args:
        cfg: the leaf's env cfg, after its tier has built ``rewards`` and ``terminations``.
        sensor_name: the leaf's single-target hand sensor -- ``grip_contact`` on the legs that
            hold the bulb, ``release_contact`` on the leg that lets go of it. Its left-hand
            mirror is watched too, since either hand can crush the bulb (issue #151).
    """
    params = {
        "sensor_cfg": SceneEntityCfg(sensor_name),
        "other_sensor_cfg": SceneEntityCfg(f"{sensor_name}_left"),
        "limit": GLASS_CONTACT_LIMIT_N,
    }
    if isinstance(cfg.rewards, SubtaskShapingRewardsCfg):
        cfg.rewards.bulb_crushed = RewTerm(
            func=mate_terms.grip_force_exceeded, weight=CRUSH_PENALTY_WEIGHT, params=params
        )
    cfg.terminations.bulb_crushed = DoneTerm(func=mate_terms.grip_force_exceeded, params=params)
