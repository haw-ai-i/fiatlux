# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The bulb's two breakage bounds, wired per leaf.

``add_bulb_crush_gate`` is the squeeze bound for the legs that carry the bulb (S04, S05, S06, S09,
S10); the five carry legs span four tiers, so it cannot be declared on one of them the way
``subtask_tiers.mate`` declares it for S03 and S11. It reads whichever single-target hand sensor
that leaf already builds.

``add_bulb_impact_gate`` is the impact bound, and every leg that handles a bulb takes it (#138).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from fiatlux_task.grasp_poses import BULB_IMPACT_SPEED_LIMIT, GLASS_CONTACT_LIMIT_N

from .. import mdp
from ..mdp import impact_terms, mate_terms
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


def add_bulb_impact_gate(cfg) -> None:
    """Terminate and penalize a leg where either bulb is struck past ``BULB_IMPACT_SPEED_LIMIT``.

    Both bulbs are gated wherever the preset spawns both: the one being carried can be thrown, and
    the one left in the crate or the socket can be kicked.

    Args:
        cfg: the leaf's env cfg, after its tier has built ``scene``, ``rewards`` and
            ``terminations``.
    """
    for name in ("old_bulb", "fresh_bulb"):
        if getattr(cfg.scene, name, None) is None:
            continue
        term = f"{name}_struck"
        setattr(
            cfg.terminations,
            term,
            DoneTerm(
                func=impact_terms.payload_struck,
                params={"asset_cfg": SceneEntityCfg(name), "limit": BULB_IMPACT_SPEED_LIMIT},
            ),
        )
        setattr(
            cfg.rewards,
            term,
            RewTerm(func=mdp.success_term_fired, weight=CRUSH_PENALTY_WEIGHT, params={"term_name": term}),
        )
