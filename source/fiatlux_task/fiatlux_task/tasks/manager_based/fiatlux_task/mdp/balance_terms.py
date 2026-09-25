# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Predicates for the balance mode (S02, S04, S10, S12 -- climbing and descending).

An older, fixed-layout approach gated on a hardcoded ``xy_center`` taken from
``TOP_ROBOT_POSITION`` / ``CLIMB_ROBOT_POSITION``, which describe the default workshop layout's
ladder. In this chain the ladder is wherever a placement subtask left it, so the gates below read
its LIVE root pose instead. That is the only structural difference; the height/proximity/speed
shape is the same three-part gate, and the speed cap serves the same purpose -- a solver-kicked
robot flying through the success region must not score.

Every function here is STATELESS. Conjunctions are assembled in the leaves with ``mdp.all_of`` and
debounced by :class:`~.gates.sustained` in the ``success`` termination alone, so a second counter
cannot exist to disagree with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .rewards import _ladder_top_point_w, _root_pos_env

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ladder_root_pos_env(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Ladder root position in the env-local frame."""
    return _root_pos_env(env, SceneEntityCfg("ladder"))


def climbed_to_ladder_top(
    env: ManagerBasedRLEnv,
    height_slack: float,
    xy_radius: float,
    max_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True where the robot's pelvis has reached the placed ladder's upper steps, under control.

    The height bound and the horizontal centre both come from the ladder's live top point, so a
    ladder standing anywhere in the room gates identically. ``height_slack`` is how far below the
    top step a solid stance may leave the pelvis.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    top = _ladder_top_point_w(env) - env.scene.env_origins
    pos = _root_pos_env(env, asset_cfg)
    high = pos[:, 2] > top[:, 2] - height_slack
    near = torch.norm((pos - top)[:, :2], dim=1) < xy_radius
    calm = asset.data.root_lin_vel_w.norm(dim=-1) < max_speed
    return high & near & calm


def descended_from_ladder(
    env: ManagerBasedRLEnv,
    maximum_height: float,
    xy_radius: float,
    max_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True where the robot has come back down to floor stance at the placed ladder's base.

    ``maximum_height`` is an absolute floor-stance pelvis height, not a ladder-relative one: the
    ladder's feet are on the same floor the robot lands on, so only the horizontal centre needs
    the live pose.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base = _ladder_root_pos_env(env)
    pos = _root_pos_env(env, asset_cfg)
    low = pos[:, 2] < maximum_height
    near = torch.norm((pos - base)[:, :2], dim=1) < xy_radius
    calm = asset.data.root_lin_vel_w.norm(dim=-1) < max_speed
    return low & near & calm
