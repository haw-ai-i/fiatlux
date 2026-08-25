# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terms for the mate mode (S06 remove-the-old-bulb, S14 screw-the-fresh-one-in).

Both subtasks manipulate a bulb at an inverted fixture while balanced on a ladder, so both need
the same two things the rest of the family does not:

* **Alignment about the mating axis must be free.** ``BULB_PLUG_AXIS == SOCKET_SEAT_AXIS ==
  (0, 0, 1)``, and screwing is rotation about exactly that axis, so a full-quaternion orientation
  error (what ``mdp.object_socket_orientation_tanh`` measures) grows as the bulb is screwed home.
  :func:`bulb_axis_alignment_tanh` scores the angle BETWEEN the two axes instead, which the screw
  leaves invariant.
* **A crush bound with teeth.** Gripping harder is the obvious way to keep hold of a bulb while
  balancing, and glass gives at ``GLASS_CONTACT_LIMIT_N``. As a success conjunct that bound would
  only be read at the scoring step, so it is a TERMINATION here: breaking the bulb ends the
  episode and puts the gate out of reach for the rest of it.

The force channel cannot attribute contact per feature -- the bulb is one rigid body, so PhysX
reports per body, not per collider -- so the glass and cap bounds are two thresholds on the same
reading rather than two separate channels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from fiatlux_task.assets import BULB_PLUG_AXIS, SOCKET_SEAT_AXIS

from .grasp_terms import _filtered_hand_force

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _axis_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, axis: tuple[float, float, float]) -> torch.Tensor:
    """An asset's local ``axis`` expressed in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    local = torch.tensor(axis, device=env.device).expand(env.num_envs, 3)
    return quat_apply(asset.data.root_quat_w, local)


def bulb_mating_axis_angle(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("bulb"),
    socket_cfg: SceneEntityCfg = SceneEntityCfg("socket"),
) -> torch.Tensor:
    """Angle (rad) between the bulb's plug axis and the socket's seat axis.

    Zero when the bulb points straight into the socket, at any roll about that axis.
    """
    plug = _axis_w(env, asset_cfg, BULB_PLUG_AXIS)
    seat = _axis_w(env, socket_cfg, SOCKET_SEAT_AXIS)
    return torch.acos((plug * seat).sum(dim=1).clamp(-1.0, 1.0))


def bulb_axis_alignment_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("bulb"),
    socket_cfg: SceneEntityCfg = SceneEntityCfg("socket"),
) -> torch.Tensor:
    """Dense alignment reward via ``1 - tanh(axis_angle / std)``; the screw itself costs nothing."""
    return 1.0 - torch.tanh(bulb_mating_axis_angle(env, asset_cfg, socket_cfg) / std)


def grip_force_exceeded(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, limit: float) -> torch.Tensor:
    """True where any hand body's filtered contact force on the payload exceeds ``limit`` (N).

    The termination form of ``grasp_terms.grasp_force_within``: crossing the bound is a physical
    event that ends the episode, not a condition re-read at scoring time.
    """
    return _filtered_hand_force(env, sensor_cfg).max(dim=1).values > limit
