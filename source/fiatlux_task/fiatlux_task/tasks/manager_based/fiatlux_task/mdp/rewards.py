# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward / success functions for the Fiatlux G1 bulb-insertion task.

The task signal is the pose error between the grasped *bulb* and the *socket*:
- distance kernels (L2 / tanh / exponential) for coarse-to-fine reaching,
- an orientation-alignment kernel,
- a sparse "seated" bonus (also reused as the success termination),
- a contact-force penalty for compliant insertion,
plus generic smoothness / joint-limit penalties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Bulb -> socket pose error helpers
# ---------------------------------------------------------------------------


def _bulb_socket_pos_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Euclidean distance (m) between the bulb and the socket centre."""
    bulb: RigidObject = env.scene["bulb"]
    socket: RigidObject = env.scene["socket"]
    return torch.norm(bulb.data.root_pos_w - socket.data.root_pos_w, dim=1)


def _bulb_socket_ori_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Shortest-path angular distance (rad) between bulb and socket frames."""
    bulb: RigidObject = env.scene["bulb"]
    socket: RigidObject = env.scene["socket"]
    return quat_error_magnitude(bulb.data.root_quat_w, socket.data.root_quat_w)


def object_socket_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 distance penalty (use with a negative weight)."""
    return _bulb_socket_pos_error(env)


def object_socket_distance_tanh(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Dense reaching reward via ``1 - tanh(d / std)``."""
    return 1.0 - torch.tanh(_bulb_socket_pos_error(env) / std)


def object_socket_distance_exp(env: ManagerBasedRLEnv, sigma: float) -> torch.Tensor:
    """Sharp seating reward via a Gaussian kernel (strong only very close-in)."""
    d = _bulb_socket_pos_error(env)
    return torch.exp(-torch.square(d) / (sigma**2))


def object_socket_orientation_tanh(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Orientation-alignment reward via ``1 - tanh(angle / std)``."""
    return 1.0 - torch.tanh(_bulb_socket_ori_error(env) / std)


# ---------------------------------------------------------------------------
# Success / sparse bonus (also used as the success termination)
# ---------------------------------------------------------------------------


def bulb_seated(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.015,
    ori_threshold: float = 0.2,
) -> torch.Tensor:
    """True where the bulb is within position *and* orientation tolerance of the socket."""
    return (_bulb_socket_pos_error(env) < pos_threshold) & (
        _bulb_socket_ori_error(env) < ori_threshold
    )


def object_dropped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float,
) -> torch.Tensor:
    """True where the object's height has fallen below ``min_height`` (m)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < min_height


# ---------------------------------------------------------------------------
# Contact / compliance
# ---------------------------------------------------------------------------


def hand_contact_force_l2(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize squared net contact force on the grasping hand (compliance)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net = sensor.data.net_forces_w  # (N, B, 3)
    return torch.sum(torch.square(net), dim=(1, 2))


# ---------------------------------------------------------------------------
# Generic smoothness / safety penalties
# ---------------------------------------------------------------------------


def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations (L2 squared) for smoother motion."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joints that exceed their soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)
