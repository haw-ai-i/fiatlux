# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the FIATLUX task (e.g. contact sensing)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_pose_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """World-frame root pose of a rigid object: position (xyz) + quaternion (wxyz).

    Used for the privileged / ground-truth observation group (bulb and socket).

    Returns:
        Tensor of shape (num_envs, 7).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.cat([asset.data.root_pos_w, asset.data.root_quat_w], dim=-1)


def replace_score_distances(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The replace task's four score-relevant distances, for the privileged group.

    Columns: ladder top -> fixture seat, fresh-bulb plug -> fixture seat, old-bulb plug
    clearance from the fixture seat, old bulb -> disposal crate. These mirror the channels
    the reward terms normalize, exposed so a cheatcode policy can read the score geometry.
    The two old-bulb columns are the RAW geometry; the reward terms gate the same
    quantities on attachment state (``old_bulb_release_clearance`` and
    ``old_bulb_disposal_distance_pinned``), so a still-attached bulb reads as seated there
    while these columns keep reporting live distance.

    Returns:
        Tensor of shape (num_envs, 4).
    """
    from . import rewards

    return torch.stack(
        [
            rewards.ladder_fixture_distance(env),
            rewards.bulb_fixture_distance(env),
            rewards.old_bulb_fixture_clearance(env),
            rewards.old_bulb_disposal_distance(env),
        ],
        dim=1,
    )


def lidar_ranges(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Per-ray hit distance from a ``MultiMeshRayCasterCfg``/``RayCasterCfg`` lidar.

    Misses report ``inf`` in ``ray_hits_w`` (no intersection within ``max_distance``);
    clamped to the sensor's own ``max_distance`` so the observation stays finite (a raw
    ``inf`` would poison downstream normalization/concatenation).

    Returns:
        Tensor of shape (num_envs, num_rays).
    """
    from isaaclab.sensors.ray_caster import RayCaster

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ranges = torch.linalg.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    return torch.nan_to_num(ranges, posinf=sensor.cfg.max_distance).clamp(max=sensor.cfg.max_distance)


def contact_net_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Net contact forces (world frame) from the contact sensor, flattened for policy obs.

    Uses the current timestep net forces (no history). Body selection is via sensor_cfg.body_ids
    if set by the manager, or sensor_cfg.body_names matched against the sensor's body_names.

    Returns:
        Tensor of shape (num_envs, num_bodies * 3) in world frame (x,y,z per body).
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net = contact_sensor.data.net_forces_w  # (N, B, 3)
    body_ids = sensor_cfg.body_ids
    if body_ids is None or body_ids == slice(None):
        if getattr(sensor_cfg, "body_names", None) is not None:
            names = [sensor_cfg.body_names] if isinstance(sensor_cfg.body_names, str) else sensor_cfg.body_names
            pattern = re.compile(names[0] if len(names) == 1 else "|".join(names))
            body_ids = [i for i, b in enumerate(contact_sensor.body_names) if pattern.search(b)]
            if body_ids:
                net = net[:, body_ids, :]
    else:
        net = net[:, body_ids, :]
    return net.reshape(env.num_envs, -1)
