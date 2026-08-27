# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Predicates and distance fns for the grasp mode (S08 grab-the-bulb).

The subtask ends in the hand taking an object's weight, so the gate has to distinguish *holding*
from merely *touching*: filtered contact force past a threshold, plus whatever geometric fact
rules out cheating the force reading. Every function here is STATELESS -- the debounce is
:class:`~.gates.sustained`, which lives only in the ``success`` termination, so a second counter
cannot exist to disagree with it.

``ladder_near_vertical`` lives here rather than with the ladder gates because it is the same
positive-form-of-a-termination shape: the balance and mate leaves carry it as an "and the ladder
is still standing" conjunct at a stricter bound than the tipping termination's.

Contact helpers read a sensor filtered to ONE target prim (see
``subtask_tiers.grasp.add_grasp_contact_sensor``), for the same reason ``place_terms`` does:
``force_matrix_w`` is zero unless the filtered target is a rigid body, and per-target columns
are only unambiguous when the sensor carries a single filter pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

from fiatlux_task.robots.g1 import G1_PALM_BODIES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# The right-hand palm body, every G1 variant this family builds against (Inspire). Grasp
# reach signals measure to this point rather than to a fingertip -- close enough for a dense
# shaping term, and one body avoids picking a "the" grasping digit that differs per variant.
_RIGHT_PALM_BODY = G1_PALM_BODIES[1]


# ---------------------------------------------------------------------------
# Reach distance -- bare (env) -> Tensor, no closures (distance_progress forbids them)
# ---------------------------------------------------------------------------


def _right_hand_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World position of the right-hand palm body."""
    robot: Articulation = env.scene["robot"]
    body_id = robot.find_bodies(_RIGHT_PALM_BODY)[0][0]
    return robot.data.body_pos_w[:, body_id, :]


def hand_bulb_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance (m) from the right hand to the fresh bulb; S08's reach signal."""
    bulb: RigidObject = env.scene["bulb"]
    return torch.norm(_right_hand_pos_w(env) - bulb.data.root_pos_w, dim=1)


def hand_old_bulb_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance (m) from the right hand to the seated old bulb; S03's reach signal."""
    old_bulb: RigidObject = env.scene["old_bulb"]
    return torch.norm(_right_hand_pos_w(env) - old_bulb.data.root_pos_w, dim=1)


# ---------------------------------------------------------------------------
# Contact force -- generic, sensor_cfg must be filtered to ONE target prim
# ---------------------------------------------------------------------------


def _filtered_hand_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Per-body force magnitude (N, B) on the sensor's filtered target."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.force_matrix_w.sum(dim=2)  # (N, B, M, 3) -> (N, B, 3)
    return force.norm(dim=-1)


def grasp_contact_bootstrap(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, saturation_force: float
) -> torch.Tensor:
    """Dense ``[0, 1)`` reward for building filtered contact force, saturating at ``saturation_force``.

    Bridges the gap between "reached it" and the sparse, sustained gate: reach-progress pays for
    approach, this pays for closing the hand once there, before the gate's own force threshold is
    within reach.
    """
    force = _filtered_hand_force(env, sensor_cfg).max(dim=1).values
    return torch.tanh(force / saturation_force)


def grasp_force_within(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, limit: float) -> torch.Tensor:
    """True where no hand body's filtered contact force exceeds ``limit`` (a fragility bound)."""
    return _filtered_hand_force(env, sensor_cfg).max(dim=1).values <= limit


def hand_bodies_in_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, min_bodies: int, force_threshold: float
) -> torch.Tensor:
    """True where at least ``min_bodies`` sensor bodies register filtered force above ``force_threshold``.

    A single contact point (a fingertip grazing the object) is touching, not holding; requiring
    several bodies is the cheap proxy for "wrapped by the hand" without needing per-finger grasp
    geometry.
    """
    return (_filtered_hand_force(env, sensor_cfg) > force_threshold).sum(dim=1) >= min_bodies


# ---------------------------------------------------------------------------
# The ladder still standing -- a conjunct for every leaf that works on or beside it
# ---------------------------------------------------------------------------


def ladder_near_vertical(
    env: ManagerBasedRLEnv, tilt_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("ladder")
) -> torch.Tensor:
    """True where the ladder's tilt from vertical is under ``tilt_limit`` (rad).

    The positive form of ``mdp.ladder_tipped`` (same up-axis angle, reversed comparison), so a
    gate can carry a STRICTER bound than the 0.6 rad tipping termination without conflating the
    two: a ladder leaning right up to the tipping point must not read as one a robot can work
    from.
    """
    ladder: RigidObject = env.scene[asset_cfg.name]
    up = torch.zeros(env.num_envs, 3, device=env.device)
    up[:, 2] = 1.0
    up_w = quat_apply(ladder.data.root_quat_w, up)
    return torch.acos(up_w[:, 2].clamp(-1.0, 1.0)) < tilt_limit


# ---------------------------------------------------------------------------
# S08 -- grab the fresh bulb
# ---------------------------------------------------------------------------


def object_lifted(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_height: float) -> torch.Tensor:
    """True where the object's root sits above ``min_height`` (world frame).

    Generic height gate for "off the surface it started on"; the leaf supplies ``min_height``
    from that surface's own rest height plus a clearance margin.
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    return obj.data.root_pos_w[:, 2] > min_height
