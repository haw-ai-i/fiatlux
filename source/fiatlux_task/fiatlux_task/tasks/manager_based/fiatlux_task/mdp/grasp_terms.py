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

import contextlib
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

from fiatlux_task.robots.g1 import G1_PALM_BODY_BY_VARIANT

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Grasp reach signals measure to the right-hand palm body rather than to a fingertip -- close
# enough for a dense shaping term, and one body avoids picking a "the" grasping digit that
# differs per variant. WHICH body that is depends on the mounted hand (Inspire names it
# ``right_hand_base_link``, Dex3 ``right_hand_palm_link``), so it is resolved from the attached
# articulation instead of hardcoded: ``swap_robot_variant`` rewrites joint names only, never
# body names, so a hardcoded palm here crashes env creation on the other variant.
_PALM_ID_CACHE_ATTR = "_fiatlux_right_palm_body_id"


def _right_palm_body_id(robot: Articulation) -> int:
    """Body index of the right palm on whichever G1 hand variant is mounted."""
    cached = getattr(robot, _PALM_ID_CACHE_ATTR, None)
    if cached is not None:
        return cached
    for name in G1_PALM_BODY_BY_VARIANT.values():
        if name in robot.body_names:
            body_id = robot.find_bodies(name)[0][0]
            # not cacheable on every object; resolving per call is still correct
            with contextlib.suppress(AttributeError):
                setattr(robot, _PALM_ID_CACHE_ATTR, body_id)
            return body_id
    raise ValueError(
        "no known G1 right-palm body on the attached robot (looked for "
        f"{sorted(G1_PALM_BODY_BY_VARIANT.values())}); its bodies are {robot.body_names}"
    )


# ---------------------------------------------------------------------------
# Reach distance -- bare (env) -> Tensor, no closures (distance_progress forbids them)
# ---------------------------------------------------------------------------


def _right_hand_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World position of the right-hand palm body."""
    robot: Articulation = env.scene["robot"]
    body_id = _right_palm_body_id(robot)
    return robot.data.body_pos_w[:, body_id, :]


def hand_bulb_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance (m) from the right hand to the fresh bulb; S08's reach signal."""
    bulb: RigidObject = env.scene["fresh_bulb"]
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


def grasp_force_within(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    limit: float,
    other_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """True where no hand body's filtered contact force exceeds ``limit`` (a fragility bound).

    With ``other_sensor_cfg`` the bound covers BOTH hands: either one can crush the bulb, so the
    limit has to hold for both (issue #151).
    """
    worst = _filtered_hand_force(env, sensor_cfg).max(dim=1).values
    if other_sensor_cfg is not None:
        worst = torch.maximum(worst, _filtered_hand_force(env, other_sensor_cfg).max(dim=1).values)
    return worst <= limit


def hand_bodies_in_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_bodies: int,
    force_threshold: float,
    other_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """True where at least ``min_bodies`` sensor bodies register filtered force above ``force_threshold``.

    A single contact point (a fingertip grazing the object) is touching, not holding; requiring
    several bodies is the cheap proxy for "wrapped by the hand" without needing per-finger grasp
    geometry.

    With ``other_sensor_cfg`` EITHER hand qualifies on its own (issue #151). Counted per hand
    rather than pooled: the bodies have to be wrapped round the object by one hand, and pooling
    would let two fingers of each count as a grasp.
    """
    held = (_filtered_hand_force(env, sensor_cfg) > force_threshold).sum(dim=1) >= min_bodies
    if other_sensor_cfg is not None:
        other = (_filtered_hand_force(env, other_sensor_cfg) > force_threshold).sum(dim=1) >= min_bodies
        held = held | other
    return held


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
