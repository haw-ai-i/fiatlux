# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Predicates for the navigate mode's carrying legs (S03, S08, S12) and bench approach (S10).

``NavigateSubtaskCfg``/``NavigateRewardsCfg`` (``subtask_env_cfg.py``) already cover a bare walk
like S01's; what a CARRYING leg adds on top is two things:

* a start state where the payload is already in hand, composed from the carrier's own (randomized)
  root pose and a frozen in-root grasp (``grasp_poses.py``) -- cfg-build-time geometry, since a
  leaf's ``__post_init__`` sets ``init_state``s before the scene exists, so there is no forward
  kinematics to read yet;
* an arrival gate that additionally requires the payload still be held, or a thrown payload that
  skids into the target radius would score. That needs an unambiguous per-target force reading,
  which the scene's shared ``hand_contact`` cannot give once it filters more than one prim (the
  replace preset always filters both bulbs) -- the same reason ``place_terms.object_released``
  needs its own single-target sensor. ``add_grip_contact_sensor`` mirrors
  ``subtask_tiers.place.add_release_contact_sensor`` for exactly that reason, read the other way.

S10 (approach the fresh bulb, hands free) needs neither: it is a bare walk like S01's, just
pointed at the bulb instead of the ladder, so it only needs the missing distance channel.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from ..scene_cfg import (
    CLIMB_ROBOT_POSITION,
    DISPOSAL_ZONE_HALF_SIZE,
    LADDER_POSITION,
    TABLETOP_BULB_POSITION,
    TABLETOP_ROBOT_POSITION,
    G1ReplaceSceneCfg,
)
from .rewards import LADDER_TILT_LIMIT, base_facing_error, base_ladder_distance, ladder_ready, ladder_tipped

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# Provisional constants
# ---------------------------------------------------------------------------

# "Close enough to start climbing" -- a foot-placement question, unlike LADDER_APPROACH_RADIUS
# (arm's reach to a rail, for grasping). PROVISIONAL: based on climb_env_cfg's own validated
# ready-to-ascend stance (CLIMB_ROBOT_POSITION, "at the step ladder's base, ready to ascend",
# tuned against verify_scene.py orbit videos) relative to the workshop preset's fixed
# LADDER_POSITION -- that stance stands this far out from the ladder's root. Used as a RADIUS
# rather than that stance's own directional offset because the replace preset samples the
# ladder's yaw, so no fixed direction generalizes.
LADDER_MOUNT_RADIUS = math.dist(CLIMB_ROBOT_POSITION[:2], LADDER_POSITION[:2])

# "Close enough to reach for the bulb next." PROVISIONAL: based on the tabletop preset's own
# authored manipulation standoff (TABLETOP_ROBOT_POSITION relative to TABLETOP_BULB_POSITION),
# the only measured robot-to-bulb approach distance on record; the replace preset's table is the
# same asset, just randomly placed.
BULB_APPROACH_RADIUS = math.dist(TABLETOP_ROBOT_POSITION[:2], TABLETOP_BULB_POSITION[:2])

# "Close enough to be standing at the crate." PROVISIONAL: no authored robot-relative standoff
# exists for the disposal crate (unlike the tabletop bench), so this reuses the crate's own
# layout zone half-extent (footprint + working clearance) as the least-arbitrary bound on record.
DISPOSAL_ARRIVAL_RADIUS = DISPOSAL_ZONE_HALF_SIZE

# Grip-presence floor for the carrying legs' arrival gate. PROVISIONAL: the same cutoff
# subtask_tiers.place.RELEASE_FORCE_THRESHOLD_N uses, read the other way -- below it counts as
# let go, at or above it counts as still held.
GRIP_FORCE_THRESHOLD_N = 1.0  # N


# ---------------------------------------------------------------------------
# Cfg-time geometry (plain Python; no torch, no env)
# ---------------------------------------------------------------------------


def _quat_apply(q: Quat, v: Vec3) -> Vec3:
    """Rotate a vector by a ``(w, x, y, z)`` quaternion, in plain Python."""
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _quat_mul(q1: Quat, q2: Quat) -> Quat:
    """Hamilton product ``q1 * q2``; rotating by the result applies ``q2`` first, then ``q1``."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def compose_carried_pose(root_pos: Vec3, root_quat: Quat, payload_in_root: tuple[Vec3, Quat]) -> tuple[Vec3, Quat]:
    """World-frame pose of a payload frozen in the carrier's root frame (``grasp_poses.py``).

    Plain Python, evaluated at cfg-build time: a leaf's ``__post_init__`` sets ``init_state``s
    before the scene exists, so the grasp is composed from the root pose alone, the way
    ``grasp_poses.py`` documents -- reading forward kinematics at reset instead put the payload
    centimetres off, since joint targets are written but not yet simulated at that point.
    """
    payload_pos, payload_quat = payload_in_root
    pos = tuple(r + o for r, o in zip(root_pos, _quat_apply(root_quat, payload_pos), strict=True))
    quat = _quat_mul(root_quat, payload_quat)
    return pos, quat


# ---------------------------------------------------------------------------
# Grip detection
# ---------------------------------------------------------------------------


def add_grip_contact_sensor(scene: G1ReplaceSceneCfg, target_prim_path: str) -> None:
    """Hand-only contact sensor filtered to ONE target prim, for grip detection.

    Mirrors ``subtask_tiers.place.add_release_contact_sensor``, read the other way: a carrying
    leg's arrival gate needs an unambiguous per-target column, and the scene's shared
    ``hand_contact`` filters more than one prim in the replace preset (both bulbs), so its columns
    cannot be attributed to one of them.
    """
    scene.grip_contact = ContactSensorCfg(
        prim_path=scene.hand_contact.prim_path,
        filter_prim_paths_expr=[target_prim_path],
        history_length=1,
        track_air_time=False,
    )


def payload_held(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float) -> torch.Tensor:
    """True where some hand body still presses on the sensor's ONE filtered target above
    ``force_threshold`` -- the positive counterpart of ``place_terms.object_released``.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.force_matrix_w.sum(dim=2)  # (N, B, M, 3) -> (N, B, 3); M == 1 here
    return force.norm(dim=-1).max(dim=1).values > force_threshold


# ---------------------------------------------------------------------------
# Distance channels (bare ``(env) -> Tensor``, so each can be a ``distance_progress`` distance_fn)
# ---------------------------------------------------------------------------


def base_disposal_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Horizontal distance (m) from the robot's root to the disposal crate's root."""
    robot: Articulation = env.scene["robot"]
    crate: RigidObject = env.scene["bin"]
    return torch.norm((robot.data.root_pos_w - crate.data.root_pos_w)[:, :2], dim=1)


def base_bulb_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Horizontal distance (m) from the robot's root to the fresh bulb's root (on the table)."""
    robot: Articulation = env.scene["robot"]
    bulb: RigidObject = env.scene["bulb"]
    return torch.norm((robot.data.root_pos_w - bulb.data.root_pos_w)[:, :2], dim=1)


# ---------------------------------------------------------------------------
# Arrival predicates
# ---------------------------------------------------------------------------


def arrived_at_bulb(
    env: ManagerBasedRLEnv,
    xy_radius: float,
    facing_tolerance: float,
    max_speed: float,
) -> torch.Tensor:
    """Same shape as ``arrived_at_ladder`` minus the tip conjunct (nothing here to tip over):
    the robot has walked to the fresh bulb and stopped, hands free."""
    near = base_bulb_distance(env) < xy_radius
    facing = base_facing_error(env, SceneEntityCfg("bulb")) < facing_tolerance
    calm = env.scene["robot"].data.root_lin_vel_w.norm(dim=-1) < max_speed
    return near & facing & calm


def arrived_carrying_ladder(
    env: ManagerBasedRLEnv,
    xy_radius: float,
    facing_tolerance: float,
    max_speed: float,
    grip_force_threshold: float,
) -> torch.Tensor:
    """The carried ladder has reached the fixture, robot facing it and standing, still gripped.

    ``near`` reuses ``ladder_ready`` -- the LADDER's position relative to the fixture (also
    upright), not the robot's own footprint, since positioning the ladder is the point of this
    leg. Without the grip conjunct a robot that flings the ladder ahead and merely walks into the
    facing/speed gate empty-handed would score.
    """
    ready = ladder_ready(env, xy_radius, LADDER_TILT_LIMIT)
    facing = base_facing_error(env, SceneEntityCfg("socket")) < facing_tolerance
    calm = env.scene["robot"].data.root_lin_vel_w.norm(dim=-1) < max_speed
    held = payload_held(env, SceneEntityCfg("grip_contact"), grip_force_threshold)
    return ready & facing & calm & held


def arrived_carrying_old_bulb(
    env: ManagerBasedRLEnv,
    xy_radius: float,
    facing_tolerance: float,
    max_speed: float,
    grip_force_threshold: float,
) -> torch.Tensor:
    """Same shape as ``arrived_at_ladder``, pointed at the disposal crate, plus the grip
    conjunct: without it a thrown bulb that skids into the crate's radius would score."""
    near = base_disposal_distance(env) < xy_radius
    facing = base_facing_error(env, SceneEntityCfg("bin")) < facing_tolerance
    calm = env.scene["robot"].data.root_lin_vel_w.norm(dim=-1) < max_speed
    held = payload_held(env, SceneEntityCfg("grip_contact"), grip_force_threshold)
    return near & facing & calm & held


def arrived_carrying_bulb(
    env: ManagerBasedRLEnv,
    xy_radius: float,
    facing_tolerance: float,
    max_speed: float,
    grip_force_threshold: float,
) -> torch.Tensor:
    """Same shape as ``arrived_at_ladder`` (upright-ladder conjunct included), plus the grip
    conjunct for the fresh bulb carried back to it."""
    near = base_ladder_distance(env) < xy_radius
    facing = base_facing_error(env, SceneEntityCfg("ladder")) < facing_tolerance
    calm = env.scene["robot"].data.root_lin_vel_w.norm(dim=-1) < max_speed
    held = payload_held(env, SceneEntityCfg("grip_contact"), grip_force_threshold)
    return near & facing & calm & held & ~ladder_tipped(env, LADDER_TILT_LIMIT)
