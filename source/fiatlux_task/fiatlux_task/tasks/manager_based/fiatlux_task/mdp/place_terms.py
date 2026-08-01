# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Predicates for the place mode (S04 place-the-ladder, S09 dispose-of-the-bulb).

Letting go is the goal in both, so the gate has to distinguish *placed* from *held in the right
place*: the object is where it belongs, it is at rest, and the hand is off it. Every function here
is STATELESS -- the debounce is :class:`~.gates.sustained`, which lives only in the ``success``
termination, so a second counter cannot exist to disagree with it.

Two shapes recur and are written once, parameterized by ``asset_cfg``, because a conjunct that is
spelled twice is a conjunct that can drift: ``object_at_rest`` and ``object_released``.

Release detection reads a contact sensor filtered to ONE target prim (see
``subtask_tiers.place.add_release_contact_sensor``). Two properties of that channel decide the
design: ``force_matrix_w`` is zero unless the filtered target is a rigid body -- against static
geometry "released" would read true forever -- and per-target columns are only unambiguous when the
sensor carries a single filter pattern, which the scene's shared ``hand_contact`` does not.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from fiatlux_task.assets import BULB_LIE_Z_OFFSET, BULB_STAND_Z_OFFSET

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

# The A-frame's step-facing direction IN ITS OWN ROOT FRAME. Local +y, from two independent
# constants in scene_cfg that agree: LADDER_YAW_DEG = 90 is documented as aiming the steps at
# world -x, and R_z(90) maps local +y to world -x; CLIMB_ROBOT_POSITION stands the robot on the
# -x side of LADDER_POSITION to mount it. The ladder's placed yaw comes from the layout draw, so
# every gate below rotates this by the LIVE quaternion rather than assuming a world direction.
LADDER_STEP_FACE_LOCAL = (0.0, 1.0, 0.0)

# The crate's interior footprint, as its measured outer footprint less one wall. Outer extent is
# the crate's real size (0.60 x 0.40 x 0.17 m, scene_cfg); the wall thickness is PROVISIONAL --
# the validation script measures it off the spawned collision meshes and it should be frozen from
# that. Only the interior FOOTPRINT discriminates: a bulb on the rim sits on the wall line.
CRATE_OUTER_FOOTPRINT = (0.60, 0.40)  # m, measured
CRATE_RIM_Z = 0.17  # m, measured: the crate's outer height, i.e. the rim it could balance on
CRATE_WALL_THICKNESS = 0.04  # m, PROVISIONAL
CRATE_INTERIOR_HALF_EXTENT = (
    CRATE_OUTER_FOOTPRINT[0] / 2.0 - CRATE_WALL_THICKNESS,
    CRATE_OUTER_FOOTPRINT[1] / 2.0 - CRATE_WALL_THICKNESS,
)
# Lowest root z a contained bulb can read, over every orientation: the root sits outside the
# geometry, so a bulb standing on its cap on the crate's inner floor puts its root BELOW that
# floor (BULB_STAND_Z_OFFSET), and one balanced on its glass dome lower still. Bounded by the
# bulb's own length rather than by an orientation, which is the point (CRITIQUE A5).
BULB_MAX_ROOT_DROP = BULB_STAND_Z_OFFSET + BULB_LIE_Z_OFFSET


# ---------------------------------------------------------------------------
# Shared conjuncts
# ---------------------------------------------------------------------------


def object_at_rest(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_vel_limit: float,
    ang_vel_limit: float,
) -> torch.Tensor:
    """True where the object's root is translating and rotating below both limits.

    Placed means settled. Without this a gate passes on the pass-through frame of an object that
    is still moving through the target region.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    slow = asset.data.root_lin_vel_w.norm(dim=-1) < lin_vel_limit
    steady = asset.data.root_ang_vel_w.norm(dim=-1) < ang_vel_limit
    return slow & steady


def object_released(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    """True where no hand body pushes on the sensor's filtered target above ``force_threshold``.

    The sensor must filter for exactly ONE prim (the manipulated object) -- summing a multi-target
    filter would let force on some other object mask the release, or the absence of force on it
    fake one.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.force_matrix_w.sum(dim=2)  # (N, B, M, 3) -> (N, B, 3)
    return force.norm(dim=-1).max(dim=1).values < force_threshold


def robot_standing(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True where the robot is neither below ``minimum_height`` nor tilted past ``limit_angle``.

    The positive form of the base's ``fell_below``/``fell_over`` pair, so a success gate can carry
    it as a conjunct. ``mdp.fall_terminated`` cannot serve: it returns a float for the penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    above = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] > minimum_height
    upright = torch.acos(-asset.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)).abs() < limit_angle
    return above & upright


# ---------------------------------------------------------------------------
# S04 -- place the ladder
# ---------------------------------------------------------------------------


def ladder_feet_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Height (m) of the ladder's root above the floor; 0 once it stands on its feet.

    The A-frame's base is authored at z=0, so the root height IS the gap under its feet. Bare
    ``(env) -> Tensor`` so it can be a ``distance_progress`` ``distance_fn``, which forbids
    lambdas and closures.
    """
    ladder: RigidObject = env.scene["ladder"]
    return ladder.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def ladder_feet_down(
    env: ManagerBasedRLEnv,
    tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ladder"),
) -> torch.Tensor:
    """True where the ladder's root sits within ``tolerance`` of the floor, i.e. on its feet."""
    ladder: RigidObject = env.scene[asset_cfg.name]
    z = ladder.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return z.abs() < tolerance


def _step_face_dir_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Horizontal unit vector the ladder's steps face, from its live root quaternion."""
    ladder: RigidObject = env.scene[asset_cfg.name]
    local = torch.tensor(LADDER_STEP_FACE_LOCAL, device=env.device).expand(env.num_envs, 3)
    face = quat_apply(ladder.data.root_quat_w, local)[:, :2]
    return face / face.norm(dim=1, keepdim=True).clamp_min(1e-6)


def robot_at_ladder_base(
    env: ManagerBasedRLEnv,
    forward_limit: float,
    lateral_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ladder"),
) -> torch.Tensor:
    """True where the robot stands in front of the ladder's steps, within mounting range.

    The successor subtask mounts the ladder, and an A-frame's steps face ONE way, so a placement
    that leaves the robot behind or beside it hands the successor a start state it cannot climb
    from (CRITIQUE C2). Tested as a rectangle on the step side -- ``forward`` along the live step
    face, ``lateral`` across it -- rather than a radius, because a radius accepts the back.
    """
    ladder: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene["robot"]
    face = _step_face_dir_w(env, asset_cfg)
    offset = (robot.data.root_pos_w - ladder.data.root_pos_w)[:, :2]
    forward = (offset * face).sum(dim=1)
    lateral = offset[:, 0] * face[:, 1] - offset[:, 1] * face[:, 0]
    return (forward > 0.0) & (forward < forward_limit) & (lateral.abs() < lateral_limit)


# ---------------------------------------------------------------------------
# S09 -- dispose of the old bulb
# ---------------------------------------------------------------------------


def object_in_container(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    interior_half_extent: tuple[float, float],
    interior_floor_z: float,
    rim_z: float,
    max_root_drop: float,
) -> torch.Tensor:
    """True where the object's root lies inside the container's interior, in ANY orientation.

    Containment, not a height equality (CRITIQUE A5): a window around the root height a bulb reads
    while *standing on its cap* excludes the same bulb lying on its glass, and excludes one wedged
    against a wall or still settling -- states that are unambiguously in the crate.

    The interior FOOTPRINT does the discriminating: a bulb balanced on the rim sits on the wall
    line, outside it, and one on the floor beside the crate is further out still. The vertical
    bounds only have to exclude "above the opening" and "under the floor", so they are given as
    the rim height and the deepest the root can sit below the interior floor over all orientations
    -- the root is outside the bulb's own geometry, so that offset is not zero.

    Evaluated in the container's own frame, so a yawed crate is handled.
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    container: RigidObject = env.scene[container_cfg.name]
    rel = quat_apply_inverse(container.data.root_quat_w, obj.data.root_pos_w - container.data.root_pos_w)
    inside_x = rel[:, 0].abs() < interior_half_extent[0]
    inside_y = rel[:, 1].abs() < interior_half_extent[1]
    below_rim = rel[:, 2] < rim_z
    above_floor = rel[:, 2] > interior_floor_z - max_root_drop
    return inside_x & inside_y & below_rim & above_floor


def old_bulb_in_bin(
    env: ManagerBasedRLEnv,
    interior_floor_z: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("old_bulb"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """``object_in_container`` bound to the old bulb and the disposal crate's measured geometry."""
    return object_in_container(
        env,
        asset_cfg=asset_cfg,
        container_cfg=bin_cfg,
        interior_half_extent=CRATE_INTERIOR_HALF_EXTENT,
        interior_floor_z=interior_floor_z,
        rim_z=CRATE_RIM_Z,
        max_root_drop=BULB_MAX_ROOT_DROP,
    )


# ---------------------------------------------------------------------------
# Cfg-time geometry (plain Python; no torch, no env)
# ---------------------------------------------------------------------------


def step_face_dir_from_yaw(yaw_rad: float) -> tuple[float, float]:
    """The world direction the ladder's steps face, for a ladder at ``yaw_rad``.

    The cfg-time twin of :func:`_step_face_dir_w`, for placing a start state against the layout
    draw's sampled ladder yaw.
    """
    fx, fy = LADDER_STEP_FACE_LOCAL[0], LADDER_STEP_FACE_LOCAL[1]
    return (
        fx * math.cos(yaw_rad) - fy * math.sin(yaw_rad),
        fx * math.sin(yaw_rad) + fy * math.cos(yaw_rad),
    )


def yaw_from_quat(quat: tuple[float, float, float, float]) -> float:
    """Yaw (rad) about +z of a ``(w, x, y, z)`` quaternion, at cfg-build time."""
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
