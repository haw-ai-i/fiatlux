# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Predicates for the place mode (S01 move-the-ladder, S06 dispose-of-the-bulb).

Leaving the object behind is the goal in both, so the gate has to distinguish *placed* from
*passing through*: the object is where it belongs and it is at rest, plus -- where letting go is
part of the claim, as it is for a bulb in a crate -- the hand is off it. Every function here
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
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude

from fiatlux_task.assets import BULB_LIE_Z_OFFSET, BULB_STAND_Z_OFFSET

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

# The ladder's step-facing direction in its own root frame: the flat rungs are on local -y, the
# back brace on local +y. The ladder's placed yaw comes from the layout draw, so every gate below
# rotates this by the live quaternion rather than assuming a world direction.
LADDER_STEP_FACE_LOCAL = (0.0, -1.0, 0.0)

CRATE_RIM_Z = 0.17  # m
# Interior half-extent + floor, measured by ray-casting the crate collision mesh (#131);
# re-measure if the crate USD changes.
CRATE_INTERIOR_HALF_EXTENT = (0.2873, 0.1876)  # m; #131
CRATE_INTERIOR_FLOOR_Z = 0.0074  # m; #131
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
    ang_vel_limit: float | None = None,
) -> torch.Tensor:
    """True where the object's root is translating below ``lin_vel_limit``, and rotating below
    ``ang_vel_limit`` where one is given.

    Placed means settled. Without this a gate passes on the pass-through frame of an object that
    is still moving through the target region.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    at_rest = asset.data.root_lin_vel_w.norm(dim=-1) < lin_vel_limit
    if ang_vel_limit is not None:
        at_rest = at_rest & (asset.data.root_ang_vel_w.norm(dim=-1) < ang_vel_limit)
    return at_rest


# Time constant (s) of the pose-derived speed estimate object_settled uses. 0.1 s = five control
# steps: long enough to average the solver's substep contact bounce out (see below), short enough
# that a bulb that starts moving again reads as moving well inside the gates' 1 s sustain window.
SETTLED_SPEED_TAU_S = 0.1
_SETTLED_STATE_ATTR = "_fiatlux_object_settled_state"


def object_settled(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_vel_limit: float,
    ang_vel_limit: float | None = None,
) -> torch.Tensor:
    """``object_at_rest`` for an object HELD IN CONTACT by a force: speeds from the pose over time.

    Same limits and meaning as :func:`object_at_rest`, but the speeds are estimated from the
    object's pose (finite difference between control steps, low-passed with
    ``SETTLED_SPEED_TAU_S``) instead of read from ``root_lin_vel_w`` / ``root_ang_vel_w``. For a
    free object the two agree (a bulb resting in a crate reads 0.02 rad/s either way). For a body
    pressed into stiff contact by an external wrench they do not: PhysX pushes it back out of the
    contact every substep (200 Hz) while the wrench pushes it in, and the 50 Hz sample of that
    bounce reads as a steady 0.05-0.1 m/s / 0.5-2 rad/s on a body whose pose is changing by
    microns (S11 insert teleop bags, issue #171: reported |w| 0.5-2.0 rad/s against 0.000 rad/s
    from the recorded poses, tilt std 0.05 deg). Used by S11's gate, where the seated bulb is
    exactly that body; S01/S06 keep ``object_at_rest`` (their objects rest free, and the
    instantaneous read has no lag).

    State lives on the env keyed by asset name, indexed by ``common_step_counter`` so that being
    evaluated more than once per step (terminations + the teleop recorder) does not read a zero
    difference on the second call. An env that just reset restarts its estimate from zero -- both
    at reset time itself (``episode_length_buf == 0``, e.g. ``gate_progress.reset()`` reading the
    new episode's baseline before ``common_step_counter`` has advanced) and on the new episode's
    first regular evaluation (``episode_length_buf == 1``, discarding the one teleport-sized diff
    that pass would otherwise compute against the dying episode's cached pose).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w
    quat = asset.data.root_quat_w
    store = getattr(env, _SETTLED_STATE_ATTR, None)
    if store is None:
        store = {}
        setattr(env, _SETTLED_STATE_ATTR, store)
    step = int(env.common_step_counter)
    st = store.get(asset_cfg.name)
    if st is None or st["step"] > step:
        st = {
            "step": step,
            "pos": pos.clone(),
            "quat": quat.clone(),
            "lin": torch.zeros(pos.shape[0], device=pos.device),
            "ang": torch.zeros(pos.shape[0], device=pos.device),
        }
        store[asset_cfg.name] = st
    # An env resets mid-step: its termination is evaluated (caching st at this step's counter),
    # then its pose is teleported to the new episode's start and gate_progress.reset() calls this
    # same term again to set the new episode's baseline -- still at the same common_step_counter,
    # so the dt block below is skipped and would otherwise hand back the dying episode's stale
    # cached speed. Snap those envs' baseline to the just-written pose directly, independent of
    # the step dedup; the next regular call, one step later, computes real motion from there.
    just_reset = env.episode_length_buf == 0
    if bool(just_reset.any()):
        st["pos"][just_reset] = pos[just_reset]
        st["quat"][just_reset] = quat[just_reset]
        st["lin"][just_reset] = 0.0
        st["ang"][just_reset] = 0.0
    if step != st["step"]:
        dt = env.step_dt * (step - st["step"])
        lin_fd = torch.norm(pos - st["pos"], dim=-1) / dt
        alpha = min(1.0, dt / SETTLED_SPEED_TAU_S)
        st["lin"] = st["lin"] + alpha * (lin_fd - st["lin"])
        if ang_vel_limit is not None:
            ang_fd = quat_error_magnitude(quat, st["quat"]) / dt
            st["ang"] = st["ang"] + alpha * (ang_fd - st["ang"])
        # just reset: the pose jump is a teleport, not motion. Not == 0 -- ManagerBasedRLEnv.step()
        # increments episode_length_buf before termination/reward terms run, on every step
        # including the first one after a reset, so this term never observes 0; 1 is the value
        # it actually sees on that first pass (see nav_terms.py's identical note).
        fresh = env.episode_length_buf == 1
        st["lin"][fresh] = 0.0
        st["ang"][fresh] = 0.0
        st["pos"].copy_(pos)
        st["quat"].copy_(quat)
        st["step"] = step
    settled = st["lin"] < lin_vel_limit
    if ang_vel_limit is not None:
        settled = settled & (st["ang"] < ang_vel_limit)
    return settled


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
# S01 -- move the ladder to the fixture
# ---------------------------------------------------------------------------


def ladder_feet_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Height (m) of the ladder's root above the floor; 0 once it stands on its feet.

    The ladder's base is authored at z=0, so the root height IS the gap under its feet. Bare
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


# ---------------------------------------------------------------------------
# S06 -- dispose of the old bulb
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
    interior_floor_z: float = CRATE_INTERIOR_FLOOR_Z,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("old_bulb"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """``object_in_container`` bound to a bulb and the disposal crate."""
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

    Cfg-time, from a frozen quaternion rather than a live one, for placing a start state against
    the layout draw's sampled ladder yaw.
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


def lean_stance_against_ladder(
    ladder_pos: tuple[float, float, float],
    ladder_rot: tuple[float, float, float, float],
    standoff: float,
    pelvis_z: float,
    lean_deg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """A pelvis pose ``standoff`` metres off the ladder's step-facing side, leaning into it.

    Which side of the ladder the steps are on depends on its yaw, so a stance written as world
    coordinates is only correct for the yaw it was authored at -- and silently faces the brace
    side at any other. Returns ``(pos, rot)`` for the ladder's own pose.
    """
    fx, fy = step_face_dir_from_yaw(yaw_from_quat(ladder_rot))
    pos = (ladder_pos[0] + fx * standoff, ladder_pos[1] + fy * standoff, pelvis_z)
    # Face back at the ladder, then pitch into it: q_z(facing) * q_y(lean).
    facing, lean = math.atan2(-fy, -fx), math.radians(lean_deg)
    cz, sz = math.cos(facing * 0.5), math.sin(facing * 0.5)
    cy, sy = math.cos(lean * 0.5), math.sin(lean * 0.5)
    return pos, (cz * cy, -sz * sy, cz * sy, sz * cy)
