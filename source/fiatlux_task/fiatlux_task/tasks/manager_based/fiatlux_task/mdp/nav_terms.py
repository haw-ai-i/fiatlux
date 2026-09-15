# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Predicates for the navigate mode's carrying legs (S05, S09) and bench approach (S07).

``NavigateSubtaskCfg``/``NavigateRewardsCfg`` (``subtask_env_cfg.py``) already cover a bare walk;
what a CARRYING leg adds on top is two things:

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

S07 (approach the fresh bulb, hands free) needs neither: it is a bare walk, just pointed at the
bulb, so it only needs the missing distance channel.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply_inverse

from fiatlux_task.assets import G1_HORIZONTAL_REACH
from fiatlux_task.grasp_poses import BULB_GLASS_RADIUS_M, BULB_IN_ROOT_STANDING

from ..scene_cfg import (
    CLIMB_ROBOT_POSITION,
    DISPOSAL_ZONE_HALF_SIZE,
    LADDER_POSITION,
    STANCE_RESET_JITTER,
    TABLE_COLLISION_HALF_EXTENT,
    TABLETOP_BULB_POSITION,
    TABLETOP_ROBOT_POSITION,
    G1ReplaceSceneCfg,
)
from .place_terms import CRATE_FOOTPRINT_HALF_EXTENT
from .rewards import base_facing_error, ladder_tipped

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# Provisional constants
# ---------------------------------------------------------------------------

# "Close enough to start climbing" -- a foot-placement question, unlike LADDER_APPROACH_RADIUS
# (arm's reach to a rail, for grasping). PROVISIONAL. A radius rather than a directional offset
# because the replace preset samples the ladder's yaw, so no fixed direction generalizes.
LADDER_MOUNT_RADIUS = math.dist(CLIMB_ROBOT_POSITION[:2], LADDER_POSITION[:2])

# "Close enough to reach for the bulb next." PROVISIONAL: the tabletop preset's own
# manipulation standoff, applied to the replace preset's randomly placed copy of that table.
BULB_APPROACH_RADIUS = math.dist(TABLETOP_ROBOT_POSITION[:2], TABLETOP_BULB_POSITION[:2])

# Which side of the table to stand on: the tabletop preset's authored approach vector, UNSCALED
# (0.626 m). A radius alone can land the robot under the table, hence a direction rather than a
# distance.
#
# This used to be rescaled to ``G1_HORIZONTAL_REACH - 0.05`` (0.4545 m) so the bulb was within
# arm's reach from a standing start. That put the robot's arms inside the bench: MEASURED at the
# staged pose against the table's real collider, 20 links overlap it at seed 0 and 25 at seed 1 --
# both wrists, both hands and nearly every finger -- with a 6.6 mm first-step depenetration jump.
# At 0.626 the same measurement gives 0 overlapping links (seed 0) and 3 shallow finger links
# (seed 1, from the +/-5 cm reset jitter), and the jump falls to 1.2 mm. Issue #104.
#
# The two constraints do not both fit: the crossover is around 0.48-0.50 m, so ANY standoff that
# clears the bench already has the bulb outside ``G1_HORIZONTAL_REACH``. Resolved in favour of
# clearance -- S08 may take a step to close the last ~0.12 m. It is a grasp task, not a walking
# one, but a policy that cannot step to its object cannot do the job either, and a start state
# whose arms begin inside the furniture is not a start state.
_TABLETOP_APPROACH_DX = TABLETOP_ROBOT_POSITION[0] - TABLETOP_BULB_POSITION[0]
_TABLETOP_APPROACH_DY = TABLETOP_ROBOT_POSITION[1] - TABLETOP_BULB_POSITION[1]
BULB_APPROACH_OFFSET = (_TABLETOP_APPROACH_DX, _TABLETOP_APPROACH_DY)

_TABLETOP_APPROACH_DIST = math.hypot(_TABLETOP_APPROACH_DX, _TABLETOP_APPROACH_DY)
_APPROACH_UX = _TABLETOP_APPROACH_DX / _TABLETOP_APPROACH_DIST
_APPROACH_UY = _TABLETOP_APPROACH_DY / _TABLETOP_APPROACH_DIST
CARRY_SWEEP_RADIUS = math.hypot(*BULB_IN_ROOT_STANDING[0][:2]) + BULB_GLASS_RADIUS_M
_CARRY_CLEARANCE = CARRY_SWEEP_RADIUS + STANCE_RESET_JITTER

_EXIT_AXIS = (
    0 if TABLE_COLLISION_HALF_EXTENT[0] / abs(_APPROACH_UX) < TABLE_COLLISION_HALF_EXTENT[1] / abs(_APPROACH_UY) else 1
)
_EXIT_U = (_APPROACH_UX, _APPROACH_UY)[_EXIT_AXIS]
CARRY_CLEARANCE_STANDOFF = (TABLE_COLLISION_HALF_EXTENT[_EXIT_AXIS] + _CARRY_CLEARANCE) / abs(_EXIT_U)
TABLE_CARRY_OFFSET = (_APPROACH_UX * CARRY_CLEARANCE_STANDOFF, _APPROACH_UY * CARRY_CLEARANCE_STANDOFF)

_OTHER_AXIS = 1 - _EXIT_AXIS
if abs((_APPROACH_UX, _APPROACH_UY)[_OTHER_AXIS] * CARRY_CLEARANCE_STANDOFF) > TABLE_COLLISION_HALF_EXTENT[_OTHER_AXIS]:
    raise ValueError(
        "The carry standoff clears a corner of the bench, not a face, so the clearance it computes "
        "is optimistic. Solve the box distance directly instead of assuming a face."
    )

# "Close enough to be standing at the crate", measured from the crate's FOOTPRINT rather than
# its origin (issue #149). Horizontal arm reach is the condition that matters: within it the
# robot can put a hand over the crate, which is what arrival is for.
DISPOSAL_ARRIVAL_CLEARANCE = G1_HORIZONTAL_REACH

# Where staging PUTS the robot to WORK at the crate, as a radius from its origin -- a different
# quantity from the gate above, which is a clearance from its footprint. PROVISIONAL: the crate's
# own layout zone half-extent; no authored robot-relative standoff exists. Safe only because the
# legs that use it also FACE the crate, so SONIC's settle backstep carries them away from it.
DISPOSAL_STANCE_RADIUS = DISPOSAL_ZONE_HALF_SIZE

# Where staging puts a robot that is DONE with the crate and turned away from it (S07). That turn
# is what makes the work radius unsafe here: facing the bench, the crate lands at any bearing,
# including straight behind, and the settle backstep then walks the robot into it. Measured over
# the 11 recorded S07 takes -- crate bearings from -134 to +147 deg, and the two furthest behind
# both ended with the base INSIDE the crate footprint, which is what the first hard walk command
# trips over (#205). So: clear of the footprint whichever way the robot happens to be turned.
# 0.45 m from the nearest wall at spawn. The settle then pulls the base back TOWARD the crate,
# so what survives is less: measured over 10 headless S07 spawns, 0.18-0.43 m once settled
# (main: 0.01-0.17 m, and inside the footprint on 1 of 5 seeds).
DISPOSAL_DEPARTURE_CLEARANCE = 0.45  # m
DISPOSAL_DEPARTURE_RADIUS = max(CRATE_FOOTPRINT_HALF_EXTENT) + DISPOSAL_DEPARTURE_CLEARANCE

# Grip-presence floor for the carrying legs' arrival gate: subtask_tiers.place's
# RELEASE_FORCE_THRESHOLD_N read the other way. PROVISIONAL.
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


def settle_carried_payload_live(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    payload_cfg: SceneEntityCfg,
    hand_variant: str = "inspire",
    reset_mode: bool = False,
) -> None:
    """Seat a carried payload directly ON the hand's LIVE, actually-simulated palm surface.

    Bulb lying across the palm (``ARM_CRADLE``'s wrist roll turns the palm up), glass over the
    finger bases, for the hand to close on. Two things this seat is NOT, both measured
    (issue #105):

    * Not a pinch seated relative to the palm body's origin: that origin is several cm off the
      visible mesh, and the finger-base centroid used instead sits on the knuckle axes INSIDE
      the hand -- the bulb started 1-2 cm deep in the collider on both hands, the solver threw
      it out at 3-6 m/s, and the 65-2366 N "grip" readings were that push. The seat now lifts
      the centroid to the measured surface (``G1_PALM_SURFACE_OFFSET_M``).
    * Not a rest on an open, uncurled palm: in the staged carry pose the palm is ~20 deg off
      level and a free bulb rolls off it within 0.5 s even with the robot pinned in place. What
      keeps it in the hand is fingers closing on it AFTER it is seated (teleop closes the grip
      from open at rest and holds at ~55 N on Dex3). Fingers already curled when the seat
      fires land inside or outside the bulb instead of around it, and ``poses.HAND_CUP``'s
      0.5 rad does not retain a surface-seated bulb under a held pose -- how the RL legs stage
      the hold is an open benchmark decision, not something this seat can settle.

    Wired as an ``interval`` event (``interval_range_s=(0.0, 0.0)``, fires every step) gated on
    ``episode_length_buf == 1``, not a ``reset``-mode event: reset-mode events fire before any
    physics/kinematics update runs this cycle, so ``body_pos_w`` there can still hold the
    previous episode's poses (the same reason ``compose_carried_pose`` cannot read it either).
    Not ``== 0`` either -- ``ManagerBasedRLEnv.step()`` increments ``episode_length_buf`` BEFORE
    dispatching interval events, on every step including the first one after a reset, so an
    interval event never observes 0; 1 is the value it actually sees on that first pass, once
    the arm has actually been simulated into its target pose (the reset's own
    ``scene.write_data_to_sim()`` + ``sim.forward()``, followed by this step's physics).
    """
    if reset_mode:
        ids = env_ids
    else:
        fresh = env.episode_length_buf[env_ids] == 1
        if not bool(fresh.any()):
            return
        ids = env_ids[fresh]

    from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

    from fiatlux_task.assets import BULB_LIE_Z_OFFSET
    from fiatlux_task.grasp_poses import BULB_GLASS_CENTRE_M
    from fiatlux_task.robots.g1 import (
        G1_FINGER_BASE_BODIES_BY_VARIANT,
        G1_PALM_BODY_BY_VARIANT,
        G1_PALM_CAP_SIDE,
        G1_PALM_LOCAL_AXES,
        G1_PALM_SURFACE_OFFSET_M,
    )

    robot: Articulation = env.scene["robot"]
    payload: RigidObject = env.scene[payload_cfg.name]
    if reset_mode:
        # Reset-mode events run before the cycle's kinematics update, so body_pos_w still holds
        # the previous episode's poses. Force the update here instead of waiting a step, so the
        # payload is seated before any physics runs on the overlap (issue #197).
        robot.write_data_to_sim()
        env.sim.forward()
        robot.update(0.0)
    palm_idx = robot.find_bodies(G1_PALM_BODY_BY_VARIANT[hand_variant])[0][0]
    palm_quat = robot.data.body_quat_w[ids, palm_idx]
    rot = matrix_from_quat(palm_quat)  # (n, 3, 3), columns are the palm's local axes in world

    # Anchor POSITION on the finger-base centroid, not the palm body's own origin -- the latter
    # renders several cm off the visible mesh, toward the wrist, so a payload placed relative to
    # it hangs over empty air with nothing underneath and falls/rolls off within a few steps.
    # The centroid itself lies on the knuckle axes inside the hand, so lift it to the measured
    # surface (``G1_PALM_SURFACE_OFFSET_M``) before stacking the bulb's own radius on top.
    base_idx = [robot.find_bodies(n)[0][0] for n in G1_FINGER_BASE_BODIES_BY_VARIANT[hand_variant]]
    palm_pos = robot.data.body_pos_w[ids][:, base_idx].mean(dim=1)

    (n_idx, n_sign), _, (a_idx, a_sign) = G1_PALM_LOCAL_AXES[hand_variant]
    normal = rot[:, :, n_idx] * n_sign
    across = rot[:, :, a_idx] * a_sign

    # The bulb's ROOT sits at the tip end (outside its own geometry, per grasp_poses.py), with
    # the entire body extending along its local +z from there -- placing the root itself at the
    # anchor left the cap balanced on the fingers and the whole glass body (the bigger, heavier
    # part) cantilevered off past the edge of the hand, unsupported. Local +z lies along the
    # palm's "across" axis, pointing from the cap side (G1_PALM_CAP_SIDE, chosen per hand so the
    # neck runs away from the thumb) toward the glass; shift the root back along -axis by
    # BULB_GLASS_CENTRE_M (root-to-glass-centre) to land the glass's centre, not the root, over
    # the palm.
    axis = -across * G1_PALM_CAP_SIDE[hand_variant]  # bulb +z runs root->glass, so the cap sits at +side
    lift = G1_PALM_SURFACE_OFFSET_M[hand_variant] + BULB_LIE_Z_OFFSET
    pos = palm_pos + normal * lift - axis * BULB_GLASS_CENTRE_M
    # Lying on its side, long axis across the palm (spans the palm, the low/stable resting
    # orientation) -- same basis convention as the pinch version, just a different seat.
    basis = torch.stack([normal, torch.linalg.cross(axis, normal), axis], dim=-1)
    quat = quat_from_matrix(basis)

    payload.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
    payload.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device), env_ids=ids)


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
    # Left mirror, so a payload carried in the left hand reads as carried (issue #151).
    scene.grip_contact_left = ContactSensorCfg(
        prim_path=scene.left_hand_contact.prim_path,
        filter_prim_paths_expr=[target_prim_path],
        history_length=1,
        track_air_time=False,
    )


def _peak_filtered_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Largest per-body force magnitude on the sensor's ONE filtered target."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.force_matrix_w.sum(dim=2)  # (N, B, M, 3) -> (N, B, 3); M == 1 here
    return force.norm(dim=-1).max(dim=1).values


def payload_held(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    other_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """True where some hand body still presses on the sensor's ONE filtered target above
    ``force_threshold`` -- the positive counterpart of ``place_terms.object_released``.

    With ``other_sensor_cfg``, carrying it in EITHER hand counts (issue #151).
    """
    peak = _peak_filtered_force(env, sensor_cfg)
    if other_sensor_cfg is not None:
        peak = torch.maximum(peak, _peak_filtered_force(env, other_sensor_cfg))
    return peak > force_threshold


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
    bulb: RigidObject = env.scene["fresh_bulb"]
    return torch.norm((robot.data.root_pos_w - bulb.data.root_pos_w)[:, :2], dim=1)


# ---------------------------------------------------------------------------
# Arrival conjuncts
# ---------------------------------------------------------------------------
# One condition each, so a leaf's gate is an ``mdp.all_of`` list rather than a function that
# ``&``s them together in its own body. A list can be read from outside: tests assert a conjunct
# is present, and ``mdp.gates.gate_progress`` counts how many an episode satisfied.


def base_near(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    xy_radius: float,
    half_extent: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """True where the robot's root is horizontally within ``xy_radius`` of the entity's FOOTPRINT.

    ``half_extent`` is that footprint in the entity's own frame; the default of zero measures to
    its origin, which is right for something small and wrong for a container. Measured to the
    origin, the 0.60 x 0.40 m disposal crate scores the approach side rather than being at it: the
    same 0.24 m gap from the crate reads 0.54 m off the short end and 0.44 m off the long face
    (issue #149).

    Evaluated in the entity's own frame, so a yawed crate is handled.
    """
    robot: Articulation = env.scene["robot"]
    target: RigidObject = env.scene[asset_cfg.name]
    rel = quat_apply_inverse(target.data.root_quat_w, robot.data.root_pos_w - target.data.root_pos_w)
    half = torch.tensor(half_extent, device=env.device)
    return (rel[:, :2].abs() - half).clamp(min=0.0).norm(dim=1) < xy_radius


def base_facing(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, facing_tolerance: float) -> torch.Tensor:
    """True where the robot's heading is within ``facing_tolerance`` of the bearing to the entity."""
    return base_facing_error(env, asset_cfg) < facing_tolerance


def base_calm(env: ManagerBasedRLEnv, max_speed: float) -> torch.Tensor:
    """True where the robot's root speed is under ``max_speed`` -- rejects scoring while still
    charging at the target."""
    return env.scene["robot"].data.root_lin_vel_w.norm(dim=-1) < max_speed


def ladder_upright(env: ManagerBasedRLEnv, tilt_limit: float) -> torch.Tensor:
    """True where the ladder has NOT tipped past ``tilt_limit``."""
    return ~ladder_tipped(env, tilt_limit)
