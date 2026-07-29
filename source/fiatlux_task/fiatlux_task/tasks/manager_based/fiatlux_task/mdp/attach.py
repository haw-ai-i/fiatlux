# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bulb attach/detach mechanic (unification spec Phase 4, issue #54).

The Replace task needs the old bulb to start "screwed into" the fixture yet be removable,
and the fresh bulb to become "screwed in" once seated -- neither of which plain rigid-body
physics provides. BEHAVIOR-1K solves this with its ``AttachedTo`` state engine (aligned
metalinks -> create a fixed joint; PhysX break event -> remove it), but that mechanism is
runtime USD stage editing and does not survive contact with Isaac Lab's GPU-vectorized
pipeline: per-env joint creation/destruction, breakable-joint events, and runtime
``kinematic_enabled`` toggles are all unsupported or broken there (see
``journal/specs/issue-54-bulb-attach-detach.md`` for the full platform survey).

What transfers from BEHAVIOR-1K is the *state machine*, not the joint plumbing. This
module implements it with per-env boolean attachment state and **pose slaving** (the
spec's Approach B): while a bulb is ATTACHED, every env step writes its seated pose and
zero velocity through the tensorized ``RigidObject`` root-state API -- the same documented
calls episode resets use, valid per-env on GPU. An attached bulb therefore behaves like
the old kinematic stand-in (immovable, collidable), but attachment is now *state we can
flip per env*, which is what makes removal and installation achievable at all. The
physical D6-joint backend (Approach A) stays a follow-up pending its on-GPU spike.

Gates (thresholds seeded from BEHAVIOR-1K, screw gate per the unification spec -- B1K
itself has no screwing; aligned bulbs snap instantly):

- **Detach (unscrew)**: palm within ``grasp_radius`` of the old bulb AND the wrist-roll
  joint has accumulated ``screw_angle`` rad of *ratcheted* motion in the unscrew
  direction (only strokes in that direction count, so back-and-forth re-grip strokes do
  not cancel; the return stroke is free, which is physically generous but strictly harder
  than B1K's zero-screw model).
- **Attach (screw-in)**: fresh bulb within the seating tolerances (the same pos/ori
  thresholds as ``bulb_seated``) AND palm within ``grasp_radius`` AND ``screw_angle`` rad
  of ratcheted wrist roll accumulated *while aligned*, AND the old bulb already detached
  (two bulbs cannot occupy one socket).

The event term must run every env step (it both integrates the wrist-roll signal and
holds attached bulbs seated), so wire it with ``mode="interval"`` and
``interval_range_s=(0.0, 0.0)``.

One ordering caveat shapes the score-channel functions below: ``ManagerBasedRLEnv.step``
computes terminations and rewards *before* it applies interval events, so those managers
see the bulb wherever that step's physics left it -- the corrective seat-pose write lands
afterwards. A held bulb is dynamic between writes, so a hard shove can transiently carry
it past the 0.10 m removal clearance within one step, and the best-progress /
paid-once latches in ``distance_progress`` / ``completion_bonus`` would make that
transient permanent. The ``old_bulb_*`` channel functions here therefore report the bulb
*at the seat pose* while it is attached (being held IS the task state; the displacement
is solver noise), and the raw-geometry terms in ``rewards.py`` stay for the tasks without
an attachment manager.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply

from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_OFFSET

from .rewards import (
    _bulb_socket_axis_error,
    _bulb_socket_pos_error,
    old_bulb_disposal_distance,
    old_bulb_disposed,
    old_bulb_dropped,
    old_bulb_fixture_clearance,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg

# The manager registers itself on the env so the predicate terms below can reach its
# state without knowing the event-term name it was wired under.
_ENV_ATTR = "_fiatlux_bulb_attachment"


def _seated_bulb_root_pose_w(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Root pose a bulb has when seated in the fixture, from the *live* socket pose."""
    socket: RigidObject = env.scene["socket"]
    quat = socket.data.root_quat_w
    seat = torch.tensor(SOCKET_SEAT_OFFSET, device=env.device).expand(env.num_envs, 3)
    plug = torch.tensor(BULB_PLUG_OFFSET, device=env.device).expand(env.num_envs, 3)
    pos = socket.data.root_pos_w + quat_apply(quat, seat) - quat_apply(quat, plug)
    return pos, quat


class bulb_attachment(ManagerTermBase):
    """Every-step event term: the bulb attach/detach state machine + seated-pose slaving.

    Owns per-env state: ``old_attached`` (True at reset -- the old bulb starts screwed
    in), ``fresh_attached`` (False at reset), and the two ratcheted wrist-roll
    accumulators. Each step it advances the gates, then writes the seated pose (derived
    from the *current* socket pose, so a re-posed fixture keeps its bulb) and zero
    velocity for every attached bulb.

    Operates on all envs regardless of the ``env_ids`` the event manager passes: the
    accumulators are only correct when integrated every step, which the zero interval
    guarantees -- a nonzero ``interval_range_s`` would silently undercount.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        setattr(env, _ENV_ATTR, self)
        robot: Articulation = env.scene["robot"]
        # Inspire names the grasping body right_hand_base_link, Dex3 right_hand_palm_link,
        # and swap_robot_variant does not rewrite palm_body -- so resolve by trying the
        # known names rather than indexing an empty match (issue #54 review).
        wanted = cfg.params.get("palm_body")
        candidates = [wanted] if wanted else ["right_hand_base_link", "right_hand_palm_link"]
        palm_ids = []
        for name in candidates:
            palm_ids, _ = robot.find_bodies(name)
            if palm_ids:
                break
        if not palm_ids:
            raise ValueError(
                f"bulb_attachment: no grasping body matched {candidates} on this robot; "
                f"pass palm_body= explicitly. Available: {robot.body_names}"
            )
        wrist_ids, _ = robot.find_joints(cfg.params.get("wrist_joint", "right_wrist_roll_joint"))
        if not wrist_ids:
            raise ValueError("bulb_attachment: no wrist-roll joint matched; pass wrist_joint=")
        self._palm_id = palm_ids[0]
        self._wrist_id = wrist_ids[0]
        n, dev = env.num_envs, env.device
        self._old_attached = torch.ones(n, dtype=torch.bool, device=dev)
        self._fresh_attached = torch.zeros(n, dtype=torch.bool, device=dev)
        self._unscrew_accum = torch.zeros(n, device=dev)
        self._screw_accum = torch.zeros(n, device=dev)
        # sim is initialized before managers, so joint state is readable here
        self._prev_roll = robot.data.joint_pos[:, self._wrist_id].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._old_attached[ids] = True
        self._fresh_attached[ids] = False
        self._unscrew_accum[ids] = 0.0
        self._screw_accum[ids] = 0.0
        # reset events run before manager resets, so this reads the post-randomization pose
        robot: Articulation = self._env.scene["robot"]
        self._prev_roll[ids] = robot.data.joint_pos[ids, self._wrist_id]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        palm_body: str | None = None,  # None -> auto-detect Inspire/Dex3 naming
        wrist_joint: str = "right_wrist_roll_joint",
        grasp_radius: float = 0.12,
        screw_angle: float = math.pi,
        unscrew_sign: float = -1.0,
        pos_threshold: float = 0.015,
        ori_threshold: float = 0.2,
    ) -> None:
        # palm_body / wrist_joint are consumed once in __init__ (index resolution)
        robot: Articulation = env.scene["robot"]
        old_bulb: RigidObject = env.scene["old_bulb"]
        fresh_bulb: RigidObject = env.scene["bulb"]

        roll = robot.data.joint_pos[:, self._wrist_id]
        delta = roll - self._prev_roll
        self._prev_roll = roll.clone()
        palm_pos = robot.data.body_link_pos_w[:, self._palm_id]

        # -- detach gate (old bulb): ratcheted unscrew roll while the palm grips it
        old_near = torch.norm(palm_pos - old_bulb.data.root_pos_w, dim=1) < grasp_radius
        active = self._old_attached & old_near
        self._unscrew_accum += torch.where(
            active, (unscrew_sign * delta).clamp(min=0.0), torch.zeros_like(delta)
        )
        self._old_attached &= self._unscrew_accum < screw_angle

        # -- attach gate (fresh bulb): ratcheted screw-in roll while seated + gripped,
        #    only into an empty socket
        # Axis alignment, NOT full-frame: rotation about the mating axis is the screw
        # gesture itself, so a full-frame check would fight the gate it gates.
        aligned = (_bulb_socket_pos_error(env) < pos_threshold) & (
            _bulb_socket_axis_error(env) < ori_threshold
        )
        fresh_near = torch.norm(palm_pos - fresh_bulb.data.root_pos_w, dim=1) < grasp_radius
        active = ~self._fresh_attached & ~self._old_attached & aligned & fresh_near
        self._screw_accum += torch.where(
            active, (-unscrew_sign * delta).clamp(min=0.0), torch.zeros_like(delta)
        )
        self._fresh_attached |= (
            ~self._old_attached & aligned & (self._screw_accum >= screw_angle)
        )

        # -- hold attached bulbs seated (pose derived from the live socket pose; the
        #    write also snaps a just-attached fresh bulb from within-tolerance to exact,
        #    mirroring B1K's teleport-on-attach)
        pos, quat = _seated_bulb_root_pose_w(env)
        for bulb, attached in (
            (old_bulb, self._old_attached),
            (fresh_bulb, self._fresh_attached),
        ):
            ids = attached.nonzero(as_tuple=False).squeeze(-1)
            if ids.numel() > 0:
                bulb.write_root_pose_to_sim(
                    torch.cat([pos[ids], quat[ids]], dim=-1), env_ids=ids
                )
                bulb.write_root_velocity_to_sim(
                    torch.zeros(ids.numel(), 6, device=env.device), env_ids=ids
                )


def _attachment(env: ManagerBasedRLEnv) -> bulb_attachment:
    mgr = getattr(env, _ENV_ATTR, None)
    if mgr is None:
        raise RuntimeError(
            "no bulb_attachment event term is configured on this env; the attachment "
            "predicates only work on tasks that wire mdp.bulb_attachment into their EventCfg"
        )
    return mgr


def old_bulb_attached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True where the old bulb is still screwed into the fixture."""
    return _attachment(env)._old_attached


def fresh_bulb_attached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True where the fresh bulb has been screwed into the fixture.

    The attach-aware replacement for the geometric ``bulb_seated``: requires the full
    screw-in gate, not just transiting the seating tolerances -- a bulb waved through
    the success zone (or knocked back out of it) no longer scores.
    """
    return _attachment(env)._fresh_attached


def old_bulb_release_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Old-bulb fixture clearance (m) that reads 0 while the bulb is still screwed in.

    Attach-aware ``old_bulb_fixture_clearance``: terminations/rewards run before the
    interval event re-seats a held bulb (see module docstring), so the raw clearance can
    transiently exceed the removal threshold while attached and latch a false
    removal-progress/removed payout. Held means seated, by definition.
    """
    clearance = old_bulb_fixture_clearance(env)
    return torch.where(old_bulb_attached(env), torch.zeros_like(clearance), clearance)


def old_bulb_removed_after_release(
    env: ManagerBasedRLEnv, clearance_threshold: float
) -> torch.Tensor:
    """Attach-aware ``old_bulb_removed``: only a *released* bulb can count as removed."""
    return old_bulb_release_clearance(env) > clearance_threshold


def old_bulb_disposal_distance_pinned(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Old-bulb -> disposal-crate distance (m), pinned to the seat pose while attached.

    Attach-aware ``old_bulb_disposal_distance``: a transient shove of a held bulb toward
    the crate must not pay disposal progress (the best-progress latch would keep it).
    """
    d = old_bulb_disposal_distance(env)
    seat_pos, _ = _seated_bulb_root_pose_w(env)
    crate: RigidObject = env.scene["bin"]
    d_held = torch.norm(seat_pos - crate.data.root_pos_w, dim=1)
    return torch.where(old_bulb_attached(env), d_held, d)


def old_bulb_dropped_after_release(
    env: ManagerBasedRLEnv, min_height: float, disposal_threshold: float
) -> torch.Tensor:
    """Attach-aware ``old_bulb_dropped``: a bulb still screwed in cannot be "dropped".

    Guards the drop penalty/termination against transient displacement of the held bulb
    (relevant for low wall mounts) before the interval event re-seats it.
    """
    return old_bulb_dropped(env, min_height, disposal_threshold) & ~old_bulb_attached(env)


def attached_replacement_success(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.015,
    ori_threshold: float = 0.2,
    disposal_threshold: float = 0.25,
) -> torch.Tensor:
    """True where the fresh bulb is screwed in AND the old bulb is in the disposal crate.

    The attach-aware ``full_replacement_success`` (also the ``success`` termination).
    ``pos_threshold`` / ``ori_threshold`` are accepted for the meta.json recording
    contract (``recording.py`` reads them off the ``success`` term); the attach gate is
    what enforces them, at attach time rather than here.
    """
    return fresh_bulb_attached(env) & old_bulb_disposed(env, disposal_threshold)
