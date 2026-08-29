# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bayonet bulb/socket constraint (issue #54; spec: ``journal/specs/issue-54-bulb-attach-detach.md``).

The socket is a bayonet mount driven entirely by the **bulb pose** relative to the
socket -- never by robot state. Each bulb is in one of three phases per env:

- ``FREE`` -- unconstrained rigid body; physics owns it entirely.
- ``AXIAL`` -- the insertion channel: only travel along the socket axis survives;
  lateral offset and all relative rotation are projected away every step.
- ``ROTATING`` -- the lock groove at full depth: position is pinned at the seat and only
  twist about the axis survives, tracked as ``theta`` in ``[0, rotation_angle]``.

Install is insert-then-rotate, removal is rotate-then-eject, and the order is
structural: the two motion regimes are mutually exclusive, so no sequence of pushes
frees a locked bulb and no fresh bulb counts as installed until it bottomed out and
turned through the lock angle. The old bulb resets locked (``ROTATING`` at
``rotation_angle``); the fresh bulb resets ``FREE``.

``insertion_depth`` and ``rotation_angle`` accept a scalar or a ``(low, high)`` range;
ranges are re-sampled independently per env at each reset (the domain-randomization
hook). Enforcement is per-step pose/velocity projection through the tensorized
``RigidObject`` root-state API -- no USD edits, no joints, valid per-env on GPU. Wire
the event term with ``mode="interval"`` and ``interval_range_s=(0.0, 0.0)`` so it runs
every step.

One ordering caveat shapes the score-channel functions below: ``ManagerBasedRLEnv.step``
computes terminations and rewards *before* interval events, so those managers see the
bulb wherever physics left it -- the corrective projection lands afterwards. A hard
mid-step shove of a constrained bulb could therefore transiently satisfy a distance
threshold and be made permanent by the best-progress / paid-once latches in
``distance_progress`` / ``completion_bonus``. The ``old_bulb_*`` functions here report
the bulb at the seat pose until it actually leaves the channel (being constrained IS the
task state; the displacement is solver noise). The raw-geometry terms in ``rewards.py``
remain for tasks without an attachment manager.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET

from .rewards import (
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

_FREE = 0
_AXIAL = 1
_ROTATING = 2
_OLD = 0  # state row of the old bulb (scene entity "old_bulb")
_FRESH = 1  # state row of the fresh bulb (scene entity "fresh_bulb")
_EPS = 1e-5

ParameterSpec = float | tuple[float, float]


def _sample_parameter(target: torch.Tensor, env_ids, spec: ParameterSpec, name: str) -> None:
    """Fill selected environments from a scalar or uniform randomization range."""
    if isinstance(spec, tuple):
        if len(spec) != 2 or spec[0] <= 0.0 or spec[1] < spec[0]:
            raise ValueError(f"{name} range must satisfy 0 < low <= high, got {spec}")
        target[env_ids] = torch.empty_like(target[env_ids]).uniform_(spec[0], spec[1])
    else:
        if spec <= 0.0:
            raise ValueError(f"{name} must be positive, got {spec}")
        target[env_ids] = spec


def _axis_angle_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    return torch.cat([torch.cos(half).unsqueeze(-1), axis * torch.sin(half).unsqueeze(-1)], dim=-1)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _signed_twist(socket_quat: torch.Tensor, bulb_quat: torch.Tensor, local_axis: torch.Tensor) -> torch.Tensor:
    """Signed bulb twist about the socket axis (swing-twist decomposition), radians."""
    relative = quat_mul(quat_inv(socket_quat), bulb_quat)
    relative = relative * torch.where(relative[:, :1] < 0.0, -1.0, 1.0)
    twist_sin = (relative[:, 1:] * local_axis).sum(dim=1)
    return 2.0 * torch.atan2(twist_sin, relative[:, 0])


def _orientation_error(socket_quat: torch.Tensor, bulb_quat: torch.Tensor) -> torch.Tensor:
    """Full-frame rotation angle between socket and bulb (bounds both tilt and twist).

    NOT the entry gate. Twist about the seat axis IS the screwing motion, so a full-frame
    comparison reads the very pose the mechanic asks for as misalignment -- see ``_tilt_error``
    and issue #90. This stays for callers that genuinely want both components bounded.
    """
    relative = quat_mul(quat_inv(socket_quat), bulb_quat)
    return 2.0 * torch.acos(relative[:, 0].abs().clamp(max=1.0))


def _tilt_error(socket_quat: torch.Tensor, bulb_quat: torch.Tensor, local_axis: torch.Tensor) -> torch.Tensor:
    """Angle between the bulb's plug axis and the socket's seat axis, ignoring twist (issue #90).

    This is what an entry gate should measure. A bayonet cap enters at any clock angle -- the
    operator holds it wherever their wrist happens to be and turns from there -- so the only
    orientation that can block entry is the bulb pointing the wrong way.

    Measured in the teleop bags of 2026-08-21: of the steps where the operator held the bulb in
    the socket and it did not capture, the tilt was within tolerance on 326 of 358, and the twist
    alone was rejecting them. One hold lasted 0.92 s at 25 degrees of twist.
    """
    plug = quat_apply(bulb_quat, local_axis.expand(bulb_quat.shape[0], 3))
    seat = quat_apply(socket_quat, local_axis.expand(socket_quat.shape[0], 3))
    cos = (plug * seat).sum(dim=1) / (plug.norm(dim=1) * seat.norm(dim=1)).clamp(min=1e-9)
    return torch.acos(cos.clamp(-1.0, 1.0))


def _seated_bulb_root_pose_w(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Root pose a bulb has when seated in the fixture, from the *live* socket pose."""
    socket: RigidObject = env.scene["socket"]
    quat = socket.data.root_quat_w
    seat = torch.tensor(SOCKET_SEAT_OFFSET, device=env.device).expand(env.num_envs, 3)
    plug = torch.tensor(BULB_PLUG_OFFSET, device=env.device).expand(env.num_envs, 3)
    pos = socket.data.root_pos_w + quat_apply(quat, seat) - quat_apply(quat, plug)
    return pos, quat


class bulb_attachment(ManagerTermBase):
    """Every-step FREE/AXIAL/ROTATING bayonet state machine for both bulbs."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        setattr(env, _ENV_ATTR, self)
        self._rotation_sign = float(cfg.params.get("rotation_sign", -1.0))
        if abs(self._rotation_sign) != 1.0:
            raise ValueError(f"rotation_sign must be -1 or 1, got {self._rotation_sign}")
        self._insertion_depth_spec = cfg.params.get("insertion_depth", 0.034)
        self._rotation_angle_spec = cfg.params.get("rotation_angle", 0.5 * math.pi)
        n, dev = env.num_envs, env.device
        # Row 0 = old bulb, row 1 = fresh bulb; the bulbs differ only in reset phase.
        self._phase = torch.zeros(2, n, dtype=torch.int8, device=dev)
        self._theta = torch.zeros(2, n, device=dev)
        # Twist of each bulb's pose as we last wrote (or spawned) it: theta integrates
        # against this, so transition and reset steps need no special-casing.
        self._prev_twist = torch.zeros(2, n, device=dev)
        # Clock angle each bulb entered the channel at (issue #90). The mechanic used to force
        # every engaged bulb to the socket's own clock angle, which teleported a gripped bulb by
        # up to a fifth of a radian in one step. `theta` is measured FROM this, so a lock is a
        # quarter turn from wherever the operator entered. The final roll varies and is invisible:
        # the bulb is a surface of revolution.
        self._entry_twist = torch.zeros(2, n, device=dev)
        self._depth = torch.zeros(n, device=dev)
        self._angle = torch.zeros(n, device=dev)
        self._axis_l = torch.tensor(SOCKET_SEAT_AXIS, device=dev).expand(n, 3)
        self._seat_offset = torch.tensor(SOCKET_SEAT_OFFSET, device=dev).expand(n, 3)
        self._plug_offset = torch.tensor(BULB_PLUG_OFFSET, device=dev).expand(n, 3)
        # Telemetry snapshot, taken at the end of every __call__ and never touched by reset().
        # `ManagerBasedRLEnv.step` runs interval events BEFORE it auto-resets finished
        # episodes, so this holds the state as of the terminating step -- which is what a
        # recorder wants for that row. Reading the live tensors there would report the *next*
        # episode's reset phases and freshly sampled limits instead.
        self._snapshot = (
            torch.zeros(2, n, device=dev),  # phase, as float
            torch.zeros(2, n, device=dev),  # theta
            torch.zeros(n, device=dev),  # sampled rotation angle
            torch.zeros(n, device=dev),  # sampled insertion depth
        )
        self.reset()
        self._take_snapshot()

    def _take_snapshot(self) -> None:
        """Copy the current lock state into the telemetry snapshot. See ``_snapshot``."""
        phase, theta, angle, depth = self._snapshot
        phase.copy_(self._phase.float())
        theta.copy_(self._theta)
        angle.copy_(self._angle)
        depth.copy_(self._depth)

    @property
    def rotation_sign(self) -> float:
        """Sign convention of the unlock twist. Issue #77 settles which face is correct.

        The wrong face is silent: the old bulb resets *at* the clamp ceiling, so a twist
        further into the lock changes nothing and ``at_lock_stop`` damps it away. Recording
        this value tells an operator which convention a run used.
        """
        return self._rotation_sign

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        _sample_parameter(self._depth, ids, self._insertion_depth_spec, "insertion_depth")
        _sample_parameter(self._angle, ids, self._rotation_angle_spec, "rotation_angle")
        self._phase[_OLD, ids] = _ROTATING
        self._theta[_OLD, ids] = self._angle[ids]
        self._phase[_FRESH, ids] = _FREE
        self._theta[_FRESH, ids] = 0.0
        # Both bulbs spawn untwisted relative to the socket (the old bulb's init rot IS
        # the fixture rot), so the first step's twist delta reads as ~0, not as -angle.
        self._prev_twist[:, ids] = 0.0
        # The old bulb spawns AT the socket's rotation, so entry twist zero reproduces the
        # pre-#90 seated pose exactly.
        self._entry_twist[:, ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        insertion_depth: ParameterSpec = 0.034,
        rotation_angle: ParameterSpec = 0.5 * math.pi,
        rotation_sign: float = -1.0,
        radial_tolerance: float = 0.015,
        tilt_tolerance: float = 0.2,
        seat_tolerance: float = 0.004,
    ) -> None:
        # insertion_depth / rotation_angle / rotation_sign are consumed from cfg.params
        # in __init__/reset. Always operates on all envs: the constraint must be
        # enforced every step, which the zero interval guarantees.
        del env_ids, insertion_depth, rotation_angle, rotation_sign
        old_bulb: RigidObject = env.scene["old_bulb"]
        fresh_bulb: RigidObject = env.scene["fresh_bulb"]
        self._advance(
            old_bulb,
            _OLD,
            socket_empty=self._phase[_FRESH] == _FREE,
            radial_tolerance=radial_tolerance,
            tilt_tolerance=tilt_tolerance,
            seat_tolerance=seat_tolerance,
        )
        self._advance(
            fresh_bulb,
            _FRESH,
            socket_empty=self._phase[_OLD] == _FREE,
            radial_tolerance=radial_tolerance,
            tilt_tolerance=tilt_tolerance,
            seat_tolerance=seat_tolerance,
        )
        self._take_snapshot()

    def _advance(
        self,
        bulb: RigidObject,
        row: int,
        socket_empty: torch.Tensor,
        radial_tolerance: float,
        tilt_tolerance: float,
        seat_tolerance: float,
    ) -> None:
        sign = self._rotation_sign
        socket: RigidObject = self._env.scene["socket"]
        socket_quat = socket.data.root_quat_w
        bulb_quat = bulb.data.root_quat_w
        axis_w = quat_apply(socket_quat, self._axis_l)
        seat = socket.data.root_pos_w + quat_apply(socket_quat, self._seat_offset)
        plug = bulb.data.root_pos_w + quat_apply(bulb_quat, self._plug_offset)
        displacement = plug - seat
        axial = (displacement * axis_w).sum(dim=1)
        lateral = torch.norm(displacement - axial.unsqueeze(1) * axis_w, dim=1)
        twist = _signed_twist(socket_quat, bulb_quat, self._axis_l)

        phase, theta, entry = self._phase[row], self._theta[row], self._entry_twist[row]
        # Lock-positive twist change since the pose we last wrote (or the spawn pose).
        delta = sign * _wrap_to_pi(twist - self._prev_twist[row])

        # -- transitions, all evaluated on the phase at step start (at most one per step)
        was_free = phase == _FREE
        was_axial = phase == _AXIAL
        was_rotating = phase == _ROTATING
        # `axial >= -seat_tolerance`, not `>= 0`: `lock` already accepts the bulb sitting that
        # far past the seat, because contact geometry stops it slightly short of the exact plane.
        # An entry gate that demands `>= 0` exactly rejects an operator who pushes a millimetre
        # too far -- the same overshoot the next transition forgives (issue #90).
        engage = (
            was_free
            & socket_empty
            & (axial >= -seat_tolerance)
            & (axial <= self._depth)
            & (lateral < radial_tolerance)
            & (_tilt_error(socket_quat, bulb_quat, self._axis_l) < tilt_tolerance)
        )
        eject = was_axial & (axial > self._depth)
        # Bottomed within seat_tolerance: contact geometry stops the bulb slightly short
        # of the exact seat, so an exact axial<=0 gate would make locking unreachable.
        lock = was_axial & ~eject & (axial <= seat_tolerance) & (delta > _EPS)
        theta_rotated = torch.minimum((theta + delta).clamp(min=0.0), self._angle)
        unlock = was_rotating & (theta_rotated <= _EPS) & (delta < 0.0)

        phase[engage] = _AXIAL
        theta[engage] = 0.0
        # Keep the clock angle the bulb arrived at. Without this the projection below writes
        # `socket_quat` outright, which teleports a gripped bulb onto the socket's own clock
        # angle in a single step -- measured at 0.155 rad in the 2026-08-21 bags (issue #90).
        entry[engage] = twist[engage]
        phase[eject] = _FREE
        phase[lock] = _ROTATING
        theta[lock] = torch.minimum(delta, self._angle)[lock]
        theta[was_rotating] = theta_rotated[was_rotating]
        phase[unlock] = _AXIAL
        theta[unlock] = 0.0

        # -- projection: remove every pose/velocity component the phase forbids
        in_axial = phase == _AXIAL
        in_rotating = phase == _ROTATING
        # The unlock step (ROTATING -> AXIAL) stays pinned at the seat like the lock step,
        # so axial travel only begins the following step -- otherwise a shove landing on
        # the same step the bulb unlocks would leak through the channel before AXIAL
        # projection engages.
        axial_travel = in_axial & ~unlock
        ids = (in_axial | in_rotating).nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            # theta == 0 throughout AXIAL, so one expression covers both phases. The written
            # twist is the ENTRY clock angle plus theta: a lock is a quarter turn from wherever
            # the bulb went in, not a turn onto the socket's own angle.
            proj_quat = quat_mul(socket_quat, _axis_angle_quat(self._axis_l, entry + sign * theta))
            proj_axial = torch.where(
                axial_travel,
                torch.minimum(axial.clamp(min=0.0), self._depth),
                torch.zeros_like(axial),
            )
            proj_pos = seat + proj_axial.unsqueeze(1) * axis_w - quat_apply(proj_quat, self._plug_offset)

            axial_speed = (bulb.data.root_lin_vel_w * axis_w).sum(dim=1)
            twist_speed = (bulb.data.root_ang_vel_w * axis_w).sum(dim=1)
            at_lock_stop = (theta >= self._angle - _EPS) & (sign * twist_speed > 0.0)
            twist_speed = torch.where(at_lock_stop, torch.zeros_like(twist_speed), twist_speed)
            zero = torch.zeros_like(axis_w)
            proj_lin = torch.where(axial_travel.unsqueeze(1), axial_speed.unsqueeze(1) * axis_w, zero)
            proj_ang = torch.where(in_rotating.unsqueeze(1), twist_speed.unsqueeze(1) * axis_w, zero)
            bulb.write_root_pose_to_sim(torch.cat([proj_pos[ids], proj_quat[ids]], dim=-1), env_ids=ids)
            bulb.write_root_velocity_to_sim(torch.cat([proj_lin[ids], proj_ang[ids]], dim=-1), env_ids=ids)

        # A constrained bulb now sits at the twist we wrote; a free bulb keeps its own.
        self._prev_twist[row] = torch.where(in_axial | in_rotating, entry + sign * theta, twist)


def _attachment(env: ManagerBasedRLEnv) -> bulb_attachment:
    mgr = getattr(env, _ENV_ATTR, None)
    if mgr is None:
        raise RuntimeError(
            "no bulb_attachment event term is configured on this env; the attachment "
            "predicates only work on tasks that wire mdp.bulb_attachment into their EventCfg"
        )
    return mgr


def attachment_manager(env: ManagerBasedRLEnv) -> bulb_attachment | None:
    """The env's ``bulb_attachment`` manager, or ``None`` when the task wires no term.

    ``_attachment()`` raises for an unwired task by design: a silent miss is the defect
    issue #76 exists to remove. Task-generic writers such as ``recording.py`` need to ask
    the question instead of answering it with an exception, so they use this accessor.
    """
    return getattr(env, _ENV_ATTR, None)


def bulb_lock_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bayonet lock state of both bulbs, for the privileged observation group (issue #77).

    Columns: old phase, old ``theta`` (rad), fresh phase, fresh ``theta`` (rad). Phase is
    0 ``FREE``, 1 ``AXIAL``, 2 ``ROTATING``, cast to float so the group concatenates.

    Until this term existed, ``_phase`` and ``_theta`` reached no observation, telemetry or
    recording path. An operator could not see whether a twist registered, which is why the
    three candidate causes in #77 could not be told apart -- nor told apart from a bad grasp.

    Returns:
        Tensor of shape (num_envs, 4).
    """
    mgr = _attachment(env)
    return torch.stack(
        [mgr._phase[_OLD].float(), mgr._theta[_OLD], mgr._phase[_FRESH].float(), mgr._theta[_FRESH]],
        dim=-1,
    )


def bulb_lock_telemetry(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Per-bulb lock state plus the per-env sampled parameters, for ``recording.py``.

    ``rotation_angle`` and ``insertion_depth`` re-sample per env at every reset, so a
    recorded ``theta`` alone is not interpretable: the same 1.4 rad is a fully locked bulb
    under one sample and a half-turned one under the next.

    Reads the snapshot rather than the live tensors, because a recorder runs after
    ``ManagerBasedRLEnv.step`` has auto-reset whichever episodes finished. The live state of a
    done env already belongs to the NEXT episode, so a terminal row would otherwise carry that
    episode's reset phases and newly sampled limits. This is the same hazard ``recording.py``
    documents for object poses, and the reason it reads termination term flags.
    """
    phase, theta, angle, depth = _attachment(env)._snapshot
    return {
        "old_bulb_phase": phase[_OLD],
        "old_bulb_theta": theta[_OLD],
        "fresh_bulb_phase": phase[_FRESH],
        "fresh_bulb_theta": theta[_FRESH],
        "lock_rotation_angle": angle,
        "lock_insertion_depth": depth,
    }


def old_bulb_attached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True while the old bulb remains in the rotation-only lock groove."""
    return _attachment(env)._phase[_OLD] == _ROTATING


def fresh_bulb_attached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True where the fresh bulb completed insertion and the full lock rotation.

    The attach-aware replacement for the geometric ``bulb_seated``: requires the whole
    bayonet sequence, not just transiting the seating tolerances.
    """
    mgr = _attachment(env)
    return (mgr._phase[_FRESH] == _ROTATING) & (mgr._theta[_FRESH] >= mgr._angle - _EPS)


def _old_bulb_constrained(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True until the old bulb has traveled out of the bayonet channel."""
    return _attachment(env)._phase[_OLD] != _FREE


def old_bulb_release_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Old-bulb fixture clearance (m) that reads 0 until the bulb exits the channel.

    Attach-aware ``old_bulb_fixture_clearance``: terminations/rewards run before the
    interval event projects a constrained bulb (see module docstring), so the raw
    clearance can transiently exceed the removal threshold and latch a false payout.
    """
    clearance = old_bulb_fixture_clearance(env)
    return torch.where(_old_bulb_constrained(env), torch.zeros_like(clearance), clearance)


def old_bulb_removed_after_release(env: ManagerBasedRLEnv, clearance_threshold: float) -> torch.Tensor:
    """Attach-aware ``old_bulb_removed``: only a *released* bulb can count as removed."""
    return old_bulb_release_clearance(env) > clearance_threshold


def old_bulb_disposal_distance_pinned(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Old-bulb -> crate distance, pinned to the seat until it exits the channel.

    A transient shove of a constrained bulb toward the crate must not pay disposal
    progress; the interval event rejects that displacement after rewards are computed.
    """
    d = old_bulb_disposal_distance(env)
    seat_pos, _ = _seated_bulb_root_pose_w(env)
    crate: RigidObject = env.scene["bin"]
    d_held = torch.norm(seat_pos - crate.data.root_pos_w, dim=1)
    return torch.where(_old_bulb_constrained(env), d_held, d)


def old_bulb_dropped_after_release(
    env: ManagerBasedRLEnv, min_height: float, disposal_threshold: float
) -> torch.Tensor:
    """Channel-aware ``old_bulb_dropped``: a constrained bulb cannot be "dropped".

    Guards the drop penalty/termination against transient displacement before the
    interval event projects the bulb back onto the bayonet channel.
    """
    return old_bulb_dropped(env, min_height, disposal_threshold) & ~_old_bulb_constrained(env)


def old_bulb_disposed_after_release(env: ManagerBasedRLEnv, disposal_threshold: float) -> torch.Tensor:
    """Channel-aware ``old_bulb_disposed``: only a *released* bulb can count as disposed.

    The paid-once disposal bonus and the ``success`` predicate read this rather than raw
    ``old_bulb_disposed``: rewards/terminations run before the interval event re-projects
    a constrained bulb, so a transient shove of a still-guided bulb into the crate radius
    could otherwise latch the payout.
    """
    return old_bulb_disposed(env, disposal_threshold) & ~_old_bulb_constrained(env)


def attached_replacement_success(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.015,
    ori_threshold: float = 0.2,
    disposal_threshold: float = 0.25,
) -> torch.Tensor:
    """True where the fresh bulb is locked in AND the old bulb is in the disposal crate.

    The attach-aware ``full_replacement_success`` (also the ``success`` termination).
    ``pos_threshold`` / ``ori_threshold`` are accepted for the meta.json recording
    contract (``recording.py`` reads them off the ``success`` term); the bayonet entry
    and projection stages enforce alignment instead.
    """
    return fresh_bulb_attached(env) & old_bulb_disposed_after_release(env, disposal_threshold)
