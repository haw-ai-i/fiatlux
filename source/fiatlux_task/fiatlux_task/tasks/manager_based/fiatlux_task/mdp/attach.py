# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bulb/socket retention (issue #167): an axial detent, not a bayonet.

An earlier version of this module (issue #54) implemented a three-phase FREE/AXIAL/ROTATING
bayonet with twist tracking, enforced by overwriting the bulb's pose and velocity every step.
That overwrite ran *after* the physics solver had already resolved contact for the step, so
whenever real bulb-socket collision was enabled the two authorities fought instead of
converging: contact pushed one way, the projection un-did it, contact pushed again -- the
"median 1067 N of socket contact" and the PR #128 crush-glitch spikes were both this pattern.
The workaround was to filter bulb-socket collision out entirely (``scene_cfg.py``'s
``_spawn_bulb_socket_filtered``, now removed) and let the scripted projection substitute for
real contact everywhere.

Two independent diagnostics (``scripts/diagnose_contact_twist.py``,
``scripts/diagnose_contact_axial.py``; see ``plans/bayonet-force-based-attachment.md``) found
that with collision genuinely enabled, real contact geometry blocked BOTH the twist-release and
axial-insertion motions the bayonet assumed were unobstructed -- the plug's radius was equal to
or larger than the bore at every relevant height. Shrinking the plug (2026-09-07) fixed that,
and a rerun under real contact (2026-09-08) confirmed insertion now works given reasonable
orientation guidance. That result removed the reason for the bayonet's own existence: with
collision back on, the socket's geometry confines the bulb laterally and angularly on its own --
nothing scripted has to. There is also no physical lug or groove in this asset (a plain round
bore), so the twist/lock semantics were never modeling a real feature, only a scripted one.

What real contact still cannot provide is RETENTION: nothing stops the bulb sliding back out of
a plain round bore under gravity or a knock. This module supplies exactly that, and nothing
else. Two states per bulb, per env:

- ``FREE`` -- unconstrained rigid body; physics owns it entirely.
- ``SEATED`` -- held by a continuous axial spring-damper FORCE (world-frame, applied through
  ``set_external_force_and_torque``), not a pose or velocity overwrite. Lateral position and
  orientation are left entirely to real contact; only axial displacement from the seat is
  corrected, and only while seated.

``FREE -> SEATED`` fires on reaching the seat (within ``seat_tolerance``) while reasonably
aligned (``radial_tolerance``, ``tilt_tolerance``) with the socket unoccupied by the other bulb.
``SEATED -> FREE`` (release) fires when real, physics-driven axial displacement from the seat
exceeds ``release_threshold`` -- a deliberate, sustained pull, not a force threshold, so a light
knock does not release it but a genuine withdrawal does.

Because retention is a continuous force rather than a per-step overwrite, the solver resolves
contact and the spring together in one solve every step -- there is no second authority for it
to disagree with, and so no equivalent of the crush-glitch pattern to reintroduce.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply

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
_SEATED = 1
_OLD = 0  # state row of the old bulb (scene entity "old_bulb")
_FRESH = 1  # state row of the fresh bulb (scene entity "fresh_bulb")

# How near the seat a bulb must SPAWN to be treated as starting seated in the fixture
# (``_resolve_spawn_phase``). Only has to separate "seated" from "put somewhere else"; carried
# over unchanged from the bayonet version -- see its git history for the measurements behind
# this value (the in-hand carry stage spawns ~0.10-0.13 m from the seat, ~2x this tolerance).
_SEATED_SPAWN_TOLERANCE = 0.05  # m


def _tilt_error(socket_quat: torch.Tensor, bulb_quat: torch.Tensor, local_axis: torch.Tensor) -> torch.Tensor:
    """Angle between the bulb's plug axis and the socket's seat axis, ignoring twist (issue #90).

    A push-fit bulb enters at any clock angle -- only the bulb pointing the wrong way should
    block entry. Carried over unchanged from the bayonet version; still the right entry gate
    even with no lock to turn into.
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
    """Every-step FREE/SEATED axial-detent retention for both bulbs.

    ``old_bulb`` is optional -- insert-only scenes (nothing to remove) omit it, and this term
    then only ever manages ``fresh_bulb``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        setattr(env, _ENV_ATTR, self)
        n, dev = env.num_envs, env.device
        # Row 0 = old bulb, row 1 = fresh bulb; the bulbs differ only in reset phase.
        self._phase = torch.zeros(2, n, dtype=torch.int8, device=dev)
        # Envs whose spawn phase has not been read off the scene yet. See _resolve_spawn_phase.
        self._pending = torch.zeros(n, dtype=torch.bool, device=dev)
        self._axis_l = torch.tensor(SOCKET_SEAT_AXIS, device=dev).expand(n, 3)
        self._seat_offset = torch.tensor(SOCKET_SEAT_OFFSET, device=dev).expand(n, 3)
        self._plug_offset = torch.tensor(BULB_PLUG_OFFSET, device=dev).expand(n, 3)
        # Telemetry snapshot, taken at the end of every __call__ and never touched by reset().
        # `ManagerBasedRLEnv.step` auto-resets finished episodes before the next call, so a
        # terminal row must read this snapshot rather than the live tensor, or it would report
        # the NEXT episode's reset phase instead (same hazard the bayonet version documented).
        self._snapshot = torch.zeros(2, n, device=dev)  # phase, as float
        self.reset()
        self._take_snapshot()

    def _take_snapshot(self) -> None:
        self._snapshot.copy_(self._phase.float())

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        # Both bulbs reset FREE here and are seated, if they belong seated, by
        # `_resolve_spawn_phase` on the first step -- see that method for why the decision
        # cannot be made here.
        self._phase[:, ids] = _FREE
        self._pending[ids] = True

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        radial_tolerance: float = 0.015,
        tilt_tolerance: float = 0.2,
        seat_tolerance: float = 0.004,
        release_threshold: float = 0.02,
        spring_k: float = 20.0,
        spring_d: float = 1.7,
        max_force: float = 5.0,
    ) -> None:
        # Always operates on all envs: retention must be enforced every step, which the zero
        # interval guarantees.
        #
        # spring_k/spring_d starting point: critically damped (d = 2*sqrt(k*m)) at the bulb's
        # ~0.035 kg mass, kept well under the semi-implicit stability bound k < mass/step_dt^2
        # (~87 N/m at step_dt=0.02s) with margin for the fact this is a rough starting gain,
        # not a derived one -- retune against real teleop bags before trusting it in production.
        # max_force 5 N is an arbitrary "light detent" ceiling, several times the bulb's own
        # ~0.34 N weight; also needs real tuning. release_threshold 2 cm is comfortably past
        # seat_tolerance (4 mm) so an unheld, resting bulb (which sags a little under gravity)
        # never self-releases, but well short of a real withdrawal.
        del env_ids
        # old_bulb is absent from insert-only scenes (e.g. the tabletop preset, which has
        # nothing to remove) -- self._phase[_OLD] then never leaves its _FREE default, so
        # fresh_bulb's own socket_empty check below is correct with no further change.
        old_bulb: RigidObject | None = env.scene["old_bulb"] if "old_bulb" in env.scene.keys() else None
        fresh_bulb: RigidObject = env.scene["fresh_bulb"]
        self._resolve_spawn_phase(env, old_bulb, fresh_bulb)
        if old_bulb is not None:
            self._advance(
                old_bulb,
                _OLD,
                socket_empty=self._phase[_FRESH] == _FREE,
                radial_tolerance=radial_tolerance,
                tilt_tolerance=tilt_tolerance,
                seat_tolerance=seat_tolerance,
                release_threshold=release_threshold,
                spring_k=spring_k,
                spring_d=spring_d,
                max_force=max_force,
            )
        self._advance(
            fresh_bulb,
            _FRESH,
            socket_empty=self._phase[_OLD] == _FREE,
            radial_tolerance=radial_tolerance,
            tilt_tolerance=tilt_tolerance,
            seat_tolerance=seat_tolerance,
            release_threshold=release_threshold,
            spring_k=spring_k,
            spring_d=spring_d,
            max_force=max_force,
        )
        self._take_snapshot()

    def _resolve_spawn_phase(
        self, env: ManagerBasedEnv, old_bulb: RigidObject | None, fresh_bulb: RigidObject
    ) -> None:
        """Seat whichever bulb the task actually SPAWNED at the seat, and only that one.

        Reading the phase off the scene means a preset that moves a bulb has said everything
        it needs to say -- carried over unchanged from the bayonet version (issue #109/#108).
        Deferred to the first step rather than done in ``reset()`` because the phase depends on
        the bulb's spawned pose, and the event that restores it (``reset_scene_to_default``) is
        a sibling reset term whose ordering against this one is not ours to rely on.

        ``old_bulb`` is ``None`` on insert-only scenes with nothing to remove; that row is left
        alone (permanently ``_FREE``, its default).
        """
        ids = self._pending.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        seat_pos, _ = _seated_bulb_root_pose_w(env)
        for row, bulb in ((_OLD, old_bulb), (_FRESH, fresh_bulb)):
            if bulb is None:
                continue
            at_seat = torch.norm(bulb.data.root_pos_w - seat_pos, dim=1) < _SEATED_SPAWN_TOLERANCE
            seated = ids[at_seat[ids]]
            if seated.numel() > 0:
                self._phase[row, seated] = _SEATED
        self._pending[ids] = False

    def _advance(
        self,
        bulb: RigidObject,
        row: int,
        socket_empty: torch.Tensor,
        radial_tolerance: float,
        tilt_tolerance: float,
        seat_tolerance: float,
        release_threshold: float,
        spring_k: float,
        spring_d: float,
        max_force: float,
    ) -> None:
        socket: RigidObject = self._env.scene["socket"]
        socket_quat = socket.data.root_quat_w
        bulb_quat = bulb.data.root_quat_w
        axis_w = quat_apply(socket_quat, self._axis_l)
        seat = socket.data.root_pos_w + quat_apply(socket_quat, self._seat_offset)
        plug = bulb.data.root_pos_w + quat_apply(bulb_quat, self._plug_offset)
        displacement = plug - seat
        axial = (displacement * axis_w).sum(dim=1)
        lateral = torch.norm(displacement - axial.unsqueeze(1) * axis_w, dim=1)

        phase = self._phase[row]
        was_free = phase == _FREE
        was_seated = phase == _SEATED

        seat_now = (
            was_free
            & socket_empty
            & (axial.abs() <= seat_tolerance)
            & (lateral < radial_tolerance)
            & (_tilt_error(socket_quat, bulb_quat, self._axis_l) < tilt_tolerance)
        )
        # A real, physics-driven excursion past the seat -- not a force threshold -- releases
        # it. This is deliberately readable straight off real contact: nothing here overwrites
        # position, so `axial` is exactly what the solver produced.
        release = was_seated & (axial.abs() > release_threshold)

        phase[seat_now] = _SEATED
        phase[release] = _FREE

        # Continuous axial spring-damper, world-frame, applied alongside (never instead of)
        # real contact -- zero force where not seated, so a bulb that just released or was
        # never seated is untouched by this term. Lateral and orientation are left to real
        # contact entirely; nothing here corrects them.
        seated_now = (phase == _SEATED) & ~release
        axial_rate = (bulb.data.root_lin_vel_w * axis_w).sum(dim=1)
        spring_force = (-spring_k * axial - spring_d * axial_rate).clamp(-max_force, max_force)
        applied = torch.where(seated_now, spring_force, torch.zeros_like(spring_force))
        forces = (applied.unsqueeze(1) * axis_w).unsqueeze(1)
        torques = torch.zeros((self._env.num_envs, 1, 3), device=self._env.device)
        bulb.set_external_force_and_torque(forces, torques, is_global=True)


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
    """Seated state of both bulbs, for the privileged observation group (issue #77).

    Columns: old seated (0/1), fresh seated (0/1), cast to float so the group concatenates.
    Two columns, not the bayonet version's four -- there is no theta to report anymore.

    Returns:
        Tensor of shape (num_envs, 2).
    """
    mgr = _attachment(env)
    return torch.stack([mgr._phase[_OLD].float(), mgr._phase[_FRESH].float()], dim=-1)


def bulb_lock_telemetry(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Per-bulb seated state, for ``recording.py``.

    Reads the snapshot rather than the live tensors, because a recorder runs after
    ``ManagerBasedRLEnv.step`` has auto-reset whichever episodes finished. The live state of a
    done env already belongs to the NEXT episode, so a terminal row would otherwise carry that
    episode's reset phase. This is the same hazard ``recording.py`` documents for object poses.
    """
    snapshot = _attachment(env)._snapshot
    return {
        "old_bulb_phase": snapshot[_OLD],
        "fresh_bulb_phase": snapshot[_FRESH],
    }


def old_bulb_attached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True while the old bulb remains seated."""
    return _attachment(env)._phase[_OLD] == _SEATED


def fresh_bulb_attached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True while the fresh bulb remains seated.

    The attach-aware replacement for the geometric ``bulb_seated``: requires the seat
    admission gate (alignment + socket-empty) to have fired, not just current proximity.
    """
    return _attachment(env)._phase[_FRESH] == _SEATED


def _old_bulb_constrained(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True while the old bulb is seated (retained)."""
    return _attachment(env)._phase[_OLD] != _FREE


def old_bulb_release_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Old-bulb fixture clearance (m) that reads 0 until the bulb is released.

    Attach-aware ``old_bulb_fixture_clearance``: a transient shove of a seated bulb must not
    read as clearance, since the retention spring (not a projection) still owns the axial
    error and will pull it back.
    """
    clearance = old_bulb_fixture_clearance(env)
    return torch.where(_old_bulb_constrained(env), torch.zeros_like(clearance), clearance)


def old_bulb_removed_after_release(env: ManagerBasedRLEnv, clearance_threshold: float) -> torch.Tensor:
    """Attach-aware ``old_bulb_removed``: only a *released* bulb can count as removed."""
    return old_bulb_release_clearance(env) > clearance_threshold


def old_bulb_disposal_distance_pinned(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Old-bulb -> crate distance, pinned to the seat until it is released.

    A transient shove of a seated bulb toward the crate must not pay disposal progress.
    """
    d = old_bulb_disposal_distance(env)
    seat_pos, _ = _seated_bulb_root_pose_w(env)
    crate: RigidObject = env.scene["bin"]
    d_held = torch.norm(seat_pos - crate.data.root_pos_w, dim=1)
    return torch.where(_old_bulb_constrained(env), d_held, d)


def old_bulb_dropped_after_release(env: ManagerBasedRLEnv, min_height: float) -> torch.Tensor:
    """Channel-aware ``old_bulb_dropped``: a seated bulb cannot be "dropped"."""
    return old_bulb_dropped(env, min_height) & ~_old_bulb_constrained(env)


def old_bulb_disposed_after_release(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Channel-aware ``old_bulb_disposed``: only a *released* bulb can count as disposed.

    The paid-once disposal bonus and the ``success`` predicate read this rather than raw
    ``old_bulb_disposed``, so a transient shove of a still-seated bulb into the crate cannot
    latch the payout.
    """
    return old_bulb_disposed(env) & ~_old_bulb_constrained(env)


def attached_replacement_success(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.015,
    ori_threshold: float = 0.2,
) -> torch.Tensor:
    """True where the fresh bulb is seated AND the old bulb is in the disposal crate.

    The attach-aware ``full_replacement_success`` (also the ``success`` termination).
    ``pos_threshold``/``ori_threshold`` are accepted for the meta.json recording contract
    (``recording.py`` reads them off the ``success`` term); the seat admission gate enforces
    alignment instead.
    """
    return fresh_bulb_attached(env) & old_bulb_disposed_after_release(env)
