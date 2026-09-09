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
- ``SEATED`` -- held by a continuous spring-damper WRENCH (world-frame, applied through
  ``set_external_force_and_torque``), not a pose or velocity overwrite: a full-strength axial
  term plus a deliberately much gentler lateral + tilt centering term (issue #171).

``FREE -> SEATED`` fires on reaching the seat (within ``seat_tolerance``) while reasonably
aligned (``radial_tolerance``, ``tilt_tolerance``) with the socket unoccupied by the other bulb.
``SEATED -> FREE`` (release) fires when real, physics-driven axial displacement from the seat
exceeds ``release_threshold`` -- a deliberate, sustained pull, not a force threshold, so a light
knock does not release it but a genuine withdrawal does.

Because retention is a continuous force rather than a per-step overwrite, the solver resolves
contact and the wrench together in one solve every step -- there is no second authority for it
to disagree with, and so no equivalent of the crush-glitch pattern to reintroduce.

**Lateral + tilt centering (issue #171, added after the mechanism above shipped)**: the first
version left lateral position and orientation entirely to real contact, on the theory that with
collision back on the socket confines the bulb on its own. Real teleop evidence found otherwise:
a seated bulb visibly tilts/swings, because the bore's authored radial clearance (2.69mm,
``assets/omniverse_bulb/LightBulb_bulb_z_rigid.usda``'s 0.84 plug scale) is real, necessary
slop, not a defect -- ``scripts/diagnose_contact_axial.py`` confirmed that tightening it even
to 1.86mm (0.88 scale) breaks force-driven insertion outright (the bore has no lead-in
chamfer; any reduction makes the plug catch on the sharp rim), so the wobble cannot be fixed by
shrinking the gap. Instead, a SEATED bulb now also gets a lateral spring-damper (pulling the
plug back toward the seat axis line) and a tilt spring-damper torque (aligning the plug axis
back to the seat axis, twist left free as always) -- both far weaker than the axial term and
its own real contact, `deliberately` gentle so they damp wobble without fighting or substituting
for contact the way the old bayonet's pose overwrite did. Gated on ``seated_now`` exactly like
the axial term, so a FREE bulb (including one still mid-insertion) is completely unaffected --
this cannot touch insertion dynamics at all.

**Axial retention is a magnet, not a spring (issue #171, revised)**: a first pass held the
bulb with a linear spring (``F = -k*axial``) -- weakest exactly at the seat, growing the
farther out the bulb is pulled. That was flagged as physically backwards for what this is
meant to model: a magnetic (or friction/detent) catch is STRONGEST at contact and falls off
with distance, the opposite shape. The practical difference matters -- a spring makes removal
progressively harder the more you pull (hardest right before release); a magnet gives a firm
breakaway resistance right at full contact, then gets EASIER to keep separating once past it,
and the force-to-zero jump at ``release_threshold`` is smaller since it was already decaying
into that boundary rather than sitting near its peak.

The axial force is now ``F(axial) = -tanh(axial / deadband) * hold_force / (1 + |axial| /
hold_range) - spring_d * axial_rate``: magnitude peaks at ``hold_force`` exactly at the seat
and decays toward 0 as ``|axial|`` grows past ``hold_range`` (the "half-strength distance");
``tanh(.../deadband)`` supplies the sign smoothly (a plain ``sign(axial)`` would flip
direction at full magnitude across an infinitesimal crossing of ``axial=0``, a chatter risk at
rest -- ``deadband`` is far smaller than any displacement that matters, so this is
indistinguishable from a sign flip everywhere except in a sub-millimeter dead zone at the
seat). ``spring_d`` (still linear velocity damping, critically-damped for the LOCAL stiffness
``hold_force/hold_range`` at ``axial=0``, the region where the bulb actually spends most of
its time) is unchanged in form.

An earlier version of the ceiling-mount fix (before this model change) added a feedforward
that cancelled gravity's axial component outright, making steady-state sag ~0 at every
orientation. That was rejected on review: a passive mechanism doesn't know about gravity and
null it out, so a bulb hanging against gravity SHOULD sag more than one resting with it, the
same as any real spring/friction/magnet would -- that asymmetry is physically correct, not a
bug. The magnet model's own gravity margin is sized the same way the spring's was: ``hold_force``
and ``hold_range`` are chosen so the worst-case (ceiling, full weight, gravity entirely along
the release direction) steady-state sag sits well clear of ``release_threshold``, while keeping
the local stiffness at ``axial=0`` well under the documented semi-implicit stability ceiling
(``~mass/step_dt^2``). That ceiling is also why ``hold_force`` ends up modest in absolute terms
(a fraction of a newton, not the old spring's several-newton cap): a magnet-like law's peak
force occurs exactly where its local stiffness is evaluated (``axial=0``), unlike a linear
spring whose cap can sit far out along an otherwise-gentle-near-zero curve -- there is no way
to get a strong peak AND fast falloff AND stay under the same stability ceiling at this
control loop's rate (50 Hz) all at once; this trades some peak strength for real gravity
margin and a stable, chatter-free hold. A table/wall-mounted bulb still sags less than a
ceiling-mounted one under its own weight, correctly.

**Twist friction (issue #171, third finding)**: a ceiling-mounted, seated bulb was found
spinning about the seat axis at 1-19 rad/s (~2-3 rev/s) for a sustained ~2.9s -- present
almost from the moment it seats, never decaying, then abruptly destabilizing into a real
ejection (axial and tilt both blowing up together) with no operator and no hand contact the
entire time. Cause: the tilt torque's damping term (``-tilt_d * ang_vel``) used the FULL
angular velocity, including whatever twist (rotation about the seat axis itself) the bulb
happened to be carrying -- but that damping shared ``max_torque`` with the tilt-ALIGNMENT
term (``tilt_k * cross(...)``), a deliberately tiny budget (0.05 N*m) sized to stay gentle
against real contact. Arresting even 10 rad/s of twist needs roughly ``tilt_d * 10`` of
torque, already several times that budget -- so the damping saturated at a small fraction of
what the spin needed every single step, doing essentially nothing to it, while contact kept
the spin fed. Twist was always deliberately left uncorrected FOR ALIGNMENT (issue #90: any
clock angle is a valid entry, there is no lug/groove to turn into) -- but "no target clock
angle" was never meant to also mean "no bound on how fast it may spin forever, frictionlessly."

A first fix split angular velocity into its twist component (along the seat axis) and its
perpendicular (tilt) component, and damped twist separately with its own, much larger torque
budget -- but as a VISCOUS term (``-twist_d * twist_rate``, proportional to speed, the same
shape as every other damping term here). Under real contact this settled into a stable but
nonzero equilibrium spin rather than arresting it, and -- tellingly -- raising the gain made
the equilibrium WORSE (a higher, not lower, residual rate) at some tested magnitudes, evidence
of a genuinely nonlinear/chaotic coupling with real contact, not just an under-sized gain.
Replaced with FRICTION instead: real contact friction is Coulomb-like, roughly CONSTANT
magnitude opposing the direction of sliding, not scaling with speed -- ``twist_friction`` is
that constant magnitude, its sign supplied smoothly by ``tanh(twist_rate / twist_deadband)``
(the same chatter-avoidance reasoning as the axial term's direction) rather than a literal
``sign()``. Both the viscous and the friction versions reliably stop the actual reported
failure (self-ejection: verified holding for 5+ simulated seconds at spin rates spanning the
full 1-19 rad/s reported range, ``scripts/verify_twist_damping.py``) -- but NEITHER reliably
drives the residual spin itself to zero; it persists at some nonzero, sometimes noisy rate
that this single-axis torque law does not have consistent authority over. That residual spin
looks like a real 3D contact effect (a loosely-toleranced cylindrical plug precessing/rattling
in the bore) rather than something a twist-only force law can fully resolve -- flagged as an
open follow-up, not treated as solved. ``max_twist_torque`` is a defensive outer clamp, not
normally binding (``twist_friction`` is already comfortably under it by choice). Safe to make
this whole term far stronger than tilt's alignment torque specifically because it is PURE
dissipation, not a restoring force toward any target -- it can only remove existing rotational
energy, never fight a real contact equilibrium the way a stiff position/orientation lock would
(the crush-glitch pattern this whole design already avoids elsewhere). Tilt's own alignment
torque and its (now twist-free) damping are unchanged in spirit, still gentle.
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


def _clamp_vector_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Scale each row of ``v`` down to ``max_norm`` if it exceeds it, preserving direction.

    The axial spring is a scalar (one DOF), so a plain ``.clamp(-max, max)`` saturates it
    correctly. Lateral force and tilt torque are 3-vectors; clamping components independently
    would distort their direction when only one axis saturates, so this scales the whole vector
    down uniformly instead, by its norm.
    """
    norm = v.norm(dim=-1, keepdim=True)
    scale = (max_norm / norm.clamp(min=1e-9)).clamp(max=1.0)
    return v * scale


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
        hold_force: float = 0.5,
        hold_range: float = 0.01,
        spring_d: float = 2.65,
        max_force: float = 5.0,
        lateral_k: float = 5.0,
        lateral_d: float = 0.85,
        max_lateral_force: float = 1.0,
        tilt_k: float = 0.05,
        tilt_d: float = 0.01,
        max_torque: float = 0.05,
        twist_friction: float = 0.5,
        twist_deadband: float = 0.1,
        max_twist_torque: float = 1.0,
    ) -> None:
        # Always operates on all envs: retention must be enforced every step, which the zero
        # interval guarantees.
        #
        # hold_force/hold_range (issue #171, revised): a magnet-like axial law, not a spring --
        # see the module docstring for why. hold_force is the peak/breakaway force AT the seat
        # (axial=0); hold_range is the distance at which it has decayed to half that. Their
        # ratio is the LOCAL stiffness at axial=0 (hold_force/hold_range = 50 N/m here), kept
        # well under the semi-implicit stability bound mass/step_dt^2 (~87 N/m at
        # step_dt=0.02s; 50 is ~57% of it) -- spring_d is critically damped for that local
        # stiffness, same derivation as the original spring (d = 2*sqrt(k*m) at the bulb's
        # ~0.035 kg mass). Sized for the WORST CASE static load, not the average one: a ceiling
        # mount is inverted, so the bulb's own ~0.34 N weight acts entirely along the (outward)
        # seat axis there, and the resulting steady-state sag needs to sit clear of
        # release_threshold with real margin for transients -- at this gain that sag is ~4.6mm
        # (worst case), against a 20mm threshold. A table/wall-mounted bulb sags less than this
        # under its own weight, correctly -- that asymmetry is what a real passive retention
        # mechanism (spring, friction, magnet) would also show; nothing here is gravity-aware or
        # orientation-aware, on purpose. hold_force ends up modest in absolute terms (~1.5x the
        # bulb's own weight) because a magnet-like law's peak occurs exactly where its local
        # stiffness is evaluated (axial=0) -- there is no way to get a much stronger peak AND
        # keep the gravity margin AND stay under the same stability ceiling at this control
        # loop's 50 Hz rate all at once. Still rough starting gains, not derived ones -- retune
        # against real teleop bags before trusting them in production. max_force 5 N is an
        # overall safety clamp on the combined position+damping force (rarely the binding
        # constraint from the position term alone, since that never exceeds hold_force by
        # construction; matters mainly for a large damping contribution at high velocity).
        # release_threshold 2 cm is comfortably past seat_tolerance (4 mm) so an unheld, resting
        # bulb (which sags a little under gravity) never self-releases, but well short of a real
        # withdrawal.
        #
        # lateral_k/lateral_d/max_lateral_force (issue #171): deliberately far gentler than the
        # axial term -- capped at ~3x the bulb's own weight, critically damped the same way --
        # so this damps wobble without fighting real contact or acting as a second authority
        # over lateral position (real contact, not this term, is what actually has to bear a
        # sideways load). Kept as a plain spring-damper (not magnet-shaped): it is an assist to
        # real contact, not the primary retention the axial term's magnet law models.
        # tilt_k/tilt_d/max_torque have no inertia measurement behind them (unlike the axial
        # term's mass-derived critical damping); they are rough starting points at the same
        # order of magnitude as diagnose_contact_axial.py's own orientation-hold trial
        # (k_p=0.02, k_d=0.002) shown to function as a soft aligner there, needing the same
        # real-teleop retuning as every other gain in this term.
        #
        # twist_friction/twist_deadband/max_twist_torque (issue #171, third finding): FRICTION
        # for rotation about the seat axis itself (twist), split out from tilt_d so it gets
        # its own, much larger torque budget instead of sharing tilt's deliberately tiny
        # max_torque. Found necessary after a ceiling-seated bulb sustained a 1-19 rad/s twist
        # for ~2.9s with no decay under a first (viscous, -tilt_d*ang_vel-shaped) attempt --
        # arresting even 10 rad/s needs several times tilt's max_torque budget, so that shared
        # clamp was saturating uselessly every step regardless of spin speed. A viscous
        # twist-only damping term with its own larger budget still wasn't right, though: it
        # settled into a chaotic, non-monotonic equilibrium spin under real contact that got
        # WORSE (not better) as the gain was raised -- evidence of a wrong force LAW, not just
        # an under-sized one. twist_friction models real contact friction instead: roughly
        # CONSTANT magnitude opposing the direction of rotation (Coulomb-like), not scaling
        # with speed the way viscous damping does, matching how solid-on-solid friction
        # actually behaves. Safe to size much larger than the tilt-alignment budget
        # specifically because this is PURE dissipation, not a restoring force toward any
        # target -- twist has no target angle (issue #90: any clock angle is a valid entry),
        # so this can only ever remove existing rotational energy, never fight a real contact
        # equilibrium the way a stiff position/orientation lock would. Still rough starting
        # values -- needs real-teleop validation that it's enough to out-damp whatever
        # contact keeps feeding the spin, without making ordinary handling feel unexpectedly
        # stiff.
        del env_ids
        # old_bulb is absent from insert-only scenes (e.g. the tabletop preset, which has
        # nothing to remove) -- self._phase[_OLD] then never leaves its _FREE default, so
        # fresh_bulb's own socket_empty check below is correct with no further change.
        old_bulb: RigidObject | None = env.scene["old_bulb"] if "old_bulb" in env.scene.keys() else None
        fresh_bulb: RigidObject = env.scene["fresh_bulb"]
        self._resolve_spawn_phase(env, old_bulb, fresh_bulb)
        gains = dict(
            radial_tolerance=radial_tolerance,
            tilt_tolerance=tilt_tolerance,
            seat_tolerance=seat_tolerance,
            release_threshold=release_threshold,
            hold_force=hold_force,
            hold_range=hold_range,
            spring_d=spring_d,
            max_force=max_force,
            lateral_k=lateral_k,
            lateral_d=lateral_d,
            max_lateral_force=max_lateral_force,
            tilt_k=tilt_k,
            tilt_d=tilt_d,
            max_torque=max_torque,
            twist_friction=twist_friction,
            twist_deadband=twist_deadband,
            max_twist_torque=max_twist_torque,
        )
        if old_bulb is not None:
            self._advance(old_bulb, _OLD, socket_empty=self._phase[_FRESH] == _FREE, **gains)
        self._advance(fresh_bulb, _FRESH, socket_empty=self._phase[_OLD] == _FREE, **gains)
        self._take_snapshot()

    def _resolve_spawn_phase(self, env: ManagerBasedEnv, old_bulb: RigidObject | None, fresh_bulb: RigidObject) -> None:
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
        hold_force: float,
        hold_range: float,
        spring_d: float,
        max_force: float,
        lateral_k: float,
        lateral_d: float,
        max_lateral_force: float,
        tilt_k: float,
        tilt_d: float,
        max_torque: float,
        twist_friction: float,
        twist_deadband: float,
        max_twist_torque: float,
    ) -> None:
        socket: RigidObject = self._env.scene["socket"]
        socket_quat = socket.data.root_quat_w
        bulb_quat = bulb.data.root_quat_w
        axis_w = quat_apply(socket_quat, self._axis_l)
        seat = socket.data.root_pos_w + quat_apply(socket_quat, self._seat_offset)
        plug = bulb.data.root_pos_w + quat_apply(bulb_quat, self._plug_offset)
        displacement = plug - seat
        axial = (displacement * axis_w).sum(dim=1)
        lateral_vec = displacement - axial.unsqueeze(1) * axis_w
        lateral = torch.norm(lateral_vec, dim=1)

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

        # Continuous spring-damper wrench, world-frame, applied alongside (never instead of)
        # real contact -- zero everywhere a bulb is not seated, so a bulb that just released or
        # was never seated is completely untouched by this term (including mid-insertion: this
        # cannot affect the FREE-phase dynamics diagnose_contact_axial.py exercises).
        seated_now = (phase == _SEATED) & ~release
        seated_mask = seated_now.unsqueeze(1)

        # Axial: full-strength retention, shaped like a magnet's force/gap curve, not a
        # spring's (issue #171 -- see the module docstring for the physical reasoning and the
        # gravity-margin derivation behind hold_force/hold_range). Magnitude peaks at
        # hold_force exactly at the seat and decays toward 0 as |axial| grows past hold_range;
        # direction comes from a smoothed sign (tanh over a deadband far smaller than any
        # displacement that matters -- seat_tolerance/4 -- rather than a literal sign(), which
        # would flip direction at full magnitude across an infinitesimal crossing of axial=0,
        # a chatter risk exactly at rest).
        deadband = seat_tolerance / 4.0
        direction = torch.tanh(axial / deadband)
        magnitude = hold_force / (1.0 + axial.abs() / hold_range)
        axial_rate = (bulb.data.root_lin_vel_w * axis_w).sum(dim=1)
        axial_force = (-direction * magnitude - spring_d * axial_rate).clamp(-max_force, max_force)
        axial_force_vec = axial_force.unsqueeze(1) * axis_w

        # Lateral: a much gentler spring-damper pulling the plug back toward the seat axis
        # line (issue #171 -- see the module docstring for why this is a software fix, not a
        # tighter bore).
        lin_vel = bulb.data.root_lin_vel_w
        lateral_vel = lin_vel - (lin_vel * axis_w).sum(dim=1, keepdim=True) * axis_w
        lateral_force_vec = _clamp_vector_norm(-lateral_k * lateral_vec - lateral_d * lateral_vel, max_lateral_force)

        # Tilt vs. twist: split angular velocity into its component along the seat axis
        # (twist -- rotation about the plug's own axis, which has no target angle, issue #90)
        # and everything perpendicular to it (tilt -- misalignment of the axis itself, which
        # DOES have a target: aligned with the socket's). Conflating them into one damping
        # term sharing tilt's tiny max_torque let a real spin go essentially undamped (issue
        # #171, third finding): arresting it needed several times that budget every step.
        ang_vel = bulb.data.root_ang_vel_w
        twist_rate = (ang_vel * axis_w).sum(dim=1)
        tilt_ang_vel = ang_vel - twist_rate.unsqueeze(1) * axis_w

        # Tilt: small torque aligning the plug axis back to the seat axis -- the standard
        # small-angle "rotate A onto B" construction (magnitude ~ sin(tilt), direction the
        # correct rotation axis) -- damped against ONLY the tilt component of angular
        # velocity now, so twist can get its own, separately-sized budget below.
        plug_axis_w = quat_apply(bulb_quat, self._axis_l)
        tilt_correction = torch.cross(plug_axis_w, axis_w, dim=-1)
        tilt_torque_vec = _clamp_vector_norm(tilt_k * tilt_correction - tilt_d * tilt_ang_vel, max_torque)

        # Twist damping: models real socket FRICTION, not a restoring torque toward any target
        # angle (any clock angle is a valid entry, issue #90 -- "rotation is free" means no
        # target, it was never meant to mean frictionless). Real contact friction is
        # Coulomb-like -- roughly CONSTANT magnitude opposing the direction of sliding,
        # largely independent of speed -- unlike a spring's velocity-proportional (viscous)
        # damping, which is the wrong shape for this: an earlier viscous-only version showed
        # a chaotic, non-monotonic equilibrium spin under real contact that got WORSE, not
        # better, when the gain was raised, evidence this needed a different force law rather
        # than a bigger version of the same one. twist_direction supplies the sign smoothly
        # (tanh over twist_deadband, a velocity scale far below any spin that matters) rather
        # than a literal sign(), the same chatter-avoidance reasoning as the axial term's
        # direction. Split from tilt's alignment torque so it gets its own, much larger budget
        # (max_twist_torque): PURE dissipation, safe to size larger than a restoring force
        # since it can only remove existing rotational energy, never fight a real contact
        # equilibrium the way a target-seeking torque would.
        twist_direction = torch.tanh(twist_rate / twist_deadband)
        twist_torque = (-twist_friction * twist_direction).clamp(-max_twist_torque, max_twist_torque)
        twist_torque_vec = twist_torque.unsqueeze(1) * axis_w

        forces = torch.where(seated_mask, axial_force_vec + lateral_force_vec, torch.zeros_like(axial_force_vec))
        torques = torch.where(seated_mask, tilt_torque_vec + twist_torque_vec, torch.zeros_like(tilt_torque_vec))
        bulb.set_external_force_and_torque(forces.unsqueeze(1), torques.unsqueeze(1), is_global=True)


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
