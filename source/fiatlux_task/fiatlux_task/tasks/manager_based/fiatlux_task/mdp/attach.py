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

Two independent diagnostics -- one for twist, one for axial insertion
(``scripts/diagnose_contact_axial.py``) -- found that with collision genuinely enabled, real
contact geometry blocked BOTH the twist-release and axial-insertion motions the bayonet assumed
were unobstructed -- the plug's radius was equal to or larger than the bore at every relevant
height. Shrinking the plug (2026-09-07) fixed that, and a rerun under real contact (2026-09-08)
confirmed insertion now works given reasonable orientation guidance. That result removed the
reason for the bayonet's own existence: with collision back on, the socket's geometry confines
the bulb laterally and angularly on its own -- nothing scripted has to. There is also no
physical lug or groove in this asset (a plain round bore), so the twist/lock semantics were
never modeling a real feature, only a scripted one.

What real contact still cannot provide is RETENTION: nothing stops the bulb sliding back out of
a plain round bore under gravity or a knock. This module supplies exactly that, and nothing
else. Two states per bulb, per env:

- ``FREE`` -- unconstrained rigid body; physics owns it entirely.
- ``SEATED`` -- held by a continuous WRENCH (world-frame, applied through
  ``set_external_force_and_torque``), not a pose or velocity overwrite: a full-strength axial
  magnet pulling the plug to the bottom of the bore, plus a deliberately much gentler lateral
  + tilt centering term (issue #171). The wrench is gated on being IN THE BORE as well as
  SEATED -- a magnet in the bore bottom acts on a plug inside that bore and nothing else.

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

**Axial retention is a magnet at the bottom of the bore (issue #171, revised twice)**: this
models a magnetic catch, so the force is an ATTRACTION toward the bottom of the bore --
strongest there, fading as the plug withdraws, and absent once the plug is out of the bore
altogether. Two earlier shapes were wrong, in opposite ways, and both are worth recording
because the second failure is subtle.

A first pass used a linear spring (``F = -k*axial``): weakest exactly at the seat, growing the
farther out you pull. Backwards for a magnet, which is strongest at contact. It also makes
removal progressively harder the more you pull, where a magnet gives a firm breakaway right at
contact and then gets easier.

The second pass fixed the shape but not the ARITHMETIC: ``F = -sign(axial) * hold_force /
(1 + |axial|/hold_range)`` with ``hold_force = 0.5 N``, ``hold_range = 0.01 m``. Because that
magnitude DECREASES with distance while gravity does not, its balance point against gravity is
an UNSTABLE one -- inside it the bulb is pulled home, outside it gravity wins and the bulb
accelerates out, with nothing to catch it. At those gains the peak was only 1.46x the bulb's
0.343 N weight and the crossing sat at 4.6 mm. The version of this docstring that shipped it
described that 4.6 mm as a "worst-case steady-state sag" sitting safely inside a 20 mm
``release_threshold``; it is not a sag the bulb settles at, it is a cliff edge, and on a
ceiling mount at the reported seed the bulb went over it and fell out in 0.64 s. A decaying
attraction is only a retention mechanism if it stays above the load it has to hold across the
whole travel where it is supposed to hold -- otherwise the shape is right and the mechanism
still does not work.

The law now is::

    depth = max(axial, 0)  # outward travel only
    falloff = max(1 - depth / bore_depth, 0)  # 1 at the bore bottom, 0 at the mouth
    F = (-hold_force * falloff - spring_d * axial_rate) * seat_axis  # always inward

Three properties, each load-bearing:

- **Always inward, never signed.** The magnitude never changes direction, so unlike the old
  ``sign()``/``tanh()`` construction there is no direction flip to chatter across, at the seat
  or anywhere else. What stops the plug going deeper is real contact bottoming it out, exactly
  as a real plug bottoms out on a real magnet's face -- the attraction is then balanced by the
  contact normal force, and the bulb sits at ``axial`` ~ 2 mm rather than at a force
  equilibrium. ``depth`` clamps at 0 so being pressed slightly past the seat reference does not
  weaken the hold.
- **Above the load everywhere it has to hold.** ``hold_force`` (1.5 N) is 4.4x the bulb's
  weight at the bore bottom and still 1.75x it at ``release_threshold`` (15 mm). Solving
  ``F(a) = weight`` puts the gravity crossing at 19.3 mm, outside the release threshold, so on
  a ceiling mount gravity alone cannot walk the bulb out at all, let alone reach release. That
  is the margin the previous version lacked. (The threshold was 8 mm, where the same law gives
  3.0x weight, until the brush finding below traded some of that margin for reach.)
- **Zero outside the bore.** ``bore_depth`` (0.025 m) is measured, not chosen:
  ``scripts/measure_bore_geometry.py`` profiles both meshes in this seat frame and finds the
  socket's 20.2 mm throat spanning axial +29.1..+34.2 mm and the plug's guided body (r ~17.0
  -17.5 mm) spanning +9..+35.6 mm when home, so the plug clears the throat after ~25 mm of
  withdrawal. Past that there is no bore to be inside of and the whole wrench is zero. (With
  ``release_threshold`` at 15 mm the phase gate still fires first, so this bound is a safety
  net rather than an operating condition -- it binds only if someone configures a release
  threshold deeper than the bore.)

``release_threshold`` came down from 20 mm to 8 mm as part of this, and it had to at the time:
holding 2x the weight all the way out to 20 mm while still vanishing by the bore mouth at 25 mm
needs a falloff slope of ~137 N/m, well past the semi-implicit stability ceiling
``mass/step_dt^2`` (~88 N/m at 50 Hz). What that reasoning fixed was the SLOPE; the threshold
itself only has to sit where the magnet can still hold the load, which is the gravity crossing
at 19.3 mm, not the 2x-weight point at 8 mm. It went back out to 15 mm for the brush finding
below, at an unchanged 60 N/m slope (69% of the ceiling).

``spring_d`` (1.5 N*s/m) is capped by ``mass / step_dt`` = 1.75, NOT by critical damping for
the 60 N/m slope (which would be 2.90): one control step of damping must not be able to
reverse the velocity it opposes, which is the same impulse-vs-momentum rule the twist-friction
bug below is a case study in. It is therefore deliberately under-damped on paper; real contact
supplies the rest of the dissipation.

Nothing here is gravity-aware or orientation-aware, on purpose: an earlier ceiling fix added a
feedforward that cancelled gravity's axial component outright, and that was rejected on review
because a passive mechanism doesn't know about gravity and null it out. The asymmetry is kept
-- what changed is that the hold is now strong enough that the asymmetry shows up as a
slightly different contact pressure rather than as the bulb leaving.

**No twist torque at all, and why one was tried twice and removed (issue #171, third
finding)**: a ceiling-mounted, seated bulb was found "spinning" about the seat axis at 1-19
rad/s for a sustained ~2.9s -- present from the moment it seats, never decaying, then abruptly
destabilizing into a real ejection (axial and tilt blowing up together) with no operator and
no hand contact the entire time. Two fixes were attempted, on the theory that twist was
under-damped: first a viscous term (``-twist_d * twist_rate``) split off from tilt's damping
so it could have its own, much larger torque budget than tilt's deliberately tiny
``max_torque``; then, when that settled into a nonzero equilibrium spin that got WORSE as its
gain rose, a Coulomb-style constant-magnitude friction (``-twist_friction *
tanh(twist_rate/twist_deadband)``, ``twist_friction = 0.5 N*m``).

Both were wrong, and the second one WAS the bug. A term-by-term ablation of this whole wrench
(``scripts/ablate_attach_forces.py``, seed 3, forced ceiling mount, zero action, no injected
spin) found that zeroing ``twist_friction`` -- and no other term -- removes the spin: 95-97
rad/s with it, 0.7-3.0 rad/s without, holding 4/5 repeats instead of 0/5, while every other
single-term ablation still spun at ~95 rad/s. With NOTHING applied at all the bulb shows only
~1.2 rad/s, so the spin was never a contact phenomenon this term failed to suppress; the term
was generating it.

The mechanism is a unit-scale error, and it is instructive. This bulb's moment of inertia
about the seat axis is ``2.9e-05 kg*m^2`` -- four orders of magnitude below its 0.035 kg mass,
because the plug is only ~18 mm in radius. A wrench set through
``set_external_force_and_torque`` persists for the whole control step, so one application of
0.5 N*m changes the twist rate by ``0.5 / 2.9e-05 * 0.02 = 344 rad/s``, while the law reverses
sign whenever ``|twist_rate|`` crosses ``twist_deadband = 0.1 rad/s``. It therefore overshoots
zero by ~3400x the deadband every step, flips sign, and re-accelerates the other way: a
sign-flipping friction law is only dissipative if its impulse cannot exceed the momentum it is
opposing (``tau <= I*|omega|/dt``), and this one exceeded it by ~350x. Coulomb chatter, not
friction. The original teleop bag confirms this directly rather than by analogy -- twist
changes sign on 138 of 140 consecutive control steps at ±16 rad/s, a clean alternation at
exactly the 50 Hz control rate. What the operator saw as a spin was this alternation, and
a gain sweep tracks the predicted ``tau/I*dt`` overshoot closely at every magnitude tried
(0.05 -> 30 rad/s, 0.005 -> 3.7, 0.0015 -> 0.65).

So there is no twist term here now, in any form. Rotation about the seat axis is left entirely
to the socket's real contact friction, which -- verified, not assumed -- the asset already
supplies: ``assets/omniverse_bulb/LightBulb_collision.usda`` binds a physics material with
static friction 1.2 / dynamic 1.0 and it resolves at runtime onto all 2 bulb and all 8 socket
colliders. A scripted torque was never needed to provide what the material provides, and
issue #90's "rotation is free" (any clock angle is a valid entry, there is no lug or groove to
turn into) stands as written. Should a twist term ever be wanted again, it must bound its
impulse by ``I*|omega|/step_dt`` so friction cannot reverse the rotation it opposes.

Angular velocity is still SPLIT into its twist component (along the seat axis) and its
perpendicular tilt component, and tilt damping still uses only the latter. That split was the
one sound part of the first attempt: damping the full angular velocity through tilt's tiny
0.05 N*m budget meant any carried twist saturated that clamp every step, spending the whole
tilt-alignment budget on a rotation it had no authority over and no target for. Tilt's own
alignment torque and its (twist-free) damping are otherwise unchanged, still gentle.

**Wall-mount tilt was too weak to matter, and recovering it does not un-jam a bulb already
tilted too far (issue #171, fourth finding)**: ``a321061`` fixed the ceiling case, where gravity
acts along the seat axis. On a WALL mount the seat axis is roughly horizontal, so gravity instead
loads the lateral/tilt centering terms this docstring's earlier section calls "deliberately far
gentler than the axial term... rough starting points... needing the same real-teleop retuning as
every other gain." Real S11 insert teleop (wall mount, seed 2, 4 independent episodes) found
every one of 6 SEATED windows (0.2-5.7 s) settled with tilt parked at 0.15-0.26 rad against this
term's own 0.2 rad ``tilt_tolerance`` -- right at the boundary the seat admission gate uses, with
no margin -- and every window ended in a release triggered by a single-step axial jump
coincident with a hand-contact transient (see the release-debounce finding below), from which the
bulb could never re-seat because re-seating also needs ``tilt < tilt_tolerance`` and that budget
was already spent.

``scripts/verify_wall_hold.py`` (new) isolates whether that near-tolerance tilt is a PASSIVE
property of the mechanism or an artifact of teleop noise, the same isolation
``verify_ceiling_hold.py`` did for the axial defect: force a wall mount, spawn the bulb already
seated and perfectly aligned, and watch under gravity alone, no hand, no disturbance. That
baseline is NOT the bug -- a perfectly-aligned spawn holds tilt under 0.01 rad indefinitely at
the old gains. The bug shows up once the bulb is perturbed even slightly: a one-time bump of
just 0.05-0.10 rad (``--tilt_perturb``, applied about world Y after the aligned spawn, standing
in for a hand knock or an imperfect approach during insertion) does not decay back toward the
spawn tilt at the old gains -- it GROWS to 0.16-0.21 rad and sits there, matching the bags almost
exactly. A larger 0.15 rad bump is actively unstable and ejects the bulb outright at 0.1 s. So
the old tilt/lateral gains were not merely gentle, they failed to actually restore alignment
after any real disturbance at all, and their own docstring already flagged them as unvalidated.

Raising ``tilt_k``/``max_torque`` from 0.05 to 0.25 (``tilt_d`` from 0.01 to 0.003, both within
the stability bounds recorded next to the gains above) fixes RECOVERY from a small bump: the same
0.10 rad perturbation that grew to 0.21 rad at the old gains now decays to 0.07-0.10 rad instead.
But sweeping ``tilt_k``/``max_torque`` up to 0.5 against a LARGER starting tilt (0.15-0.19 rad,
already in the band the bags got stuck at) makes essentially no difference -- max tilt stays
within a few percent of the starting value regardless of how much torque authority is given.
That is real contact GEOMETRY, not gain magnitude: past some angle the plug mechanically wedges
against the bore rim (the same category of effect the module docstring's radial-clearance section
already documents for lateral -- "the wobble cannot be fixed by shrinking the gap" -- but for
tilt, and this time not fixable by a bigger torque either). So this fix raises the basin a small
disturbance can be recovered from (making it less likely insertion noise tips the bulb into the
jammed regime to begin with); it does not claim to rescue a bulb that is already jammed there.
That is the release-debounce fix's job, below, and a bulb sitting jammed but STILL SEATED (as
every evidence-bag window in fact was, for up to 5.7 s) is not itself a failure.

**Release was a single-frame position test despite being documented as sustained (issue #171,
fifth finding)**: `` release = was_seated & (axial.abs() > release_threshold) `` fires the instant
one control step reads past threshold, with no debounce -- despite this module's own long-standing
language calling ``release_threshold`` "a deliberate, sustained pull... so a light knock does not
release it but a genuine withdrawal does." That was aspirational, not implemented. Every one of
the 6 real S11 SEATED windows above ends the same way: axial sits at 5-8 mm (under the 8 mm
threshold) for the whole hold, then jumps past it in exactly ONE 20 ms step, coincident with the
hand's recorded contact force either spiking (up to 123 N) or dropping to ~0 N as it lets go --
never a multi-step trend. ``_release_streak`` now counts consecutive over-threshold steps per
bulb and only releases at ``release_debounce_steps`` (3, 60 ms) of them, resetting to 0 on any
step that reads back under threshold -- so a one-frame contact jolt cannot release the bulb, but
a real withdrawal (which stays past threshold for many steps as the hand keeps pulling) still
does, just 60 ms later.

**The magnet was switching off mid-brush, which is what actually cost the task (issue #171,
sixth finding)**: with the two fixes above in, real VR teleop confirmed the mechanism itself
holds -- the bulb seats and stays locked for up to 10.6 s. What still scored 0 is that a dex3
operator cannot let go cleanly. Across 33 s of seated time in four takes the fingers were fully
off the bulb for 0.6 s total (19-22 separate touches per take, up to 15 in one 6 s stretch), so
every hold ends in a brush, and the brush ended the hold.

The bags say precisely how. Over all 14 knock-outs, the excursion the brush actually caused
peaked at 8.9, 9.8, 9.9, 10.5, 10.8, 10.8, 11.2, 11.3, 11.6, 11.8, 12.9 and 13.8 mm -- twelve of
them, every one inside 15 mm. (The remaining two reached 78 and 139 mm: the hand had 32-36 N on
the bulb just beforehand and genuinely threw it out. Those must release, and still do.) And 11
of the 14 crossed the threshold with under 1 N of hand force on the bulb -- 7 of them with
exactly 0.00 N. The hand was already gone; the bulb was coasting on momentum from a brush that
had ended.

A magnet recaptures a coast like that. This one could not, because at 8 mm the phase had already
flipped to FREE and the whole wrench with it -- so the bulb crossed 8 mm still moving outward
(0.10 m/s at the crossing typically, 0.25 m/s worst) with nothing left to decelerate it, and
parked out at 10-14 mm where, had the magnet still been on, it would have been pulled home by
0.66-0.90 N, 1.9-2.6x its own weight. The defect was not too little holding force. It was
switching the holding force off in the middle of the excursion it existed to arrest.

So ``release_threshold`` goes 8 mm -> 15 mm, and nothing else changes: same law, same
``hold_force``, same 60 N/m slope, same debounce. What sets the ceiling on it is the magnet's own
gravity crossing at 19.3 mm -- past there the attraction really is weaker than the bulb's weight
and a ceiling-mounted bulb really is on its way out, so releasing is correct. 15 mm keeps 4.3 mm
of margin under that, still holds 1.75x weight at the threshold itself, is still well inside the
25 mm bore mouth, and is still ~4x ``seat_tolerance`` so seat and release stay firmly separated.
Pushing it further buys nothing measurable anyway: 18 mm and 20 mm avoid exactly the same 12
knock-outs 15 mm does, while spending the margin.

Deliberately NOT done, though the bags support it: gating release on the hand still being in
contact with the bulb (which would have caught 11 of the 14 on its own). It needs a contact
sensor this module does not take and would not have on every scene that wires it; it needs a
geometric override anyway, or a bulb flung 139 mm clear with no hand on it stays SEATED forever
and ``fresh_bulb_attached`` reports an installed bulb lying on the floor; and it would make a
passive detent's behaviour depend on whether an agent happens to be touching it, which is the
same objection that got the gravity feedforward rejected further up this docstring. The
threshold change covers all 12 recoverable cases without any of that. Raising ``hold_force``
was also rejected: it would lift the removal breakout past the 1.5 N that ``verify_no_twist_spin
.py --pull_n`` requires to still take the bulb OUT, trading an insert failure for a remove one.
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
        # Consecutive steps a SEATED bulb has read axial past release_threshold (issue #171,
        # wall-mount finding). See _advance's release-debounce comment.
        self._release_streak = torch.zeros(2, n, dtype=torch.int32, device=dev)
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
        self._release_streak[:, ids] = 0
        self._pending[ids] = True

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        radial_tolerance: float = 0.015,
        tilt_tolerance: float = 0.2,
        seat_tolerance: float = 0.004,
        release_threshold: float = 0.015,
        release_debounce_steps: int = 3,
        hold_force: float = 1.5,
        bore_depth: float = 0.025,
        spring_d: float = 1.5,
        max_force: float = 5.0,
        lateral_k: float = 5.0,
        lateral_d: float = 0.85,
        max_lateral_force: float = 1.0,
        tilt_k: float = 0.25,
        tilt_d: float = 0.003,
        max_torque: float = 0.25,
    ) -> None:
        # Always operates on all envs: retention must be enforced every step, which the zero
        # interval guarantees.
        #
        # hold_force/bore_depth (issue #171, revised twice): the axial term is a MAGNET AT THE
        # BOTTOM OF THE BORE -- an attraction that is strongest there, fades linearly as the
        # plug withdraws, and is gone once the plug has left the bore. See the module docstring
        # for the two shapes this replaces and why the second one (a decaying magnitude whose
        # gravity balance point is UNSTABLE) let a ceiling-mounted bulb fall out despite having
        # the right shape. Sizing, against the bulb's 0.035 kg / 0.343 N:
        #
        #   hold_force 1.5 N   = 4.4x weight at the bore bottom, 1.75x at release_threshold.
        #                        Solving F(a) = weight puts the gravity crossing at 19.3 mm,
        #                        outside release_threshold -- so on a ceiling mount (worst case:
        #                        inverted, full weight along the outward seat axis) gravity
        #                        alone cannot walk the bulb out, which is the whole point.
        #   bore_depth 0.025 m = MEASURED, not chosen (scripts/measure_bore_geometry.py): the
        #                        socket's 20.2 mm throat spans axial +29.1..+34.2 mm and the
        #                        plug's guided body spans +9..+35.6 mm when home, so the plug
        #                        clears the throat after ~25 mm of withdrawal. The attraction
        #                        reaches 0 exactly there because past it there is no bore for
        #                        the plug to be inside of.
        #   slope 60 N/m       = hold_force/bore_depth, the steepest this law ever gets. Under
        #                        the semi-implicit stability bound mass/step_dt^2 (~88 N/m at
        #                        step_dt=0.02 s, 50 Hz) with 31% to spare.
        #   spring_d 1.5       = capped by mass/step_dt (1.75), NOT by critical damping for
        #                        60 N/m (2.90): one control step of damping must not be able to
        #                        reverse the velocity it opposes. Same impulse-vs-momentum rule
        #                        the twist-friction bug violated by ~350x. Deliberately
        #                        under-damped on paper; real contact supplies the rest.
        #   release_threshold 0.015 m = 0.02 -> 0.008 -> 0.015. The 8 mm step fixed the SLOPE
        #                        problem (2x weight out to 20 mm needs ~137 N/m, past the
        #                        stability bound); it also switched the magnet off mid-brush,
        #                        which is what cost the insert task in real VR. Real bags: 12 of
        #                        14 knock-outs were finger-brushes parking at 8.9-13.8 mm, every
        #                        one inside 15 mm, with the hand already off the bulb (under 1 N)
        #                        at the crossing in 11 of the 14 -- coasts a still-engaged magnet
        #                        would have pulled home at 0.66-0.90 N. Bounded above by the
        #                        gravity crossing (19.3 mm): past there the attraction is under
        #                        the bulb's weight and releasing is correct. 18 and 20 mm avoid
        #                        exactly the same 12, so the extra reach buys nothing. Still
        #                        ~4x seat_tolerance, so seat and release stay well separated.
        #   release_debounce_steps 3 (issue #171, wall-mount finding) = requires axial to read
        #                        past release_threshold for 3 CONSECUTIVE steps (60 ms at the
        #                        50 Hz control rate) before releasing, not just one. Real S11
        #                        insert teleop (wall mount, seed 2, 4 episodes, 6 SEATED windows)
        #                        found every release was a single-step axial jump coincident
        #                        with a hand-contact transient (a grip spike up to 123 N, or the
        #                        hand losing contact entirely) -- axial sat at 5-8 mm for the
        #                        whole hold, then jumped past 8 mm in exactly one 20 ms step. A
        #                        genuine withdrawal stays past threshold for many steps as the
        #                        hand keeps pulling, so it is unaffected; a one-frame jolt is not.
        #
        # max_force 5 N is an overall safety clamp on the combined attraction + damping (the
        # attraction alone never exceeds hold_force by construction; this matters mainly for a
        # large damping contribution at high velocity). Nothing here is gravity-aware or
        # orientation-aware, on purpose -- see the docstring on the rejected feedforward.
        #
        # lateral_k/lateral_d/max_lateral_force (issue #171): deliberately far gentler than the
        # axial term -- capped at ~3x the bulb's own weight, critically damped the same way --
        # so this damps wobble without fighting real contact or acting as a second authority
        # over lateral position (real contact, not this term, is what actually has to bear a
        # sideways load). Kept as a plain spring-damper (not magnet-shaped): it is an assist to
        # real contact, not the primary retention the axial term's magnet law models.
        # tilt_k/max_torque 0.25 N*m/0.25 N*m, tilt_d 0.003 (issue #171, wall-mount finding,
        # up from 0.05/0.05/0.01): the original values were rough starting points, and real S11
        # insert teleop (wall mount) found them too weak to matter -- see the module docstring
        # for scripts/verify_wall_hold.py's measurements. Not sized by a closed-form margin like
        # the axial term (real bore contact dominates the tilt dynamics far more than it does the
        # axial ones, so this was tuned empirically against that script rather than derived), but
        # bounded the same way: 0.25 stays under the semi-implicit slope ceiling I_perp/step_dt^2
        # (~0.16-2.56 N*m/rad depending on whether the control or physics step is the relevant
        # one; 0.25 sits inside both readings tried) and tilt_d 0.003 is near I_perp/step_dt
        # (~0.0032), the same impulse-vs-momentum cap the axial spring_d respects, using
        # I_perp ~= 6.4e-5 kg*m^2 (the tilt-plane principal moments, see the twist finding below).
        #
        # There is deliberately NO twist gain here (issue #171, third finding). A 0.5 N*m
        # Coulomb twist friction used to sit alongside these, and it turned out to BE the
        # reported ceiling "spin": against this bulb's 2.9e-05 kg*m^2 twist inertia, one
        # control step of it swings the twist rate by 344 rad/s while its sign flips at 0.1
        # rad/s, so it chattered instead of dissipating. See the module docstring for the
        # ablation and the bag evidence. Twist is left to the socket's real contact friction
        # (static 1.2 / dynamic 1.0, bound in the asset), which needs no help from here.
        del env_ids
        # old_bulb is absent from insert-only scenes (e.g. the tabletop preset, which has
        # nothing to remove) -- self._phase[_OLD] then never leaves its _FREE default, so
        # fresh_bulb's own socket_empty check below is correct with no further change.
        old_bulb: RigidObject | None = env.scene["old_bulb"] if "old_bulb" in env.scene.rigid_objects else None
        fresh_bulb: RigidObject = env.scene["fresh_bulb"]
        self._resolve_spawn_phase(env, old_bulb, fresh_bulb)
        # Shared by both bulbs on a two-bulb (Replace) task -- both sit at the SAME socket in
        # the SAME step, so compute this once here rather than once per _advance call.
        socket: RigidObject = env.scene["socket"]
        socket_quat = socket.data.root_quat_w
        axis_w = quat_apply(socket_quat, self._axis_l)
        seat = socket.data.root_pos_w + quat_apply(socket_quat, self._seat_offset)
        gains = dict(
            radial_tolerance=radial_tolerance,
            tilt_tolerance=tilt_tolerance,
            seat_tolerance=seat_tolerance,
            release_threshold=release_threshold,
            release_debounce_steps=release_debounce_steps,
            hold_force=hold_force,
            bore_depth=bore_depth,
            spring_d=spring_d,
            max_force=max_force,
            lateral_k=lateral_k,
            lateral_d=lateral_d,
            max_lateral_force=max_lateral_force,
            tilt_k=tilt_k,
            tilt_d=tilt_d,
            max_torque=max_torque,
        )
        if old_bulb is not None:
            self._advance(
                old_bulb, _OLD, socket_quat, axis_w, seat, socket_empty=self._phase[_FRESH] == _FREE, **gains
            )
        self._advance(fresh_bulb, _FRESH, socket_quat, axis_w, seat, socket_empty=self._phase[_OLD] == _FREE, **gains)
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
        socket_quat: torch.Tensor,
        axis_w: torch.Tensor,
        seat: torch.Tensor,
        socket_empty: torch.Tensor,
        radial_tolerance: float,
        tilt_tolerance: float,
        seat_tolerance: float,
        release_threshold: float,
        release_debounce_steps: int,
        hold_force: float,
        bore_depth: float,
        spring_d: float,
        max_force: float,
        lateral_k: float,
        lateral_d: float,
        max_lateral_force: float,
        tilt_k: float,
        tilt_d: float,
        max_torque: float,
    ) -> None:
        bulb_quat = bulb.data.root_quat_w
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
        #
        # DEBOUNCED (issue #171, wall-mount finding): real teleop evidence found every observed
        # release was a single control-STEP axial jump coincident with a hand-contact transient
        # (a grip spike or the hand losing contact entirely as it opened), not a sustained
        # withdrawal -- see the module docstring. The module's own long-standing description of
        # release_threshold as "a deliberate, sustained pull... so a light knock does not release
        # it" was aspirational until now: the check was a bare single-frame position test, so any
        # one-step contact jolt big enough already released it, exactly as those bags show.
        # `_release_streak` counts CONSECUTIVE steps a SEATED bulb has read past threshold and
        # only releases once that streak reaches `release_debounce_steps`; it resets to 0 the
        # instant a step reads back under threshold, so a momentary jolt cannot accumulate across
        # separate excursions. A genuine withdrawal stays past threshold for many steps as the
        # hand keeps pulling, so it still releases -- just `release_debounce_steps` steps later.
        # Directional, not absolute: only OUTWARD travel past release_threshold releases. axial
        # is signed positive outward (see the magnet law above, `depth = axial.clamp(min=0.0)`),
        # so `axial.abs()` here used to also release on excessive INWARD travel -- a hard crush
        # driving the plug past the seat reference in the other direction, which real contact is
        # supposed to arrest (see "what stops the plug going deeper" above), not this term. That
        # reintroduced the exact "retention switches off under a disturbance" failure this
        # debounce was built to fix, just from a crush instead of a withdrawal.
        streak = self._release_streak[row]
        over_threshold = was_seated & (axial > release_threshold)
        streak.copy_(torch.where(over_threshold, streak + 1, torch.zeros_like(streak)))
        release = over_threshold & (streak >= release_debounce_steps)
        streak[release] = 0

        phase[seat_now] = _SEATED
        phase[release] = _FREE

        # Continuous wrench, world-frame, applied alongside (never instead of) real contact --
        # zero everywhere a bulb is not seated, so a bulb that just released or was never
        # seated is completely untouched by this term (including mid-insertion: this cannot
        # affect the FREE-phase dynamics diagnose_contact_axial.py exercises).
        #
        # IN THE BORE is a second, geometric gate on top of the phase: a magnet sunk in the
        # bottom of a bore acts on a plug inside that bore and on nothing else, so the wrench
        # is zero once the plug has cleared the throat axially (`bore_depth`, measured -- see
        # the module docstring) or wandered outside it radially. With release_threshold (8 mm)
        # far inside bore_depth (25 mm) the phase gate always fires first in practice, and the
        # bore itself pins `lateral` to a couple of mm against a 15 mm radial_tolerance, so
        # neither half of this normally binds -- it is here so the force cannot outlive the
        # geometry that justifies it if either bound is ever reconfigured.
        in_bore = (axial < bore_depth) & (lateral < radial_tolerance)
        seated_now = (phase == _SEATED) & ~release & in_bore
        seated_mask = seated_now.unsqueeze(1)

        # Axial: the magnet. An ATTRACTION toward the bottom of the bore -- always inward,
        # never signed, strongest at the bottom, fading linearly to nothing at the bore mouth
        # (issue #171; see the module docstring for the two earlier shapes and the sizing).
        # `depth` clamps at 0 so being pressed slightly past the seat reference does not weaken
        # the hold, and because the magnitude never changes direction there is no sign flip to
        # chatter across -- what stops the plug going deeper is real contact bottoming it out,
        # the same way a plug bottoms out on a real magnet's face.
        depth = axial.clamp(min=0.0)
        falloff = (1.0 - depth / bore_depth).clamp(min=0.0)
        axial_rate = (bulb.data.root_lin_vel_w * axis_w).sum(dim=1)
        axial_force = (-hold_force * falloff - spring_d * axial_rate).clamp(-max_force, max_force)
        axial_force_vec = axial_force.unsqueeze(1) * axis_w

        # Lateral: a much gentler spring-damper pulling the plug back toward the seat axis
        # line (issue #171 -- see the module docstring for why this is a software fix, not a
        # tighter bore).
        lin_vel = bulb.data.root_lin_vel_w
        lateral_vel = lin_vel - (lin_vel * axis_w).sum(dim=1, keepdim=True) * axis_w
        lateral_force_vec = _clamp_vector_norm(-lateral_k * lateral_vec - lateral_d * lateral_vel, max_lateral_force)

        # Split angular velocity into its component along the seat axis (TWIST -- rotation
        # about the plug's own axis, which has no target angle, issue #90) and everything
        # perpendicular to it (TILT -- misalignment of the axis itself, which does have a
        # target: aligned with the socket's). Only the tilt part is damped below. Twist is
        # dropped here rather than damped: it gets no torque from this module at all, by
        # design (issue #171 -- see the module docstring; a 0.5 N*m Coulomb friction on this
        # axis was the cause of the reported ceiling spin, not a cure for it). Real socket
        # contact friction, which the asset genuinely carries, is what opposes twist.
        #
        # The split itself is still needed even with no twist term: damping the FULL angular
        # velocity would spend tilt's deliberately tiny max_torque budget opposing whatever
        # twist the bulb carries -- saturating the clamp every step on a rotation this module
        # has neither a target for nor authority over, and starving the tilt alignment it is
        # actually there to serve.
        ang_vel = bulb.data.root_ang_vel_w
        twist_rate = (ang_vel * axis_w).sum(dim=1)
        tilt_ang_vel = ang_vel - twist_rate.unsqueeze(1) * axis_w

        # Tilt: small torque aligning the plug axis back to the seat axis -- the standard
        # small-angle "rotate A onto B" construction (magnitude ~ sin(tilt), direction the
        # correct rotation axis) -- damped against ONLY the tilt component of angular velocity.
        plug_axis_w = quat_apply(bulb_quat, self._axis_l)
        tilt_correction = torch.cross(plug_axis_w, axis_w, dim=-1)
        tilt_torque_vec = _clamp_vector_norm(tilt_k * tilt_correction - tilt_d * tilt_ang_vel, max_torque)

        forces = torch.where(seated_mask, axial_force_vec + lateral_force_vec, torch.zeros_like(axial_force_vec))
        torques = torch.where(seated_mask, tilt_torque_vec, torch.zeros_like(tilt_torque_vec))
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
    read as clearance, since the detent (not a projection) still owns the axial
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
