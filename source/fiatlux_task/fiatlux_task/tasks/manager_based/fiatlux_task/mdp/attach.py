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
  weight at the bore bottom and still 3.0x it at ``release_threshold`` (8 mm). Solving
  ``F(a) = weight`` puts the gravity crossing at 19.3 mm -- 2.4x the release threshold -- so on
  a ceiling mount gravity alone cannot walk the bulb out at all, let alone reach release. That
  is the margin the previous version lacked.
- **Zero outside the bore.** ``bore_depth`` (0.025 m) is measured, not chosen:
  ``scripts/measure_bore_geometry.py`` profiles both meshes in this seat frame and finds the
  socket's 20.2 mm throat spanning axial +29.1..+34.2 mm and the plug's guided body (r ~17.0
  -17.5 mm) spanning +9..+35.6 mm when home, so the plug clears the throat after ~25 mm of
  withdrawal. Past that there is no bore to be inside of and the whole wrench is zero. (With
  ``release_threshold`` at 8 mm the phase gate always fires first, so this bound is a safety
  net rather than an operating condition -- it binds only if someone configures a release
  threshold deeper than the bore.)

``release_threshold`` came down from 20 mm to 8 mm as part of this, and it had to: holding 2x
the weight all the way out to 20 mm while still vanishing by the bore mouth at 25 mm needs a
falloff slope of ~137 N/m, well past the semi-implicit stability ceiling ``mass/step_dt^2``
(~88 N/m at 50 Hz). 20 mm was also 80% of the way out of a 25 mm bore -- by then the bulb has
essentially left. At 8 mm the slope is 60 N/m (69% of the ceiling), and 8 mm is still twice
``seat_tolerance``, so a genuine withdrawal reads as one and a knock does not.

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
        release_threshold: float = 0.008,
        hold_force: float = 1.5,
        bore_depth: float = 0.025,
        spring_d: float = 1.5,
        max_force: float = 5.0,
        lateral_k: float = 5.0,
        lateral_d: float = 0.85,
        max_lateral_force: float = 1.0,
        tilt_k: float = 0.05,
        tilt_d: float = 0.01,
        max_torque: float = 0.05,
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
        #   hold_force 1.5 N   = 4.4x weight at the bore bottom, 3.0x at release_threshold.
        #                        Solving F(a) = weight puts the gravity crossing at 19.3 mm,
        #                        2.4x release_threshold -- so on a ceiling mount (worst case:
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
        #   release_threshold 0.008 m = down from 0.02. Holding 2x weight out to 20 mm while
        #                        still vanishing by the 25 mm bore mouth needs a ~137 N/m
        #                        slope, past the stability bound; and 20 mm was 80% of the way
        #                        out of the bore anyway. 8 mm is still 2x seat_tolerance, so a
        #                        real withdrawal reads as one and a knock does not.
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
        # tilt_k/tilt_d/max_torque have no inertia measurement behind them (unlike the axial
        # term's mass-derived critical damping); they are rough starting points at the same
        # order of magnitude as diagnose_contact_axial.py's own orientation-hold trial
        # (k_p=0.02, k_d=0.002) shown to function as a soft aligner there, needing the same
        # real-teleop retuning as every other gain in this term.
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
        old_bulb: RigidObject | None = env.scene["old_bulb"] if "old_bulb" in env.scene.keys() else None
        fresh_bulb: RigidObject = env.scene["fresh_bulb"]
        self._resolve_spawn_phase(env, old_bulb, fresh_bulb)
        gains = dict(
            radial_tolerance=radial_tolerance,
            tilt_tolerance=tilt_tolerance,
            seat_tolerance=seat_tolerance,
            release_threshold=release_threshold,
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
