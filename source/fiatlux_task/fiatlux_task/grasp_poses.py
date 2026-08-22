# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Held-payload poses and grip geometry, shared by the grasp/carry subtasks.

A payload is three things and no more (there is no payload class): an entity name, a frozen
root pose **in the robot's root frame**, and the force bounds the thing tolerates. This module
holds the last two.

Why the root frame: a reset event composes ``robot_root_pose (x) payload_in_root`` from
quantities that are known *before* physics steps. Forward kinematics is not available there --
at reset the joint targets are written but not simulated, so ``robot.data.body_pos_w`` still
holds the previous episode's poses. Reading it put the bulb 13 cm from the palm once and 4 cm
off once.

Frame convention: robot root (pelvis) frame, +x forward, +y left, +z up, so the right hand's
payloads sit at negative y. Quaternions are ``(w, x, y, z)``. Each pose is
``(pos_xyz, quat_wxyz)``.

Calibration workflow, the same one ``poses.py`` documents: run the probe named at the constant,
iterate until the contacts register AND the rendered grasp shows the payload enclosed by the
digits, then replace the ``UNCALIBRATED`` note with ``CALIBRATED <date>``. Numbers here are
specific to the assets and actuator gains they were measured against.

**Poses marked UNCALIBRATED are geometric estimates**, composed from measured constants (palm
world position under ``ARM_CRADLE``, the bulb's own cap offsets, the ladder's footprint), which
is enough to build the envs and to render a start state, and not enough to trust: a coded
assertion that the payload is within a centimetre of the palm has passed twice on this project
while the render showed it pinched outside the hand. ``BULB_IN_ROOT_STANDING`` /
``BULB_IN_ROOT_ON_LADDER`` are the one pair actually run through the settled-hold probe end to
end (fingertips converge on the pin point, contact sustains after release, force stays under the
break limit) -- treat the rest as estimates until they get the same treatment.
"""

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# Bulb grip geometry (metres, in the bulb's own root frame, offsets along local +z)
# ---------------------------------------------------------------------------
# MEASURED via ``verify_interactions.py --scenario hand --probe``. The bulb's root lies
# OUTSIDE its geometry (cap bottom at +0.036, i.e. ``assets.BULB_STAND_Z_OFFSET``), so a
# placement seats a FEATURE and backs the root out along the axis.
#
# The cap is the grip feature, not the glass: the glass is 8 cm across against the Dex3
# thumb's 6.5 cm reach off the palm plane, so the thumb cannot close over it.
BULB_CAP_RADIUS_M = 0.021
BULB_CAP_CENTRE_M = 0.054
BULB_GLASS_RADIUS_M = 0.040
BULB_GLASS_CENTRE_M = 0.131

# What the bulb tolerates, by the feature held -- one number cannot serve both. Glass: a thin
# soda-lime shell, ~50-150 N under a hard fingertip; 50 N is also ``scripts/score.py``'s
# fragility threshold. Cap: a metal E26 shell, several hundred N, and it needs to be, since a
# realistic 0.3-0.5 N.m install torque at its 13 mm radius costs tens of N of grip.
GLASS_CONTACT_LIMIT_N = 50.0
CAP_CONTACT_LIMIT_N = 300.0

# How far out along the fingers the bulb sits, per hand -- MEASURED via
# ``verify_interactions.py --scenario hand --probe``'s CLOSED-fingertip printout, not derived.
# The two variants curl to opposite regions of the palm frame, not just different amounts: Dex3
# reaches forward of the palm origin (bounded by the thumb's 2.3 cm reach against 12.4 cm of
# fingertip); Inspire's curl sweeps the fingertips BACK past the palm origin -- a cradle, not a
# fingertip pinch -- centring near f=-0.148.
PALM_GRASP_FORWARD_M_BY_VARIANT: dict[str, float] = {"dex3": 0.045, "inspire": -0.148}

# The ladder is not fragile; this bound is a runaway-solver tripwire, nothing else. A grip
# reading past it means the solver is wedging the hand against a rail, not that the ladder is
# about to break.
LADDER_GRIP_TRIPWIRE_N = 500.0

# ---------------------------------------------------------------------------
# Held-payload poses in the robot root frame
# ---------------------------------------------------------------------------

# Bulb held lying ACROSS the curled fingers, cap gripped between fingers and thumb (``ARM_CRADLE``
# + ``HAND_CRADLE``, Inspire). Consumed by S07/S08/S12/S13/S14 as their start state, and produced
# by S11.
#
# CALIBRATED 2026-08-20 against ``verify_interactions.py --scenario hand --probe --robot inspire``:
# ``HAND_CRADLE``'s curled fingers form a cradle LOOP that gravity seats a lying object into --
# they do not hold a standing one. An earlier version of this constant held the bulb upright on
# its cap instead (matching how a person actually carries a bulb) and claimed the lying-across
# orientation "squirts out"; measured, it is the reverse -- upright slips out in under a second
# (0 N contact after the pin releases, 15 cm of drift in 3 s) while lying-across holds cleanly
# (146-226 N steady contact, 1.8-3.1 cm of slip, sustained the full 150-step hold, gentle release).
# Trust the measurement over the older comment.
#
# Measured directly (not composed from separate constants) via the settled-hold probe: teleport
# the bulb to ``palm_grasp_pose()``, ramp ``HAND_CRADLE`` closed, release the pin, run 150 steps,
# then read ``quat_inv(root_quat) (x) (bulb_pos - root_pos)`` and ``quat_inv(root_quat) * bulb_quat``.
# Root-frame offsets are pose-invariant (composed back onto whatever root pose ``compose_carried_pose``
# is given), so this transfers directly from the fixed-root calibration rig onto a free, randomized
# root.
BULB_IN_ROOT_STANDING: tuple[Vec3, Quat] = (
    (0.5635, -0.1955, 0.2946),
    (-0.450560, 0.576195, 0.544127, 0.411001),
)

# Same grasp, same value. There is no longer a torso-lean correction to apply: the earlier
# on-ladder variant existed only because "upright" was a WORLD-frame constraint, which needed
# counter-pitching against ``poses.LADDER_STANCE_ROOT_ROT``'s ~12 deg forward lean. The
# lying-across grasp has no such constraint -- it is whatever ``ARM_CRADLE``/``HAND_CRADLE``
# (root-relative joint targets) naturally produce, which does not depend on the root's own
# orientation. Kept as a separate name because S07/S14 import it as such, not because the value
# differs. Consumed by S07 (old bulb) and S14 (fresh bulb).
BULB_IN_ROOT_ON_LADDER: tuple[Vec3, Quat] = BULB_IN_ROOT_STANDING

# Ladder held by BOTH rails, upright, symmetric in front of the robot, at the moment
# ``ladder_grasped`` fires. Produced by S02, consumed by S03 and S04. Two-handed by design, not
# just measurement: one rail cannot resist the gravity TORQUE about a single contact point (a
# one-hand version of this pose was tried and measured swinging/falling in
# ``verify_interactions``-style rollouts), and the Inspire fingers' torque limit (2.0 N.m,
# capped there after an earlier wedged-finger/saturated-PD catapult bug -- see ``robots.g1``)
# is not independently known to generate enough one-point friction to hold
# ``LADDER_MASS_KG`` = 7.25 at all. Two contact points removes the torque problem outright and
# roughly doubles the available normal-force budget before the actuator cap is even a question.
#
# UNCALIBRATED -- geometric estimate, not contact-verified. Composed the same way as the
# one-hand version it replaces, doubled: the rail pair's location off
# ``assets.STEP_LADDER_RIGID_USD``'s collision mesh (vertex-cloud clustering -- a rigid,
# non-foldable 4-rail A-frame, mirror-symmetric about its own local x=0), local
# (x=+-0.270, y=-0.408) at 0.50 m up from the base; and both hands' FK position with the arms
# abducted (``right_shoulder_roll_joint`` = -0.674 rad, mirrored) enough to span that rail
# spacing -- default (arms-at-side) spacing is only ~0.45 m against the rails' ~0.54 m, so some
# abduction is geometrically required, not a style choice: (0.241, +-0.270, 0.145) in the pelvis
# frame. Base ends up ~0.44 m off the floor.
#
# Probe: ``verify_interactions.py --scenario ladder_grasp --probe`` (does not exist yet; S02's
# deliverable) -- closing both hands on their rails, lifting until all feet clear the floor,
# then printing the in-root pose from the scripted grasp, never a policy rollout.
#
# Open risk: whether this grip actually holds without swinging past ``ladder_near_vertical``'s
# tolerance, and whether ``LADDER_CARRY_ARM_JOINT_POS`` (itself an uncalibrated 75%-closed
# guess, mirrored to both hands) wraps each rail rather than clipping through it.
LADDER_IN_ROOT_CARRIED: tuple[Vec3, Quat] = ((-0.166, 0.0, -0.355), (0.707107, 0.0, 0.0, 0.707107))

# Inspire-hand arm + finger pose for the two-handed carry above. Shoulder abduction is the part
# that actually matters (see ``LADDER_IN_ROOT_CARRIED``); fingers close 75% of each joint's own
# travel toward its upper limit, curl-positive, confirmed on the right hand by a rendered
# comparison against the untouched left hand before this was two-handed -- mirrored to
# ``L_*`` here on the (unverified but standard-for-a-mirrored-rig) assumption the left hand
# curls the same way. The same 75%-toward-upper-limit guess tried on Dex3's
# ``right_hand_*_joint`` rendered with the fingers still fully extended -- disconfirmed, so no
# Dex3 entries here; a Dex3 rollout keeps default (open) fingers and arms at the sides until
# someone works out its actual curl direction and re-derives the abduction angle for its own
# hand geometry.
#
# Merge into ``robot.init_state.joint_pos`` (regex-keyed, already carries the family's
# bent-knee entries) rather than assigning over it. Safe across a later ``--robot dex3`` swap:
# ``swap_robot_variant`` strips any hand-marker key foreign to the incoming variant before
# reattaching ``init_state`` (``robots.g1._drop_foreign_hand_joint_pos``).
LADDER_CARRY_ARM_JOINT_POS: dict[str, float] = {
    "right_shoulder_roll_joint": -0.674,
    "left_shoulder_roll_joint": 0.674,
    "R_index_proximal_joint": 1.275,
    "R_index_intermediate_joint": 1.275,
    "R_middle_proximal_joint": 1.275,
    "R_middle_intermediate_joint": 1.275,
    "R_pinky_proximal_joint": 1.275,
    "R_pinky_intermediate_joint": 1.275,
    "R_ring_proximal_joint": 1.275,
    "R_ring_intermediate_joint": 1.275,
    "R_thumb_proximal_yaw_joint": 0.95,
    "R_thumb_proximal_pitch_joint": 0.425,
    "R_thumb_intermediate_joint": 0.6,
    "R_thumb_distal_joint": 0.9,
    "L_index_proximal_joint": 1.275,
    "L_index_intermediate_joint": 1.275,
    "L_middle_proximal_joint": 1.275,
    "L_middle_intermediate_joint": 1.275,
    "L_pinky_proximal_joint": 1.275,
    "L_pinky_intermediate_joint": 1.275,
    "L_ring_proximal_joint": 1.275,
    "L_ring_intermediate_joint": 1.275,
    "L_thumb_proximal_yaw_joint": 0.95,
    "L_thumb_proximal_pitch_joint": 0.425,
    "L_thumb_intermediate_joint": 0.6,
    "L_thumb_distal_joint": 0.9,
}
