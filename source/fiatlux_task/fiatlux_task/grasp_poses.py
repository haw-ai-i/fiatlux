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

**Every pose below is UNCALIBRATED.** They are geometric estimates composed from measured
constants (palm world position under ``ARM_CRADLE``, the bulb's own cap offsets, the ladder's
footprint), which is enough to build the envs and to render a start state, and not enough to
trust: a coded assertion that the payload is within a centimetre of the palm has passed twice
on this project while the render showed it pinched outside the hand.
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

# The ladder is not fragile; this bound is a runaway-solver tripwire, nothing else. A grip
# reading past it means the solver is wedging the hand against a rail, not that the ladder is
# about to break.
LADDER_GRIP_TRIPWIRE_N = 500.0

# ---------------------------------------------------------------------------
# Held-payload poses in the robot root frame
# ---------------------------------------------------------------------------

# Bulb held upright on its cap in the right hand, robot standing (``ARM_CRADLE`` +
# ``HAND_CRADLE_DEX3``). Consumed by S08 and S12 as their start state, and produced by S11.
#
# UNCALIBRATED -- must be measured. Composed from: the palm at ~(0.44, -0.15, 0.26) in the
# root frame (``poses.py``'s measured (0.44, -0.15, 1.11) world at a (0, 0, 0.85) root),
# ``PALM_GRASP_FORWARD_M`` = 0.045 along the fingers, the cap seated one ``BULB_CAP_RADIUS_M``
# off the palm face, and the root backed out ``BULB_CAP_CENTRE_M`` below the cap centre.
#
# Probe: ``verify_interactions.py --scenario hand --probe --robot dex3``, extended to print the
# bulb's root pose in the robot root frame (``quat_inv(root_quat) (x) (bulb_pos - root_pos)``)
# after the hold has settled. Today that probe prints the palm frame and the grasp pose only.
#
# Upright on the cap, NOT laid across the fingers: laid across, the bulb is pinched against
# the palm and squirts out (``poses.py``, ``BULB_UPRIGHT_QUAT``). Do not re-attempt that grasp.
BULB_IN_ROOT_STANDING: tuple[Vec3, Quat] = ((0.485, -0.150, 0.237), (1.0, 0.0, 0.0, 0.0))

# The same grasp with the on-ladder stance's torso lean applied. The bulb must stay upright in
# the WORLD while the root is pitched forward ~12 deg (``poses.LADDER_STANCE_ROOT_ROT``), so in
# the root frame it is counter-pitched -12 deg about +y. Consumed by S13.
#
# UNCALIBRATED -- must be measured. The lean and the arm cradle compose; do NOT assume
# ``BULB_IN_ROOT_STANDING`` is reusable with only the rotation swapped -- the seat offsets are
# taken along world up, which is no longer a root axis here.
#
# Probe: ``verify_interactions.py --scenario ladder --probe --robot dex3`` with the cradle arm/
# hand pose applied on top of ``LADDER_STANCE_JOINTS``, printing the same in-root pose.
BULB_IN_ROOT_ON_LADDER: tuple[Vec3, Quat] = ((0.487, -0.150, 0.240), (0.994522, 0.0, -0.104528, 0.0))

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
