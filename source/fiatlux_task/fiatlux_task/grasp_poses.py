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

# Ladder held by one rail, upright, alongside the robot's right leg and clear of it, at the
# moment ``ladder_grasped`` fires. Produced by S02, consumed by S03.
#
# UNCALIBRATED -- must be measured. Composed from: the ladder's root at its base centre with
# the base authored at z=0; the pelvis 0.79 m above the feet, so a base 0.06 m clear of the
# floor sits at -0.73; the base centre 0.62 m to the robot's right, which puts the near feet at
# y = -0.62 + 0.34 = -0.28, i.e. 0.13 m outside the right foot's outer edge; and yaw 90 deg so
# the A-frame's NARROW axis (0.68 m) spans left-right and its 1.11 m spread runs fore-aft. At
# yaw 0 the spread axis would put the near feet at y = -0.065, through the legs.
#
# Probe: a scripted one-rail grasp -- ``verify_interactions.py --scenario ladder_grasp --probe``
# (does not exist yet; it is S02's deliverable) -- closing the hand on a rail, lifting until all
# four feet clear the floor, then printing the in-root pose. Take it from the scripted grasp,
# never from a policy rollout.
#
# Open risk, not resolved here: at ``LADDER_MASS_KG`` = 7.25 a one-rail grasp may not hold, in
# which case this pose is two-handed and the whole geometry changes. That is a measurement, and
# S02 must report it rather than assume it.
LADDER_IN_ROOT_CARRIED: tuple[Vec3, Quat] = ((0.150, -0.620, -0.730), (0.707107, 0.0, 0.0, 0.707107))
