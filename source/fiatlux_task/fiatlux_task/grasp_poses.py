# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Held-payload poses and grip geometry, shared by the grasp/carry subtasks.

Frame convention: robot root (pelvis) frame, +x forward, +y left, +z up. Quaternions are
``(w, x, y, z)``. Each pose is ``(pos_xyz, quat_wxyz)``.

Root frame, because a reset event composes ``robot_root_pose (x) payload_in_root`` from
quantities known *before* physics steps: at reset the joint targets are written but not
simulated, so ``robot.data.body_pos_w`` still holds the previous episode's poses.

``UNCALIBRATED`` marks a pose that has not been through its probe.
"""

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# Bulb grip geometry (metres, in the bulb's own root frame, offsets along local +z)
# ---------------------------------------------------------------------------
# The bulb's root lies OUTSIDE its geometry (cap bottom at +0.036, i.e.
# ``assets.BULB_STAND_Z_OFFSET``), so a placement seats a feature and backs the root out along
# the axis.
BULB_CAP_RADIUS_M = 0.021
BULB_CAP_CENTRE_M = 0.054
BULB_GLASS_RADIUS_M = 0.040
BULB_GLASS_CENTRE_M = 0.131

# Contact bounds by the feature held. 50 N is also ``scripts/score.py``'s fragility threshold.
GLASS_CONTACT_LIMIT_N = 50.0
CAP_CONTACT_LIMIT_N = 300.0

# Where the bulb sits along the palm frame's fingers axis, per hand variant.
PALM_GRASP_FORWARD_M_BY_VARIANT: dict[str, float] = {"dex3": 0.045, "inspire": -0.148}

# Runaway-solver tripwire, not a fragility bound: a grip past this is the solver wedging the
# hand against a rail.
LADDER_GRIP_TRIPWIRE_N = 500.0

# ---------------------------------------------------------------------------
# Held-payload poses in the robot root frame
# ---------------------------------------------------------------------------

# Produced by S11, consumed by S07/S08/S12/S13/S14.
# CALIBRATED 2026-08-23 against ``poses.ARM_CRADLE`` + ``poses.HAND_CUP`` -- the pose its
# consumers actually spawn, not the closed ``HAND_CRADLE`` the previous value was derived
# against. Settled hold: 108 N on the cap, 1.5 cm of slip, contact in 150/150 steps.
BULB_IN_ROOT_STANDING: tuple[Vec3, Quat] = (
    (0.5556, -0.2051, 0.2879),
    (-0.393290, 0.473585, 0.619048, 0.487667),
)

# Consumed by S07 (old bulb) and S14 (fresh bulb). Kept as a separate name because those two
# import it as such; the value is shared.
BULB_IN_ROOT_ON_LADDER: tuple[Vec3, Quat] = BULB_IN_ROOT_STANDING

# Produced by S02, consumed by S03 and S04.
# UNCALIBRATED, and derived from the PREVIOUS ladder asset: the rails it was composed against do
# not exist on the current one, and renders show the ladder passing through the torso.
LADDER_IN_ROOT_CARRIED: tuple[Vec3, Quat] = ((-0.166, 0.0, -0.355), (0.707107, 0.0, 0.0, 0.707107))

# Inspire only: ``swap_robot_variant`` drops all 24 finger/thumb keys below on Dex3 and
# substitutes none, so a Dex3 rollout carries the ladder with an open hand. The two
# shoulder rolls are not hand-marker keys, so the arm pose itself is identical on both.
# Merge into ``robot.init_state.joint_pos`` rather than assigning over it.
# ``swap_robot_variant`` strips hand-marker keys foreign to the incoming variant
# (``robots.g1._drop_foreign_hand_joint_pos``).
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
