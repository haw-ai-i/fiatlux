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

# ---------------------------------------------------------------------------
# Held-payload poses in the robot root frame
# ---------------------------------------------------------------------------

# Carried-bulb seat in the root frame. Produced by S08, consumed by S04/S05/S09/S10/S11.
# Baked to match ``settle_carried_payload_live``'s live-palm seat so the bulb sits on the palm from
# frame 1 (no teleport); the event still does the authoritative re-seat at step 1. Re-bake if the
# staged ``ARM_CRADLE`` wrist pose changes.
BULB_IN_ROOT_STANDING: tuple[Vec3, Quat] = (
    (0.4309, 0.0085, 0.1528),
    (0.483448, 0.522018, -0.468013, 0.524155),
)

# Consumed by S04 (old bulb) and S11 (fresh bulb). Kept as a separate name because those two
# import it as such; the value is shared.
BULB_IN_ROOT_ON_LADDER: tuple[Vec3, Quat] = BULB_IN_ROOT_STANDING

# MEASURED 2026-09-13 via a hand-probe/curl sweep (issue #196): the root-frame target the
# Dex3 palm's own grasp geometry produces under the same ``ARM_CRADLE`` the subtasks stage with.
BULB_IN_ROOT_STANDING_DEX3: tuple[Vec3, Quat] = (
    (0.2623, -0.0657, 0.0),
    (0.707107, 0.0, 0.0, 0.707107),
)
BULB_IN_ROOT_ON_LADDER_DEX3: tuple[Vec3, Quat] = BULB_IN_ROOT_STANDING_DEX3

BULB_IN_ROOT_STANDING_BY_VARIANT: dict[str, tuple[Vec3, Quat]] = {
    "inspire": BULB_IN_ROOT_STANDING,
    "dex3": BULB_IN_ROOT_STANDING_DEX3,
}
BULB_IN_ROOT_ON_LADDER_BY_VARIANT: dict[str, tuple[Vec3, Quat]] = {
    "inspire": BULB_IN_ROOT_ON_LADDER,
    "dex3": BULB_IN_ROOT_ON_LADDER_DEX3,
}
