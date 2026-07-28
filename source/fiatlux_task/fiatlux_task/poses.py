# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named, hand-calibrated joint poses for scripted interaction scenarios.

These are NOT policies: each entry is a static joint-position dict (exact joint
names, radians) used by ``scripts/verify_interactions.py`` to pose the G1 into a
known interaction configuration -- open hand cradling the bulb, fingers closed
around it, feet/hands on the step ladder -- with the root fixed
(``fix_root_link``) so no balance controller is needed.

Calibration workflow: run ``scripts/verify_interactions.py --scenario <s> --probe``,
which prints the palm/EE world pose and contact readings for the current
constants; iterate the values here until the scenario's contacts register, then
freeze them with a ``CALIBRATED`` note. The values are specific to the assets and
actuator gains they were tuned against -- when either changes, re-run the probes.

Joint name sources: ``robots/g1.py`` (``G1_ARM_JOINTS``, ``G1_HAND_JOINTS``).
"""

from .robots.g1 import G1_DEX3_RIGHT_HAND_JOINTS, G1_FINGER_JOINTS, G1_HAND_JOINTS, G1_THUMB_JOINTS

# ---------------------------------------------------------------------------
# Right arm, palm-down press poses over the bench table.
# CALIBRATED 2026-07-04 against g1_29dof_with_inspire_rev_1_0 + arm gains
# (stiffness 150 / damping 5), via `verify_interactions.py --scenario hand
# --probe` arm-grid sweep + `--video`: the palm SURFACE of right_hand_base_link
# is its local -x side (+x is the back of the hand). At a (0, 0, 0.85) fixed
# root, HOVER holds the downward-facing palm at ~(0.44, -0.15, 1.11) world,
# ~8 cm over the tabletop; PRESS/CRUSH lower it via shoulder pitch only.
# ---------------------------------------------------------------------------
# The press descends via WRIST pitch only: lowering the shoulder instead parks
# the forearm on the table edge (a fulcrum that blocks any deeper travel).
_ARM_PRESS_BASE: dict[str, float] = {
    "right_shoulder_pitch_joint": -1.35,
    "right_shoulder_roll_joint": -0.30,
    "right_shoulder_yaw_joint": 0.00,
    "right_elbow_joint": 1.20,
    # local -x (palm surface) facing down
    "right_wrist_roll_joint": -1.57,
    "right_wrist_yaw_joint": 0.00,
}
ARM_PRESS_HOVER: dict[str, float] = {**_ARM_PRESS_BASE, "right_wrist_pitch_joint": 0.00}
# gentle hold: palm target a few cm INSIDE the lying bulb -- arm PD compliance
# (stiffness 150) turns the blocked travel into a modest sustained press.
# Calibrated: much deeper tilts the palm face enough to nudge the egg away.
ARM_PRESS_DOWN: dict[str, float] = {**_ARM_PRESS_BASE, "right_wrist_pitch_joint": -0.60}
# deep press: the forearm/wrist wedges onto the kinematic slab edge via shoulder
# descent -- sustained force far past 50 N on the wrist bodies. (Wrist-only
# descent cannot do this: past ~45 deg the palm face turns sideways; and an
# egg-shaped bulb squirts out before a crush force can build on it.)
ARM_PRESS_CRUSH: dict[str, float] = {
    **_ARM_PRESS_BASE,
    "right_shoulder_pitch_joint": -0.55,
    "right_wrist_pitch_joint": 0.0,
}

# Fingers spread flat for the press (no curl -- the palm face does the work).
HAND_FLAT: dict[str, float] = dict.fromkeys(G1_HAND_JOINTS, 0.0)

# Right arm/hand, palm-UP cradle: the bulb held IN the hand, not pressed against the
# bench. CALIBRATED 2026-07-22 against the Omniverse bulb; re-probe if either changes.
# Constraints the values sit inside:
#   * wrist roll +1.57 faces the palm up (its surface is the hand link's local -x); the
#     palm still sits 9-13 deg off level, which no wrist joint removes;
#   * the hand must close around a bulb already in it -- an open palm holds nothing;
#   * curl is bounded on BOTH sides: below ~1.0 the bulb slips, at 1.3 the grip exceeds
#     the 50 N break threshold, and past 1.4 the closing fingers eject it;
#   * measure grip only with the material pinned (it is a startup randomization).
# The thumb tops out at 0.6 rad on its pitch joint, hence its own smaller target.
ARM_CRADLE: dict[str, float] = {**ARM_PRESS_HOVER, "right_wrist_roll_joint": 1.57}
HAND_CRADLE: dict[str, float] = {
    **dict.fromkeys(G1_FINGER_JOINTS, 1.0),
    **dict.fromkeys(G1_THUMB_JOINTS, 0.6),
}

# ---------------------------------------------------------------------------
# Dex3 equivalents. The benchmark scores this hand -- it is the only variant GR00T
# released a checkpoint for -- so the scenarios run on it and Inspire is the opt-in.
#
# A different hand, not a renaming: 3 digits, not 5, and every joint range is signed
# and asymmetric (thumb_2 is [-1.745, 0], so its curl is NEGATIVE while the fingers'
# is positive). Ranges, from the USD:
#   index_0/middle_0  [0, +1.571]   index_1/middle_1  [0, +1.745]
#   thumb_0 [-1.047, +1.047] (swings the thumb into opposition)
#   thumb_1 [-1.047, +0.611]   thumb_2 [-1.745, 0]
# ---------------------------------------------------------------------------
HAND_FLAT_DEX3: dict[str, float] = dict.fromkeys(G1_DEX3_RIGHT_HAND_JOINTS, 0.0)
_DEX3_CURL = 1.2
HAND_CRADLE_DEX3: dict[str, float] = {
    "right_hand_index_0_joint": _DEX3_CURL,
    "right_hand_index_1_joint": _DEX3_CURL,
    "right_hand_middle_0_joint": _DEX3_CURL,
    "right_hand_middle_1_joint": _DEX3_CURL,
    # thumb swung across the fingers, then curled onto the object
    "right_hand_thumb_0_joint": 0.9,
    "right_hand_thumb_1_joint": 0.3,
    "right_hand_thumb_2_joint": -_DEX3_CURL,
}

HAND_FLAT_BY_VARIANT: dict[str, dict[str, float]] = {"inspire": HAND_FLAT, "dex3": HAND_FLAT_DEX3}
HAND_CRADLE_BY_VARIANT: dict[str, dict[str, float]] = {"inspire": HAND_CRADLE, "dex3": HAND_CRADLE_DEX3}

# Bulb release orientation (w, x, y, z): lying on its side on the table (the
# ymomhw bulb is an elongated ~25 cm body; upright it topples). Long axis along
# world y so any settling roll runs along x, away from the near table edge.
BULB_LYING_QUAT: tuple[float, float, float, float] = (0.7071068, 0.7071068, 0.0, 0.0)
# Upright on its screw cap: the bulb's own stable axis, and the only orientation the hand
# retains -- laid across the fingers it is pinched against the palm and squirts out.
BULB_UPRIGHT_QUAT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# ---------------------------------------------------------------------------
# Ladder stance: FREE-root lean -- feet on the ground at the A-frame's base,
# torso pitched forward, both palms on a chest-height step. The force balance
# self-calibrates (hands carry a natural fraction of body weight); a welded
# root instead turns every mm of overlap into a kN wedge. Tuned against
# STEP_LADDER_USD at scene_cfg.LADDER_POSITION (1.6, 0, 0), steps facing -x.
# ---------------------------------------------------------------------------
LADDER_STANCE_ROOT_POS: tuple[float, float, float] = (1.00, 0.0, 0.74)
# forward pitch ~12 deg (w, x, y, z about +y)
LADDER_STANCE_ROOT_ROT: tuple[float, float, float, float] = (0.9945, 0.0, 0.1045, 0.0)
LADDER_STANCE_JOINTS: dict[str, float] = {
    # legs: standing, feet on the ground (stock standing bends)
    ".*_hip_pitch_joint": -0.05,
    ".*_knee_joint": 0.20,
    ".*_ankle_pitch_joint": -0.15,
    # arms: forward, palms pressed down onto a chest-height step
    "right_shoulder_pitch_joint": -0.55,
    "left_shoulder_pitch_joint": -0.55,
    "right_elbow_joint": 0.85,
    "left_elbow_joint": 0.85,
    "right_wrist_pitch_joint": -0.25,
    "left_wrist_pitch_joint": -0.25,
}
