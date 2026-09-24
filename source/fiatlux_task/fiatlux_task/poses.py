# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named joint poses for scripted interaction scenarios.

Each entry is a static joint-position dict (exact joint names, radians), staged directly by
subtask cfgs to hold a hand/arm pose (e.g. ``ARM_CRADLE``/``HAND_CUP`` in S04/S09, to hold a
payload) as well as by diagnostic scripts.

Joint name sources: ``robots/g1.py`` (``G1_ARM_JOINTS``, ``G1_HAND_JOINTS``).
"""

from .robots.g1 import G1_DEX3_RIGHT_HAND_JOINTS, G1_FINGER_JOINTS, G1_HAND_JOINTS, G1_THUMB_JOINTS

# ---------------------------------------------------------------------------
# Right arm, over the bench table. The palm surface of right_hand_base_link is its local -x side.
# ---------------------------------------------------------------------------
_ARM_PRESS_BASE: dict[str, float] = {
    "right_shoulder_pitch_joint": -1.35,
    "right_shoulder_roll_joint": -0.30,
    "right_shoulder_yaw_joint": 0.00,
    "right_elbow_joint": 1.20,
    "right_wrist_roll_joint": -1.57,
    "right_wrist_yaw_joint": 0.00,
}
ARM_PRESS_HOVER: dict[str, float] = {**_ARM_PRESS_BASE, "right_wrist_pitch_joint": 0.00}
ARM_PRESS_DOWN: dict[str, float] = {**_ARM_PRESS_BASE, "right_wrist_pitch_joint": -0.60}
ARM_PRESS_CRUSH: dict[str, float] = {
    **_ARM_PRESS_BASE,
    "right_shoulder_pitch_joint": -0.55,
    "right_wrist_pitch_joint": 0.0,
}

HAND_FLAT: dict[str, float] = dict.fromkeys(G1_HAND_JOINTS, 0.0)

# Finger curl is bounded on both sides: below ~1.0 the bulb slips, at 1.3 the grip exceeds the
# 50 N break threshold, past 1.4 the closing fingers eject it. The thumb's pitch joint tops out
# at 0.6 rad.
#
# The carry arm mirrors the RESTING left arm (just the wrist rolls palm-up to cradle the bulb) so
# the payload rides at the torso. The old bench press-hover pose held it extended at chest height,
# which staged the arm/bulb inside wall fixtures and knocked the settling robot off the ladder (#127).
ARM_CRADLE: dict[str, float] = {
    "right_shoulder_pitch_joint": -0.35,
    "right_shoulder_roll_joint": 0.00,
    "right_shoulder_yaw_joint": 0.00,
    "right_elbow_joint": 0.35,
    "right_wrist_roll_joint": 1.57,
    "right_wrist_pitch_joint": 0.00,
    "right_wrist_yaw_joint": 0.00,
}
HAND_CRADLE: dict[str, float] = {
    **dict.fromkeys(G1_FINGER_JOINTS, 0.9),
    **dict.fromkeys(G1_THUMB_JOINTS, 0.6),
}

# The most open palm that still retains the bulb. Below this it drops: at 0.44 the grip peaks
# at 210 N and the bulb is 29 cm gone within 3 s, at 0.35 it never registers contact at all.
# Higher curl only costs grip force (0.65 -> 179 N, 0.80 -> 269 N against the cap's 300 N bound).
# CALIBRATED 2026-08-23 via a hand-probe/curl sweep.
HAND_CUP: dict[str, float] = {
    **dict.fromkeys(G1_FINGER_JOINTS, 0.50),
    **dict.fromkeys(G1_THUMB_JOINTS, 0.333),
}

# ---------------------------------------------------------------------------
# Dex3: 3 digits, not 5, and every joint range is signed and asymmetric. Ranges:
#   index_0/middle_0  [0, +1.571]   index_1/middle_1  [0, +1.745]
#   thumb_0 [-1.047, +1.047]   thumb_1 [-1.047, +0.611]   thumb_2 [-1.745, 0]
# ---------------------------------------------------------------------------
HAND_FLAT_DEX3: dict[str, float] = dict.fromkeys(G1_DEX3_RIGHT_HAND_JOINTS, 0.0)
_DEX3_CURL = 1.2
_DEX3_CUP_CURL = 0.667  # issue #196
HAND_CRADLE_DEX3: dict[str, float] = {
    "right_hand_index_0_joint": _DEX3_CURL,
    "right_hand_index_1_joint": _DEX3_CURL,
    "right_hand_middle_0_joint": _DEX3_CURL,
    "right_hand_middle_1_joint": _DEX3_CURL,
    "right_hand_thumb_0_joint": 0.9,
    "right_hand_thumb_1_joint": 0.3,
    "right_hand_thumb_2_joint": -_DEX3_CURL,
}

def hand_cup_dex3(curl: float = _DEX3_CUP_CURL) -> dict[str, float]:
    return {
        "right_hand_index_0_joint": curl,
        "right_hand_index_1_joint": curl,
        "right_hand_middle_0_joint": curl,
        "right_hand_middle_1_joint": curl,
        "right_hand_thumb_0_joint": 0.9,
        "right_hand_thumb_1_joint": 0.3,
        "right_hand_thumb_2_joint": -curl,
    }


HAND_CUP_DEX3: dict[str, float] = hand_cup_dex3()

HAND_FLAT_BY_VARIANT: dict[str, dict[str, float]] = {"inspire": HAND_FLAT, "dex3": HAND_FLAT_DEX3}
HAND_CRADLE_BY_VARIANT: dict[str, dict[str, float]] = {"inspire": HAND_CRADLE, "dex3": HAND_CRADLE_DEX3}
HAND_CUP_BY_VARIANT: dict[str, dict[str, float]] = {"inspire": HAND_CUP, "dex3": HAND_CUP_DEX3}

# (w, x, y, z). The ymomhw bulb is an elongated ~25 cm body; upright it topples.
BULB_LYING_QUAT: tuple[float, float, float, float] = (0.7071068, 0.7071068, 0.0, 0.0)
BULB_UPRIGHT_QUAT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# ---------------------------------------------------------------------------
# Ladder stance, free root: a welded root turns every mm of overlap into a kN wedge, so the
# force balance is left to settle itself.
#
# Relative to the ladder, not in world coordinates. The previous values were world coordinates
# authored for one ladder at one yaw, and at the yaw the ladder actually spawns at they put the
# robot against its BRACE side -- so a scenario whose whole purpose is step contact was reading
# the rear frame. Compose with mdp.place_terms.lean_stance_against_ladder, which resolves the
# step-facing side from the ladder's own pose.
# ---------------------------------------------------------------------------
LADDER_STANCE_STANDOFF: float = 0.60  # m out from the ladder's root
LADDER_STANCE_PELVIS_Z: float = 0.74  # m
LADDER_STANCE_LEAN_DEG: float = 12.0  # forward pitch into the steps
LADDER_STANCE_JOINTS: dict[str, float] = {
    ".*_hip_pitch_joint": -0.05,
    ".*_knee_joint": 0.20,
    ".*_ankle_pitch_joint": -0.15,
    "right_shoulder_pitch_joint": -0.55,
    "left_shoulder_pitch_joint": -0.55,
    "right_elbow_joint": 0.85,
    "left_elbow_joint": 0.85,
    "right_wrist_pitch_joint": -0.25,
    "left_wrist_pitch_joint": -0.25,
}
