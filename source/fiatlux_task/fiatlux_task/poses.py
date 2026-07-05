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
HAND_FLAT: dict[str, float] = dict.fromkeys(
    (
        "R_index_proximal_joint",
        "R_index_intermediate_joint",
        "R_middle_proximal_joint",
        "R_middle_intermediate_joint",
        "R_ring_proximal_joint",
        "R_ring_intermediate_joint",
        "R_pinky_proximal_joint",
        "R_pinky_intermediate_joint",
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_intermediate_joint",
        "R_thumb_distal_joint",
    ),
    0.0,
)

# Bulb release orientation (w, x, y, z): lying on its side on the table (the
# ymomhw bulb is an elongated ~25 cm body; upright it topples). Long axis along
# world y so any settling roll runs along x, away from the near table edge.
BULB_LYING_QUAT: tuple[float, float, float, float] = (0.7071068, 0.7071068, 0.0, 0.0)

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
