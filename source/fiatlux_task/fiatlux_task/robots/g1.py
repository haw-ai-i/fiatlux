# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable Unitree G1 articulation config and joint-name constants.

This mirrors Isaac Lab's ``isaaclab_assets`` pattern: the robot is defined once,
task-agnostically, and tasks reference it via ``G1_INSPIRE_CFG.replace(...)``.
The joint-name and end-effector constants live here too, since they describe the
G1 itself rather than any particular task. A Dex3-hand variant slots in alongside
``G1_INSPIRE_CFG`` when needed (different USD + hand joint names).

The cfg deliberately leaves ``prim_path`` unset (``MISSING``); each scene supplies
it via ``.replace(prim_path=...)`` so the same robot can be reused across tasks.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone

from ..assets import G1_USD

# The Inspire hand is authored with collision meshes that interpenetrate their
# non-joint-connected neighbors at the default pose (PhysX adjacency filtering
# only exempts pairs sharing a joint): the camera housing sits 15-17 mm inside
# the wrist-pitch and palm colliders (~8-9 kN permanent wedge), and the thumb
# proximal link overlaps the palm across the thumb-yaw link (~300 N). With
# self-collisions enabled these saturate every contact reading on the hand.
# Filter exactly those pairs at spawn; a fixed-joint merge of the asset would
# cover only the camera housing and requires re-authoring the USD.
_G1_FILTERED_PAIRS = {
    "{side}_hand_camera_base_link": ("{side}_wrist_pitch_link", "{side}_hand_base_link"),
    "{S}_thumb_proximal": ("{side}_hand_base_link",),
}


@clone
def _spawn_g1_with_filtered_hand_mounts(prim_path, cfg, translation=None, orientation=None):
    from pxr import UsdPhysics

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    stage = prim.GetStage()
    for side in ("left", "right"):
        fmt = {"side": side, "S": side[0].upper()}
        for body, targets in _G1_FILTERED_PAIRS.items():
            api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/{body.format(**fmt)}"))
            rel = api.GetFilteredPairsRel()
            for target in targets:
                rel.AddTarget(f"{prim_path}/{target.format(**fmt)}")
    return prim


# ---------------------------------------------------------------------------
# Joint / body names (standard Unitree G1 naming)
# ---------------------------------------------------------------------------

# Right-arm joints used for the manipulation subtasks.
G1_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# Right Inspire-hand finger joints (12 DoF) so the policy can actually grasp.
G1_HAND_JOINTS = [
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
]
# End-effector body the wrist camera mounts on / eef pose is read from (exists in
# all G1 variants). The Inspire hand links hang off this via right_hand_palm_link.
G1_EE_BODY = "right_wrist_yaw_link"

# Climb-family limbs and joint groups (probe-verified against the Inspire USD via
# `verify_interactions.py --scenario ladder --probe`, 2026-07-06).
G1_FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
# The Inspire palm body; its surface is the local -x side (see fiatlux_task/poses.py).
G1_PALM_BODIES = ["left_hand_base_link", "right_hand_base_link"]
G1_TORSO_BODY = "torso_link"
# Joint-name patterns for reward scoping (match the actuator groups below).
G1_WAIST_JOINT_PATTERNS = ["waist_.*_joint"]
G1_FINGER_JOINT_PATTERNS = ["[LR]_.*_joint"]


# ---------------------------------------------------------------------------
# Actuator model
# ---------------------------------------------------------------------------

# Rotor inertia reflected through the gearbox, per Unitree motor type.
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

_LEG_STIFFNESS = {".*_hip_.*_joint": 150.0, ".*_knee_joint": 200.0, ".*_ankle_.*_joint": 40.0}
_LEG_DAMPING = {".*_hip_.*_joint": 2.0, ".*_knee_joint": 4.0, ".*_ankle_.*_joint": 2.0}
_LEG_EFFORT = {
    ".*_hip_yaw_joint": 88.0,
    ".*_hip_roll_joint": 139.0,
    ".*_hip_pitch_joint": 139.0,
    ".*_knee_joint": 139.0,
    ".*_ankle_.*_joint": 50.0,
}
_LEG_ARMATURE = {
    ".*_hip_yaw_joint": ARMATURE_7520_14,
    ".*_hip_roll_joint": ARMATURE_7520_22,
    ".*_hip_pitch_joint": ARMATURE_7520_22,
    ".*_knee_joint": ARMATURE_7520_22,
    ".*_ankle_.*_joint": 2.0 * ARMATURE_5020,
}
_ARM_STIFFNESS = {
    ".*_shoulder_pitch_joint": 100.0,
    ".*_shoulder_roll_joint": 100.0,
    ".*_shoulder_yaw_joint": 50.0,
    ".*_elbow_joint": 50.0,
    ".*_wrist_.*_joint": 20.0,
}
_ARM_DAMPING = {
    ".*_shoulder_.*_joint": 2.0,
    ".*_elbow_joint": 2.0,
    ".*_wrist_.*_joint": 1.0,
}
_ARM_EFFORT = {
    ".*_shoulder_.*_joint": 25.0,
    ".*_elbow_joint": 25.0,
    ".*_wrist_roll_joint": 25.0,
    ".*_wrist_pitch_joint": 5.0,
    ".*_wrist_yaw_joint": 5.0,
}
_ARM_ARMATURE = {
    ".*_shoulder_.*_joint": ARMATURE_5020,
    ".*_elbow_joint": ARMATURE_5020,
    ".*_wrist_roll_joint": ARMATURE_5020,
    ".*_wrist_pitch_joint": ARMATURE_4010,
    ".*_wrist_yaw_joint": ARMATURE_4010,
}

# ---------------------------------------------------------------------------
# Articulation config (legged / free base, Inspire hand)
# ---------------------------------------------------------------------------

# The robot spawns standing for the insertion subtask, but keeps its legs so the
# same asset can locomote and climb in later roadmap subtasks.
G1_INSPIRE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_USD,
        func=_spawn_g1_with_filtered_hand_mounts,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
        activate_contact_sensors=True,
    ),
    # Spawn standing (feet on the floor at the bent-knee pose; settled height 0.787 m).
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.79),
        # Only the bent leg joints are listed; every other joint defaults to 0.0.
        # (A ``".*"`` catch-all here would also match these and trip Isaac Lab's
        # one-regex-per-joint resolver in ``resolve_matching_names_values``.)
        joint_pos={
            ".*_hip_pitch_joint": -0.05,
            ".*_knee_joint": 0.2,
            ".*_ankle_pitch_joint": -0.15,
        },
    ),
    # Disjoint actuator groups covering every joint. NOTE the arm regex is anchored to
    # shoulder/elbow/wrist so it does not also grab the *leg* joints (which share the
    # left_/right_ prefix).
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"],
            effort_limit_sim=_LEG_EFFORT,
            stiffness=_LEG_STIFFNESS,
            damping=_LEG_DAMPING,
            armature=_LEG_ARMATURE,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim={"waist_yaw_joint": 88.0, "waist_(roll|pitch)_joint": 50.0},
            stiffness=250.0,
            damping=5.0,
            armature={
                "waist_yaw_joint": ARMATURE_7520_14,
                "waist_(roll|pitch)_joint": 2.0 * ARMATURE_5020,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_(shoulder|elbow|wrist).*_joint"],
            effort_limit_sim=_ARM_EFFORT,
            stiffness=_ARM_STIFFNESS,
            damping=_ARM_DAMPING,
            armature=_ARM_ARMATURE,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_.*_joint"],
            # Real Inspire fingers produce ~1-2 N.m; 2.0 is plenty for a 60 g bulb. The
            # old 100 N.m limit let a wedged finger's saturated PD torque catapult the
            # whole robot off furniture (hundreds of m/s -- verify_scene finding).
            effort_limit_sim=2.0,
            stiffness=1000.0,
            damping=15.0,
        ),
    },
)
