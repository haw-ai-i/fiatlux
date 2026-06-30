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

from ..assets import G1_USD

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


# ---------------------------------------------------------------------------
# Articulation config (legged / free base, Inspire hand)
# ---------------------------------------------------------------------------

# The robot spawns standing for the insertion subtask, but keeps its legs so the
# same asset can locomote and climb in later roadmap subtasks.
G1_INSPIRE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_USD,
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
    # Spawn standing (matching Unitree's reference init): pelvis at ~0.75 m with the
    # legs slightly bent so the feet rest on the ground.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        # Only the bent leg joints are listed; every other joint defaults to 0.0.
        # (A ``".*"`` catch-all here would also match these and trip Isaac Lab's
        # one-regex-per-joint resolver in ``resolve_matching_names_values``.)
        joint_pos={
            ".*_hip_pitch_joint": -0.05,
            ".*_knee_joint": 0.2,
            ".*_ankle_pitch_joint": -0.15,
        },
    ),
    # Disjoint actuator groups covering every joint. Only the right arm + right hand
    # are driven by policy actions; the rest hold their standing pose. NOTE the arm
    # regex is anchored to shoulder/elbow/wrist so it does not also grab the right
    # *leg* joints (which also start with ``right_``).
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"],
            effort_limit_sim=300.0,
            stiffness=200.0,
            damping=10.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim=200.0,
            stiffness=200.0,
            damping=10.0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_(shoulder|elbow|wrist).*_joint"],
            effort_limit_sim=88.0,
            stiffness=40.0,
            damping=2.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["right_(shoulder|elbow|wrist).*_joint"],
            effort_limit_sim=88.0,
            stiffness=150.0,
            damping=5.0,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_.*_joint"],
            effort_limit_sim=100.0,
            stiffness=1000.0,
            damping=15.0,
        ),
    },
)
