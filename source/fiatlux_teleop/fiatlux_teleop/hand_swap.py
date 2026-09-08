"""Generic Inspire -> DEX3 hand swap for free-base (walking) teleop cfgs.

Any teleop task authored with Inspire hands can be driven with ``--hand dex3`` through this
one helper -- it is task-agnostic. The two cases it deliberately does NOT cover:

* Insert-Teleop: bolts the robot to the floor, so it needs the FIXED-base dex3 variant --
  it keeps its own ``insert_teleop_env_cfg.apply_dex3_hands``.
* Tasks already dex3-native (Carry/Gallery): nothing to swap -- the driver detects this
  from the robot USD path and skips the call.

``swap_robot_variant`` re-points reward/termination joint references along with the asset;
the binary grip actions are re-pointed here because the name remap does not cover their
open/close command expressions. Grip terms are discovered by shape (joint_names +
open_command_expr) rather than by attribute name, so any env's action layout works.
"""

from __future__ import annotations


def _repoint_grips(env_cfg, left_joints, left_open, left_grasp, right_joints, right_open, right_grasp):
    """Re-point every binary grip action (found by shape, not name) to the given hand's joints."""
    for _term in vars(env_cfg.actions).values():
        if not (hasattr(_term, "joint_names") and hasattr(_term, "open_command_expr")):
            continue  # not a binary grip (arm IK terms etc.)
        _left = any("left" in _j for _j in (_term.joint_names or []))
        _term.joint_names = list(left_joints if _left else right_joints)
        _term.open_command_expr = dict(left_open if _left else right_open)
        _term.close_command_expr = dict(left_grasp if _left else right_grasp)


def apply_dex3_hands(env_cfg) -> None:
    """Switch a *parsed*, Inspire-native, free-base teleop cfg to DEX3 hands, in place."""
    from fiatlux_task.robots.g1 import (
        G1_DEX3_HAND_GRASP,
        G1_DEX3_HAND_OPEN,
        G1_DEX3_LEFT_HAND_GRASP,
        G1_DEX3_LEFT_HAND_JOINTS,
        G1_DEX3_LEFT_HAND_OPEN,
        G1_DEX3_RIGHT_HAND_JOINTS,
        swap_robot_variant,
    )

    swap_robot_variant(env_cfg, "dex3")
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    _repoint_grips(env_cfg,
                   G1_DEX3_LEFT_HAND_JOINTS, G1_DEX3_LEFT_HAND_OPEN, G1_DEX3_LEFT_HAND_GRASP,
                   G1_DEX3_RIGHT_HAND_JOINTS, G1_DEX3_HAND_OPEN, G1_DEX3_HAND_GRASP)


def apply_inspire_hands(env_cfg) -> None:
    """Switch a *parsed*, Dex3-native teleop cfg to INSPIRE hands, in place (mirror of above)."""
    from fiatlux_task.robots.g1 import (
        G1_HAND_GRASP,
        G1_HAND_JOINTS,
        G1_HAND_OPEN,
        G1_LEFT_HAND_JOINTS,
        swap_robot_variant,
    )

    # left-hand open/grasp are the right-hand presets mirrored (same derivation as
    # insert_teleop_env_cfg's module constants; g1.py only ships the right-hand dicts)
    _left_open = dict.fromkeys(G1_LEFT_HAND_JOINTS, 0.0)
    _left_grasp = {k.replace("R_", "L_", 1): v for k, v in G1_HAND_GRASP.items()}
    swap_robot_variant(env_cfg, "inspire")
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    _repoint_grips(env_cfg,
                   G1_LEFT_HAND_JOINTS, _left_open, _left_grasp,
                   G1_HAND_JOINTS, G1_HAND_OPEN, G1_HAND_GRASP)
