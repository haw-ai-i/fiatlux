"""Teleop variants of the 15 benchmark subtasks, from one recipe.

Every subtask (``FIATLUX-S01..S15-*-v0``) derives from the same ``SubtaskEnvCfg`` and uses the
same RL action space (``joint_pos``: whole-body joint targets). They differ only in scene
staging, success gates and rewards -- so a teleop twin is the SAME three swaps every hand-written
teleop cfg does, and is applied here generically rather than copied 15 times:

1. **actions** -> bimanual arm IK + binary grip (legs/waist stay SONIC's, driven by the driver)
2. **XR** -> pelvis-anchored follow camera + the ``controller_rel`` device with the four
   retargeters (arm pose + grip, per hand)
3. **terminations** -> all disabled, so a session is operator-paced and never auto-resets

Each subtask has its own thin cfg file in ``fiatlux_teleop/subtasks/`` that subclasses the RL
cfg and calls :func:`apply_subtask_teleop` -- mirroring how the benchmark writes its subtasks
(explicit file per task, shared behaviour in a common module). Put per-task teleop tweaks in
those files; keep this recipe generic.

Ids are the subtask id with ``-Teleop-v0``, e.g. ``FIATLUX-S05-ClimbLadder-Teleop-v0``, and they
run on the existing driver unchanged:

    FIATLUX_TASK=FIATLUX-S05-ClimbLadder-Teleop-v0 bash scripts/teleop/restart_sonic_teleop.sh
"""

from __future__ import annotations

import os

from fiatlux_task.robots.g1 import (
    swap_robot_variant,
    G1_ARM_JOINTS,
    G1_DEX3_HAND_GRASP,
    G1_DEX3_HAND_OPEN,
    G1_DEX3_LEFT_HAND_GRASP,
    G1_DEX3_LEFT_HAND_JOINTS,
    G1_DEX3_LEFT_HAND_OPEN,
    G1_DEX3_RIGHT_HAND_JOINTS,
    G1_EE_BODY,
    G1_HAND_GRASP,
    G1_HAND_JOINTS,
    G1_HAND_OPEN,
    G1_LEFT_HAND_JOINTS,
)
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import XrAnchorRotationMode, XrCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.utils import configclass

# left-arm joint/EE names live in the Insert teleop module (same source Carry uses)
from .insert_teleop_env_cfg import (
    G1_LEFT_ARM_JOINTS,
    G1_LEFT_EE_BODY,
    G1_LEFT_HAND_GRASP,
    G1_LEFT_HAND_OPEN,
)
from .xr_controller_retargeters import (
    ControllerGripperRetargeterCfg,
    Se3RelControllerRetargeterCfg,
)

# (task id stem, module under fiatlux_task...subtasks, cfg class)
SUBTASKS: tuple[tuple[str, str, str], ...] = (
    ("FIATLUX-S01-ApproachLadder", "s01_approach_ladder_env_cfg", "S01ApproachLadderEnvCfg"),
    ("FIATLUX-S02-GrabLadder", "s02_grab_ladder_env_cfg", "S02GrabLadderEnvCfg"),
    ("FIATLUX-S03-CarryLadder", "s03_carry_ladder_env_cfg", "S03CarryLadderEnvCfg"),
    ("FIATLUX-S04-PlaceLadder", "s04_place_ladder_env_cfg", "S04PlaceLadderEnvCfg"),
    ("FIATLUX-S05-ClimbLadder", "s05_climb_ladder_env_cfg", "S05ClimbLadderEnvCfg"),
    ("FIATLUX-S06-RemoveOldBulb", "s06_remove_old_bulb_env_cfg", "S06RemoveOldBulbEnvCfg"),
    ("FIATLUX-S07-DescendWithBulb", "s07_descend_with_bulb_env_cfg", "S07DescendWithBulbEnvCfg"),
    ("FIATLUX-S08-CarryBulbToDisposal", "s08_carry_bulb_to_disposal_env_cfg",
     "S08CarryBulbToDisposalEnvCfg"),
    ("FIATLUX-S09-DisposeBulb", "s09_dispose_bulb_env_cfg", "S09DisposeBulbEnvCfg"),
    ("FIATLUX-S10-ApproachNewBulb", "s10_approach_new_bulb_env_cfg", "S10ApproachNewBulbEnvCfg"),
    ("FIATLUX-S11-GrabNewBulb", "s11_grab_new_bulb_env_cfg", "S11GrabNewBulbEnvCfg"),
    ("FIATLUX-S12-CarryBulbToLadder", "s12_carry_bulb_to_ladder_env_cfg",
     "S12CarryBulbToLadderEnvCfg"),
    ("FIATLUX-S13-ClimbWithBulb", "s13_climb_with_bulb_env_cfg", "S13ClimbWithBulbEnvCfg"),
    ("FIATLUX-S14-ScrewInBulb", "s14_screw_in_bulb_env_cfg", "S14ScrewInBulbEnvCfg"),
    ("FIATLUX-S15-ClimbDown", "s15_climb_down_env_cfg", "S15ClimbDownEnvCfg"),
)

_IK = DifferentialIKControllerCfg(
    command_type="pose", use_relative_mode=False, ik_method="dls",
    # heavier DLS damping so the arm relaxes to a natural rest instead of holding the elbow
    # tucked at 90 deg (matches Insert/Carry teleop)
    ik_params={"lambda_val": 0.05},
)


# Grip joint sets per hand variant. The subtask's OWN hand is kept -- see apply_subtask_teleop.
_GRIPS = {
    "inspire": (G1_HAND_JOINTS, G1_HAND_OPEN, G1_HAND_GRASP,
                G1_LEFT_HAND_JOINTS, G1_LEFT_HAND_OPEN, G1_LEFT_HAND_GRASP),
    "dex3": (G1_DEX3_RIGHT_HAND_JOINTS, G1_DEX3_HAND_OPEN, G1_DEX3_HAND_GRASP,
             G1_DEX3_LEFT_HAND_JOINTS, G1_DEX3_LEFT_HAND_OPEN, G1_DEX3_LEFT_HAND_GRASP),
}


def _make_actions_cfg(hand: str):
    """Bimanual absolute-EE-pose IK + binary grip, for the hand the robot actually has."""
    rj, ro, rg, lj, lo, lg = _GRIPS[hand]

    @configclass
    class _Cfg:
        arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=G1_ARM_JOINTS, body_name=G1_EE_BODY,
            controller=_IK, scale=1.0,
        )
        hand_action = BinaryJointPositionActionCfg(
            asset_name="robot", joint_names=list(rj),
            open_command_expr=dict(ro), close_command_expr=dict(rg),
        )
        left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=G1_LEFT_ARM_JOINTS, body_name=G1_LEFT_EE_BODY,
            controller=_IK, scale=1.0,
        )
        left_hand_action = BinaryJointPositionActionCfg(
            asset_name="robot", joint_names=list(lj),
            open_command_expr=dict(lo), close_command_expr=dict(lg),
        )

    return _Cfg()


def apply_subtask_teleop(cfg) -> None:
    """Convert a *parsed* subtask cfg into its teleop twin, in place."""
    # HAND VARIANT: default to the one the subtask authored, override with FIATLUX_TELEOP_HAND
    # (the driver sets it from --hand). The swap must happen HERE, before the action terms are
    # built below, so the grips are authored against the target hand's joints rather than
    # remapped after the fact.
    #
    # This used to be pinned to the task's own hand because the benchmark's grasp terms
    # hardcoded the Inspire palm body, which broke every grasp subtask under Dex3 with
    # "Not all regular expressions are matched: right_hand_base_link". `mdp/grasp_terms.py`
    # now resolves the palm from the mounted articulation, so either hand works.
    native = "dex3" if "dex3" in str(getattr(cfg.scene.robot.spawn, "usd_path", "") or "") else "inspire"
    hand = os.environ.get("FIATLUX_TELEOP_HAND", native).lower()
    if hand not in ("inspire", "dex3"):
        raise ValueError(f"FIATLUX_TELEOP_HAND must be 'inspire' or 'dex3', got {hand!r}")
    if hand != native:
        swap_robot_variant(cfg, hand)
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False

    cfg.actions = _make_actions_cfg(hand)

    # Operator-paced: disable the FAILURE and timeout terms (falls, drops, tipped ladders,
    # time_out) so a recoverable mistake does not end the take. Subtasks each declare their own
    # set, so clear by exclusion rather than naming them.
    #
    # KEEP `success`. It is not just a stopping rule -- it is the only place the task's success
    # predicate is evaluated, and `recording.term_flag` records it per step as `success_term`,
    # which is what `scripts/score.py` reads to decide whether an episode succeeded. Clearing it
    # made term_flag fall back to its absent-term default (an all-False vector), so every teleop
    # demo scored success_rate 0.0 no matter how well it was performed. Ending the episode when
    # the operator achieves the goal is also the natural boundary for a recorded take.
    for _name in list(vars(cfg.terminations)):
        if _name.startswith("_") or _name == "success":
            continue
        if getattr(cfg.terminations, _name, None) is not None:
            setattr(cfg.terminations, _name, None)

    # Follow camera: anchor the XR view to the pelvis so it tracks the robot as SONIC walks it
    # (z offset drops the play-space floor so the operator's eyes land at head height).
    cfg.xr = XrCfg(
        anchor_pos=(0.0, 0.0, -0.9),
        anchor_rot=(0.0, 0.0, 0.0, 1.0),
        anchor_prim_path="/World/envs/env_0/Robot/pelvis",
        fixed_anchor_height=True,
        anchor_rotation_mode=XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
    )

    rp = cfg.scene.robot.init_state.pos
    cfg.teleop_devices = DevicesCfg(devices={
        "controller_rel": OpenXRDeviceCfg(
            retargeters=[
                Se3RelControllerRetargeterCfg(
                    bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                    root_pos=(rp[0], rp[1], rp[2] - 0.05), sim_device=cfg.sim.device),
                ControllerGripperRetargeterCfg(
                    bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=cfg.sim.device),
                Se3RelControllerRetargeterCfg(
                    bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                    root_pos=(rp[0], rp[1], rp[2] - 0.05), sim_device=cfg.sim.device),
                ControllerGripperRetargeterCfg(
                    bound_hand=DeviceBase.TrackingTarget.HAND_LEFT, sim_device=cfg.sim.device),
            ],
            sim_device=cfg.sim.device, xr_cfg=cfg.xr,
        ),
    })

