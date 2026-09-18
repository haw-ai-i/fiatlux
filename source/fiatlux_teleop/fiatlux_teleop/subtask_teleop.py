"""Teleop variants of the 15 benchmark subtasks, from one recipe.

Every subtask (``FIATLUX-S01..S12-*-v0``) derives from the same ``SubtaskEnvCfg`` and uses the
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
    swap_robot_variant,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import add_ego_camera

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
    ("FIATLUX-S01-MoveLadder", "s01_move_ladder_env_cfg", "S01MoveLadderEnvCfg"),
    ("FIATLUX-S02-ClimbLadder", "s02_climb_ladder_env_cfg", "S02ClimbLadderEnvCfg"),
    ("FIATLUX-S03-RemoveOldBulb", "s03_remove_old_bulb_env_cfg", "S03RemoveOldBulbEnvCfg"),
    ("FIATLUX-S04-DescendWithBulb", "s04_descend_with_bulb_env_cfg", "S04DescendWithBulbEnvCfg"),
    ("FIATLUX-S05-CarryBulbToDisposal", "s05_carry_bulb_to_disposal_env_cfg", "S05CarryBulbToDisposalEnvCfg"),
    ("FIATLUX-S06-DisposeBulb", "s06_dispose_bulb_env_cfg", "S06DisposeBulbEnvCfg"),
    ("FIATLUX-S07-ApproachNewBulb", "s07_approach_new_bulb_env_cfg", "S07ApproachNewBulbEnvCfg"),
    ("FIATLUX-S08-GrabNewBulb", "s08_grab_new_bulb_env_cfg", "S08GrabNewBulbEnvCfg"),
    ("FIATLUX-S09-CarryBulbToLadder", "s09_carry_bulb_to_ladder_env_cfg", "S09CarryBulbToLadderEnvCfg"),
    ("FIATLUX-S10-ClimbWithBulb", "s10_climb_with_bulb_env_cfg", "S10ClimbWithBulbEnvCfg"),
    ("FIATLUX-S11-ScrewInBulb", "s11_screw_in_bulb_env_cfg", "S11ScrewInBulbEnvCfg"),
    ("FIATLUX-S12-ClimbDown", "s12_climb_down_env_cfg", "S12ClimbDownEnvCfg"),
)

_IK = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=False,
    ik_method="dls",
    # heavier DLS damping so the arm relaxes to a natural rest instead of holding the elbow
    # tucked at 90 deg (matches Insert/Carry teleop)
    #
    # IK damping (lambda). With a fixed EE target the arm still drifts -- the extra DOF of the
    # 7-DOF arm lets the shoulder wander (up to ~2 rad) as SONIC sways the torso. A higher lambda
    # damps that drift so the arm holds its pose, without writing joint state (which tipped the
    # robot, #126). 0.05 is the operator default; FIATLUX_ARM_IK_LAMBDA raises it for hands-off.
    ik_params={"lambda_val": float(os.environ.get("FIATLUX_ARM_IK_LAMBDA", "0.05"))},
)


# Grip joint sets per hand variant. The subtask's OWN hand is kept -- see apply_subtask_teleop.
_GRIPS = {
    "inspire": (
        G1_HAND_JOINTS,
        G1_HAND_OPEN,
        G1_HAND_GRASP,
        G1_LEFT_HAND_JOINTS,
        G1_LEFT_HAND_OPEN,
        G1_LEFT_HAND_GRASP,
    ),
    "dex3": (
        G1_DEX3_RIGHT_HAND_JOINTS,
        G1_DEX3_HAND_OPEN,
        G1_DEX3_HAND_GRASP,
        G1_DEX3_LEFT_HAND_JOINTS,
        G1_DEX3_LEFT_HAND_OPEN,
        G1_DEX3_LEFT_HAND_GRASP,
    ),
}


def _make_actions_cfg(hand: str):
    """Bimanual absolute-EE-pose IK + binary grip, for the hand the robot actually has."""
    rj, ro, rg, lj, lo, lg = _GRIPS[hand]

    @configclass
    class _Cfg:
        arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=G1_ARM_JOINTS,
            body_name=G1_EE_BODY,
            controller=_IK,
            scale=1.0,
        )
        hand_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(rj),
            open_command_expr=dict(ro),
            close_command_expr=dict(rg),
        )
        left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=G1_LEFT_ARM_JOINTS,
            body_name=G1_LEFT_EE_BODY,
            controller=_IK,
            scale=1.0,
        )
        left_hand_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(lj),
            open_command_expr=dict(lo),
            close_command_expr=dict(lg),
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
    # Self-collisions stay as the benchmark authors them (True in robots/g1.py): with them off,
    # the fingers close through the thumb, and the teleop twin's physics diverges from the RL
    # env it mirrors. The original teleop env disabled them to calm Dex3 finger self-contact
    # jitter; re-check there if it resurfaces.

    # ROBOT USD OVERRIDE (teleop only): FIATLUX_TELEOP_ROBOT_USD swaps in an alternative robot
    # asset, e.g. the Inspire variant with the re-authored thumb yaw frame
    # (``scripts/omniverse/inspire_thumb_frame.py``). The benchmark envs never see this.
    robot_usd = os.environ.get("FIATLUX_TELEOP_ROBOT_USD")
    if robot_usd:
        if not os.path.isfile(robot_usd):
            raise FileNotFoundError(f"FIATLUX_TELEOP_ROBOT_USD={robot_usd!r} does not exist")
        cfg.scene.robot.spawn.usd_path = robot_usd
        # With a thumb that can stand perpendicular to the palm (the re-authored frame; the
        # stock model tops out at 55 deg and leans over the palm), the in-hand legs stage the
        # Inspire hand OPEN with the thumb raised: the bulb is seated on the flat fingers, then
        # the driver's closed rest grip brings the thumb down over it. Staged pre-curled
        # (``poses.HAND_CUP``) the proximal links start inside the glass and the thumb pins the
        # bulb on release.
        # FIATLUX_TELEOP_INHAND_THUMB_YAW (rad, default 1.0 = the grasp preset's yaw, so the
        # close does not swing the thumb) sets the staged thumb rotation; "cup" keeps the task's
        # own HAND_CUP staging for A/B runs.
        staged_yaw = os.environ.get("FIATLUX_TELEOP_INHAND_THUMB_YAW", "1.0")
        if hand == "inspire" and getattr(cfg.events, "settle_bulb", None) is not None and staged_yaw != "cup":
            cfg.scene.robot.init_state.joint_pos = {
                **cfg.scene.robot.init_state.joint_pos,
                **G1_HAND_OPEN,
                # 1.3 rad is the joint limit and Isaac Lab rejects a default position AT a limit
                "R_thumb_proximal_yaw_joint": min(float(staged_yaw), 1.29),
            }

    # Head camera on EVERY subtask. The benchmark only adds it in the balance tier
    # (climb / descend / mate), so six of the twelve had no robot-mounted view at all -- and the
    # ego view is the one a policy has to act from, so a dataset that carries it on half the
    # chain is awkward to train from. Added here rather than in the tiers to keep the benchmark
    # untouched; if the RL envs should carry it too, that is a benchmark decision.
    if getattr(cfg.scene, "ego_camera", None) is None:
        add_ego_camera(cfg.scene)
    # 512x512 rather than the sensor module's 256: the ego view doubles as the operator's
    # review footage, and 256 is unreadable for that. Kept an exact 2x of GR00T's expected
    # shortest_image_edge=256, so a recorded demo downsamples to the checkpoint's input size
    # without resampling artefacts. Overridden HERE, not in fiatlux_task.sensors -- the RL envs
    # and GR00T inference keep the 256 the checkpoint was trained against.
    cfg.scene.ego_camera.height = 512
    cfg.scene.ego_camera.width = 512

    cfg.actions = _make_actions_cfg(hand)
    # In-hand legs on the re-authored Inspire thumb: FIATLUX_TELEOP_INHAND_CLOSE="<fingers>:<thumb>"
    # scales the CLOSE preset's finger curl and thumb curl (thumb yaw untouched) for the settle:
    # the fingers only need to support the seated glass while the thumb opposes it. Default
    # 0.2:1.0 = fingers stop at 0.30 rad, thumb full; "1:1" restores the full preset. This is
    # the settle-time grip only -- the driver swaps the operator's close back to the full grasp
    # preset at "Teleop ready".
    close_scale = os.environ.get("FIATLUX_TELEOP_INHAND_CLOSE", "0.2:1.0")
    if robot_usd and hand == "inspire" and getattr(cfg.events, "settle_bulb", None) is not None:
        f_scale, t_scale = (float(v) for v in close_scale.split(":"))
        close = dict(cfg.actions.hand_action.close_command_expr)
        for name in close:
            if name.startswith("R_thumb_proximal_yaw"):
                continue
            close[name] = round(close[name] * (t_scale if name.startswith("R_thumb") else f_scale), 3)
        cfg.actions.hand_action.close_command_expr = close

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
    # Legs that start with the bulb already in the right hand (they re-seat it on the live palm
    # via ``settle_bulb``) begin with that grip closed on it; the trigger takes over once pulled.
    right_starts_closed = getattr(cfg.events, "settle_bulb", None) is not None
    cfg.teleop_devices = DevicesCfg(
        devices={
            "controller_rel": OpenXRDeviceCfg(
                retargeters=[
                    Se3RelControllerRetargeterCfg(
                        bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                        root_pos=(rp[0], rp[1], rp[2] - 0.05),
                        sim_device=cfg.sim.device,
                    ),
                    ControllerGripperRetargeterCfg(
                        bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                        sim_device=cfg.sim.device,
                        start_closed=right_starts_closed,
                    ),
                    Se3RelControllerRetargeterCfg(
                        bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                        root_pos=(rp[0], rp[1], rp[2] - 0.05),
                        sim_device=cfg.sim.device,
                    ),
                    ControllerGripperRetargeterCfg(
                        bound_hand=DeviceBase.TrackingTarget.HAND_LEFT, sim_device=cfg.sim.device
                    ),
                ],
                sim_device=cfg.sim.device,
                xr_cfg=cfg.xr,
            ),
        }
    )
