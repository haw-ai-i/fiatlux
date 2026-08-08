# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Carry-Teleop-v0`` -- whole-body teleop of the ladder Carry task.

Same treatment as :mod:`insert_teleop_env_cfg`, on the Carry ladder scene: the RL whole-body joint
action is replaced by an **arm-IK + binary-grip** interface (bimanual) and a ``controller_rel``
teleop device, so ``scripts/teleop/sonic_teleop.py`` drives it (SONIC balances + walks the legs; you
teleop the arms). **Dex3** hand by default. The Carry base is already free (it's a walking task),
so no un-bolting is needed -- unlike Insert.

Caveat: the ``controller_rel`` retargeter accumulates the EE target in the WORLD frame (baked to a
fixed base pose), so the arm teleop is accurate when the robot is standing at the ladder; walking
far re-references it. Root pose defaults to the Carry robot spawn.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import XrAnchorRotationMode, XrCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.utils import configclass

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

from fiatlux_task.assets import FIATLUX_ASSETS_DIR

from fiatlux_task.tasks.manager_based.fiatlux_task.carry_env_cfg import CarryEnvCfg
from .insert_teleop_env_cfg import G1_LEFT_ARM_JOINTS, G1_LEFT_EE_BODY, G1_LEFT_HAND_GRASP, G1_LEFT_HAND_OPEN
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import _quat_z_deg, _spawn_usd_as_rigid_body_frictional
from .xr_controller_retargeters import (
    ControllerGripperRetargeterCfg,
    Se3RelControllerRetargeterCfg,
)


@configclass
class CarryTeleopActionsCfg:
    """Bimanual absolute-EE-pose IK + binary Dex3 grip (replaces the RL whole-body joint action).
    SONIC drives the legs+waist directly in the driver; these actions cover the arms + hands."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=G1_ARM_JOINTS,
        body_name=G1_EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls",
            # Match Insert-Teleop: heavier DLS damping so the arm relaxes to a natural rest instead of
            # holding the elbow tucked up at 90 deg (default lambda 0.01 is near-undamped).
            ik_params={"lambda_val": 0.05},
        ),
        scale=1.0,
    )
    hand_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_DEX3_RIGHT_HAND_JOINTS,
        open_command_expr=G1_DEX3_HAND_OPEN,
        close_command_expr=G1_DEX3_HAND_GRASP,
    )
    left_arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=G1_LEFT_ARM_JOINTS,
        body_name=G1_LEFT_EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls",
            # Match Insert-Teleop: heavier DLS damping so the arm relaxes to a natural rest instead of
            # holding the elbow tucked up at 90 deg (default lambda 0.01 is near-undamped).
            ik_params={"lambda_val": 0.05},
        ),
        scale=1.0,
    )
    left_hand_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_DEX3_LEFT_HAND_JOINTS,
        open_command_expr=G1_DEX3_LEFT_HAND_OPEN,
        close_command_expr=G1_DEX3_LEFT_HAND_GRASP,
    )


@configclass
class CarryTeleopEnvCfg(CarryEnvCfg):
    """``FIATLUX-Carry-v0`` with a teleop action interface (bimanual arm IK-rel + Dex3 grip)."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Dex3 hand as the default for teleop (Unitree 3-finger). swap_robot_variant re-points the
        # finger-scoped reward/termination terms too; calm the finger self-contact like the Insert env.
        swap_robot_variant(self, "dex3")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

        # Scene arrangement (TELEOP ONLY -- RL Carry-v0 keeps apply_position_preset's placement):
        # park the ladder in a corner of the Simple Room and yaw it to face the robot.
        # Room walls span x=[-4.52,4.52], y=[-3.4,4.86]; ~0.7 m standoff keeps the A-frame off them.
        # Corners: back-right (3.8,4.1) back-left (-3.8,4.1) front-right (3.8,-2.7) front-left (-3.8,-2.7).
        # Front-right, nudged toward center so the A-frame's legs clear the walls (was clipping at 3.8,-2.7).
        _lad = (3.3, -2.3, 0.0)
        _rp = self.scene.robot.init_state.pos
        _yaw = math.degrees(math.atan2(_rp[1] - _lad[1], _rp[0] - _lad[0])) + 90.0  # face robot, then +90 deg
        self.scene.ladder.init_state.pos = _lad
        self.scene.ladder.init_state.rot = _quat_z_deg(_yaw)

        # (Ladder collision -- SDF exact surface + a ~6 mm contact offset -- is authored into the
        # asset by scripts/omniverse/omniverse_ladder_collision.py, so no code-side override is
        # needed here; see docs/ladder_collision_tightening.md.)

        # TEST LADDERS (teleop only): three DIFFERENT A-frame step-ladder designs placed within
        # reach in front of the robot, each spawned with its FILE collision (the tightened
        # convexDecomposition from omniverse_ladder_collision.py) rather than the corner ladder's SDF
        # override -- walk up and grab them to feel the asset-level fix across designs. All are
        # authored base-at-z=0 so a 1 cm drop seats them; spread ~1.2-1.3 m apart in y so their
        # footprints (up to ~1 m deep) do not overlap at spawn and depenetrate into a tip-over.
        # Free-standing A-frames only (footprint depth >= ~0.5 m); narrow leaning designs like
        # FRP_Step_A (0.25 m deep) topple on their own, so they are avoided here.
        _test_ladders = [
            ("AlumStepDouble_A", "AluminiumStepDoubleLadder_A01_PR_NVD_01_collision_rigid.usd", (1.30, 1.00)),
            ("AlumStep_B", "AluminumStepLadder_B01_PR_NVD_01_collision_rigid.usd", (1.30, 0.00)),
            ("AlumStep_D", "AluminumStepLadder_D01_PR_NVD_01_collision_rigid.usd", (1.30, -1.30)),
        ]
        for _i, (_family, _usd, (_x, _y)) in enumerate(_test_ladders):
            setattr(
                self.scene,
                f"test_ladder_{_i}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/TestLadder{_i}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=os.path.join(FIATLUX_ASSETS_DIR, "omniverse_ladder", _family, _usd),
                        func=_spawn_usd_as_rigid_body_frictional,
                        scale=(0.01, 0.01, 0.01),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=False,
                            solver_position_iteration_count=16,
                            solver_velocity_iteration_count=1,
                            max_depenetration_velocity=1.0,
                            sleep_threshold=0.005,
                            stabilization_threshold=0.001,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(_x, _y, 0.01)),
                ),
            )

        # teleop action interface (arms + hands); legs+waist are SONIC's, driven in sonic_teleop.py.
        self.actions = CarryTeleopActionsCfg()

        # operator-paced: no automatic terminations (they would auto-reset mid-teleop).
        for _t in ("time_out", "success", "ladder_tipped", "ladder_dropped", "fell_below", "fell_over"):
            if getattr(self.terminations, _t, None) is not None:
                setattr(self.terminations, _t, None)

        # FOLLOW CAMERA: anchor the XR view to the robot's pelvis so it tracks the robot as SONIC walks
        # it around -- position + smoothed yaw, at a fixed comfortable height. Without this the view is
        # pinned at the spawn and the robot walks out of frame. (Mirrors Isaac Lab's G1 loco-manip teleop.)
        rp = self.scene.robot.init_state.pos
        self.xr = XrCfg(
            # anchor_pos is an OFFSET added to the anchored pelvis prim (not an absolute pose). For a
            # FIRST-PERSON view: (x,y)=0 so the operator is AT the robot, and z=-0.9 drops the play-space
            # floor ~0.9 m so the operator's eyes land at the robot's head height (~1.3 m) instead of
            # floating ~2.3 m above it (which read as a top-down third-person). Tune z if too high/low.
            anchor_pos=(0.0, 0.0, -0.9),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
            anchor_prim_path="/World/envs/env_0/Robot/pelvis",
            fixed_anchor_height=True,
            anchor_rotation_mode=XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
        )

        # controller_rel teleop device: bimanual relative-IK arm + binary grip (the recommended
        # Insert device). Root pose baked to the Carry robot spawn so the world<->root transform
        # is right where the robot stands.
        self.teleop_devices = DevicesCfg(
            devices={
                "controller_rel": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelControllerRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            root_pos=(rp[0], rp[1], rp[2] - 0.05),
                            sim_device=self.sim.device,
                        ),
                        ControllerGripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                        Se3RelControllerRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                            root_pos=(rp[0], rp[1], rp[2] - 0.05),
                            sim_device=self.sim.device,
                        ),
                        ControllerGripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )


def apply_inspire_hands(env_cfg) -> None:
    """Switch a *parsed* Carry-Teleop env cfg from its native Dex3 hands to INSPIRE, in place.

    Carry-Teleop is Dex3-native (see ``CarryTeleopEnvCfg.__post_init__``); this is the mirror of the
    Insert env's ``apply_dex3_hands``. Call from the launcher when ``--hand inspire``. ``swap_robot_variant``
    re-points the reward/termination/action joint-name references (Dex3 -> Inspire via ``_HAND_REMAPS``);
    we additionally repoint the binary grips to the Inspire finger joints + Inspire open/grasp presets,
    which the joint-name remap alone does not cover.
    """
    swap_robot_variant(env_cfg, "inspire")
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    env_cfg.actions.hand_action.joint_names = list(G1_HAND_JOINTS)
    env_cfg.actions.hand_action.open_command_expr = dict(G1_HAND_OPEN)
    env_cfg.actions.hand_action.close_command_expr = dict(G1_HAND_GRASP)
    env_cfg.actions.left_hand_action.joint_names = list(G1_LEFT_HAND_JOINTS)
    env_cfg.actions.left_hand_action.open_command_expr = dict(G1_LEFT_HAND_OPEN)
    env_cfg.actions.left_hand_action.close_command_expr = dict(G1_LEFT_HAND_GRASP)
