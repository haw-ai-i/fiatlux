# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Carry-Teleop-v0`` -- whole-body teleop of the ladder Carry task.

Same treatment as :mod:`insert_teleop_env_cfg`, on the Carry ladder scene: the RL whole-body joint
action is replaced by an **arm-IK + binary-grip** interface (bimanual) and a ``controller_rel``
teleop device, so ``scripts/sonic_teleop.py`` drives it (SONIC balances + walks the legs; you
teleop the arms). **Dex3** hand by default. The Carry base is already free (it's a walking task),
so no un-bolting is needed -- unlike Insert.

Caveat: the ``controller_rel`` retargeter accumulates the EE target in the WORLD frame (baked to a
fixed base pose), so the arm teleop is accurate when the robot is standing at the ladder; walking
far re-references it. Root pose defaults to the Carry robot spawn.
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import XrCfg
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
    swap_robot_variant,
)

from .carry_env_cfg import CarryEnvCfg
from .insert_teleop_env_cfg import G1_LEFT_ARM_JOINTS, G1_LEFT_EE_BODY
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
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
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
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
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

        # teleop action interface (arms + hands); legs+waist are SONIC's, driven in sonic_teleop.py.
        self.actions = CarryTeleopActionsCfg()

        # operator-paced: no automatic terminations (they would auto-reset mid-teleop).
        for _t in ("time_out", "success", "ladder_tipped", "ladder_dropped", "fell_below", "fell_over"):
            if getattr(self.terminations, _t, None) is not None:
                setattr(self.terminations, _t, None)

        # XR anchor near the Carry robot spawn (tune in-headset if the operator ends up off-axis).
        rp = self.scene.robot.init_state.pos
        self.xr = XrCfg(anchor_pos=(rp[0], rp[1], 0.0), anchor_rot=(0.0, 0.0, 0.0, 1.0))

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
