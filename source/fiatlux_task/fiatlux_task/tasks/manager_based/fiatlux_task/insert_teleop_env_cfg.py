# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperation variant of ``FIATLUX-Insert-v0`` (issue #51, Phase 1).

Swaps the Insert task's joint-position **arm** action for a **differential-IK** action, so a 6-DoF
SE(3) device (keyboard / SpaceMouse) drives the right-hand end-effector, plus a **binary open/close**
grip on the Inspire hand. Everything else -- scene (tabletop preset), observations, rewards,
terminations -- is inherited unchanged from :class:`G1BulbInsertEnvCfg`, so only the *action
interface* differs and recorded demos stay compatible with the RL env's observation / reward defs.

Drive it with ``scripts/insert_teleop.py --task FIATLUX-Insert-Teleop-v0`` (add ``--teleop_device spacemouse``
for a SpaceMouse).
"""

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_abs_retargeter import Se3AbsRetargeterCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from fiatlux_task.assets import OMNI_BULB_USD, OMNI_SOCKET_USD
from fiatlux_task.robots.g1 import G1_ARM_JOINTS, G1_EE_BODY, G1_HAND_GRASP, G1_HAND_JOINTS, G1_HAND_OPEN

from .g1_bulb_env_cfg import G1BulbInsertEnvCfg
from .scene_cfg import _quat_x_deg, _spawn_usd_as_rigid_body
from .xr_controller_retargeters import (
    ControllerGripperRetargeterCfg,
    Se3AbsControllerRetargeterCfg,
    Se3RelControllerRetargeterCfg,
)

# Left-arm mirror of the right teleop constants (robots/g1.py only defines the right side). Used to
# add optional bimanual control: the left controller drives the left arm + grip.
G1_LEFT_ARM_JOINTS = [j.replace("right_", "left_", 1) for j in G1_ARM_JOINTS]
G1_LEFT_EE_BODY = "left_wrist_yaw_link"
G1_LEFT_HAND_JOINTS = [j.replace("R_", "L_", 1) for j in G1_HAND_JOINTS]
G1_LEFT_HAND_OPEN = dict.fromkeys(G1_LEFT_HAND_JOINTS, 0.0)
G1_LEFT_HAND_GRASP = {k.replace("R_", "L_", 1): v for k, v in G1_HAND_GRASP.items()}


@clone
def _spawn_omni_rigid(prim_path, cfg, translation=None, orientation=None):
    """Spawn an Omniverse SimReady asset as a *dynamic* rigid body.

    The curated OMNI bulb ships colliders but no ``RigidBodyAPI``, and its referenced geometry is
    USD-**instanceable** -- so the stock :func:`_spawn_usd_as_rigid_body` can apply the API on the
    root but then cannot author rigid/mass props onto the (instanced) collider prims. De-instancing
    the spawned prim first makes the whole subtree editable, so the rigid body actually forms.
    """
    from pxr import UsdPhysics

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    prim.SetInstanceable(False)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim.GetPath(), cfg.rigid_props)
    if cfg.mass_props is not None:
        UsdPhysics.MassAPI.Apply(prim)
        schemas.modify_mass_properties(prim.GetPath(), cfg.mass_props)
    return prim


@configclass
class G1BulbInsertTeleopEnvCfg(G1BulbInsertEnvCfg):
    """``FIATLUX-Insert-v0`` with a teleop action interface (arm IK-rel + binary grip)."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Bolt the pelvis to the world: teleop is stationary tabletop manipulation, and the free
        # legged base has no balance controller (it would sag/tip). A fixed base also gives the
        # arm IK a clean fixed-manipulator chain. The human just drives the hand.
        self.scene.robot.spawn.articulation_props.fix_root_link = True

        # Disable robot self-collision for teleop. The Inspire hand's finger collision meshes
        # interpenetrate at the spawn pose (known issue -- see robots/g1.py), which wedges the
        # capped-torque fingers into a high-speed contact limit-cycle ("random" finger jitter). Self
        # collision is only needed for the RL contact rewards; finger-vs-bulb contact (the grasp)
        # is object collision and stays enabled, so disabling this just calms the hand.
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

        # Robot stand-off: pulled in to (0.50, 0.50) so the props at the table's near edge sit at
        # ~50% arm extension -- a well-conditioned workspace. Farther back (y=0.60/0.80) put the
        # bench near the arm's reach limit, where the IK bifurcates into unreachable dead zones and
        # the grasp fails. Pelvis stays >~0.20 m clear of the table edge (y=0.281) so it doesn't
        # collide during teleop.
        self.scene.robot.init_state.pos = (0.50, 0.70, 0.75)

        # Shrink the bench height: the stock packing table's ~0.99 m top sits too high for the
        # bolted (fixed-base) G1 to reach. Scale Z only -- footprint (and props on it) stay put
        # while the surface drops to ~0.88 m. This fixed-base arm has a BIFURCATED workspace at the
        # bench distance (near-singular, ~96% extension): a high branch (~0.90 m, good resolution
        # 0.85-0.92) and a low branch (~0.68 m), with an unreachable dead zone (~0.73-0.88) between.
        # The props sit in the high branch where pick-and-place tracks cleanly. (Still 0.11 m below
        # the original 0.99 m bench.) Kinematic table -> non-uniform collider scale is harmless.
        self.scene.table.spawn.scale = (1.0, 1.0, 0.91)

        # Swap the tabletop fixture to the clean Omniverse bulb + socket (Y-up cm -> scale 0.007 +
        # a +90 deg X rotation to stand it upright, bulb up). Socket kinematic; bulb dynamic so it
        # can be grasped -- the OMNI asset ships colliders but no RigidBodyAPI and its geometry is
        # USD-instanced, so the bulb goes through _spawn_omni_rigid (de-instance, then apply the
        # rigid body) while the static socket can use the stock _spawn_usd_as_rigid_body. Positions
        # rest on the ~1.0 m table surface, near the robot's front so the arm reaches them.
        upright = _quat_x_deg(90.0)
        self.scene.socket.spawn = sim_utils.UsdFileCfg(
            usd_path=OMNI_SOCKET_USD,
            func=_spawn_usd_as_rigid_body,
            scale=(0.007, 0.007, 0.007),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        )
        self.scene.socket.init_state.pos = (0.48, 0.12, 0.88)
        self.scene.socket.init_state.rot = upright
        self.scene.bulb.spawn = sim_utils.UsdFileCfg(
            usd_path=OMNI_BULB_USD,
            func=_spawn_omni_rigid,
            scale=(0.007, 0.007, 0.007),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
        )
        # Spawn just above the surface so it barely drops -- a taller drop lets the thin bulb bounce
        # and roll to an unpredictable spot, breaking the open-loop grasp. y kept inside the table's
        # near edge (y=0.281) so it rests on the surface instead of teetering off it.
        self.scene.bulb.init_state.pos = (0.34, 0.24, 0.93)
        self.scene.bulb.init_state.rot = upright

        # NOTE: use the robot's tuned per-joint arm gains (_ARM_STIFFNESS/_ARM_DAMPING/_ARM_ARMATURE
        # from robots/g1.py). A previous blanket override (stiffness=2000, damping=100) replaced those
        # per-joint dicts with uniform values while leaving the tuned armature in place -- the
        # stiffness/armature mismatch made the IK arm oscillate ("random" motion) even holding still.

        # Arm: ABSOLUTE EE pose IK. Relative mode re-anchors to the *current* pose each step, so a
        # compliant arm ratchets/drifts (it never actively returns to a target); absolute mode holds
        # a fixed target pose and drives back to it. The teleop script integrates the device's
        # deltas into the absolute target (command = [pos(3), quat(4)]).
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=G1_ARM_JOINTS,
            body_name=G1_EE_BODY,
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            scale=1.0,
        )
        # Hand: the device's gripper toggle -> a binary open/close grip on the Inspire fingers.
        self.actions.hand_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=G1_HAND_JOINTS,
            open_command_expr=G1_HAND_OPEN,
            close_command_expr=G1_HAND_GRASP,
        )

        # LEFT arm + grip (optional bimanual): mirror of the right actions on the left limb, so the
        # left controller can drive the left hand. Added AFTER the right actions, so the action tensor
        # is [right_arm(7), right_grip(1), left_arm(7), left_grip(1)] -- matching the controller_rel
        # device's retargeter order (ActionManager iterates cfg fields in insertion order).
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=G1_LEFT_ARM_JOINTS,
            body_name=G1_LEFT_EE_BODY,
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            scale=1.0,
        )
        self.actions.left_hand_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=G1_LEFT_HAND_JOINTS,
            open_command_expr=G1_LEFT_HAND_OPEN,
            close_command_expr=G1_LEFT_HAND_GRASP,
        )

        # Teleop is operator-paced: disable ALL automatic terminations. Isaac Lab's env.step()
        # auto-resets any terminated env internally, so without this the scene would reset itself
        # out from under the operator -- on the 15 s time-out, on an accidental bulb knock/drop
        # (bulb_dropped), or on a stray success. The human resets explicitly with 'R'.
        self.terminations.time_out = None
        self.terminations.success = None
        self.terminations.bulb_dropped = None

        # XR scene anchor: the sim-world pose (on the floor) that maps to the origin of the headset's
        # local frame -- i.e. where the scene appears relative to the standing operator. Set near the
        # robot base so the G1 + tabletop appear in front of the user; anchor_rot is a 180 deg yaw
        # (w,x,y,z)=(0,0,0,1) so the operator faces the table (props at -Y) instead of away from it.
        # NOTE: still tune in-headset -- if the operator ends up 90 deg off, adjust the yaw quaternion.
        self.xr = XrCfg(anchor_pos=(0.5, 0.7, 0.0), anchor_rot=(0.0, 0.0, 0.0, 1.0))

        # Devices the teleop script can instantiate for this env. ``handtracking`` drives the same
        # absolute-IK arm + binary grip via a Meta Quest / Pico headset through CloudXR (see
        # vr_teleop/vr_teleop_setup.md): Se3AbsRetargeter maps the right-hand wrist pose -> EE target,
        # GripperRetargeter maps thumb-index pinch -> open/close. Launch with Isaac Lab's
        # ``scripts/environments/teleoperation/teleop_se3_agent.py --teleop_device handtracking`` (it
        # auto-enables ``--xr``); keyboard/spacemouse remain for the flat-screen path.
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02, rot_sensitivity=0.05, sim_device=self.sim.device
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05, rot_sensitivity=0.05, sim_device=self.sim.device
                ),
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3AbsRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                # Motion-controller variant: same absolute-IK arm + binary grip, but driven by the
                # headset CONTROLLER instead of hand tracking (the Quest/Pico CloudXR web client streams
                # controllers, not optical hand joints -- see vr_teleop/vr_teleop_setup.md). Right grip
                # pose -> EE target, trigger -> grip. Pick it with ``--teleop_device controller``.
                "controller": OpenXRDeviceCfg(
                    retargeters=[
                        Se3AbsControllerRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            sim_device=self.sim.device,
                        ),
                        ControllerGripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                # Relative/incremental controller variant (recommended): move the controller to *nudge*
                # the EE from its current target -- no need to hold your hand in the robot's workspace.
                # Pick with ``--teleop_device controller_rel``.
                "controller_rel": OpenXRDeviceCfg(
                    retargeters=[
                        # RIGHT arm + grip (defaults start from the right rest pose).
                        Se3RelControllerRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            sim_device=self.sim.device,
                        ),
                        ControllerGripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                        # LEFT arm + grip (bimanual): left controller drives the left EE, starting from
                        # the probed left rest pose; workspace shifted to the left arm's reach (x~0.65).
                        Se3RelControllerRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                            initial_position=(0.6497, 0.5043, 0.8242),
                            initial_orientation=(0.7086, 0.05, 0.0479, -0.7022),
                            workspace_min=(0.35, 0.05, 0.78),
                            workspace_max=(0.85, 0.60, 1.10),
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
