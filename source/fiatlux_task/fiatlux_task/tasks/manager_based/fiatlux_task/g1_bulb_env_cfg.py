# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fiatlux benchmark: Unitree G1 humanoid inserting a light bulb into a socket.

This is the *insertion* subtask (manipulation only): the G1 starts at the
fixture holding a bulb and must align and seat it into the socket. It is built
as a standard Isaac Lab ``ManagerBasedRLEnvCfg`` so it slots into the usual
train / play / teleop / eval scripts.

The scene is the shared family world (``scene_cfg.G1ReplaceSceneCfg``) in its
**tabletop preset**: packing table, socket-lamp on top, bulb at hand height, no
ladder. Dressing randomization is off here -- homogeneous envs keep
``replicate_physics=True`` for training scale.

Design notes (kept deliberately simple and hardware-minded for later sim-to-real):
- **Actions** are joint-position targets on the G1 arm (optionally hand), which
  map directly onto commands the Unitree SDK can consume on the real robot.
- **Observations** are split into a default *sensor-realizable* ``policy`` group
  (proprioception + wrist camera + contact forces) and a separate *privileged*
  group (ground-truth bulb/socket pose) used only by the critic and scripted
  baselines -- never as the sole interface.

The Actions/Observations/Rewards/Terminations cfgs below stay in this module for
now; they get factored into shared manipulation blocks when the at-fixture
Install task becomes their second consumer (unification spec, Phase 4).
"""

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from fiatlux_task.robots.g1 import (
    G1_ARM_JOINTS,
    G1_EE_BODY,
    G1_HAND_JOINTS,
)

from . import mdp
from .scene_cfg import G1ReplaceSceneCfg, apply_tabletop_preset

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Joint-position targets on the G1 arm (hardware-realizable for sim-to-real)."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_ARM_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )
    # Right Inspire-hand finger targets, so the policy can grasp/release the bulb.
    hand_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_HAND_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Two groups: sensor-realizable policy obs and privileged ground-truth obs."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Sensor-realizable observations (available on the real robot)."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=G1_ARM_JOINTS + G1_HAND_JOINTS
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=G1_ARM_JOINTS + G1_HAND_JOINTS
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        eef_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=G1_EE_BODY)},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        hand_contact = ObsTerm(
            func=mdp.contact_net_forces,
            scale=0.1,
            params={"sensor_cfg": SceneEntityCfg("hand_contact")},
        )
        wrist_rgb = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Ground-truth ("cheat") observations for the critic / scripted baselines."""

        bulb_pose = ObsTerm(
            func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("bulb")}
        )
        socket_pose = ObsTerm(
            func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("socket")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Reset-time randomization."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )

    reset_socket = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("socket"),
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.05, 0.05), "z": (-0.03, 0.03)},
            "velocity_range": {},
        },
    )

    reset_bulb = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("bulb"),
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            "velocity_range": {},
        },
    )

    # Intensity only: the dome carries an HDRI sky texture, and color-tinting a texture
    # reads as a render bug rather than useful domain randomization (see mdp.events).
    randomize_light = EventTerm(
        func=mdp.randomize_light_properties,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("dome_light"),
            "intensity_range": (500.0, 2000.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms: drive the bulb into the socket, gently."""

    # -- Bulb -> socket position tracking (coarse / fine / sharp) --
    align_position = RewTerm(
        func=mdp.object_socket_distance, weight=-1.0,
    )
    align_position_tanh = RewTerm(
        func=mdp.object_socket_distance_tanh, weight=0.5, params={"std": 0.1},
    )
    seat_position_exp = RewTerm(
        func=mdp.object_socket_distance_exp, weight=1.0, params={"sigma": 0.02},
    )
    # -- Orientation alignment (bulb axis vs socket axis) --
    align_orientation = RewTerm(
        func=mdp.object_socket_orientation_tanh, weight=0.3, params={"std": 0.3},
    )
    # -- Sparse seated bonus --
    seated_bonus = RewTerm(
        func=mdp.bulb_seated,
        weight=5.0,
        params={"pos_threshold": 0.015, "ori_threshold": 0.2},
    )
    # -- Compliance / safety: penalize hard contact forces --
    contact_penalty = RewTerm(
        func=mdp.hand_contact_force_l2,
        weight=-1.0e-3,
        params={"sensor_cfg": SceneEntityCfg("hand_contact")},
    )
    # -- Smoothness / safety --
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS)},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS)},
    )


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.bulb_seated,
        params={"pos_threshold": 0.015, "ori_threshold": 0.2},
    )
    bulb_dropped = DoneTerm(
        func=mdp.object_dropped,
        params={"asset_cfg": SceneEntityCfg("bulb"), "min_height": 0.4},
    )


##
# Environment configuration
##


@configclass
class G1BulbInsertEnvCfg(ManagerBasedRLEnvCfg):
    """Fiatlux insertion subtask: G1 seats a bulb into a socket (family tabletop preset)."""

    # Homogeneous envs (no per-env dressing randomization) -> replicated physics for
    # training scale. Cloning stays in USD (not fabric): the hand_contact sensor's PhysX
    # contact-reporter API cannot attach to fabric-cloned env prims.
    scene: G1ReplaceSceneCfg = G1ReplaceSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Family scene -> manipulation bench layout; no random ceiling fixture (that
        # would force replicate_physics=False; see FamilyBaseEnvCfg's dressing flag).
        apply_tabletop_preset(self.scene)
        self.scene.fixture = None

        # Wrist-mounted RGB camera (sensor-realizable observation). Requires launching
        # with --enable_cameras.
        self.scene.wrist_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + G1_EE_BODY + "/wrist_camera",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=22.48,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 5.0),
            ),
            height=224,
            width=224,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"
            ),
        )

        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0
        # Control-rate parity with the pre-unification env (30 Hz); the family base runs
        # 50 Hz -- reconciling is a deliberate, separate decision (unification spec).
        self.sim.dt = 1.0 / 120.0
        # PhysX solver floors from the family base: without them the uncontrolled robot
        # picks up multi-hundred-m/s kicks against the kinematic table (verify_scene
        # finding, Phase 0). Stabilization further damps the PD-vs-table wedge impulses
        # when the robot lies collapsed against the furniture.
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 4
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_stabilization = True
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.45, 0.0, 1.1)
