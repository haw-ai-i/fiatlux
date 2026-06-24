# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fiatlux benchmark: Unitree G1 humanoid inserting a light bulb into a socket.

This is the *insertion* subtask (manipulation only): the G1 starts at the
fixture holding a bulb and must align and seat it into the socket. It is built
as a standard Isaac Lab ``ManagerBasedRLEnvCfg`` so it slots into the usual
train / play / teleop / eval scripts.

Design notes (kept deliberately simple and hardware-minded for later sim-to-real):
- **Actions** are joint-position targets on the G1 arm (optionally hand), which
  map directly onto commands the Unitree SDK can consume on the real robot.
- **Observations** are split into a default *sensor-realizable* ``policy`` group
  (proprioception + wrist camera + contact forces) and a separate *privileged*
  group (ground-truth bulb/socket pose) used only by the critic and scripted
  baselines -- never as the sole interface.

Asset-dependent identifiers (USD paths, joint/body names) are collected as
constants at the top of this file. They follow the standard Unitree G1 naming;
adjust them to match the exact USD pulled by ``assets/download_assets.sh``.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .mdp.events import randomize_dome_light

# ---------------------------------------------------------------------------
# Asset locations and names (adjust to match the pulled USDs)
# ---------------------------------------------------------------------------

# Assets are synced into the repo-root ``assets/`` dir by ``download_assets.sh``.
# Override with the FIATLUX_ASSETS_DIR env var if they live elsewhere.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, *([os.pardir] * 6)))
FIATLUX_ASSETS_DIR = os.environ.get(
    "FIATLUX_ASSETS_DIR", os.path.join(_REPO_ROOT, "assets")
)

G1_USD = os.path.join(FIATLUX_ASSETS_DIR, "unitree_g1", "g1.usd")
BULB_USD = os.path.join(FIATLUX_ASSETS_DIR, "bulb_socket", "bulb.usd")
SOCKET_USD = os.path.join(FIATLUX_ASSETS_DIR, "bulb_socket", "socket.usd")

# Standard Unitree G1 right-arm joints used for the insertion subtask.
G1_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# End-effector body the bulb is rigidly grasped at (hand/wrist link).
G1_EE_BODY = "right_wrist_yaw_link"


##
# Scene definition
##


@configclass
class G1BulbSceneCfg(InteractiveSceneCfg):
    """Scene: Unitree G1, a graspable bulb, a socket fixture, ground and light."""

    # -- Unitree G1 humanoid --
    # NOTE: the insertion subtask uses a fixed/standing base. Use a fixed-base G1
    # USD variant (or set fix_root_link in the USD) so the arm can be trained in
    # isolation before locomotion/climbing is added.
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
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
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["right_.*_joint"],
                effort_limit_sim=88.0,
                stiffness=150.0,
                damping=5.0,
            ),
        },
    )

    # -- Light bulb: graspable rigid body --
    bulb: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bulb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BULB_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.20, 1.05)),
    )

    # -- Socket / lamp fixture: kinematic so it can be re-posed on reset --
    socket: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Socket",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SOCKET_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 1.20)),
    )

    # -- Lights --
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # -- Ground --
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # -- Contact sensor on the grasping hand (force/torque safety + obs) --
    hand_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_.*",
        history_length=1,
        track_air_time=False,
    )

    def __post_init__(self):
        super().__post_init__()

        # Wrist-mounted RGB camera (sensor-realizable observation).
        self.wrist_camera = TiledCameraCfg(
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


@configclass
class ObservationsCfg:
    """Two groups: sensor-realizable policy obs and privileged ground-truth obs."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Sensor-realizable observations (available on the real robot)."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS)},
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

    randomize_light = EventTerm(
        func=randomize_dome_light,
        mode="reset",
        params={
            "intensity_range": (1500.0, 3500.0),
            "color_range": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
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
    """Fiatlux insertion subtask: G1 seats a bulb into a socket."""

    scene: G1BulbSceneCfg = G1BulbSceneCfg(num_envs=1, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0
        self.sim.dt = 1.0 / 120.0
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.45, 0.0, 1.1)
