# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Descend-v0`` -- bipedal ladder descent (functional RL task).

The mirror image of ``FIATLUX-Climb-v0``: the G1 starts at the top of the kinematic
A-frame step ladder (family *at-height* preset, robot at the upper steps) and must
descend under control back to the ladder's base. Built as a standard
``ManagerBasedRLEnvCfg`` so it slots into the usual train / play / eval scripts.

Design notes (mirrors ``climb_env_cfg`` -- see its module docstring for the parts that
carry over unchanged: actions, sensor suite, fall gate, smoothness terms):
- **Rewards** pay progressive height *loss* (``mdp.descend_height_progress``, each
  centimetre of new depth paid once) plus the same limb-on-ladder contact bootstrap and
  CoM-sway/wobble/fall/smoothness penalties Climb uses. ``flat_orientation_l2`` stays
  absent for the same reason: a controlled descent down an A-frame needs the same
  sustained lean climbing does.
- **Terminations**: the family fall-detection gate (shared thresholds, imported from
  ``climb_env_cfg``) and a ``success`` term (``mdp.descended_to_target``) gated on
  height, horizontal proximity to the ladder's base, and speed -- the same three-part
  gate ``climbed_to_target`` uses, inverted.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from fiatlux_task.robots.g1 import G1_FINGER_JOINT_PATTERNS, G1_WAIST_JOINT_PATTERNS

from . import mdp
from .climb_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT
from .scene_cfg import (
    CLIMB_ROBOT_POSITION,
    ROOM_ENV_SPACING,
    G1ReplaceSceneCfg,
    add_ego_camera,
    add_ladder_contact_sensor,
    add_mid360_lidar,
    apply_at_height_preset,
)

##
# Task geometry
##

# Success gate derived from the ladder's base (climb's own start pose): slightly above
# standing height so a solid stance on the bottom steps/floor scores.
SUCCESS_HEIGHT = CLIMB_ROBOT_POSITION[2] + 0.15  # ~0.94 m
SUCCESS_XY = (CLIMB_ROBOT_POSITION[0], CLIMB_ROBOT_POSITION[1])
SUCCESS_XY_RADIUS = 0.6  # m; matches Climb's tolerance
SUCCESS_MAX_SPEED = 1.5  # m/s

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Whole-body joint-position targets (hardware-realizable for sim-to-real)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Two groups: sensor/estimator-realizable policy obs and privileged ground truth
    (identical to Climb's -- same at-height preset, same ladder, same sensor suite)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """IMU + estimator + proprioception + limb contact (see module docstring)."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        ladder_limb_contact = ObsTerm(
            func=mdp.contact_net_forces,
            scale=0.1,
            params={"sensor_cfg": SceneEntityCfg("ladder_contact")},
        )
        ego_rgb = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("ego_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )
        lidar_ranges = ObsTerm(
            func=mdp.lidar_ranges,
            scale=0.1,
            params={"sensor_cfg": SceneEntityCfg("mid360_lidar")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Ground-truth states for the critic / scripted baselines only."""

        root_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("robot")})
        root_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("robot")})
        ladder_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("ladder")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Reset-time randomization (kept within verify_scene's 0.10 m drift tolerance)."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )
    reset_robot_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.1, 0.1)},
            "velocity_range": {},
        },
    )
    randomize_sky_intensity = EventTerm(
        func=mdp.randomize_light_properties,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("dome_light"), "intensity_range": (600.0, 1400.0)},
    )
    randomize_key_light = EventTerm(
        func=mdp.randomize_light_properties,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("key_light"), "intensity_range": (800.0, 2200.0)},
    )


@configclass
class RewardsCfg:
    """Descent + stability shaping, mirroring Climb's ascent recipe exactly."""

    # -- main task --
    descend_progress = RewTerm(func=mdp.descend_height_progress, weight=500.0)
    success_bonus = RewTerm(
        func=mdp.descended_to_target,
        weight=500.0,
        params={
            "maximum_height": SUCCESS_HEIGHT,
            "xy_center": SUCCESS_XY,
            "xy_radius": SUCCESS_XY_RADIUS,
            "max_speed": SUCCESS_MAX_SPEED,
        },
    )
    # -- bootstrap: stay on the ladder while descending (max ~0.005/step) --
    ladder_contact = RewTerm(
        func=mdp.ladder_contact_fraction,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("ladder_contact"), "threshold": 1.0},
    )
    # -- stability shaping (no flat_orientation term: descending an A-frame needs a lean) --
    com_sway = RewTerm(func=mdp.com_sway_l2, weight=-0.5)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    termination_penalty = RewTerm(
        func=mdp.fall_terminated,
        weight=-200.0,
        params={"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT},
    )
    # -- smoothness / joint discipline (Climb's recipe) --
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    ankle_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_WAIST_JOINT_PATTERNS)},
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_FINGER_JOINT_PATTERNS)},
    )


@configclass
class TerminationsCfg:
    """Episode-ending conditions: horizon, fall detection, and success."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_below = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": FALL_MIN_HEIGHT})
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": FALL_TILT_LIMIT})
    # Contract name: recording.py / score.py read the `success` term.
    success = DoneTerm(
        func=mdp.descended_to_target,
        params={
            "maximum_height": SUCCESS_HEIGHT,
            "xy_center": SUCCESS_XY,
            "xy_radius": SUCCESS_XY_RADIUS,
            "max_speed": SUCCESS_MAX_SPEED,
        },
    )


##
# Environment configuration
##


@configclass
class DescendEnvCfg(ManagerBasedRLEnvCfg):
    """Ladder-descent environment (at-height preset, robot at the top)."""

    scene_preset: str = "descend"
    orbit_center: tuple[float, float, float] = (1.4, 0.0, 1.6)
    orbit_radius: float = 3.4
    orbit_height: float = 2.6

    # Homogeneous envs: the at-height preset drops the random dressing fixture itself, so
    # physics replication is safe for training scale (mirrors ClimbEnvCfg).
    scene: G1ReplaceSceneCfg = G1ReplaceSceneCfg(
        num_envs=1, env_spacing=ROOM_ENV_SPACING, replicate_physics=True, clone_in_fabric=False
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_at_height_preset(self.scene, robot_at="top")
        add_ladder_contact_sensor(self.scene)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)

        # family control rate (50 Hz)
        self.decimation = 4
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0

        # PhysX floors + stabilization (family finding, see ClimbEnvCfg)
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 1  # floor, not a target:
        # per-body counts above it are kept; see FamilyBaseEnvCfg.solver_velocity_iterations
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_stabilization = True

        self.viewer.eye = (3.5, 3.5, 2.5)
        self.viewer.lookat = (1.35, 0.0, 1.6)

    def disable_randomization(self) -> None:
        """Deterministic canonical spawns (debug / basic testing; ``--no_randomize``).

        Strips the reset-time randomization terms; ``reset_all`` stays -- restoring
        default state between episodes is correctness, not noise.
        """
        self.events.reset_robot_joints = None
        self.events.reset_robot_root = None
        self.events.randomize_sky_intensity = None
        self.events.randomize_key_light = None
