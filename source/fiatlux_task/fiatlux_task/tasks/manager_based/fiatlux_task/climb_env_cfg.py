# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Climb-v0`` -- bipedal ladder ascent (functional RL task).

The G1 starts at the base of the kinematic A-frame step ladder (family *at-height*
preset: elevated chandelier above the ladder, bulb parked on the floor) and must
climb until its pelvis reaches the upper steps -- the working height for the
at-fixture manipulation tasks. Built as a standard ``ManagerBasedRLEnvCfg`` so it
slots into the usual train / play / eval scripts.

Design notes (mirrors ``g1_bulb_env_cfg`` where the tasks overlap):
- **Actions** are whole-body joint-position targets (all ~53 DoF incl. fingers; the
  per-phase action-space split is a family decision, see the unification spec).
- **Observations**: the ``policy`` group holds IMU terms, proprioception, last
  action, the limb-on-ladder contact forces, a head-mounted RGB camera, and a
  head-mounted lidar (ground + ladder ranges) -- the family's standard whole-body sensor suite
  (``fiatlux_task.sensors``), the same one Carry/Replace attach. Estimated base
  height and linear velocity are included as a documented *estimator-realizable*
  exception to the sensor-only contract -- the real G1 publishes both from its
  kinematic-inertial estimator (the same argument Isaac Lab's velocity tasks make).
  Ground-truth poses (robot root, ladder) stay in the ``privileged`` group, which
  reaches only the critic (see ``ClimbPPORunnerCfg.obs_groups``).
- **Rewards** pay progressive height gain (each centimetre once) plus a small
  limb-on-ladder contact bootstrap, and penalize CoM sway, wobble, falls, and the
  usual smoothness/limit terms. ``flat_orientation_l2`` is deliberately absent:
  climbing an A-frame requires a sustained forward lean.
- **Terminations** implement the family's fall-detection RL gate (base height +
  tilt thresholds end solver-kick episodes immediately) and a ``success`` term
  (name consumed by ``recording.py``/``score.py``).

The env runs at the family control rate (50 Hz, like every task in the family --
the GEAR-WBC decoders enforce it). A shared RL family base becomes worthwhile
once Descend upgrades to a second RL member (unification spec, Phase 4).
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
from .scene_cfg import (
    ROOM_ENV_SPACING,
    TOP_ROBOT_POSITION,
    G1ReplaceSceneCfg,
    add_ego_camera,
    add_ladder_contact_sensor,
    add_mid360_lidar,
    apply_at_height_preset,
)

##
# Task geometry
##

# Success gate derived from the descend task's start pose (pelvis at the upper
# steps): slightly below it so a solid stance on the top steps scores.
SUCCESS_HEIGHT = TOP_ROBOT_POSITION[2] - 0.15  # 1.70 m
SUCCESS_XY = (TOP_ROBOT_POSITION[0], TOP_ROBOT_POSITION[1])
SUCCESS_XY_RADIUS = 0.6  # m; with the max-speed cap, rejects ballistic fly-throughs
SUCCESS_MAX_SPEED = 1.5  # m/s

# Fall thresholds, shared by the terminations and the fall penalty (which recomputes
# them; see mdp.fall_terminated). Standing pelvis is 0.75 m, a deep mounting crouch
# stays > 0.45 m, a collapsed/draped robot reads < 0.30 m; the climb lean is
# ~12-35 deg while beyond ~57 deg a position-controlled G1 cannot recover.
FALL_MIN_HEIGHT = 0.35  # m, world frame (the floor is flat)
FALL_TILT_LIMIT = 1.0  # rad

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
    """Two groups: sensor/estimator-realizable policy obs and privileged ground truth."""

    @configclass
    class PolicyCfg(ObsGroup):
        """IMU + estimator + proprioception + limb contact (see module docstring)."""

        # IMU-realizable
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # Estimator-realizable exceptions (kinematic-inertial state estimate).
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        # Proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # Net contact force per climb limb (feet + palms), world frame.
        ladder_limb_contact = ObsTerm(
            func=mdp.contact_net_forces,
            scale=0.1,
            params={"sensor_cfg": SceneEntityCfg("ladder_contact")},
        )
        # Exteroception: ego RGB (features) + head lidar ranges. Requires
        # launching with --enable_cameras.
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
        """Ground-truth states for the critic / scripted baselines only.

        No socket term: the chandelier is visual dressing for this task (success is
        geometric), and ``verify_interactions.py``'s ladder scenario drops it.
        """

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
    # Runs after reset_all (cfg order) so it re-poses only the robot root.
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

    # Grip friction for the hands (startup, through the PhysX view -- see
    # mdp.hand_grip_material_event for why this cannot be a USD material bind).
    randomize_hand_material = mdp.hand_grip_material_event()


@configclass
class RewardsCfg:
    """Ascent + stability shaping. Rewards accrue as ``func * weight * step_dt``
    (dt = 0.02 s), so the main-task episode total is ~+11 (full 1.1 m climb) +10
    (success bonus); every shaping penalty is sized well below that."""

    # -- main task --
    climb_progress = RewTerm(func=mdp.climb_height_progress, weight=500.0)
    success_bonus = RewTerm(
        func=mdp.climbed_to_target,
        weight=500.0,
        params={
            "minimum_height": SUCCESS_HEIGHT,
            "xy_center": SUCCESS_XY,
            "xy_radius": SUCCESS_XY_RADIUS,
            "max_speed": SUCCESS_MAX_SPEED,
        },
    )
    # -- bootstrap: touch the ladder (max ~0.005/step; unfarmable vs the task total) --
    ladder_contact = RewTerm(
        func=mdp.ladder_contact_fraction,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("ladder_contact"), "threshold": 1.0},
    )
    # -- stability shaping (no flat_orientation term: climbing needs a sustained lean) --
    com_sway = RewTerm(func=mdp.com_sway_l2, weight=-0.5)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # Fires exactly on the fell_below/fell_over step (recomputed predicates; neither
    # mdp.is_terminated -- would also penalize success -- nor mdp.is_terminated_term
    # -- reads the sticky which-term-fired log, repeating the penalty every step
    # after an env's first fall -- does the right thing here).
    termination_penalty = RewTerm(
        func=mdp.fall_terminated,
        weight=-200.0,
        params={"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT},
    )
    # -- smoothness / joint discipline (weights per the G1 rough-locomotion recipe) --
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
    # Fall detection (the family RL gate: end solver-kick episodes immediately).
    # Thresholds shared with the fall penalty; rationale at FALL_* above.
    fell_below = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": FALL_MIN_HEIGHT})
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": FALL_TILT_LIMIT})
    # Contract name: recording.py / score.py read the `success` term.
    success = DoneTerm(
        func=mdp.climbed_to_target,
        params={
            "minimum_height": SUCCESS_HEIGHT,
            "xy_center": SUCCESS_XY,
            "xy_radius": SUCCESS_XY_RADIUS,
            "max_speed": SUCCESS_MAX_SPEED,
        },
    )


##
# Environment configuration
##


@configclass
class ClimbEnvCfg(ManagerBasedRLEnvCfg):
    """Ladder-climbing environment (at-height preset, robot at the base)."""

    scene_preset: str = "climb"
    # frame the ladder from its feet up to the chandelier
    orbit_center: tuple[float, float, float] = (1.4, 0.0, 1.5)
    orbit_radius: float = 3.4
    orbit_height: float = 2.6

    # Homogeneous envs: the at-height preset drops the random dressing fixture
    # itself, so physics replication is safe for training scale. USD cloning (not
    # fabric) keeps the PhysX contact reporters attachable.
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
        apply_at_height_preset(self.scene, robot_at="base")
        add_ladder_contact_sensor(self.scene)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)

        # family control rate (50 Hz)
        self.decimation = 4
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0

        # PhysX floors + stabilization (family finding: an uncontrolled free-base G1
        # against kinematic furniture accumulates solver kicks; see unification spec)
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 1  # floor, not a target:
        # per-body counts above it are kept; see FamilyBaseEnvCfg.solver_velocity_iterations
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_stabilization = True

        self.viewer.eye = (3.5, 3.5, 2.5)
        self.viewer.lookat = (1.4, 0.0, 1.2)

    def disable_randomization(self) -> None:
        """Deterministic canonical spawns (debug / basic testing; ``--no_randomize``).

        Strips the reset-time randomization terms; ``reset_all`` stays -- restoring
        default state between episodes is correctness, not noise.
        """
        self.events.reset_robot_joints = None
        self.events.reset_robot_root = None
        self.events.randomize_sky_intensity = None
        self.events.randomize_key_light = None
