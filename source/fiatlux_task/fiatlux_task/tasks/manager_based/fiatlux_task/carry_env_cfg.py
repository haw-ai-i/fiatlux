# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``CarryEnvCfg`` -- the G1 grasps a ladder and positions it upright at a target.

The ladder-handling / positioning subtask: a **dynamic, high-friction** ladder starts out in
front of the G1; the robot **walks to it**, grasps a rail (physics friction, no weld), and
carries it upright to beneath the ceiling light fixture. **Whole-body** control (`joint_names=
[".*"]`, like the full Replace task) so locomotion + manipulation are both available. Built as a
standard ``ManagerBasedRLEnvCfg`` so it slots into train / play / eval.

Scoring **reuses the full Replace task's ladder terms** (one shared source of truth):
``ladder_fixture_distance`` progress + the ``ladder_ready`` completion/success predicate
(ladder top horizontally within reach of the fixture, upright) + a ``ladder_tipped`` penalty,
plus fall + the Climb/Replace whole-body stability shaping (CoM sway, ankle limits, ...).
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
    LADDER_READY_FACING_TOLERANCE,
    LADDER_READY_REACH,
    ROOM_ENV_SPACING,
    SUBTASK_EPISODE_LENGTH_S,
    G1ReplaceSceneCfg,
    add_ego_camera,
    add_mid360_lidar,
    apply_position_preset,
)

# Ladder-positioning tolerances -- reused from the full Replace task's ladder scoring so the
# subtask and the full task judge the ladder identically (mdp.ladder_ready / ladder_tipped).
LADDER_TILT_LIMIT = 0.6  # rad; the upright ladder stands at ~0
FALL_MIN_HEIGHT = 0.4  # m; robot-fall gate (shared by the penalty + termination)
FALL_TILT_LIMIT = 1.0  # rad


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Whole-body joint-position targets: the task spans locomotion + manipulation (the robot
    walks to the ladder, then grasps and carries it), mirroring the full Replace task."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Two groups: sensor-realizable policy obs and privileged ground-truth obs."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Sensor-realizable whole-body observations (IMU + estimator + proprioception),
        mirroring the Climb / Replace locomotion recipe."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        hand_contact = ObsTerm(
            func=mdp.contact_net_forces,
            scale=0.1,
            params={"sensor_cfg": SceneEntityCfg("hand_contact")},
        )
        left_hand_contact = ObsTerm(
            func=mdp.contact_net_forces,
            scale=0.1,
            params={"sensor_cfg": SceneEntityCfg("left_hand_contact")},
        )
        # Exteroception: ego RGB (features) + head lidar ranges (the ladder being
        # carried is the salient thing to range). Requires --enable_cameras.
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

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Ground-truth ("cheat") observations for the critic: exact robot / ladder / fixture poses."""

        robot_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("robot")})
        ladder_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("ladder")})
        fixture_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("socket")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Reset-time events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_sky_intensity = EventTerm(
        func=mdp.randomize_light_properties,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("dome_light"),
            "intensity_range": (600.0, 1400.0),
            "rotation_range_deg": {"yaw": (0.0, 360.0)},
        },
    )
    # Replicate-safe visual DR: key-light orientation and a global albedo tint on the shared
    # room.
    randomize_key_light = EventTerm(
        func=mdp.randomize_light_properties,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("key_light"),
            "rotation_range_deg": {"pitch": (-15.0, 15.0), "yaw": (-30.0, 30.0)},
        },
    )
    randomize_material_tint = EventTerm(
        func=mdp.randomize_material_tint,
        mode="reset",
        params={"asset_cfgs": [SceneEntityCfg("room")]},
    )
    # small start-pose DR on the ladder (position + yaw) for robustness
    reset_ladder = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ladder"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.2, 0.2)},
            "velocity_range": {},
        },
    )

    # Grip friction for the hands (startup, through the PhysX view -- see
    # mdp.hand_grip_material_event for why this cannot be a USD material bind).
    randomize_hand_material = mdp.hand_grip_material_event()


@configclass
class RewardsCfg:
    """Score channels -- reuse the full Replace task's ladder scoring: dense normalized
    progress of the ladder toward the fixture, a sparse 'ladder positioned upright within
    reach' completion bonus, and a tip penalty; plus fall + the Climb/Replace whole-body
    stability / smoothness shaping (the robot must stay balanced while walking + carrying)."""

    # -- dense normalized progress: ladder top -> fixture socket seat (randomization-fair) --
    ladder_progress = RewTerm(
        func=mdp.distance_progress,
        weight=250.0,
        params={"distance_fn": mdp.ladder_fixture_distance},
    )
    # -- sparse completion: ladder positioned upright within reach of the fixture (pays once) --
    ladder_ready = RewTerm(
        func=mdp.completion_bonus,
        weight=100.0,
        params={
            "predicate_fn": mdp.ladder_ready,
            "predicate_params": {
                "reach": LADDER_READY_REACH,
                "facing_tolerance": LADDER_READY_FACING_TOLERANCE,
                "tilt_limit": LADDER_TILT_LIMIT,
            },
        },
    )
    # -- penalties (each predicate also terminates -> fires once) --
    ladder_tipped = RewTerm(func=mdp.ladder_tipped, weight=-200.0, params={"tilt_limit": LADDER_TILT_LIMIT})
    robot_fall = RewTerm(
        func=mdp.fall_terminated,
        weight=-200.0,
        params={"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT},
    )
    contact_penalty = RewTerm(
        func=mdp.hand_contact_force_l2, weight=-1.0e-4, params={"sensor_cfg": SceneEntityCfg("hand_contact")}
    )
    # -- whole-body stability / smoothness shaping --
    com_sway = RewTerm(func=mdp.com_sway_l2, weight=-0.1)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
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
    """Horizon, success (ladder positioned upright within reach), and fall/tip violations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # success: the Replace task's ladder-ready predicate (palm can reach, stance faces it, upright)
    success = DoneTerm(
        func=mdp.ladder_ready,
        params={
            "reach": LADDER_READY_REACH,
            "tilt_limit": LADDER_TILT_LIMIT,
            "facing_tolerance": LADDER_READY_FACING_TOLERANCE,
        },
    )
    ladder_tipped = DoneTerm(func=mdp.ladder_tipped, params={"tilt_limit": LADDER_TILT_LIMIT})
    # No height-gate "dropped" check here: this ladder's
    # root frame sits at ~0 m when resting upright on the floor (verified via a live probe --
    # see the groot-scoring debug notes), the same as its correctly-resting state, so any
    # min_height threshold above 0 trips on step 1 of every episode regardless of policy.
    # Replace's own ladder handling relies on orientation alone (`ladder_tipped`) for exactly
    # this reason; Carry follows suit.
    # robot fall detection (built-in bool terms; end solver-kick episodes immediately)
    fell_below = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": FALL_MIN_HEIGHT})
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": FALL_TILT_LIMIT})


@configclass
class CarryEnvCfg(ManagerBasedRLEnvCfg):
    """Ladder-handling / positioning subtask: grasp the ladder and plant it upright at a target."""

    scene_preset: str = "carry"
    # orbit framing (verify_scene --record): wide spread (robot, ladder ~2 m out, ceiling
    # light ~3 m up), so the full approach-and-carry layout stays in frame.
    orbit_center: tuple[float, float, float] = (0.55, 0.0, 1.9)
    orbit_radius: float = 4.6
    orbit_height: float = 1.5

    # Homogeneous envs -> replicated physics for training scale (USD cloning, not fabric, so
    # the hand_contact sensor's PhysX contact-reporter attaches; same as Insert).
    scene: G1ReplaceSceneCfg = G1ReplaceSceneCfg(
        num_envs=1, env_spacing=ROOM_ENV_SPACING, replicate_physics=True, clone_in_fabric=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        apply_position_preset(self.scene)
        add_ego_camera(self.scene)
        add_mid360_lidar(self.scene)

        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = SUBTASK_EPISODE_LENGTH_S
        # family control rate (50 Hz; the GEAR-WBC decoders enforce it)
        self.sim.dt = 1.0 / 200.0
        # PhysX solver floors + stabilization (uncontrolled free base against props; Insert finding)
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 8
        # low velocity-iteration floor: high TGS velocity iters jitter resting bodies (ladder)
        self.sim.physx.min_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_stabilization = True
        self.viewer.eye = (2.5, 2.5, 2.0)
        self.viewer.lookat = (0.5, 0.0, 0.9)

    def disable_randomization(self) -> None:
        """Deterministic canonical spawns (debug / basic testing; ``--no_randomize``).

        Strips the reset-time randomization terms; ``reset_all`` stays -- restoring
        default state between episodes is correctness, not noise.
        """
        self.events.reset_ladder = None
        self.events.randomize_sky_intensity = None
        # A randomization too, though not a reset term.
        self.events.randomize_hand_material = mdp.hand_grip_material_event(randomize=False)
