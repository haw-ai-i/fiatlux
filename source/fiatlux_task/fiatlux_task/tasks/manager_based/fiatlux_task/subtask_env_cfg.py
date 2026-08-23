# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared base for the subtask family.

Each subtask is its own module, class and versioned gym id; this base holds what is identical
across all of them and fixes the canonical name for each shared channel. ``replace_env_cfg``
documents the per-term episode sums in ``extras['log']`` as the score breakdown, so one concept
under two names cannot be aggregated across tasks.

Layers, of which this file is the first two:

    SubtaskEnvCfg          obs, actions, events, shaping rewards, solver, fall gates
      NavigateSubtaskCfg   walk-to-a-target channels and the arrival gate's shape
        <leaf>             the thresholds, the predicate, the preset, the horizon

Leaves supply their behaviour through the declared hook fields (``success_predicate``,
``progress_distance_fn``), which ``__post_init__`` validates and wires in one place: the
``success`` termination owns the gate, and ``success_bonus`` reads that term's flag rather than
re-evaluating it. A leaf that forgets a hook fails at construction.

``ClassVar`` does not work for those hooks: ``@configclass`` ignores the annotation and makes
them ordinary dataclass fields. Plain functions work as defaults without binding, and
``to_dict()`` skips the callables.
"""
import os
from collections.abc import Callable

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
from .scene_cfg import ROOM_ENV_SPACING, G1ReplaceSceneCfg

# Fall gates, one definition for every subtask. Standing pelvis is 0.79 m, a deep mounting crouch
# stays above 0.35 m, a collapsed robot reads under 0.30 m; beyond ~57 deg a position-controlled
# G1 cannot recover.
FALL_MIN_HEIGHT = 0.35  # m, world frame (the floor is flat)
FALL_TILT_LIMIT = 1.0  # rad

# Arrival gate shape, shared by every navigation subtask.
ARRIVAL_FACING_TOLERANCE = 0.5  # rad
ARRIVAL_MAX_SPEED = 1.0  # m/s; rejects scoring while still charging at the target


@configclass
class SubtaskActionsCfg:
    """Whole-body joint-position targets. Every subtask spans enough of the body to need them."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class SubtaskObservationsCfg:
    """Two groups: sensor-realizable ``policy`` and ground-truth ``privileged``.

    The ``policy`` group is FROZEN across all subtasks and must not be extended by a subclass. The
    argument is sim-to-real, not tidiness: this is the sensor-realizable mode, and real hardware
    does not change its sensor suite between subtasks. One observation space for the family also
    means a single policy can attempt any of them and GR00T's adapter needs no per-subtask change.
    It stays fixed-width because the filtered hand-contact channel sums over its filter targets,
    giving ``(N, B, 3)`` whichever objects a subtask filters for -- only the meaning changes.

    Extend ``privileged`` instead: it reaches only the critic, and rsl_rl's ``obs_groups`` routing
    is per-task anyway.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # IMU-realizable
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # Estimator-realizable: the real G1 publishes both from its kinematic-inertial estimator.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        # Proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # Force on the manipulated objects (filtered channel, not the unfiltered net force).
        hand_contact = ObsTerm(
            func=mdp.contact_net_forces, scale=0.1, params={"sensor_cfg": SceneEntityCfg("hand_contact")}
        )
        # Exteroception; needs --enable_cameras.
        ego_rgb = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("ego_camera"), "data_type": "rgb", "model_name": "resnet18"},
        )
        lidar_ranges = ObsTerm(func=mdp.lidar_ranges, scale=0.1, params={"sensor_cfg": SceneEntityCfg("mid360_lidar")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        root_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("robot")})
        root_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class SubtaskEventCfg:
    """Reset and randomization. Declaration order IS execution order for the event manager."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )
    # After reset_all, so it re-poses only the robot root.
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
        params={
            "asset_cfg": SceneEntityCfg("dome_light"),
            "intensity_range": (600.0, 1400.0),
            "rotation_range_deg": {"yaw": (0.0, 360.0)},
        },
    )
    randomize_key_light = EventTerm(
        func=mdp.randomize_light_properties,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("key_light"),
            "intensity_range": (800.0, 2200.0),
            "rotation_range_deg": {"pitch": (-15.0, 15.0), "yaw": (-30.0, 30.0)},
        },
    )
    randomize_material_tint = EventTerm(
        func=mdp.randomize_material_tint, mode="reset", params={"asset_cfgs": [SceneEntityCfg("room")]}
    )
    randomize_hand_material = mdp.hand_grip_material_event()


@configclass
class SubtaskRewardsCfg:
    """Shaping and discipline, identical for every subtask. Task channels go in a subclass.

    ``robot_fall`` is the canonical name for the fall penalty. It recomputes the fall predicates
    rather than using ``mdp.is_terminated`` (which would also punish success) or
    ``mdp.is_terminated_term`` (which reads a sticky log and would repeat every step after the
    first fall).

    ``success_bonus`` is the canonical name for the completion bonus. It reads the ``success``
    termination's flag rather than re-evaluating the gate, so the two cannot drift even when the
    gate is stateful, and it pays exactly on the terminating step.
    """

    success_bonus = RewTerm(func=mdp.success_term_fired, weight=500.0)
    robot_fall = RewTerm(
        func=mdp.fall_terminated,
        weight=-200.0,
        params={"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT},
    )
    com_sway = RewTerm(func=mdp.com_sway_l2, weight=-0.5)
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
class SubtaskTerminationsCfg:
    """Horizon, fall detection, and success. ``success`` is a contract name that
    ``recording.py`` / ``score.py`` read; its ``func`` is filled from the leaf's hook."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_below = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": FALL_MIN_HEIGHT})
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": FALL_TILT_LIMIT})
    # Placeholder func; SubtaskEnvCfg.__post_init__ replaces it with the leaf's predicate.
    success = DoneTerm(func=mdp.time_out)


@configclass
class SubtaskEnvCfg(ManagerBasedRLEnvCfg):
    """Everything the fifteen subtasks share. Subclass per mode, then per subtask."""

    # -- leaf contract; __post_init__ raises if a leaf leaves one unset --
    success_predicate: Callable | None = None
    success_params: dict | None = None

    scene: G1ReplaceSceneCfg = G1ReplaceSceneCfg(
        num_envs=1, env_spacing=ROOM_ENV_SPACING, replicate_physics=True, clone_in_fabric=False
    )
    observations: SubtaskObservationsCfg = SubtaskObservationsCfg()
    actions: SubtaskActionsCfg = SubtaskActionsCfg()
    events: SubtaskEventCfg = SubtaskEventCfg()
    rewards: SubtaskRewardsCfg = SubtaskRewardsCfg()
    terminations: SubtaskTerminationsCfg = SubtaskTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.success_predicate is None:
            raise ValueError(f"{type(self).__name__} must set success_predicate")
        # Only the termination carries the gate. success_bonus reads its result (see mdp/gates.py),
        # so there is one definition of success even when the gate is stateful.
        self.terminations.success.func = self.success_predicate
        self.terminations.success.params = dict(self.success_params or {})

        # Family control rate (50 Hz).
        self.decimation = 4
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation

        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_stabilization = True

        # FIATLUX_DEBUG_NO_RANDOMIZE: deterministic spawns without a caller asking, matching
        # scene_cfg's layout-seed default.
        if os.environ.get("FIATLUX_DEBUG_NO_RANDOMIZE"):
            self.disable_randomization()

    def disable_randomization(self) -> None:
        """Deterministic spawns (``--no_randomize``). ``reset_all`` stays: restoring default state
        between episodes is correctness, not noise."""
        self.events.reset_robot_joints = None
        self.events.reset_robot_root = None
        self.events.randomize_sky_intensity = None
        self.events.randomize_key_light = None
        self.events.randomize_hand_material = mdp.hand_grip_material_event(randomize=False)


@configclass
class NavigateRewardsCfg(SubtaskRewardsCfg):
    """Walk-to-a-target channels. ``distance_fn`` and the predicate come from the leaf."""

    approach_progress = RewTerm(
        func=mdp.distance_progress, weight=500.0, params={"distance_fn": mdp.base_ladder_distance}
    )


@configclass
class NavigateSubtaskCfg(SubtaskEnvCfg):
    """A subtask whose job is to walk somewhere and stand there.

    Progress is normalized per episode by ``distance_progress``, so a layout draw that spawns the
    robot close cannot outscore a far one -- both saturate at 1.0 on arrival.
    """

    progress_distance_fn: Callable | None = None
    rewards: NavigateRewardsCfg = NavigateRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.progress_distance_fn is None:
            raise ValueError(f"{type(self).__name__} must set progress_distance_fn")
        self.rewards.approach_progress.params["distance_fn"] = self.progress_distance_fn
