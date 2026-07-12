# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Carry-v0`` -- the G1 grasps a ladder and positions it upright at a target.

The ladder-handling / positioning subtask: a **dynamic, high-friction** ladder starts out in
front of the G1; the robot grasps a rail (physics friction, no weld) and positions it upright
beneath the ceiling light fixture. Built as a standard ``ManagerBasedRLEnvCfg`` (like the
Insert task) so it slots into train / play / eval.

Scoring **reuses the full Replace task's ladder terms** (one shared source of truth):
``ladder_fixture_distance`` progress + the ``ladder_ready`` completion/success predicate
(ladder top horizontally within reach of the fixture, upright) + a ``ladder_tipped`` penalty,
plus a hand↔ladder grasp-contact bootstrap and the standard fall / smoothness shaping. Actions
are the right arm + Inspire hand; the free legged base holds the standing pose via PD.
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

from fiatlux_task.robots.g1 import G1_ARM_JOINTS, G1_EE_BODY, G1_HAND_JOINTS

from . import mdp
from .scene_cfg import (
    G1ReplaceSceneCfg,
    apply_position_preset,
)

# Ladder-positioning tolerances -- reused from the full Replace task's ladder scoring so the
# subtask and the full task judge the ladder identically (mdp.ladder_ready / ladder_tipped).
LADDER_TILT_LIMIT = 0.6  # rad; the upright A-frame stands at ~0
LADDER_READY_XY_RADIUS = 0.9  # m; ladder top horizontally within working reach of the fixture
FALL_MIN_HEIGHT = 0.4  # m; robot-fall gate (shared by the penalty + termination)
FALL_TILT_LIMIT = 1.0  # rad


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Joint-position targets on the G1 right arm + Inspire hand (base held by PD)."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=G1_ARM_JOINTS, scale=0.5, use_default_offset=True
    )
    hand_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=G1_HAND_JOINTS, scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Two groups: sensor-realizable policy obs and privileged ground-truth obs."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Sensor-realizable observations (available on the real robot)."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS + G1_HAND_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS + G1_HAND_JOINTS)},
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
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Ground-truth ("cheat") observations for the critic / scripted baselines."""

        ladder_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("ladder")})

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
        params={"asset_cfg": SceneEntityCfg("dome_light"), "intensity_range": (600.0, 1400.0)},
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


@configclass
class RewardsCfg:
    """Score channels -- reuse the full Replace task's ladder scoring: dense normalized
    progress of the ladder toward the fixture, a sparse 'ladder positioned upright within
    reach' completion bonus, and a tip penalty; plus a grasp-contact bootstrap and the
    standard compliance / smoothness / fall shaping."""

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
                "xy_radius": LADDER_READY_XY_RADIUS,
                "tilt_limit": LADDER_TILT_LIMIT,
            },
        },
    )
    # -- penalties (each predicate also terminates -> fires once) --
    ladder_tipped = RewTerm(
        func=mdp.ladder_tipped, weight=-200.0, params={"tilt_limit": LADDER_TILT_LIMIT}
    )
    robot_fall = RewTerm(
        func=mdp.fall_terminated,
        weight=-200.0,
        params={"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT},
    )
    # -- compliance / smoothness --
    contact_penalty = RewTerm(
        func=mdp.hand_contact_force_l2, weight=-1.0e-4, params={"sensor_cfg": SceneEntityCfg("hand_contact")}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1.0e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS)}
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-1.0e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS)}
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS)}
    )


@configclass
class TerminationsCfg:
    """Horizon, success (ladder positioned upright within reach), and fall/tip/drop violations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # success: the Replace task's ladder-ready predicate (xy within reach + upright)
    success = DoneTerm(
        func=mdp.ladder_ready,
        params={"xy_radius": LADDER_READY_XY_RADIUS, "tilt_limit": LADDER_TILT_LIMIT},
    )
    ladder_tipped = DoneTerm(func=mdp.ladder_tipped, params={"tilt_limit": LADDER_TILT_LIMIT})
    ladder_dropped = DoneTerm(
        func=mdp.object_dropped, params={"asset_cfg": SceneEntityCfg("ladder"), "min_height": 0.2}
    )
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
        num_envs=1, env_spacing=6.0, replicate_physics=True, clone_in_fabric=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        apply_position_preset(self.scene)

        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        # PhysX solver floors + stabilization (uncontrolled free base against props; Insert finding)
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 8
        # low velocity-iteration floor: high TGS velocity iters jitter resting bodies (ladder)
        self.sim.physx.min_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_stabilization = True
        self.viewer.eye = (2.5, 2.5, 2.0)
        self.viewer.lookat = (0.5, 0.0, 0.9)
