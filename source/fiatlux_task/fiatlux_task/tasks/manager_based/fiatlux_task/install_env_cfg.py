# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Install-v0`` -- insert/screw in a new bulb (functional RL task).

The screw-in counterpart of ``FIATLUX-Insert-v0`` on the same bench world: the table
lamp's socket starts EMPTY and the fresh bulb rests in a parts crate on the floor
beside the bench (instead of Insert's bulb-at-hand-height tabletop start).

Since Install's preset (``apply_install_preset``) uses the exact same scene entity
names as Insert's tabletop preset (``bulb``, ``socket``) and a plain *dynamic*
(non-kinematic) bulb -- unlike Remove's kinematic stand-in -- this task's Actions /
Observations / Rewards / Terminations are Insert's own, unchanged: the extra distance
from the floor crate to the socket is exactly what ``mdp.object_socket_distance*``
already rewards, just over a larger starting gap. See ``g1_bulb_env_cfg.py`` for the
full design rationale (kept in sync deliberately rather than imported, matching how
Climb/Carry/Replace each carry their own copy of the shared action-space convention).
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
from .climb_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT
from .scene_cfg import (
    BIN_BULB_POSITION,
    ROOM_ENV_SPACING,
    G1ReplaceSceneCfg,
    add_ego_camera,
    add_wrist_camera,
    apply_install_preset,
)

##
# Drop detection
##

# Unlike Insert's bulb (starts at hand height on the table, min_height=0.4 catches any
# floor-level drop), Install's bulb legitimately STARTS at floor level in the parts crate
# (BIN_BULB_POSITION z=0.15 m) -- a bare height gate would fire on the very first step.
# Reuses old_bulb_dropped's height-AND-away-from-bin logic (same "resting in a floor
# container isn't a drop" case Remove/Replace already handle) instead of object_dropped's
# single height gate.
BULB_DROP_HEIGHT = BIN_BULB_POSITION[2] + 0.05  # m; just above the crate's resting height
BIN_CLEARANCE = 0.25  # m; matches Remove/Replace's disposal-crate proximity scale

##
# MDP settings (Insert's own -- see g1_bulb_env_cfg.py)
##


@configclass
class ActionsCfg:
    """Whole-body joint-position targets (family convention: one term sized to every DoF)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Two groups: sensor-realizable policy obs and privileged ground-truth obs."""

    @configclass
    class PolicyCfg(ObsGroup):
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
        bulb_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("bulb")})
        socket_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("socket")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Reset-time randomization (Insert's own layering: reset_all first, then re-pose)."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

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
        func=mdp.randomize_light_properties,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("dome_light"), "intensity_range": (500.0, 2000.0)},
    )

    # Grip friction for the hands (startup, through the PhysX view -- see
    # mdp.hand_grip_material_event for why this cannot be a USD material bind).
    randomize_hand_material = mdp.hand_grip_material_event()


@configclass
class RewardsCfg:
    """Reward terms: drive the bulb into the socket, gently (Insert's own set, unchanged)."""

    align_position = RewTerm(func=mdp.object_socket_distance, weight=-1.0)
    align_position_tanh = RewTerm(func=mdp.object_socket_distance_tanh, weight=0.5, params={"std": 0.1})
    seat_position_exp = RewTerm(func=mdp.object_socket_distance_exp, weight=1.0, params={"sigma": 0.02})
    align_orientation = RewTerm(func=mdp.object_socket_orientation_tanh, weight=0.3, params={"std": 0.3})
    seated_bonus = RewTerm(
        func=mdp.bulb_seated,
        weight=5.0,
        params={"pos_threshold": 0.015, "ori_threshold": 0.2},
    )
    contact_penalty = RewTerm(
        func=mdp.hand_contact_force_l2,
        weight=-1.0e-3,
        params={"sensor_cfg": SceneEntityCfg("hand_contact")},
    )
    robot_fall = RewTerm(
        func=mdp.fall_terminated,
        weight=-200.0,
        params={"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT},
    )
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
    """Termination terms (Insert's own set, unchanged)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.bulb_seated, params={"pos_threshold": 0.015, "ori_threshold": 0.2})
    bulb_dropped = DoneTerm(
        func=mdp.old_bulb_dropped,
        params={
            "min_height": BULB_DROP_HEIGHT,
            "disposal_threshold": BIN_CLEARANCE,
            "asset_cfg": SceneEntityCfg("bulb"),
            "bin_cfg": SceneEntityCfg("bin"),
        },
    )
    fell_below = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": FALL_MIN_HEIGHT})
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": FALL_TILT_LIMIT})


##
# Environment configuration
##


@configclass
class InstallEnvCfg(ManagerBasedRLEnvCfg):
    """Bulb-installation environment (bench preset, fresh bulb in the floor crate)."""

    scene_preset: str = "install"
    # bench framing, wide enough to include the floor crate
    orbit_center: tuple[float, float, float] = (0.4, -0.2, 1.0)
    orbit_radius: float = 3.4
    orbit_height: float = 2.2

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
        apply_install_preset(self.scene)
        self.scene.fixture = None

        add_wrist_camera(self.scene)
        # GrootPolicy looks up ``scene["ego_camera"]`` unconditionally (same gap
        # g1_bulb_env_cfg had before it was fixed); every family member needs it.
        add_ego_camera(self.scene)

        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0
        self.sim.dt = 1.0 / 200.0
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 1  # floor, not a target:
        # per-body counts above it are kept; see FamilyBaseEnvCfg.solver_velocity_iterations
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_stabilization = True
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.4, -0.2, 1.0)

    def disable_randomization(self) -> None:
        """Deterministic canonical spawns (debug / basic testing; ``--no_randomize``).

        Strips the reset-time randomization terms; ``reset_all`` stays -- restoring
        default state between episodes is correctness, not noise.
        """
        self.events.reset_robot_joints = None
        self.events.reset_socket = None
        self.events.reset_bulb = None
        self.events.randomize_light = None
        # Not a reset term, but a randomization all the same: unpinned, every grasp
        # force measured downstream is seed-dependent.
        self.events.randomize_hand_material = mdp.hand_grip_material_event(randomize=False)
