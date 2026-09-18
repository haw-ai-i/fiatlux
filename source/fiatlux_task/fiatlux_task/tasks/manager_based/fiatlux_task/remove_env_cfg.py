# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Remove-v0`` -- unscrew/remove the existing bulb (standalone RL task).

Start layout: Insert's bench world with the OLD BULB seated in the table lamp -- DYNAMIC,
resting in the socket's open hole under gravity (see ``apply_remove_preset``) -- and an
empty parts crate beside the bench as its destination.

**Honesty note:** the bulb lifts straight out -- nothing gates unscrewing here, so the
task is "pick it up and bin it" rather than "unscrew it". ``FIATLUX-Replace-v0`` now gates
removal on ``mdp.bulb_attachment`` (issue #167, which superseded the issue #54 bayonet);
porting that term here is the remaining work (unification spec Phase 4, issue #76 Step 2).
Note the mechanic is a continuous WRENCH while seated plus a release past an axial
threshold -- not a joint: the revolute/screw and make/break-fixed-joint designs the original
spec proposed were dropped, and rotation gates nothing, since this asset has no lug or
groove to turn into. The reward/termination code below is real, not a placeholder -- it is
Replace's own ``old_bulb_removed`` / ``old_bulb_disposed`` channels, parametrized
to point at this scene's ``bulb`` entity instead of Replace's ``old_bulb``. A policy can
solve it today by lifting the bulb out and binning it; what the mechanic would add is the
requirement to *unscrew* first, which is how Replace now scores the same channels.
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
    ROOM_ENV_SPACING,
    G1ReplaceSceneCfg,
    add_ego_camera,
    add_wrist_camera,
    apply_remove_preset,
)

##
# Task thresholds (Replace's own values for the same mechanic)
##

BULB_ENTITY = SceneEntityCfg("old_bulb")  # this scene's single, dynamic "old" bulb
REMOVAL_CLEARANCE = 0.10  # m; plug this far from the seat counts as removed
OLD_BULB_DROP_HEIGHT = 0.15  # m; must clear a bulb resting *inside* the floor crate (~0.1)

##
# MDP settings
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
    """Two groups: sensor-realizable policy obs and privileged ground-truth obs
    (identical shape to Insert's -- same bench, same entity names)."""

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
        bulb_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": BULB_ENTITY})
        socket_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("socket")})
        bin_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("bin")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Reset-time randomization (mirrors Insert's layering: reset_all first, then re-pose)."""

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
    """Removal + disposal shaping: Replace's own channels, pointed at this scene's ``bulb``."""

    # -- dense normalized progress (Replace's distance_progress scheme) --
    removal_progress = RewTerm(
        func=mdp.distance_progress,
        weight=250.0,
        params={
            "distance_fn": mdp.old_bulb_fixture_clearance,
            "away_threshold": REMOVAL_CLEARANCE,
        },
    )
    disposal_progress = RewTerm(
        func=mdp.distance_progress,
        weight=500.0,
        params={"distance_fn": mdp.old_bulb_disposal_distance},
    )
    # -- sparse completions (each pays once per episode) --
    removed_bonus = RewTerm(
        func=mdp.completion_bonus,
        weight=150.0,
        params={
            "predicate_fn": mdp.old_bulb_removed,
            "predicate_params": {"clearance_threshold": REMOVAL_CLEARANCE, "asset_cfg": BULB_ENTITY},
        },
    )
    # Full success (disposed) terminates the episode on the same step; the raw predicate pays once.
    success_bonus = RewTerm(
        func=mdp.old_bulb_disposed,
        weight=500.0,
        params={"asset_cfg": BULB_ENTITY},
    )
    # -- penalties --
    robot_fall = RewTerm(
        func=mdp.fall_terminated,
        weight=-200.0,
        params={"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT},
    )
    bulb_dropped = RewTerm(
        func=mdp.old_bulb_dropped,
        weight=-100.0,
        params={"min_height": OLD_BULB_DROP_HEIGHT, "asset_cfg": BULB_ENTITY},
    )
    contact_penalty = RewTerm(
        func=mdp.hand_contact_force_l2,
        weight=-1.0e-3,
        params={"sensor_cfg": SceneEntityCfg("hand_contact")},
    )
    # -- smoothness / safety --
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
    # Contract name: recording.py / score.py read the `success` term. Disposal implies
    # removal (Replace's own logic), so only the disposal predicate is checked here.
    success = DoneTerm(
        func=mdp.old_bulb_disposed,
        params={"asset_cfg": BULB_ENTITY},
    )
    bulb_dropped = DoneTerm(
        func=mdp.old_bulb_dropped,
        params={"min_height": OLD_BULB_DROP_HEIGHT, "asset_cfg": BULB_ENTITY},
    )
    fell_below = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": FALL_MIN_HEIGHT})
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": FALL_TILT_LIMIT})


##
# Environment configuration
##


@configclass
class RemoveEnvCfg(ManagerBasedRLEnvCfg):
    """Bulb-removal environment (bench preset, old bulb seated in the table lamp)."""

    scene_preset: str = "remove"
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
        apply_remove_preset(self.scene)
        # Tabletop preset has no ladder; the top-level SceneEntityCfg would fail to resolve.
        self.events.randomize_ladder_scale = None
        # TODO(#76 Step 2): wire ``mdp.bulb_attachment`` here (as Replace does via
        #   ``bulb_attachment_event``) so the bulb is held seated and must be genuinely pulled
        #   out, rather than lifting straight out. Not a joint -- see the module docstring.

        add_wrist_camera(self.scene)
        # GrootPolicy looks up ``scene["ego_camera"]`` unconditionally (same gap
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
        self.events.randomize_light = None
        # A randomization too, though not a reset term.
        self.events.randomize_hand_material = mdp.hand_grip_material_event(randomize=False)
