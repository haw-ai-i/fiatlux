# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base (non-RL) ``ManagerBasedEnvCfg`` wiring the Fiatlux ladder scene to its managers.

Deliberately uses the **non-RL** ``ManagerBasedEnv`` base: it has observation, action and event
managers only -- NO reward / termination / command / curriculum managers and no policy/training
code. The five ladder-family tasks subclass :class:`G1LadderEnvCfg` and add their own task
logic later (switching to ``ManagerBasedRLEnvCfg``; see ``docs/roadmap.md``).

The event manager carries one *enabled* reset term plus a *disabled* randomization scaffold:
every future randomization knob is written out, commented, wired to the right
``SceneEntityCfg`` and tagged ``TODO`` -- enabling randomization later is just uncommenting.
"""

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg  # noqa: F401  -- used by the disabled randomization scaffold below
from isaaclab.utils import configclass

from . import mdp
from .scene_cfg import G1LadderSceneCfg


##
# Observations -- generic G1 proprioception (no task-specific terms yet).
##


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive observations, concatenated into a single vector for later use."""

        # joint state (relative to the default standing pose)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # root state. NOTE: root_pos_w is world-frame (includes the per-env origin); fine for a
        # foundation, swap to base-frame terms when you define task observations.
        root_pos = ObsTerm(func=mdp.root_pos_w)
        root_quat = ObsTerm(func=mdp.root_quat_w)
        root_lin_vel = ObsTerm(func=mdp.root_lin_vel_w)
        root_ang_vel = ObsTerm(func=mdp.root_ang_vel_w)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Actions -- position control of the G1, default pose as offset (zero action == hold standing).
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,  # offset == default standing pose -> zero action holds it
    )


##
# Events -- one enabled reset term + a disabled, ready-to-uncomment randomization scaffold.
##


@configclass
class EventCfg:
    """Configuration for events."""

    # -------- ENABLED: return robot + props to their configured default states on reset --------
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # =====================================================================================
    #  DISABLED randomization scaffold -- uncomment + tune to enable. Built-in mdp terms are
    #  preferred; lighting uses the project stub mdp.randomize_light_properties. Each term is
    #  already wired to the correct SceneEntityCfg(name=...).
    # =====================================================================================

    # ---- (A) asset SCALE: ladder / lamp / bulb -- mdp.randomize_rigid_body_scale ----
    #   TODO: scale runs on the USD stage, so use mode="prestartup". Replicate per prop.
    # randomize_ladder_scale = EventTerm(
    #     func=mdp.randomize_rigid_body_scale,
    #     mode="prestartup",
    #     params={"asset_cfg": SceneEntityCfg("ladder"), "scale_range": {"x": (0.95, 1.05), "y": (0.95, 1.05), "z": (0.95, 1.1)}},
    # )
    # randomize_lamp_scale = EventTerm(func=mdp.randomize_rigid_body_scale, mode="prestartup",
    #     params={"asset_cfg": SceneEntityCfg("lamp"), "scale_range": (0.9, 1.1)})
    # randomize_bulb_scale = EventTerm(func=mdp.randomize_rigid_body_scale, mode="prestartup",
    #     params={"asset_cfg": SceneEntityCfg("bulb"), "scale_range": (0.9, 1.1)})

    # ---- (B) STARTING POSE: root pose + joint offsets -- mdp.reset_root_state_uniform / reset_joints_by_offset ----
    # randomize_robot_root = EventTerm(
    #     func=mdp.reset_root_state_uniform, mode="reset",
    #     params={"asset_cfg": SceneEntityCfg("robot"),
    #             "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
    #             "velocity_range": {}},
    # )
    # randomize_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset, mode="reset",
    #     params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    # )
    # randomize_ladder_pose = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
    #     params={"asset_cfg": SceneEntityCfg("ladder"), "pose_range": {"x": (-0.05, 0.05), "yaw": (-0.1, 0.1)}, "velocity_range": {}})

    # ---- (C) COLOR / VISUAL MATERIAL -- mdp.randomize_visual_color ----
    #   TODO: target a specific child mesh of the USD asset ("mesh_name" depends on the
    #   BEHAVIOR-1K prim layout -- inspect the spawned stage to pick one).
    # randomize_ladder_color = EventTerm(
    #     func=mdp.randomize_visual_color, mode="reset",
    #     params={"asset_cfg": SceneEntityCfg("ladder"), "mesh_name": "<child mesh path>",
    #             "colors": {"r": (0.3, 0.7), "g": (0.2, 0.5), "b": (0.1, 0.3)}, "event_name": "ladder_color"},
    # )

    # ---- (D) PHYSICS MATERIAL: friction / restitution -- mdp.randomize_rigid_body_material ----
    # randomize_ladder_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material, mode="reset",
    #     params={"asset_cfg": SceneEntityCfg("ladder"),
    #             "static_friction_range": (0.7, 1.2), "dynamic_friction_range": (0.6, 1.0),
    #             "restitution_range": (0.0, 0.1), "num_buckets": 64},
    # )
    # randomize_bulb_material = EventTerm(func=mdp.randomize_rigid_body_material, mode="reset",
    #     params={"asset_cfg": SceneEntityCfg("bulb"), "static_friction_range": (0.5, 0.9),
    #             "dynamic_friction_range": (0.5, 0.9), "restitution_range": (0.0, 0.1), "num_buckets": 64})

    # ---- (E) LIGHTING: intensity / color / orientation -- project stub (no built-in) ----
    #   TODO: see mdp.randomize_light_properties (per-env lights + orientation still to do).
    # randomize_dome_light = EventTerm(
    #     func=mdp.randomize_light_properties, mode="reset",
    #     params={"asset_cfg": SceneEntityCfg("dome_light"), "intensity_range": (600.0, 1000.0), "color": (0.9, 0.9, 0.95)},
    # )


##
# Environment configuration.
##


@configclass
class G1LadderEnvCfg(ManagerBasedEnvCfg):
    """Base manager-based (non-RL) environment for the Fiatlux ladder scene."""

    # -- simulation knobs (humanoid-friendly defaults; all overridable e.g. via Hydra) --
    physics_dt: float = 1.0 / 200.0
    control_decimation: int = 4  # -> 50 Hz control
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    solver_position_iterations: int = 8
    solver_velocity_iterations: int = 4
    episode_length_s: float = 20.0
    """Nominal episode length. NOTE: ManagerBasedEnv (non-RL) has no episode horizon; this is a
    documented knob the RL/task layer will consume once terminations are added."""

    # -- scene + managers --
    scene: G1LadderSceneCfg = G1LadderSceneCfg(
        num_envs=4, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        # control / physics rates
        self.decimation = self.control_decimation
        self.sim.dt = self.physics_dt
        self.sim.render_interval = self.decimation
        self.sim.gravity = self.gravity
        # PhysX solver: floor the per-body iteration counts for humanoid stability
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = self.solver_position_iterations
        self.sim.physx.min_velocity_iteration_count = self.solver_velocity_iterations
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default viewer framing
        self.viewer.eye = (4.5, 4.5, 3.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
