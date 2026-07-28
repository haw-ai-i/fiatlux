# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base (non-RL) ``ManagerBasedEnvCfg`` wiring the Fiatlux family scene to its managers.

Deliberately uses the **non-RL** ``ManagerBasedEnv`` base: it has observation, action and event
managers only -- NO reward / termination / command / curriculum managers and no policy/training
code. The scaffold tasks subclass :class:`FamilyBaseEnvCfg` and add their own task logic later
(switching to ``ManagerBasedRLEnvCfg``; see ``docs/roadmap.md``). RL family members (the Insert
task) use ``ManagerBasedRLEnvCfg`` directly on the same scene + presets.

The event manager carries the reset term and *enabled* per-reset light randomization, plus a
*disabled* randomization scaffold: every future randomization knob is written out, commented,
wired to the right ``SceneEntityCfg`` and tagged ``TODO`` -- enabling it later is just
uncommenting. (Per-env asset randomization -- the ceiling ``fixture`` -- happens at spawn time
in the scene cfg, not here.)
"""

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .scene_cfg import (
    FIXTURE_USDS,
    ROOM_ENV_SPACING,
    G1ReplaceSceneCfg,
    add_ego_camera,
    apply_workshop_preset,
)

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

    # -------- ENABLED: per-reset light randomization. Intensity only: the dome carries an
    # HDRI sky texture, and color-tinting a texture reads as a render bug rather than useful
    # domain randomization (see mdp.events). One global sample per reset (global light prims). --------
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

    # -------- ENABLED: grip friction on the hands, at startup. Without it every grasp in the
    # benchmark is made on the PhysX default 0.5/0.5. See mdp.hand_grip_material_event for why
    # this is an event term and not a material bound in the robot spawner. --------
    randomize_hand_material = mdp.hand_grip_material_event()

    # =====================================================================================
    #  DISABLED randomization scaffold -- uncomment + tune to enable. Built-in mdp terms are
    #  preferred; lighting uses the project stub mdp.randomize_light_properties. Each term is
    #  already wired to the correct SceneEntityCfg(name=...).
    # =====================================================================================

    # ---- (A) asset SCALE: ladder / socket / bulb -- mdp.randomize_rigid_body_scale ----
    #   TODO: scale runs on the USD stage, so use mode="prestartup". Replicate per prop.
    # randomize_ladder_scale = EventTerm(
    #     func=mdp.randomize_rigid_body_scale,
    #     mode="prestartup",
    #     params={"asset_cfg": SceneEntityCfg("ladder"),
    #             "scale_range": {"x": (0.95, 1.05), "y": (0.95, 1.05), "z": (0.95, 1.1)}},
    # )
    # randomize_socket_scale = EventTerm(func=mdp.randomize_rigid_body_scale, mode="prestartup",
    #     params={"asset_cfg": SceneEntityCfg("socket"), "scale_range": (0.9, 1.1)})
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
    #     params={"asset_cfg": SceneEntityCfg("ladder"),
    #             "pose_range": {"x": (-0.05, 0.05), "yaw": (-0.1, 0.1)}, "velocity_range": {}})

    # ---- (C) COLOR / VISUAL MATERIAL ----
    #   TODO: Isaac Lab 2.3.2 ships no built-in visual-color/texture randomization term --
    #   write a custom mdp term (a USD material-attribute write on the asset's child mesh,
    #   in the style of mdp.randomize_light_properties) when task-phase DR needs it.

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

    # ---- (E) LIGHTING ----
    #   Intensity randomization is ENABLED above (randomize_sky_intensity / randomize_key_light).
    #   TODO: light *orientation* (the key light's direction) and per-env lights are still
    #   open in the mdp.randomize_light_properties stub.


##
# Environment configuration.
##


@configclass
class FamilyBaseEnvCfg(ManagerBasedEnvCfg):
    """Base manager-based (non-RL) environment for the Fiatlux family scene (workshop preset)."""

    # -- simulation knobs (humanoid-friendly defaults; all overridable e.g. via Hydra) --
    physics_dt: float = 1.0 / 200.0
    control_decimation: int = 4  # -> 50 Hz control
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    solver_position_iterations: int = 8
    # PhysX applies this as a FLOOR over every actor, not a default: any per-body
    # ``solver_velocity_iteration_count`` below it is raised to it. Keep it at 1 -- the bulb,
    # old bulb and carried ladder ask for 1 because high velocity-iteration counts make the
    # TGS solver inject energy into resting contacts. Bodies asking for more still get more.
    solver_velocity_iterations: int = 1
    episode_length_s: float = 20.0
    """Nominal episode length. NOTE: ManagerBasedEnv (non-RL) has no episode horizon; this is a
    documented knob the RL/task layer will consume once terminations are added."""

    enable_dressing_randomization: bool = True
    """Per-env random ceiling fixture (visual domain randomization). Heterogeneous per-env
    assets require ``replicate_physics=False``, which is fine at scaffold scale but wrong at
    RL-training scale -- training cfgs set this False to get replicated physics back."""

    scene_preset: str = "workshop"
    """Which family-scene preset this task uses (verify_scene dispatches its presence
    expectations on this; subclasses override alongside their preset call)."""

    # -- orbit-recording framing (verify_scene --record); subclasses override to frame
    #    their own layout. Radius must keep the camera inside the Simple Room (~4.5 m walls).
    orbit_center: tuple[float, float, float] = (0.1, 0.0, 1.3)
    orbit_radius: float = 4.0
    orbit_height: float = 2.8

    # -- scene + managers --
    scene: G1ReplaceSceneCfg = G1ReplaceSceneCfg(
        num_envs=4, env_spacing=ROOM_ENV_SPACING, replicate_physics=False, clone_in_fabric=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        apply_workshop_preset(self.scene)
        # GrootPolicy looks up ``scene["ego_camera"]`` unconditionally (same gap g1_bulb_env_cfg
        add_ego_camera(self.scene)
        # Dressing randomization: drop the fixture when disabled, or when its (opt-in)
        # assets are not downloaded (`download_assets.sh --scene-dressing`) so the env
        # still loads from a clean clone.
        if self.enable_dressing_randomization and not FIXTURE_USDS:
            print("[fiatlux] no behavior1k_* ceiling-fixture assets found -- spawning without the 'fixture' entity.")
        if not self.enable_dressing_randomization or not FIXTURE_USDS:
            self.scene.fixture = None
        if not self.enable_dressing_randomization:
            # Homogeneous envs again -> replicated physics is safe and fast. Cloning
            # stays in USD (not fabric): the hand_contact sensor's PhysX contact-reporter
            # API cannot attach to fabric-cloned env prims.
            self.scene.replicate_physics = True
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

    def disable_randomization(self) -> None:
        """Deterministic canonical spawns (debug / basic testing; ``--no_randomize``).

        Strips the reset-time randomization terms; ``reset_all`` stays -- restoring
        default state between episodes is correctness, not noise. RL subclasses with
        their own task-specific reset terms override this rather than call it.
        """
        self.events.randomize_sky_intensity = None
        self.events.randomize_key_light = None
        # A randomization too, though not a reset term.
        self.events.randomize_hand_material = mdp.hand_grip_material_event(randomize=False)


# Backwards-compat alias (pre-unification name).
G1LadderEnvCfg = FamilyBaseEnvCfg
