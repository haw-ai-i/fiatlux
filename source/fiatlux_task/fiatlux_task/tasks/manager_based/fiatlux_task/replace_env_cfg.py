# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-Replace-v0`` -- the full light-bulb-replacement task (the primary benchmark).

One flat RL episode over the whole randomized world (``apply_replace_preset``): robot,
ladder, table+fresh bulb, disposal crate, and the elevated fixture with the old bulb seated
in it, each randomized into its own non-overlapping floor zone, fixture randomly ceiling- or
wall-mounted. The goal is the full replacement: insert the fresh bulb into the fixture,
remove the old bulb, and place it in the disposal crate. No policy stitching or
staged-curriculum chaining -- that is solution structure, not benchmark structure (the
issue-#20 descoping that still stands).

Design notes (full-task benchmark plan, ``journal/specs/full-task-benchmark-plan.md``):

- **Scoring** uses *normalized* distance progress -- ``(d0 - d) / d0`` clamped to [0, 1],
  per episode, paid as best-progress increments -- so randomized spawn distances cannot
  dominate the score (a lucky close spawn and an unlucky far one both cap at 1.0). Old-bulb
  removal is clearance-from-the-fixture against an absolute threshold (its d0 is ~0).
  Sparse completion bonuses pay once per episode; penalties cover robot falls, ladder
  tipping, and bulb drops. Every channel is its own named reward/termination term, so the
  per-term episode sums in ``extras['log']`` *are* the score breakdown.
- **Observation modes**: the ``policy`` group is the *standard* (sensor-realizable) mode --
  IMU, estimated base state, proprioception, hand contact, a torso RGB camera, previous
  action. The ``privileged`` group is the *cheatcode* mode -- exact robot/object/fixture/
  target poses and the score-relevant distances. The group names stay ``policy``/
  ``privileged`` because rsl_rl's ``obs_groups`` routing is keyed to them (see
  ``ClimbPPORunnerCfg``); the benchmark docs map standard->policy, cheatcode->privileged.
- **The ladder is dynamic** (only here): knocking it over is a real, penalized, episode-
  ending physical event. **The old bulb is kinematic** -- the "screwed in" stand-in until
  the attach/detach mechanic exists (unification spec Phase 4), so its removal/disposal
  terms are scored-but-not-yet-achievable, the same gap Remove/Install carry.
"""

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from fiatlux_task.robots.g1 import (
    G1_FINGER_JOINT_PATTERNS,
    G1_TORSO_BODY,
    G1_WAIST_JOINT_PATTERNS,
)

from . import mdp
from .climb_env_cfg import FALL_MIN_HEIGHT, FALL_TILT_LIMIT
from .scene_cfg import G1ReplaceSceneCfg, apply_replace_preset

##
# Task thresholds
##

LADDER_TILT_LIMIT = 0.6  # rad; the A-frame stands at 0, real climbing wobble stays well under
LADDER_READY_XY_RADIUS = 0.9  # m; ladder top horizontally within working reach of the fixture
REMOVAL_CLEARANCE = 0.10  # m; old-bulb plug this far from the seat counts as removed
DISPOSAL_THRESHOLD = 0.25  # m; old bulb within this of the crate origin counts as disposed
SEAT_POS_THRESHOLD = 0.015  # m; fresh-bulb seating tolerance (Insert's validated values)
SEAT_ORI_THRESHOLD = 0.2  # rad
FRESH_BULB_DROP_HEIGHT = 0.4  # m; the fresh bulb's working heights are table (~1.0) and up
OLD_BULB_DROP_HEIGHT = 0.15  # m; must clear a bulb resting *inside* the floor crate (~0.1)

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Whole-body joint-position targets: the task spans locomotion, climbing, manipulation."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Two groups: ``policy`` = the *standard* (sensor-realizable) observation mode,
    ``privileged`` = the *cheatcode* (exact simulator state) mode."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Standard mode: IMU + estimator + proprioception + hand contact + torso RGB."""

        # IMU-realizable
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        # Estimator-realizable exceptions (kinematic-inertial state estimate; the same
        # documented exception Climb makes).
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        # Proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # Contact / force sensing on the grasping hand.
        hand_contact = ObsTerm(
            func=mdp.contact_net_forces,
            scale=0.1,
            params={"sensor_cfg": SceneEntityCfg("hand_contact")},
        )
        # Camera-derived features (torso-mounted RGB; requires --enable_cameras).
        torso_rgb = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("torso_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Cheatcode mode: exact poses of everything score-relevant + the score distances."""

        robot_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("robot")})
        ladder_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("ladder")})
        fixture_pose = ObsTerm(
            func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("socket")}
        )
        fresh_bulb_pose = ObsTerm(
            func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("bulb")}
        )
        old_bulb_pose = ObsTerm(
            func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("old_bulb")}
        )
        disposal_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("bin")})
        score_distances = ObsTerm(func=mdp.replace_score_distances)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Reset-time randomization (the room layout itself randomizes per scene build)."""

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


@configclass
class RewardsCfg:
    """Score channels. Rewards accrue as ``func * weight * step_dt`` (dt = 0.02 s); the
    dense terms pay best-progress increments and the completion bonuses pay once, so each
    channel's episode total is bounded by ``weight * dt`` regardless of episode length:
    dense +5/+10/+5/+10, completions +2/+5/+3/+5, success +10 -- ~55 for a perfect run,
    with every shaping penalty sized well below that."""

    # -- dense normalized progress (randomization-fair; see mdp.distance_progress) --
    ladder_progress = RewTerm(
        func=mdp.distance_progress,
        weight=250.0,
        params={"distance_fn": mdp.ladder_fixture_distance},
    )
    fresh_bulb_progress = RewTerm(
        func=mdp.distance_progress,
        weight=500.0,
        params={"distance_fn": mdp.bulb_fixture_distance},
    )
    old_bulb_removal = RewTerm(
        func=mdp.distance_progress,
        weight=250.0,
        params={
            "distance_fn": mdp.old_bulb_fixture_clearance,
            "away_threshold": REMOVAL_CLEARANCE,
        },
    )
    old_bulb_disposal_progress = RewTerm(
        func=mdp.distance_progress,
        weight=500.0,
        params={"distance_fn": mdp.old_bulb_disposal_distance},
    )
    # -- sparse completions (each pays once per episode; see mdp.completion_bonus) --
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
    fresh_bulb_inserted = RewTerm(
        func=mdp.completion_bonus,
        weight=250.0,
        params={
            "predicate_fn": mdp.bulb_seated,
            "predicate_params": {
                "pos_threshold": SEAT_POS_THRESHOLD,
                "ori_threshold": SEAT_ORI_THRESHOLD,
            },
        },
    )
    old_bulb_removed = RewTerm(
        func=mdp.completion_bonus,
        weight=150.0,
        params={
            "predicate_fn": mdp.old_bulb_removed,
            "predicate_params": {"clearance_threshold": REMOVAL_CLEARANCE},
        },
    )
    old_bulb_disposed = RewTerm(
        func=mdp.completion_bonus,
        weight=250.0,
        params={
            "predicate_fn": mdp.old_bulb_disposed,
            "predicate_params": {"distance_threshold": DISPOSAL_THRESHOLD},
        },
    )
    # Full success terminates the episode on the same step, so the raw predicate pays once.
    success_bonus = RewTerm(
        func=mdp.full_replacement_success,
        weight=500.0,
        params={
            "pos_threshold": SEAT_POS_THRESHOLD,
            "ori_threshold": SEAT_ORI_THRESHOLD,
            "disposal_threshold": DISPOSAL_THRESHOLD,
        },
    )
    # -- penalties (each predicate also terminates, so it fires once; see fall_terminated) --
    robot_fall = RewTerm(
        func=mdp.fall_terminated,
        weight=-200.0,
        params={"minimum_height": FALL_MIN_HEIGHT, "limit_angle": FALL_TILT_LIMIT},
    )
    ladder_tipped = RewTerm(
        func=mdp.ladder_tipped, weight=-200.0, params={"tilt_limit": LADDER_TILT_LIMIT}
    )
    fresh_bulb_dropped = RewTerm(
        func=mdp.object_dropped,
        weight=-100.0,
        params={"asset_cfg": SceneEntityCfg("bulb"), "min_height": FRESH_BULB_DROP_HEIGHT},
    )
    old_bulb_dropped = RewTerm(
        func=mdp.old_bulb_dropped,
        weight=-100.0,
        params={
            "min_height": OLD_BULB_DROP_HEIGHT,
            "disposal_threshold": DISPOSAL_THRESHOLD,
        },
    )
    contact_penalty = RewTerm(
        func=mdp.hand_contact_force_l2,
        weight=-1.0e-3,
        params={"sensor_cfg": SceneEntityCfg("hand_contact")},
    )
    # -- stability / smoothness shaping (Climb's recipe; milder CoM-sway: walking sways) --
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
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            )
        },
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
    """Horizon, fall/tip/drop constraint violations, and full-replacement success."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Fall detection (the family RL gate; thresholds shared with the fall penalty).
    fell_below = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": FALL_MIN_HEIGHT}
    )
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": FALL_TILT_LIMIT})
    ladder_tipped = DoneTerm(func=mdp.ladder_tipped, params={"tilt_limit": LADDER_TILT_LIMIT})
    fresh_bulb_dropped = DoneTerm(
        func=mdp.object_dropped,
        params={"asset_cfg": SceneEntityCfg("bulb"), "min_height": FRESH_BULB_DROP_HEIGHT},
    )
    old_bulb_dropped = DoneTerm(
        func=mdp.old_bulb_dropped,
        params={"min_height": OLD_BULB_DROP_HEIGHT, "disposal_threshold": DISPOSAL_THRESHOLD},
    )
    # Contract name: recording.py / score.py / eval.py read the `success` term.
    success = DoneTerm(
        func=mdp.full_replacement_success,
        params={
            "pos_threshold": SEAT_POS_THRESHOLD,
            "ori_threshold": SEAT_ORI_THRESHOLD,
            "disposal_threshold": DISPOSAL_THRESHOLD,
        },
    )


##
# Environment configuration
##


@configclass
class ReplaceEnvCfg(ManagerBasedRLEnvCfg):
    """The full replacement task (randomized room layout, standard/cheatcode obs modes)."""

    scene_preset: str = "replace"
    # wide framing: the room-scale layout, not a fixed bench corner. The Simple Room is NOT
    # centered on the world origin (measured wall bbox: x=[-4.52,4.52], y=[-3.4,4.86] -- see
    # scene_cfg.py's ROOM_FLOOR_MIN/MAX comment), so the orbit center is offset to the room's
    # actual y-midpoint and the radius stays well inside the nearer (south) wall.
    orbit_center: tuple[float, float, float] = (0.0, 0.7, 2.2)
    orbit_radius: float = 3.0
    orbit_height: float = 3.0

    couple_ladder_to_fixture: bool = False
    """Debug/curriculum aid: place the ladder's zone reachably relative to wherever the
    fixture mounted instead of sampling it independently. Off by default -- positioning the
    ladder is part of the task."""

    # Homogeneous envs (the preset drops the random dressing fixture), so physics
    # replication is safe at training scale. USD cloning (not fabric) keeps the
    # hand_contact PhysX contact reporters attachable.
    scene: G1ReplaceSceneCfg = G1ReplaceSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_replace_preset(
            self.scene, couple_ladder_to_fixture=self.couple_ladder_to_fixture
        )

        # Torso-mounted RGB camera (the standard mode's exteroception; --enable_cameras).
        # Same pinhole spawn as Insert's wrist camera, longer clip for room-scale views.
        self.scene.torso_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + G1_TORSO_BODY + "/torso_camera",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=22.48,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 20.0),
            ),
            height=224,
            width=224,
            data_types=["rgb"],
            # "world" convention: identity rot looks along the torso's +X (forward).
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.08, 0.0, 0.42), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
            ),
        )

        # family control rate (50 Hz); a longer horizon than any subtask -- the episode
        # spans approach + ladder work + insert + removal + disposal
        self.decimation = 4
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 40.0

        # PhysX floors + stabilization (family finding; more load-bearing here than
        # anywhere: an uncontrolled G1, kinematic furniture, AND a dynamic ladder)
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 4
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_stabilization = True

        self.viewer.eye = (4.5, 4.5, 3.0)
        self.viewer.lookat = (0.0, 0.7, 1.0)
