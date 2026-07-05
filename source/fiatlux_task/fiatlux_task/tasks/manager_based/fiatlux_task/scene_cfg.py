# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The single source of truth for the Fiatlux task-family scene: one world, preset layouts.

This ``InteractiveSceneCfg`` subclass assembles every element as a **named scene entity** so
each is addressable via ``SceneEntityCfg`` for later randomization: ``ground``, ``dome_light``,
``key_light``, ``room``, ``robot``, ``ladder``, ``socket``, ``bulb``, ``table``, ``fixture``,
``hand_contact``.

Every task in the family shares this scene; a *preset* selects the layout for the task's
phase of the light-bulb-replacement story:

- :func:`apply_workshop_preset` -- the default floor layout: climb ladder + socket-lamp and
  bulb on the floor (Base / Carry / Climb / Descend / Remove / Install scaffolds).
- :func:`apply_tabletop_preset` -- the manipulation bench: packing table, socket-lamp on the
  tabletop, bulb at hand height, no ladder (the Insert task).

Presets are plain functions called from an env cfg's ``__post_init__`` --
``InteractiveSceneCfg`` treats *every* dataclass field as a scene entity, so preset knobs
cannot live here as fields. Optional entities (``table``, ``ladder``, ``fixture``) are
dropped by setting them to ``None``; ``InteractiveScene`` skips ``None`` entities.

All assets come from the ``gs://fiatlux`` bucket (see ``fiatlux_task.assets`` and
``assets/download_assets.sh``): the same Inspire-hand G1 and BEHAVIOR-1K bulb/lamp
everywhere, plus the primary BEHAVIOR-1K climb ladder (``shfvtl``). The room dressing (the
Simple Room backdrop and the PolyHaven HDRI sky) comes from ``DressedSceneCfg``, plus one
*randomly chosen* BEHAVIOR-1K ceiling fixture per env (``fixture``; needs the opt-in
``download_assets.sh --scene-dressing`` asset group, and is dropped automatically when
those assets are absent).
"""

import glob
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from fiatlux_task.assets import (
    BULB_USD,
    CRATE_USD,
    ELEVATED_SOCKET_USD,
    FIATLUX_ASSETS_DIR,
    LADDER_USD,
    SOCKET_USD,
    STEP_LADDER_USD,
    TABLE_USD,
)
from fiatlux_task.robots.g1 import G1_INSPIRE_CFG
from fiatlux_task.scenes import DressedSceneCfg, spawn_b1k_single_body

# -- default (workshop) placement (module constants, not scene fields; override via each
#    entity's init_state). BEHAVIOR-1K USDs are authored with the origin near the bbox
#    center, so a prop resting on the floor sits at roughly half its height. Tuned against
#    scripts/verify_scene.py --record orbit videos; adjust the same way. --
ROBOT_POSITION = (0.0, 0.0, 0.75)  # G1_INSPIRE_CFG's standing pelvis height
# Work-site step ladder (STEP_LADDER_USD, probe: 0.68 wide x 1.11 deep x 1.75 tall, base
# authored at z=0). Yawed 90 deg so its steps face -x (toward the robot's approach).
LADDER_POSITION = (1.6, 0.0, 0.0)
LADDER_YAW_DEG = 90.0
SOCKET_POSITION = (-0.8, 0.0, 0.20)  # bbentu socket-lamp resting on the floor
BULB_POSITION = (-0.55, -0.20, 0.05)  # loose on the floor next to the lamp
FIXTURE_POSITION = (0.0, 0.0, 2.45)  # hangs overhead in the record camera's frame, clear of robot/ladder

# -- tabletop (manipulation bench) placement: the Insert layout. The robot works
#    the bench from its +y long side, facing -y: the packing table's collision
#    volume spans x[-0.82,1.62] x y[-0.48,0.28] with an under-frame up to z=0.95,
#    so a robot standing inside that footprint spawns with its legs among the
#    frame members (kN solver wedges the moment anything touches). --
TABLE_POSITION = (0.40, -0.10, 0.0)  # authored tabletop surface is ~1.0 m above the origin
TABLETOP_ROBOT_POSITION = (0.60, 0.58, 0.75)
TABLETOP_ROBOT_YAW_DEG = -90.0
TABLETOP_SOCKET_POSITION = (0.45, 0.10, 1.20)
TABLETOP_BULB_POSITION = (0.30, 0.18, 1.05)

# -- carry preset: the B1K straight ladder (shfvtl) *stored* by the room wall in its
#    authored lying/leaning pose (probe: 2.41 long x 1.67 high, bbox bottom at -0.47 ->
#    pivot z=+0.47 rests it on the floor), robot beside it, work area across the room --
CARRY_LADDER_POSITION = (-3.2, 1.8, 0.47)  # near the Simple Room wall (~4.5 m out)
CARRY_ROBOT_POSITION = (-2.4, 1.8, 0.75)  # standing next to the stored ladder
CARRY_LADDER_YAW_DEG = 90.0  # parallel to the wall

# -- at-height presets (climb / descend): elevated fixture over the ladder. The
#    chandelier hangs above/behind the ladder's top; positions are tuned against
#    verify_scene --record orbit videos, same as the floor layout. --
ELEVATED_SOCKET_POSITION = (1.9, 0.0, 2.80)  # cage bottom clears the at-top robot's head
CLIMB_ROBOT_POSITION = (0.75, 0.0, 0.75)  # at the step ladder's base, ready to ascend
TOP_ROBOT_POSITION = (1.35, 0.0, 1.85)  # pelvis at the upper steps (descend)
PARKED_BULB_POSITION = (0.5, -0.6, 0.05)  # out of the way on the floor

# -- bench manipulation extras (remove / install share Insert's tabletop world) --
TABLETOP_SEATED_BULB_POSITION = (0.45, 0.10, 1.33)  # in the table lamp's socket seat
BIN_POSITION = (0.15, -0.75, 0.0)  # parts crate on the floor beside the bench
BIN_BULB_POSITION = (0.15, -0.75, 0.15)  # fresh bulb resting in the crate (install)

# -- per-env random ceiling fixture pool (visual dressing) --
# Ceiling-mount BEHAVIOR-1K categories only: floor-standing fixtures would invade the task
# space. These are the opt-in ``download_assets.sh --scene-dressing`` asset group; when they
# are absent the pool is empty and ``FamilyBaseEnvCfg.__post_init__`` drops the ``fixture``
# entity so the env still loads. Category dirs mix two layouts (``<id>/<id>.usd`` and
# ``<id>/usd/<id>.usd``), hence the two glob patterns.
_FIXTURE_CATEGORIES = (
    "behavior1k_chandelier",
    "behavior1k_downlight",
    "behavior1k_paper_lantern",
    "behavior1k_rectangular_light",
    "behavior1k_room_light",
    "behavior1k_square_light",
    "behavior1k_track_light",
)
FIXTURE_USDS = sorted(
    usd
    for cat in _FIXTURE_CATEGORIES
    for pattern in (os.path.join("*", "*.usd"), os.path.join("*", "usd", "*.usd"))
    for usd in glob.glob(os.path.join(FIATLUX_ASSETS_DIR, cat, pattern))
)


def _quat_y_deg(angle_deg: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) quaternion for a rotation about +Y, in degrees."""
    half = math.radians(angle_deg) / 2.0
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def _quat_z_deg(angle_deg: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) quaternion for a rotation about +Z (yaw), in degrees."""
    half = math.radians(angle_deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


# Omniverse SimReady assets author PhysX colliders but no RigidBodyAPI; Isaac Lab's
# RigidObjectCfg requires exactly one rigid-body prim (the spawner's property pass only
# *modifies* an existing API). Apply it on the root at spawn time, then apply the cfg's
# rigid props (e.g. kinematic_enabled) which would otherwise silently no-op.
@clone
def _spawn_usd_as_rigid_body(prim_path, cfg, translation=None, orientation=None):
    from pxr import UsdPhysics

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim.GetPath(), cfg.rigid_props)
    return prim


@configclass
class G1ReplaceSceneCfg(DressedSceneCfg):
    """The G1 light-bulb-replacement world (see module docstring for the preset layouts)."""

    # ------------------------------------------------------------------ ground & lighting
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
        ),
    )
    # (dome_light + room backdrop are inherited from DressedSceneCfg.)
    key_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=1500.0, color=(1.0, 0.98, 0.95), angle=0.53),
        # tilt the distant light downward; orientation is the future "light direction" knob
        init_state=AssetBaseCfg.InitialStateCfg(rot=_quat_y_deg(40.0)),
    )

    # ------------------------------------------------------------------ robot & props
    # Same reusable Inspire-hand G1 everywhere (USD, standing init pose, actuator groups);
    # here we only bind it into this scene's namespace and placement.
    robot: ArticulationCfg = G1_INSPIRE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=G1_INSPIRE_CFG.init_state.replace(pos=ROBOT_POSITION),
    )
    # Work-site step ladder: a free-standing Omniverse SimReady A-frame (the _collision
    # USD carries both render meshes and authored PhysX colliders). cm-authored -> scale
    # 0.01. Kinematic so it stays put while climbed. Dropped by the tabletop preset;
    # the carry preset swaps in the B1K straight ladder as stored cargo instead.
    ladder: RigidObjectCfg | None = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ladder",
        spawn=sim_utils.UsdFileCfg(
            usd_path=STEP_LADDER_USD,
            # SimReady asset: apply the missing RigidBodyAPI at spawn (see helper above).
            func=_spawn_usd_as_rigid_body,
            scale=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=LADDER_POSITION, rot=_quat_z_deg(LADDER_YAW_DEG)
        ),
    )
    # Socket fixture: a BEHAVIOR-1K lamp standing in for the bulb socket; kinematic so it
    # can be re-posed on reset. On the floor in the workshop preset, on the table in the
    # tabletop preset (at-height mounting is the deferred elevated preset's business).
    socket: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Socket",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SOCKET_USD,
            func=spawn_b1k_single_body,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=SOCKET_POSITION),
    )
    # Graspable bulb (dynamic).
    bulb: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bulb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BULB_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BULB_POSITION),
    )
    # Packing table (manipulation bench). Spawned only by the tabletop preset.
    table: AssetBaseCfg | None = None
    # Parts crate (bulb bin). Spawned only by the install preset.
    bin: AssetBaseCfg | None = None

    # -- Contact sensor on the grasping hand (force/torque safety + obs). Family-wide: the
    # manipulation tasks read it for rewards/recording, climbing will want contact sensing.
    # Hand bodies only: this channel feeds the recorded fragility scoring, and a broader
    # match (leg/foot bodies) would put the robot's own ground reaction (~170 N standing,
    # >> the 50 N fragility threshold) into every episode's peak contact force.
    hand_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(right_hand_.*|right_wrist_.*|R_.*)",
        history_length=1,
        track_air_time=False,
    )

    # ------------------------------------------------------------------ randomized dressing
    # Per-env random ceiling fixture: each cloned env spawns one randomly chosen BEHAVIOR-1K
    # ceiling-mount fixture from FIXTURE_USDS. AssetBaseCfg (not RigidObjectCfg) keeps it out
    # of physics entirely -- no meta__ single-body stripping needed -- and collisions are
    # disabled so task physics is untouched. Heterogeneous per-env assets require
    # ``replicate_physics=False`` (managed via ``enable_dressing_randomization`` on the env cfg).
    fixture: AssetBaseCfg | None = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Fixture",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=FIXTURE_USDS,
            random_choice=True,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FIXTURE_POSITION),
    )


# Backwards-compat alias (pre-unification name).
G1LadderSceneCfg = G1ReplaceSceneCfg


##
# Preset layouts -- call from an env cfg's ``__post_init__``.
##


def apply_workshop_preset(scene: G1ReplaceSceneCfg) -> None:
    """The default floor layout: climb ladder + socket-lamp and bulb on the floor.

    This is the scene's authored default, so the function is a documented no-op -- it
    exists so every task cfg states its preset explicitly.
    """


def apply_tabletop_preset(scene: G1ReplaceSceneCfg) -> None:
    """The manipulation bench (Insert layout): table, socket on top, bulb at hand height.

    Drops the ladder; the robot stands at the bench's +y side (clear of the
    table's collision footprint) and never locomotes.
    """
    scene.ladder = None
    scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POSITION),
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD,
            # kinematic so it cannot be pushed around
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    scene.robot.init_state.pos = TABLETOP_ROBOT_POSITION
    scene.robot.init_state.rot = _quat_z_deg(TABLETOP_ROBOT_YAW_DEG)
    scene.socket.init_state.pos = TABLETOP_SOCKET_POSITION
    scene.bulb.init_state.pos = TABLETOP_BULB_POSITION


def apply_carry_preset(scene: G1ReplaceSceneCfg) -> None:
    """Ladder-handling start: a straight ladder *stored* by the room wall, robot beside it.

    The stored cargo is the B1K straight ladder (shfvtl) in its authored lying/leaning
    pose -- the realistic start for "carry the ladder to the work site". The work area
    (the floor socket-lamp) stays across the room.
    """
    scene.ladder.spawn = sim_utils.UsdFileCfg(
        usd_path=LADDER_USD,
        # Strip meta__ helper links so the B1K object resolves to one rigid body.
        func=spawn_b1k_single_body,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
    )
    scene.ladder.init_state.pos = CARRY_LADDER_POSITION
    scene.ladder.init_state.rot = _quat_z_deg(CARRY_LADDER_YAW_DEG)
    scene.robot.init_state.pos = CARRY_ROBOT_POSITION


def apply_at_height_preset(scene: G1ReplaceSceneCfg, robot_at: str = "base") -> None:
    """Elevated-fixture layout shared by climb / descend.

    The socket entity becomes a ceiling chandelier hung above the ladder; the floor lamp
    is gone. ``robot_at="base"`` starts the robot at the ladder's feet (climb),
    ``robot_at="top"`` starts it at the upper steps (descend). The random dressing
    ``fixture`` is dropped -- the task chandelier owns the ceiling.
    """
    scene.socket.spawn.usd_path = ELEVATED_SOCKET_USD
    scene.socket.init_state.pos = ELEVATED_SOCKET_POSITION
    scene.bulb.init_state.pos = PARKED_BULB_POSITION
    scene.robot.init_state.pos = CLIMB_ROBOT_POSITION if robot_at == "base" else TOP_ROBOT_POSITION
    scene.fixture = None


def _add_parts_bin(scene: G1ReplaceSceneCfg) -> None:
    """Spawn the kinematic parts crate on the floor beside the bench (remove + install)."""
    scene.bin = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bin",
        init_state=AssetBaseCfg.InitialStateCfg(pos=BIN_POSITION),
        spawn=sim_utils.UsdFileCfg(
            usd_path=CRATE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )


def apply_remove_preset(scene: G1ReplaceSceneCfg) -> None:
    """Bulb-removal start: Insert's bench with the OLD BULB SEATED in the table lamp.

    Same world as the Insert/Install bench; the bulb starts kinematic in the lamp's
    socket seat (a stand-in for "screwed in" until the task-phase attach joint exists,
    spec Phase 4), and the empty parts crate beside the bench is its destination.
    """
    apply_tabletop_preset(scene)
    _add_parts_bin(scene)
    scene.bulb.init_state.pos = TABLETOP_SEATED_BULB_POSITION
    scene.bulb.spawn.rigid_props.kinematic_enabled = True


def apply_install_preset(scene: G1ReplaceSceneCfg) -> None:
    """Bulb-installation start: Insert's bench, empty lamp socket, fresh bulb in the crate."""
    apply_tabletop_preset(scene)
    _add_parts_bin(scene)
    scene.bulb.init_state.pos = BIN_BULB_POSITION
