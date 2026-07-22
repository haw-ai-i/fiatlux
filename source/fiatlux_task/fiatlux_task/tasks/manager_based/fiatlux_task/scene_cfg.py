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
- :func:`apply_replace_preset` -- the full replacement task world (issue #20 scene, promoted
  to the primary benchmark by the full-task plan): robot, ladder, table+fresh bulb, disposal
  crate, and the elevated fixture (old bulb seated in it) each randomized into their own
  non-overlapping floor "safe zone", fixture randomly ceiling- or wall-mounted. The ladder is
  a *dynamic* rigid body here (it can genuinely tip/fall, which the task penalizes) --
  kinematic everywhere else.

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
import random
from typing import Literal, cast

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from fiatlux_task.assets import (
    BULB_STAND_Z_OFFSET,
    BULB_USD,
    CRATE_USD,
    ELEVATED_SOCKET_USD,
    FIATLUX_ASSETS_DIR,
    LADDER_USD,
    SOCKET_USD,
    STEP_LADDER_RIGID_USD,
    STEP_LADDER_USD,
    TABLE_USD,
)
from fiatlux_task.robots.g1 import G1_INSPIRE_CFG
from fiatlux_task.scenes import DressedSceneCfg, spawn_b1k_single_body
from fiatlux_task.sensors import ego_camera_cfg, mid360_lidar_cfg, wrist_camera_cfg

# -- default (workshop) placement (module constants, not scene fields; override via each
#    entity's init_state). BEHAVIOR-1K USDs are authored with the origin near the bbox
#    center, so a prop resting on the floor sits at roughly half its height. Tuned against
#    scripts/verify_scene.py --record orbit videos; adjust the same way. --
_ROBOT_Z = G1_INSPIRE_CFG.init_state.pos[2]  # standing pelvis height (feet on the floor)
ROBOT_POSITION = (0.0, 0.0, _ROBOT_Z)
# Work-site step ladder (STEP_LADDER_USD, probe: 0.68 wide x 1.11 deep x 1.75 tall, base
# authored at z=0). Yawed 90 deg so its steps face -x (toward the robot's approach).
LADDER_POSITION = (1.6, 0.0, 0.0)
LADDER_YAW_DEG = 90.0
# The fixture's origin IS its floor-contact plane (SOCKET_BASE_Z_OFFSET = 0), so it rests
# on a surface at exactly the surface height -- no half-height guessing.
SOCKET_POSITION = (-0.8, 0.0, 0.0)  # socket-fixture standing on the floor
BULB_POSITION = (-0.55, -0.20, -BULB_STAND_Z_OFFSET)  # standing on its cap on the floor

# -- tabletop (manipulation bench) placement: the Insert layout. The robot works
#    the bench from its +y long side, facing -y: the packing table's collision
#    volume spans x[-0.82,1.62] x y[-0.48,0.28] with an under-frame up to z=0.95.
#    The standoff must clear the robot's WHOLE spawn envelope, not just its legs:
#    the default arm dangle puts the right fingers ~0.40 m in front of the base at
#    z~0.9, and fingers inside the under-frame volume take a depenetration kick at
#    every reset that topples the robot (probe-verified: base y=0.58 falls in ~16
#    steps under zero action, y>=0.73 stands indefinitely). --
TABLE_POSITION = (0.40, -0.10, 0.0)
# The packing table's WORK SURFACE, measured at runtime off its main collider
# (SM_HeavyDutyPackingTable_C02_01, top z=0.9941). Do NOT take this from the table's overall
# bbox: that reads 1.0829, which is the top of the mesh tray sitting on the table
# (Cube_01..04, x=[0.656,1.394] y=[-0.443,0.055]), not a surface anything can rest on.
# Props are placed clear of that tray footprint.
TABLETOP_SURFACE_Z = 0.9941
TABLETOP_ROBOT_POSITION = (0.60, 0.73, _ROBOT_Z)
TABLETOP_ROBOT_YAW_DEG = -90.0
# The fixture stands on the bench: its origin is its own floor-contact plane.
TABLETOP_SOCKET_POSITION = (0.45, 0.10, TABLETOP_SURFACE_Z)
# Standing on its screw cap, which is the bulb's only stable rest pose: laid on its side it
# tips onto the flat cap within a second, so a "lying" spawn is a settling event, not a rest
# state. Note the asset's root origin sits BELOW its own geometry (the cap bottom is at
# +BULB_STAND_Z_OFFSET in the root frame), so resting on a surface puts the root under it.
TABLETOP_BULB_POSITION = (0.30, 0.18, TABLETOP_SURFACE_Z - BULB_STAND_Z_OFFSET)

# -- carry preset: the B1K straight ladder (shfvtl) *stored* by the room wall in its
#    authored lying/leaning pose (probe: 2.41 long x 1.67 high, bbox bottom at -0.47 ->
#    pivot z=+0.47 rests it on the floor), robot beside it, work area across the room --
CARRY_LADDER_POSITION = (-3.2, 1.8, 0.47)  # near the Simple Room wall (~4.5 m out)
CARRY_ROBOT_POSITION = (-2.4, 1.8, _ROBOT_Z)  # standing next to the stored ladder
CARRY_LADDER_YAW_DEG = 90.0  # parallel to the wall

# -- position (ladder-handling) subtask: FIATLUX-Carry-v0. The ladder is the free-standing
#    Omniverse A-frame step ladder (STEP_LADDER_USD), made DYNAMIC + high-friction + graspable.
#    It STARTS upright but off-target, a short distance IN FRONT of the robot (the robot
#    stands back from it); the robot approaches, grasps a rail, and repositions it to the
#    fixed upright TARGET pose directly under the light. The A-frame's base is authored at
#    z=0, so start/target z=0. Tuned against verify_scene --record --hold_base (the free
#    robot collapses under a zero policy). --
POSITION_ROBOT_POSITION = (-0.20, -0.20, 0.75)
# Ladder STARTS well out in front, a long distance from the robot AND far from under the
# fixture; the robot approaches it, then carries it back to the TARGET, which is directly
# UNDER the ceiling light (the fixture is mounted at TARGET_LADDER_POSITION's x,y on the
# ceiling). ~2 m robot->ladder, ~1.5 m ladder->light.
POSITION_LADDER_START_POS = (1.50, 0.85, 0.0)
POSITION_LADDER_START_YAW = 30.0
TARGET_LADDER_POSITION = (0.55, -0.30, 0.0)  # directly beneath the ceiling fixture
LADDER_MASS = 3.0  # modest, so one arm can move it (real step ladders are heavier)
# The light fixture the positioned ladder leads to is the SAME validated BEHAVIOR-1K lamp +
# bulb (SOCKET_USD / BULB_USD) the Insert/Replace tasks use, mounted on the ceiling directly
# above the target and flipped bulb-down -- exactly how apply_replace_preset mounts its
# ceiling fixture. See apply_position_preset (no OMNI-specific constants needed anymore).

# -- at-height presets (climb / descend): elevated fixture over the ladder. The
#    chandelier hangs above/behind the ladder's top; positions are tuned against
#    verify_scene --record orbit videos, same as the floor layout. --
ELEVATED_SOCKET_POSITION = (1.9, 0.0, 2.80)  # cage bottom clears the at-top robot's head
CLIMB_ROBOT_POSITION = (0.75, 0.0, _ROBOT_Z)  # at the step ladder's base, ready to ascend
TOP_ROBOT_POSITION = (1.35, 0.0, 1.85)  # pelvis at the upper steps (descend)
PARKED_BULB_POSITION = (0.5, -0.6, -BULB_STAND_Z_OFFSET)  # standing out of the way on the floor

# -- bench manipulation extras (remove / install share Insert's tabletop world) --
# Both halves are authored ASSEMBLED AT IDENTITY, so a seated bulb is simply the fixture's
# own pose -- no offset arithmetic, and the seated start state is exactly the asset's rest
# pose, i.e. zero interpenetration at reset by construction.
TABLETOP_SEATED_BULB_POSITION = TABLETOP_SOCKET_POSITION
BIN_POSITION = (0.15, -0.75, 0.0)  # parts crate on the floor beside the bench
# Fresh bulb standing upright on its cap inside the crate. The crate's INNER floor is at
# 0.055 m -- measured by letting the bulb settle, not taken from the crate's outer bbox
# (whose rim top reads 0.17). Requires the hollow-collider spawner: against the stock
# single convex collider a bulb placed in here is depenetrated straight out onto the floor.
BIN_BULB_INTERIOR_Z = 0.055
BIN_BULB_POSITION = (0.15, -0.75, BIN_BULB_INTERIOR_Z - BULB_STAND_Z_OFFSET)

# -- replace preset (issue #20): the whole family world at once, robot / table+bulb / ladder
# each randomized into their own non-overlapping floor "safe zone", fixture ceiling- or
# wall-mounted. Room extent measured directly off the Simple Room USD (``UsdGeom.BBoxCache``
# over its ``Towel_Room01_wall_*``/``floor_*`` prims, excluding the oversized decorative
# ``Floor2-5``/light helper prims): walls span roughly x=[-4.52,4.52], y=[-3.4,4.86],
# z=[-0.58,3.22]. These are inset from that measured box; tuned against
# ``verify_scene.py --record`` like every other placement constant in this file.
ROOM_FLOOR_MIN = (-4.0, -3.0)
ROOM_FLOOR_MAX = (4.0, 4.2)

# Heights, measured off the room asset AFTER ``_spawn_room_backdrop`` aligns its walking
# surface to z=0 (the asset is authored tabletop-at-origin, so everything shifts +0.7696):
#   floor 0.000 | baseboard 0.000-0.190 | walls 0.190-3.989 | ceiling slab 4.179-4.197.
# ROOM_CEILING_Z is the ceiling's underside -- a real surface a fixture can be mounted to,
# not an inset guess. It is far above reach, so ceiling fixtures hang from it on a pendant
# (CEILING_FIXTURE_Z + add_ceiling_pendant) rather than floating at the reach height, which
# is what they did while this constant still held the pre-alignment 3.0.
ROOM_CEILING_Z = 4.179
ROOM_WALL_TOP_Z = 3.989
CEILING_FIXTURE_Z = 3.0  # fixture height: reachable from the step ladder's top step
WALL_MOUNT_Z = 2.2  # reach height for a wall-mounted fixture
# Env spacing must clear the room's own wall box (9.04 x 8.26 m) now that each env carries
# its own colliding room. Overlap would be physically harmless -- filter_collisions=True
# puts every env in its own collision group -- but it makes any render with num_envs > 1
# unreadable, which is how the floating-props bug stayed invisible for so long.
ROOM_ENV_SPACING = 10.0
PENDANT_RADIUS = 0.012  # the rod a ceiling fixture hangs from
# Decorative per-env ceiling fixture: flush against the real ceiling, because these are
# ceiling-MOUNT BEHAVIOR-1K assets. It used to hang at 2.45 m purely to sit in the record
# camera's frame, which put a ceiling light floating 1.7 m below the ceiling in every shot.
FIXTURE_POSITION = (0.0, 0.0, ROOM_CEILING_Z)

# Zone half-sizes (m) -- each occupant's own "safe square" half-extent, footprint + a bit of
# working clearance. Table's is a square bound around its actual (elongated) footprint --
# collision volume x[-0.82,1.62] x y[-0.48,0.28] is centered on TABLE_POSITION (0.40,-0.10),
# so its own origin already IS its footprint center; half-extent is 1.22 m (x) / 0.38 m (y).
ROBOT_ZONE_HALF_SIZE = 0.6
TABLE_ZONE_HALF_SIZE = 1.5
LADDER_ZONE_HALF_SIZE = 1.0
DISPOSAL_ZONE_HALF_SIZE = 0.5  # the old-bulb disposal crate (crate footprint ~0.6 m + clearance)
ZONE_MARGIN = 0.5  # minimum gap left between any two zones' bounding squares
LADDER_WALL_STANDOFF = 0.4  # extra gap between the (coupled) ladder zone edge and the wall

# Dynamic-ladder mass (replace preset only). Without an authored MassAPI PhysX derives mass
# from collider volume at 1000 kg/m^3, which lands a hollow A-frame at furniture-crushing
# tens of kg; a real 1.75 m fiberglass step ladder is ~12 kg.
LADDER_MASS_KG = 12.0

# -- prop masses. Without an authored MassAPI PhysX derives mass from collider volume at
#    1000 kg/m^3, which lands a hollow crate at tens of kg. --
BULB_MASS_KG = 0.035  # a real A19 incandescent/LED is 30-45 g
SOCKET_MASS_KG = 0.30  # the fixture half; kinematic, so this only matters for reporting
CRATE_MASS_KG = 1.5  # 0.60 x 0.40 x 0.17 m plastic parts crate

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


Quat = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]


def _quat_mul(q1: Quat, q2: Quat) -> Quat:
    """Hamilton product ``q1 * q2`` (w, x, y, z); rotating by the result applies ``q2``
    first, then ``q1`` -- the same convention as composing rotation matrices."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


# Omniverse SimReady assets author PhysX colliders but no RigidBodyAPI (nor MassAPI);
# Isaac Lab's RigidObjectCfg requires exactly one rigid-body prim (the spawner's property
# pass only *modifies* an existing API). Apply them on the root at spawn time, then apply
# the cfg's rigid/mass props (e.g. kinematic_enabled) which would otherwise silently no-op.
@clone
def _spawn_usd_as_rigid_body(prim_path, cfg, translation=None, orientation=None):
    from pxr import UsdPhysics

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim.GetPath(), cfg.rigid_props)
    if cfg.mass_props is not None:
        UsdPhysics.MassAPI.Apply(prim)
        schemas.modify_mass_properties(prim.GetPath(), cfg.mass_props)
    return prim


@clone
def _spawn_open_container(prim_path, cfg, translation=None, orientation=None):
    """``_spawn_usd_as_rigid_body`` + exact-triangle-mesh colliders, so a container is
    genuinely HOLLOW and a prop can rest inside it.

    The crate USD's authored collision is a single convex volume spanning the whole box
    (measured: one collider, z=[0, 0.17] across the full footprint), so anything placed in
    the crate starts inside solid geometry and is depenetrated straight out onto the floor.
    An exact triangle mesh (``physics:approximation = "none"``) keeps the interior open --
    the same fix the socket half of the LightBulb carries for its screw hole. Legal here
    because the crate is kinematic; PhysX rejects trimesh colliders on *dynamic* bodies.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim.GetPath(), cfg.rigid_props)
    if cfg.mass_props is not None:
        UsdPhysics.MassAPI.Apply(prim)
        schemas.modify_mass_properties(prim.GetPath(), cfg.mass_props)
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set("none")
    return prim


@clone
def _spawn_usd_as_rigid_body_frictional(prim_path, cfg, translation=None, orientation=None):
    """Tune rigid/mass props on a *preconfigured* rigid asset + bind a high-friction grip material.

    For the graspable ladder in ``FIATLUX-Carry-v0``, whose ``_collision_rigid`` USD already
    carries a single dynamic ``RigidBodyAPI`` + ``MassAPI``: this only *modifies* the existing
    body (solver/sleep props, mass override) and creates + binds a high-friction material
    (``UsdFileCfg`` has no ``physics_material`` field) so it can be held by hand friction -- no
    weld. (No ``RigidBodyAPI``/``MassAPI`` ``Apply`` needed; the asset ships them.)
    """
    from isaaclab.sim.utils import bind_physics_material

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim.GetPath(), cfg.rigid_props)
    if cfg.mass_props is not None:
        schemas.modify_mass_properties(prim.GetPath(), cfg.mass_props)
    grip = sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.2, restitution=0.0)
    grip.func(f"{prim_path}/physicsMaterial", grip)
    bind_physics_material(prim_path, f"{prim_path}/physicsMaterial")
    return prim


def _spawn_invisible_ground_plane(prim_path, cfg, translation=None, orientation=None):
    """The stock ``GroundPlaneCfg`` for its physics collider only.

    ``spawn_ground_plane`` always renders Isaac Sim's generic grid-texture mesh
    (``.../Environments/Grid/default_environment.usd``, tinted by ``cfg.color``); with the
    Simple Room backdrop's own floor also in the scene, that grid sits on top of and hides
    it. Hide the mesh here so the room's floor is what's actually visible; the plane prim
    still carries the collider the family relies on for floor physics.
    """
    from pxr import UsdGeom

    prim = sim_utils.spawn_ground_plane(prim_path, cfg, translation, orientation)
    env_prim = prim.GetStage().GetPrimAtPath(f"{prim_path}/Environment")
    if env_prim.IsValid():
        UsdGeom.Imageable(env_prim).MakeInvisible()
    return prim


@configclass
class G1ReplaceSceneCfg(DressedSceneCfg):
    """The G1 light-bulb-replacement world (see module docstring for the preset layouts)."""

    # ------------------------------------------------------------------ ground & lighting
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            func=_spawn_invisible_ground_plane,
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=LADDER_POSITION, rot=_quat_z_deg(LADDER_YAW_DEG)),
    )
    # Socket fixture: a BEHAVIOR-1K lamp standing in for the bulb socket; kinematic so it
    # can be re-posed on reset. On the floor in the workshop preset, on the table in the
    # tabletop preset (at-height mounting is the deferred elevated preset's business).
    # KINEMATIC ONLY, permanently: the fixture's colliders are an exact triangle mesh (that is
    # what keeps the screw hole open), and PhysX does not allow a trimesh collider on a dynamic
    # body. Do not add a mass-randomization or "knock the lamp over" knob here without first
    # swapping the collider approximation -- it would fail at parse time.
    socket: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Socket",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SOCKET_USD,
            func=_spawn_usd_as_rigid_body,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=SOCKET_MASS_KG),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=SOCKET_POSITION),
    )
    # Graspable bulb (dynamic). The wrapper layer already carries RigidBodyAPI + MassAPI, so the
    # plain spawner suffices. contact_offset is cut from the PhysX default 0.02 m, which is half
    # the screw cap's diameter -- it would start generating contacts 2 cm before touch, the
    # classic cause of a seated bulb buzzing in the hole instead of resting.
    bulb: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bulb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BULB_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0, torsional_patch_radius=0.005
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BULB_MASS_KG),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BULB_POSITION),
    )
    # Old bulb, seated in the elevated fixture (replace preset only). Dynamic, so removing
    # it is a real physical event; built by apply_replace_preset.
    old_bulb: RigidObjectCfg | None = None
    # Rod a ceiling-mounted fixture hangs from (see add_ceiling_pendant). Only the presets
    # that mount overhead spawn it; wall mounts and the bench have no pendant.
    pendant: AssetBaseCfg | None = None
    # Packing table (manipulation bench). Spawned only by the tabletop preset.
    table: AssetBaseCfg | None = None
    # Parts crate: bulb bin (install/remove presets) or old-bulb disposal target (replace
    # preset). Always the kinematic RigidObjectCfg built by _add_parts_bin -- the disposal
    # reward/obs terms read its pose, which an AssetBaseCfg (XformPrimView) cannot serve.
    bin: RigidObjectCfg | None = None

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
        # STATIC, not kinematic: the USD authors colliders but no RigidBodyAPI, so it is
        # already a static collider -- immovable, and cheaper than a kinematic body. The
        # kinematic_enabled=True this used to carry was a silent no-op
        # (modify_rigid_body_properties returns False on a prim without a RigidBodyAPI).
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_USD),
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


def apply_position_preset(scene: G1ReplaceSceneCfg) -> None:
    """Ladder-positioning start (FIATLUX-Carry-v0): a DYNAMIC, high-friction, graspable ladder
    standing upright out in front of the robot; the robot grasps a rail and carries it to
    TARGET_LADDER_POSITION, directly beneath the ceiling light fixture. Scored by reusing the
    Replace task's ladder terms (see carry_env_cfg).
    """
    scene.ladder.spawn = sim_utils.UsdFileCfg(
        usd_path=STEP_LADDER_RIGID_USD,
        # The A-frame step ladder's PRECONFIGURED `_collision_rigid` variant -- already a single
        # dynamic RigidBodyAPI + MassAPI, so the spawner does not stamp the rigid body; it just
        # tunes the solver/mass props on the existing body and binds a high-friction grip
        # material (UsdFileCfg has no physics_material field). cm-authored -> scale 0.01; mass
        # overridden below to a modest value so one arm can move it.
        func=_spawn_usd_as_rigid_body_frictional,
        scale=(0.01, 0.01, 0.01),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            # High POSITION iters for stable resting contact; LOW velocity iters -- high
            # velocity-iteration counts make PhysX's TGS solver inject energy and jitter the
            # ladder at rest (the ">4 velocity iterations" warning). 1 keeps it steady.
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=1.0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=LADDER_MASS),
    )
    scene.ladder.init_state.pos = POSITION_LADDER_START_POS
    scene.ladder.init_state.rot = _quat_z_deg(POSITION_LADDER_START_YAW)
    scene.robot.init_state.pos = POSITION_ROBOT_POSITION
    scene.fixture = None
    # The light fixture the positioned ladder leads to: the same socket + bulb the bench tasks
    # use, mounted on the ceiling above the target and flipped bulb-down -- identical to how
    # apply_replace_preset mounts its ceiling fixture. The bulb stays KINEMATIC here: it is
    # visual context only (this task scores the ladder pose), so it needs no retention.
    # Seated == the fixture's own pose, both halves being authored assembled at identity.
    fixture_pos = (TARGET_LADDER_POSITION[0], TARGET_LADDER_POSITION[1], CEILING_FIXTURE_Z)
    fixture_quat = _quat_y_deg(180.0)
    scene.socket.init_state.pos = fixture_pos
    scene.socket.init_state.rot = fixture_quat
    scene.bulb.init_state.pos = fixture_pos
    scene.bulb.init_state.rot = fixture_quat
    scene.bulb.spawn.rigid_props.kinematic_enabled = True  # overhead context, not the manipuland
    add_ceiling_pendant(scene, fixture_pos[0], fixture_pos[1], fixture_pos[2])
    # hand_contact stays for the net-force obs + compliance penalty; no ladder force-matrix
    # filter (Carry scores the ladder via the Replace task's pose terms, not a grasp reward).


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
    add_ceiling_pendant(scene, ELEVATED_SOCKET_POSITION[0], ELEVATED_SOCKET_POSITION[1], ELEVATED_SOCKET_POSITION[2])


def add_ladder_contact_sensor(scene: G1ReplaceSceneCfg) -> None:
    """Feet + palms filtered against the kinematic ladder (climb / descend tasks).

    Not a class field: presets without a ``Ladder`` prim (tabletop) could not resolve
    the filter expression. One multi-body sensor suffices — per-body ``force_matrix_w``
    against a *single* filter body works in this stack (proven by
    ``verify_interactions.py``'s whole-robot ``limb_ladder_contact`` sensor), so the
    per-link-sensor workaround from the upstream ContactSensor docstring is not needed.
    """
    scene.ladder_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*(ankle_roll|hand_base)_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ladder"],
        history_length=1,
        track_air_time=False,
    )


def add_wrist_camera(scene: G1ReplaceSceneCfg) -> None:
    """Attach the standard wrist-mounted RGB camera (manipulation subtasks)."""
    scene.wrist_camera = wrist_camera_cfg()


def add_ego_camera(scene: G1ReplaceSceneCfg) -> None:
    """Attach the standard head-mounted RGB camera (whole-body subtasks; GR00T's ego view)."""
    scene.ego_camera = ego_camera_cfg()


def add_mid360_lidar(scene: G1ReplaceSceneCfg) -> None:
    """Attach the standard head-mounted lidar, ray-casting the ground and the ladder
    (when this preset has one -- the tabletop preset drops ``scene.ladder``)."""
    scene.mid360_lidar = mid360_lidar_cfg(include_ladder=scene.ladder is not None)


def add_ceiling_pendant(scene: G1ReplaceSceneCfg, x: float, y: float, fixture_z: float) -> None:
    """Hang an overhead fixture from the ceiling on a rod instead of from nothing.

    The room's ceiling underside is ROOM_CEILING_Z (4.179 m, measured), which is well out
    of a ladder-top reach, so overhead fixtures sit at a reachable CEILING_FIXTURE_Z and
    this spans the gap. Static geometry, like the room: it is structure, not a prop.
    """
    drop = ROOM_CEILING_Z - fixture_z
    if drop <= 0.0:
        return
    scene.pendant = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pendant",
        spawn=sim_utils.CylinderCfg(
            radius=PENDANT_RADIUS,
            height=drop,
            axis="Z",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05), metallic=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, fixture_z + drop / 2.0)),
    )


def _add_parts_bin(scene: G1ReplaceSceneCfg, position: Vec3 = BIN_POSITION) -> None:
    """Spawn the kinematic parts crate on the floor (the every-preset crate definition).

    A kinematic ``RigidObjectCfg``, never ``AssetBaseCfg``: the disposal reward/termination
    channels (``mdp.old_bulb_disposal_distance`` and friends) and the privileged obs group
    read the crate's pose via ``scene["bin"].data``, and ``InteractiveScene`` files every
    ``AssetBaseCfg`` into ``extras`` as an ``XformPrimView``, which has no ``.data`` at all
    -- so an AssetBaseCfg crate makes those terms raise ``AttributeError`` on the first step.
    The crate USD authors colliders but no RigidBodyAPI, hence the SimReady spawn helper.
    """
    scene.bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bin",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CRATE_USD,
            func=_spawn_open_container,
            # cm-authored prop (real crate 0.60 x 0.40 x 0.17 m); unscaled it spawns as a
            # 60 m colossus filling the whole room.
            scale=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=CRATE_MASS_KG),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def apply_remove_preset(scene: G1ReplaceSceneCfg) -> None:
    """Bulb-removal start: Insert's bench with the bulb SEATED in the table lamp.

    Same world as the Insert/Install bench, with the empty parts crate beside it as the
    bulb's destination. The bulb stays DYNAMIC and is held only by gravity and contact:
    the fixture is upright here and its screw hole is an open triangle-mesh collider, so a
    seated bulb nests and rests, and the robot can lift it straight out. That is what makes
    this task solvable -- it used to spawn kinematic (a "screwed in" stand-in for an
    attach/detach joint that never landed), which meant no action could move it at all.
    """
    apply_tabletop_preset(scene)
    _add_parts_bin(scene)
    scene.bulb.init_state.pos = TABLETOP_SEATED_BULB_POSITION


def apply_install_preset(scene: G1ReplaceSceneCfg) -> None:
    """Bulb-installation start: Insert's bench, empty lamp socket, fresh bulb in the crate."""
    apply_tabletop_preset(scene)
    _add_parts_bin(scene)
    scene.bulb.init_state.pos = BIN_BULB_POSITION


##
# Replace preset (issue #20): randomized full-scene layout.
##

# Wall lookup for the fixture's random mount: name -> (fixed axis index (0=x, 1=y), the
# fixed coordinate on that wall, its inward-facing unit normal, yaw so local +X faces inward).
_WALLS: dict[str, tuple[int, float, tuple[float, float], float]] = {
    "west": (0, ROOM_FLOOR_MIN[0], (1.0, 0.0), 0.0),
    "east": (0, ROOM_FLOOR_MAX[0], (-1.0, 0.0), 180.0),
    "south": (1, ROOM_FLOOR_MIN[1], (0.0, 1.0), 90.0),
    "north": (1, ROOM_FLOOR_MAX[1], (0.0, -1.0), -90.0),
}


def _sample_fixture_mount(
    rng: random.Random,
) -> tuple[Literal["ceiling", "wall"], Vec3, Quat, tuple[float, float] | None]:
    """Randomly mount the fixture on the ceiling or a wall.

    Returns ``(mount_kind, position, orientation, wall_inward_normal)`` -- the normal is
    ``None`` for a ceiling mount (nothing to stand off from). ``ehjsdz`` is authored as an
    upright desk lamp (socket opening up, base on a horizontal surface), so each mount kind
    needs a reorienting rotation: ceiling flips it ~180 deg so the shade/socket point down
    like a pendant light; wall rotates it ~90 deg so it projects outward from the wall face.
    """
    margin = 0.5
    if rng.random() < 0.5:
        x = rng.uniform(ROOM_FLOOR_MIN[0] + margin, ROOM_FLOOR_MAX[0] - margin)
        y = rng.uniform(ROOM_FLOOR_MIN[1] + margin, ROOM_FLOOR_MAX[1] - margin)
        return "ceiling", (x, y, CEILING_FIXTURE_Z), _quat_y_deg(180.0), None

    wall_name = rng.choice(list(_WALLS))
    axis, value, normal, yaw = _WALLS[wall_name]
    if axis == 0:
        along = rng.uniform(ROOM_FLOOR_MIN[1] + margin, ROOM_FLOOR_MAX[1] - margin)
        pos = (value, along, WALL_MOUNT_Z)
    else:
        along = rng.uniform(ROOM_FLOOR_MIN[0] + margin, ROOM_FLOOR_MAX[0] - margin)
        pos = (along, value, WALL_MOUNT_Z)
    # +90 (not -90): local +Z (the shade/socket opening) must map to local +X so the
    # per-wall yaw (chosen so "local +X faces inward") ends up pointing the shade into the
    # room. -90 was checked numerically and puts the shade dot(inward_normal) = -1.0 --
    # exactly backwards, facing into the wall with only the lamp's base in the room.
    quat = _quat_mul(_quat_z_deg(yaw), _quat_y_deg(90.0))
    return "wall", pos, quat, normal


def _sample_nonoverlapping_centers(
    rng: random.Random,
    half_sizes: list[float],
    bounds_min: tuple[float, float],
    bounds_max: tuple[float, float],
    fixed: list[tuple[float, float] | None] | None = None,
    margin: float = ZONE_MARGIN,
    max_tries: int = 500,
) -> list[tuple[float, float]]:
    """Rejection-sample 2D zone centers (axis-aligned squares of half-extent ``half_sizes[i]``)
    so every pair stays >= the sum of their half-sizes + ``margin`` apart on at least one
    axis (the bounding squares, plus margin, never overlap), each inset from the room bounds
    by its own half-size. ``fixed[i]``, if given, pins zone ``i`` to that center instead of
    sampling it (used for the ladder in ``couple_ladder_to_fixture`` mode) -- other zones
    are still sampled to avoid it. Runs once at cfg-build time (plain Python, no torch).

    Free zones are placed largest-first (a fixed zone, e.g. a coupled ladder, still goes in
    first regardless of size): stress-tested at 5000 random layouts against this room/these
    zone sizes with ~0.1% placement failures, vs. ~1.3% with left-to-right order (small
    zones sampled first can strand a later, larger one with nowhere left to fit).
    """
    n = len(half_sizes)
    centers: list[tuple[float, float] | None] = list(fixed) if fixed else [None] * n
    order = sorted((i for i in range(n) if centers[i] is None), key=lambda i: -half_sizes[i])

    def overlaps(i: int, c: tuple[float, float]) -> bool:
        for j, other in enumerate(centers):
            if other is None or j == i:
                continue
            min_sep = half_sizes[i] + half_sizes[j] + margin
            # bounding boxes overlap iff BOTH axis gaps are under min_sep
            if max(abs(c[0] - other[0]), abs(c[1] - other[1])) < min_sep:
                return True
        return False

    for i in order:
        hs = half_sizes[i]
        lo_x, lo_y = bounds_min[0] + hs, bounds_min[1] + hs
        hi_x, hi_y = bounds_max[0] - hs, bounds_max[1] - hs
        candidate = (rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y))
        for _ in range(max_tries):
            candidate = (rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y))
            if not overlaps(i, candidate):
                break
        centers[i] = candidate  # best-effort: accept the last sample rather than raise
    return cast(list[tuple[float, float]], centers)  # every slot filled: fixed, or by the loop


def apply_replace_preset(
    scene: G1ReplaceSceneCfg,
    rng: random.Random | None = None,
    couple_ladder_to_fixture: bool = False,
) -> None:
    """The full replacement-task layout: robot, ladder, table+fresh bulb, disposal crate, and
    the elevated socket/lamp ("fixture") with the OLD BULB seated in it -- each floor occupant
    randomized into its own non-overlapping "safe zone", and the fixture randomly ceiling- or
    wall-mounted. Randomized once per scene build (this function's own ``rng`` draw), not
    re-sampled every episode reset.

    The fixture reuses ``SOCKET_USD`` (``ehjsdz``/``kfmkwd``, the validated bulblampF/M pair
    already used by the tabletop Insert task) rather than the decorative ``ELEVATED_SOCKET_USD``
    chandelier (used only by climb/descend, which never validated a socket metalink on it) --
    this scene needs a genuinely insertible bulb+socket at height.

    Unlike every other preset the ladder spawns *dynamic* (mass ``LADDER_MASS_KG``): a
    knocked-over ladder is a real, penalized event in this task. The old bulb starts seated
    and DYNAMIC at the fixture's own pose (both halves are authored assembled at identity,
    so no offset arithmetic is needed at any mount orientation). Because the fixture is
    inverted here, the bulb is held by the seat constraint rather than by gravity.

    Args:
        couple_ladder_to_fixture: place the ladder's zone reachably relative to wherever the
            fixture mounted (beneath a ceiling point, or standing off from a mounted wall)
            instead of sampling it fully independently. Off by default -- positioning the
            ladder is part of the task; coupling is a debug/curriculum aid only.
    """
    # Default rng derives from the (seedable) global stream: scripts that call
    # `random.seed(seed)` before cfg construction (eval.py, record_run.py) get a
    # deterministic layout -- the benchmark's same-seed-same-numbers contract.
    rng = rng or random.Random(random.getrandbits(64))

    # Table: holds the fresh bulb. No separate tabletop socket -- the elevated fixture is the
    # real insertion target in this scene.
    scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_USD),  # static collider; see apply_tabletop_preset
        init_state=AssetBaseCfg.InitialStateCfg(),
    )
    scene.fixture = None  # the task fixture owns the ceiling/wall in this scene

    # Fixture mount (socket stays on its default SOCKET_USD -- the validated lamp).
    mount_kind, fixture_pos, fixture_quat, wall_normal = _sample_fixture_mount(rng)
    scene.socket.init_state.pos = fixture_pos
    scene.socket.init_state.rot = fixture_quat
    if mount_kind == "ceiling":
        add_ceiling_pendant(scene, fixture_pos[0], fixture_pos[1], fixture_pos[2])

    # Ladder zone: independent by default; coupled (reachable from the fixture) as an
    # explicit debug/curriculum opt-in.
    fixed_ladder = None
    ladder_yaw = rng.uniform(0.0, 360.0)
    if couple_ladder_to_fixture:
        if mount_kind == "ceiling":
            fixed_ladder = (fixture_pos[0], fixture_pos[1])
        else:
            assert wall_normal is not None  # non-None on every non-ceiling mount
            standoff = LADDER_ZONE_HALF_SIZE + LADDER_WALL_STANDOFF
            fixed_ladder = (
                fixture_pos[0] + wall_normal[0] * standoff,
                fixture_pos[1] + wall_normal[1] * standoff,
            )
            # steps face back toward the wall/fixture
            ladder_yaw = math.degrees(math.atan2(-wall_normal[1], -wall_normal[0]))

    robot_center, table_center, ladder_center, disposal_center = _sample_nonoverlapping_centers(
        rng,
        half_sizes=[
            ROBOT_ZONE_HALF_SIZE,
            TABLE_ZONE_HALF_SIZE,
            LADDER_ZONE_HALF_SIZE,
            DISPOSAL_ZONE_HALF_SIZE,
        ],
        bounds_min=ROOM_FLOOR_MIN,
        bounds_max=ROOM_FLOOR_MAX,
        fixed=[None, None, fixed_ladder, None],
    )

    scene.robot.init_state.pos = (robot_center[0], robot_center[1], ROBOT_POSITION[2])
    # Face the table (the task's first target), +/- a small jitter: the ego camera's
    # 50 deg frustum must contain the work area or the standard observation mode
    # cannot see the task at all.
    facing = math.degrees(math.atan2(table_center[1] - robot_center[1], table_center[0] - robot_center[0]))
    scene.robot.init_state.rot = _quat_z_deg(facing + rng.uniform(-15.0, 15.0))
    scene.table.init_state.pos = (table_center[0], table_center[1], TABLE_POSITION[2])
    bulb_local_offset = tuple(b - t for b, t in zip(TABLETOP_BULB_POSITION, TABLE_POSITION))
    scene.bulb.init_state.pos = (
        table_center[0] + bulb_local_offset[0],
        table_center[1] + bulb_local_offset[1],
        TABLE_POSITION[2] + bulb_local_offset[2],
    )
    scene.ladder.init_state.pos = (ladder_center[0], ladder_center[1], LADDER_POSITION[2])
    scene.ladder.init_state.rot = _quat_z_deg(ladder_yaw)
    # Dynamic ladder (this preset only): tipping/falling is a scored physical event.
    scene.ladder.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=8,
        max_depenetration_velocity=1.0,
    )
    scene.ladder.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=LADDER_MASS_KG)

    # Old bulb: seated in the fixture. Seated == the fixture's own pose (both halves authored
    # assembled at identity), so it starts in exact resting contact with zero interpenetration.
    # DYNAMIC, so it can actually be grasped and removed -- the fixture is inverted here, so it
    # is held by the seat constraint (mdp/attach.py), not by gravity.
    scene.old_bulb = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OldBulb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BULB_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0, torsional_patch_radius=0.005
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BULB_MASS_KG),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=fixture_pos, rot=fixture_quat),
    )

    # Disposal crate: the old bulb's destination, in its own sampled zone.
    _add_parts_bin(scene, position=(disposal_center[0], disposal_center[1], BIN_POSITION[2]))
