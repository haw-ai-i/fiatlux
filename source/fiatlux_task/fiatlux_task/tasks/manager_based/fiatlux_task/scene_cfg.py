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
from isaaclab.envs.common import ViewerCfg
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
    G1_HORIZONTAL_REACH,
    G1_OVERHEAD_REACH,
    SOCKET_USD,
    STEP_LADDER_RIGID_USD,
    STEP_LADDER_TOP_OFFSET,
    STEP_LADDER_USD,
    TABLE_USD,
)
from fiatlux_task.robots.g1 import G1_INSPIRE_CFG, G1_LADDER_CONTACT_BODIES
from fiatlux_task.scenes import DressedSceneCfg
from fiatlux_task.sensors import ego_camera_cfg, mid360_lidar_cfg, wrist_camera_cfg

# -- default (workshop) placement (module constants, not scene fields; override via each
#    entity's init_state). BEHAVIOR-1K USDs are authored with the origin near the bbox
#    center, so a prop resting on the floor sits at roughly half its height. --
_ROBOT_Z = G1_INSPIRE_CFG.init_state.pos[2]  # standing pelvis height (feet on the floor)
ROBOT_POSITION = (0.0, 0.0, _ROBOT_Z)
# STEP_LADDER_USD (0.608 x 0.979 x 1.861 m, base authored at z=0), yawed 90 deg so its steps
# face -x, toward the robot's approach.
LADDER_POSITION = (1.6, 0.0, 0.0)
LADDER_YAW_DEG = 90.0
# The fixture's origin is its floor-contact plane (SOCKET_BASE_Z_OFFSET = 0).
SOCKET_POSITION = (-0.8, 0.0, 0.0)  # socket-fixture standing on the floor
BULB_POSITION = (-0.55, -0.20, -BULB_STAND_Z_OFFSET)  # standing on its cap on the floor

# -- tabletop (manipulation bench) placement: the Insert layout. The robot works
#    the bench from its +y long side, facing -y: the packing table's collision
#    volume spans x[-0.82,1.62] x y[-0.48,0.28] with an under-frame up to z=0.95.
#    The standoff must clear the robot's whole spawn envelope, not just its legs:
#    the default arm dangle puts the right fingers ~0.40 m in front of the base at
#    z~0.9, and fingers inside the under-frame volume take a depenetration kick at
#    every reset. --
TABLE_POSITION = (0.40, -0.10, 0.0)
# The table's work surface, from its main collider, not its overall bbox: that reads 1.0829,
# the top of the mesh tray sitting on the table. Props are placed clear of the tray.
TABLETOP_SURFACE_Z = 0.9941
TABLETOP_ROBOT_POSITION = (0.60, 0.73, _ROBOT_Z)
TABLETOP_ROBOT_YAW_DEG = -90.0
# The fixture's origin is its own floor-contact plane.
TABLETOP_SOCKET_POSITION = (0.45, 0.10, TABLETOP_SURFACE_Z)
# Standing on its screw cap: laid on its side the bulb tips onto the cap within a second. The
# asset's root origin sits below its own geometry (cap bottom at +BULB_STAND_Z_OFFSET in the
# root frame), so resting on a surface puts the root under it.
TABLETOP_BULB_POSITION = (0.30, 0.18, TABLETOP_SURFACE_Z - BULB_STAND_Z_OFFSET)

# -- position (ladder-handling) subtask: FIATLUX-Carry-v0. The ladder is dynamic,
#    high-friction and graspable, starts upright but off-target, and is carried to the fixed
#    upright target under the light. Base authored at z=0, so start/target z=0. --
POSITION_ROBOT_POSITION = (-0.20, -0.20, 0.75)
# ~2 m robot->ladder, ~1.5 m ladder->light. The fixture is mounted at
# TARGET_LADDER_POSITION's x,y on the ceiling.
POSITION_LADDER_START_POS = (1.50, 0.85, 0.0)
POSITION_LADDER_START_YAW = 30.0
TARGET_LADDER_POSITION = (0.55, -0.30, 0.0)  # directly beneath the ceiling fixture
# Same SOCKET_USD / BULB_USD as the Insert/Replace tasks, ceiling-mounted above the target and
# flipped bulb-down. See apply_position_preset.

# -- at-height presets (climb / descend): elevated fixture over the ladder. --
ELEVATED_SOCKET_POSITION = (1.9, 0.0, 2.80)  # cage bottom clears the at-top robot's head
CLIMB_ROBOT_POSITION = (0.75, 0.0, _ROBOT_Z)  # at the step ladder's base, ready to ascend
# Pelvis on the ladder's standing tread (descend). STEP_LADDER_TOP_OFFSET's local x,y are
# both zero, so the stance sits over the ladder's root at any yaw.
TOP_ROBOT_POSITION = (LADDER_POSITION[0], LADDER_POSITION[1], STEP_LADDER_TOP_OFFSET[2] + _ROBOT_Z)
PARKED_BULB_POSITION = (0.5, -0.6, -BULB_STAND_Z_OFFSET)  # standing out of the way on the floor

# -- bench manipulation extras (remove / install share Insert's tabletop world) --
# Both halves are authored assembled at identity, so a seated bulb is the fixture's own pose:
# no offset arithmetic, and zero interpenetration at reset by construction.
TABLETOP_SEATED_BULB_POSITION = TABLETOP_SOCKET_POSITION
BIN_POSITION = (0.15, -0.75, 0.0)  # parts crate on the floor beside the bench
# The crate's inner floor is at 0.055 m, not the outer bbox's 0.17 rim top. Requires the
# hollow-collider spawner: against the stock single convex collider a bulb placed here is
# depenetrated straight out onto the floor.
BIN_BULB_INTERIOR_Z = 0.055
BIN_BULB_POSITION = (0.15, -0.75, BIN_BULB_INTERIOR_Z - BULB_STAND_Z_OFFSET)

# -- replace preset: robot / table+bulb / ladder each randomized into their own
# non-overlapping floor "safe zone", fixture ceiling- or wall-mounted. Inset from the room's own
# wall span (x=[-4.52,4.52], y=[-3.4,4.86], z=[-0.58,3.22]).
ROOM_FLOOR_MIN = (-4.0, -3.0)
ROOM_FLOOR_MAX = (4.0, 4.2)

# Heights after ``_spawn_room_backdrop`` aligns the walking surface to z=0 (the asset is
# authored tabletop-at-origin, so everything shifts +0.7696):
#   floor 0.000 | baseboard 0.000-0.190 | walls 0.190-3.989 | ceiling slab 4.179-4.197.
# ROOM_CEILING_Z is the ceiling's underside -- a real surface to mount to. It is far above
# reach, so ceiling fixtures hang from it on a pendant (add_ceiling_pendant) at
# CEILING_FIXTURE_Z rather than floating unsupported at the reach height.
ROOM_CEILING_Z = 4.179
ROOM_WALL_TOP_Z = 3.989
CEILING_FIXTURE_Z = 2.2  # fixture height, both mount kinds: workable from LADDER_WORK_FOOT_Z
WALL_MOUNT_Z = 2.2
# The stance the at-height tasks work from: the pelvis height Climb's success gate accepts, and
# the foot height that implies. Reach is taken from here, not from the tread, because a
# policy that only just clears the climb gate still has to do the job.
LADDER_TOP_STANCE_TOLERANCE = 0.15  # m; Climb's SUCCESS_HEIGHT slack below TOP_ROBOT_POSITION
LADDER_WORK_PELVIS_Z = TOP_ROBOT_POSITION[2] - LADDER_TOP_STANCE_TOLERANCE
LADDER_WORK_FOOT_Z = LADDER_WORK_PELVIS_Z - _ROBOT_Z
# Highest point a fixture may be mounted at and still be worked on. Asserted at import.
# Vertical only: nothing here bounds the HORIZONTAL distance to a wall-mounted fixture.
MAX_REACHABLE_MOUNT_Z = LADDER_WORK_FOOT_Z + G1_OVERHEAD_REACH
if max(CEILING_FIXTURE_Z, WALL_MOUNT_Z) > MAX_REACHABLE_MOUNT_Z:
    raise ValueError(
        f"fixture mount heights (ceiling {CEILING_FIXTURE_Z} m, wall {WALL_MOUNT_Z} m) exceed the "
        f"reach from the ladder working stance ({MAX_REACHABLE_MOUNT_Z:.3f} m, feet at "
        f"{LADDER_WORK_FOOT_Z:.3f} m): the at-height tasks would be unsolvable by construction"
    )
# A fixture must still require the ladder.
if min(CEILING_FIXTURE_Z, WALL_MOUNT_Z) <= G1_OVERHEAD_REACH:
    raise ValueError(
        f"fixture mount heights (ceiling {CEILING_FIXTURE_Z} m, wall {WALL_MOUNT_Z} m) are within "
        f"standing floor reach ({G1_OVERHEAD_REACH:.3f} m): the ladder would be unnecessary"
    )
# Horizontal tolerance on the ladder placement (m): the reach left over the fixture's vertical
# gap above the working stance. Derived from the CEILING mount only.
LADDER_READY_MARGIN = 0.05  # m, held back off the geometric bound
LADDER_READY_XY_RADIUS = (
    math.sqrt(G1_OVERHEAD_REACH**2 - (CEILING_FIXTURE_Z - LADDER_WORK_FOOT_Z) ** 2) - LADDER_READY_MARGIN
)
if LADDER_READY_XY_RADIUS <= 0.0:
    raise ValueError(
        f"no horizontal slack left for the ladder placement: a {CEILING_FIXTURE_Z} m fixture eats "
        f"the whole {G1_OVERHEAD_REACH:.3f} m reach from the {STEP_LADDER_TOP_OFFSET[2]} m ladder top"
    )
# How far from the ladder's root the robot may stand and still reach a rail: horizontal arm
# reach plus the root-to-near-rail offset, less a margin. The narrow axis is the conservative
# choice, since the ladder's yaw is sampled. Mounting the ladder is a foot-placement question
# with its own constant.
LADDER_NEAR_RAIL_OFFSET = 0.304  # m, half AlumStep_D's narrow footprint axis (0.608 m x-dim)
LADDER_APPROACH_RADIUS = G1_HORIZONTAL_REACH + LADDER_NEAR_RAIL_OFFSET - 0.05

# Must clear the room's own wall box (9.04 x 8.26 m): each env carries its own room. Overlap is
# harmless (filter_collisions=True isolates each env) but makes num_envs > 1 renders unreadable.
ROOM_ENV_SPACING = 10.0
PENDANT_RADIUS = 0.012  # the rod a ceiling fixture hangs from
# Decorative per-env ceiling fixture, flush against the ceiling: these are ceiling-mount
# BEHAVIOR-1K assets.
FIXTURE_POSITION = (0.0, 0.0, ROOM_CEILING_Z)

SCORED_BODY_SLEEP_THRESHOLD = 0.0  # #121

TABLE_COLLISION_HALF_EXTENT = (1.24, 0.38)

# Zone half-sizes (m): each occupant's "safe square" half-extent, footprint plus working
# clearance. The table's is a square bound around its elongated footprint.
ROBOT_ZONE_HALF_SIZE = 0.6
TABLE_ZONE_HALF_SIZE = 1.5
LADDER_ZONE_HALF_SIZE = 1.0
DISPOSAL_ZONE_HALF_SIZE = 0.5  # the old-bulb disposal crate (crate footprint ~0.6 m + clearance)
ZONE_MARGIN = 0.5  # minimum gap left between any two zones' bounding squares
# The nearest a wall mount's reserved ladder anchor can physically be to the wall face: the
# ladder's footprint is 0.979 m deep and the coupled draw turns that depth normal to the wall, so
# the root cannot come nearer than 0.49 m plus clearance. The previous standoff was
# LADDER_ZONE_HALF_SIZE + 0.4 = 1.4 m, a zone-packing number that left every wall draw well beyond
# reach and went unasserted, because both reach guards below are derived from the ceiling mount
# and MAX_REACHABLE_MOUNT_Z checks height only.
LADDER_LADDER_HALF_DEPTH = 0.49  # m, half the ladder's 0.979 m footprint depth
# How far the ladder's reserved anchor sits from the fixture in the floor plane -- from the wall
# face, or from the point directly beneath a ceiling mount. Both mount kinds: an anchor under a
# ceiling fixture puts the on-ladder working stance's chest inside it.
# A wall draw has the ladder's own depth behind it; a ceiling draw has nothing there, so the two
# are bounded by different things and get different values (issue #130).
LADDER_FIXTURE_STANDOFF_WALL = 0.60  # m; leaves 0.11 m behind the ladder -- the closest a wall
# draw can sit, since the ladder's own depth is what sits behind it.
# MEASURED 2026-09-07 by FK from the on-ladder stance, layout seeds 1-8: shoulder-to-palm at full
# extension is 0.419 m while the socket sat 0.49-0.52 m away, so the fingertips reached it and the
# palm never did -- touch, not grasp (issue #130). This closes that gap with the arm short of full
# extension; only the torso-clearance floor bounds it below.
LADDER_FIXTURE_STANDOFF_CEILING = 0.48  # m; nothing sits behind the ladder at a ceiling mount, so
# it comes closer than the wall draw, keeping the socket inside palm reach.
# The stance must clear the fixture, not just reach it: the working stance's pelvis is at 1.967 m
# and the fixture at 2.200 m, so the fixture is at chest height and the torso is what collides.
# MEASURED 2026-08-26, ``scripts/verify_ladder_stance.py --measure`` at layout seed 1: the widest
# non-arm body in a +/-0.15 m band around fixture height is right_shoulder_yaw_link at 0.152 m
# from the pelvis axis; the mounted socket's AABB half-extents are 0.074 x 0.081 m.
G1_STANCE_TORSO_HALF_EXTENT = 0.152  # m
FIXTURE_HALF_EXTENT = 0.081  # m
# reset_robot_root jitters the stance +/-5 cm on each floor axis, so the worst case is the diagonal.
STANCE_RESET_JITTER = math.hypot(0.05, 0.05)  # m
LADDER_FIXTURE_MIN_STANDOFF = G1_STANCE_TORSO_HALF_EXTENT + FIXTURE_HALF_EXTENT + STANCE_RESET_JITTER
# Horizontal slack left over once the fixture's height above LADDER_WORK_FOOT_Z is accounted
# for -- the gate-tolerant foot height, not the tread itself, so the ceiling and wall guards
# agree on where the robot is standing. This is the SPHERICAL bound -- G1_OVERHEAD_REACH is a
# vertical fingertip measurement used here as a radius -- so it is optimistic. The honest forward reach
# (G1_HORIZONTAL_REACH) is 0.5045 m, which is less than the ladder's own half-depth: no standoff
# satisfies both, and a fixture at this height is reachable only if the arm does better
# reaching up-and-out than straight out. Re-measure by FK from the on-ladder stance before
# treating such a draw's score as meaningful.
LADDER_FIXTURE_REACH_SLACK = math.sqrt(max(G1_OVERHEAD_REACH**2 - (WALL_MOUNT_Z - LADDER_WORK_FOOT_Z) ** 2, 0.0))
if not LADDER_LADDER_HALF_DEPTH < LADDER_FIXTURE_STANDOFF_WALL <= LADDER_FIXTURE_REACH_SLACK:
    raise ValueError(
        f"the wall fixture standoff ({LADDER_FIXTURE_STANDOFF_WALL} m) must clear the ladder's own "
        f"half-depth ({LADDER_LADDER_HALF_DEPTH} m) and stay inside the reach slack from the "
        f"platform ({LADDER_FIXTURE_REACH_SLACK:.3f} m); no value satisfies both if the fixture "
        "is too high"
    )
for _kind, _standoff in (("wall", LADDER_FIXTURE_STANDOFF_WALL), ("ceiling", LADDER_FIXTURE_STANDOFF_CEILING)):
    if _standoff <= LADDER_FIXTURE_MIN_STANDOFF:
        raise ValueError(
            f"the {_kind} fixture standoff ({_standoff} m) leaves the working stance's torso "
            f"inside the fixture: it must exceed {LADDER_FIXTURE_MIN_STANDOFF:.3f} m "
            f"(torso {G1_STANCE_TORSO_HALF_EXTENT} + fixture {FIXTURE_HALF_EXTENT} + jitter "
            f"{STANCE_RESET_JITTER:.3f})"
        )
# The anchor's own reserved zone, small enough that clamping it to the floor box (itself inset
# 0.41 m from the side walls) does not shove the ladder back out of reach. The ladder's real
# footprint half-diagonal is 0.576 m; ZONE_MARGIN (0.5 m) leaves 0.11 m of true separation even
# so, which is why the difference is safe rather than merely small.
LADDER_ANCHOR_HALF_SIZE = 0.18

# Quarter turn between the on-tread stance and the ladder's own frame; the coupled draw below
# turns the ladder back by the same amount. See ``subtask_tiers.balance.stand_robot_on_ladder_top``.
TOP_STANCE_YAW_OFFSET_DEG = 90.0

# Ladder mass, every preset. Without an authored MassAPI PhysX derives mass from collider
# volume at 1000 kg/m^3, which lands a hollow ladder at tens of kg.
#
# MEASURED, not guessed: AlumStep_D01 is a 4-step aluminium platform stepladder, and its four
# defining dimensions match Werner's P400-4 to within an inch -- platform height 1.18 m vs 4 ft,
# overall 1.861 m vs 6 ft, base width 0.608 m vs 24.5 in, platform depth 0.40 m vs 15 in. That
# ladder's catalogue net weight is 24 lb = 10.9 kg. The previous 7.58 came from
# ``omniverse_ladder_rigid.mass_for``'s shape-blind ``2 + 3*height`` heuristic and was ~30% light.
#
# MUST match the mass authored into the asset's rigid overlay (``omniverse_ladder_rigid.py
# --mass``): that overlay also authors a centre of mass and an inertia tensor, and PhysX does NOT
# rescale an authored inertia when a code-side override changes the mass -- mismatch here gives
# the body one object's mass and another's inertia.
LADDER_MASS_KG = 10.9

# -- prop masses. Without an authored MassAPI PhysX derives mass from collider volume at
#    1000 kg/m^3, which lands a hollow crate at tens of kg. --
BULB_MASS_KG = 0.035  # a real A19 incandescent/LED is 30-45 g
SOCKET_MASS_KG = 0.30  # the fixture half; kinematic, so this only matters for reporting
CRATE_MASS_KG = 1.5  # 0.60 x 0.40 x 0.17 m plastic parts crate

# -- per-env random ceiling fixture pool (visual dressing) --
# Ceiling-mount BEHAVIOR-1K categories only. Opt-in via ``download_assets.sh --scene-dressing``;
# when absent the pool is empty and ``FamilyBaseEnvCfg.__post_init__`` drops the ``fixture``
# entity. Category dirs mix ``<id>/<id>.usd`` and ``<id>/usd/<id>.usd``, hence two globs.
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


def _quat_x_deg(angle_deg: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) quaternion for a rotation about +X, in degrees (Y-up -> Z-up assets)."""
    half = math.radians(angle_deg) / 2.0
    return (math.cos(half), math.sin(half), 0.0, 0.0)


Quat = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]
# (mount kind, fixture position, fixture orientation, anchor bearing, ladder anchor)
# The bearing is the floor-plane unit vector from the fixture toward the ladder's reserved
# anchor: a wall's inward normal, or a sampled direction for a ceiling mount. Never None --
# a ceiling mount needs a direction to stand off in exactly as much as a wall one does.
FixtureMount = tuple[Literal["ceiling", "wall"], Vec3, Quat, Vec2, Vec2]


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
    (one collider, z=[0, 0.17] across the full footprint), so anything placed in
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
def _spawn_collidable_bench(prim_path, cfg, translation=None, orientation=None):
    """Spawn the packing table with colliders on its OWN geometry (issue #176).

    As authored, only ``container_h20``'s five box colliders (z 0.993-1.083, the tray on the
    work surface) collide. The table's frame, legs and lower shelf, and every crate and box on
    that shelf, are drawn but have no collider: a bulb or an arm passes straight through them.
    A camera therefore sees a loaded bench the solver does not have.

    The asset does carry ``CollisionAPI`` on ``SM_HeavyDutyPackingTable_C02_01``, but it sits on
    an Xform whose meshes are instance proxies, so it never reaches real geometry. De-instancing
    first is what makes the collider authorable.

    Exact mesh (``none``): the crates are thin-walled, and ``convexDecomposition`` leaves their
    floors porous -- a prop dropped into a crate falls through it. Legal because the table is
    static (``AssetBaseCfg``, no RigidBodyAPI); PhysX rejects triangle meshes only on dynamic
    bodies, the same reasoning ``_spawn_open_container`` records for the crate in #131.

    Each corrugated box carries its body mesh plus two ``trans__decal__*`` overlay meshes
    (trim + print, both within ~1-2% of the body's own bounding box -- not thin decals, near-full
    duplicates of it). Colliding all three would stack 3 coincident exact-mesh colliders on one
    box, which PhysX resolves as redundant/conflicting contact normals; only the body mesh needs
    one.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    for p in Usd.PrimRange(prim):
        if p.IsInstanceable():
            p.SetInstanceable(False)
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh) and not p.HasAPI(UsdPhysics.CollisionAPI) and "decal" not in p.GetName().lower():
            UsdPhysics.CollisionAPI.Apply(p)
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
    # (The ~6 mm contact offset now lives in the asset -- authored by omniverse_ladder_collision.py --
    # so no code-side trim is needed here; the ladder USD carries both the tight shape and the offset.)
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


def _make_bulb_cfg(prim_path: str, pos: Vec3, rot: Quat | None = None, *, kinematic: bool = False) -> RigidObjectCfg:
    """One bulb entity, whichever role it plays (issue #76 Step 1).

    ``fresh_bulb`` and ``old_bulb`` are the same asset with the same physics. Only the prim path,
    the spawn pose and whether it is kinematic differ, so a factory retires the duplicate spawn
    block that the Replace preset used to carry.

    ``contact_offset`` is cut from the PhysX default 0.02 m, half the screw cap's diameter, which
    would otherwise generate contacts 2 cm before touch and buzz a seated bulb in the hole.

    Bulb-socket collision is real (issue #167) -- no filter. An earlier version filtered this
    pair out because the bayonet lock's per-step pose overwrite fought the contact solver over
    the same body; that lock (and the overwrite) is gone, replaced by a continuous retention
    force in ``mdp.bulb_attachment`` that never disagrees with the solver the way the overwrite
    did, so there is nothing left for the filter to protect. This also required shrinking the
    plug's radius (``assets/omniverse_bulb/CHANGES.md``): the bore and the plug had zero or
    negative clearance at every height they overlapped, independent of collision approximation.
    """
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=BULB_USD,
            # No custom spawn func: the asset already carries RigidBodyAPI/MassAPI
            # (LightBulb_bulb_z_rigid.usda), and no collision filter is applied anymore.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=kinematic,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
                sleep_threshold=SCORED_BODY_SLEEP_THRESHOLD,
                stabilization_threshold=0.001,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0, torsional_patch_radius=0.005
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BULB_MASS_KG),
            activate_contact_sensors=True,
        ),
        init_state=(
            RigidObjectCfg.InitialStateCfg(pos=pos) if rot is None else RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot)
        ),
    )


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
    # cm-authored -> scale 0.01. Kinematic so it stays put while climbed. Dropped by the
    # tabletop preset; the carry preset swaps in its own ladder.
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
    # KINEMATIC ONLY: the fixture's colliders are an exact triangle mesh (which is what keeps
    # the screw hole open), and PhysX does not allow a trimesh collider on a dynamic body.
    # Anything that makes it dynamic fails at parse time.
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
    # Bulbs are named by PLACEMENT, not by task (issue #76 Step 1): a bulb seated in the socket
    # is the ``old_bulb``, a bulb anywhere else is the ``fresh_bulb``. Both are optional and
    # every preset builds what it owns with ``_make_bulb_cfg``, so no task inherits a bulb it
    # does not want. Replace is the only preset that builds both.
    #
    # The class carries no bulb of its own. It used to, and ``apply_workshop_preset`` was an
    # empty function because that default WAS the workshop layout -- which is exactly how Remove
    # ended up calling its single seated bulb ``bulb`` and the scoring layer ``old_bulb``.
    fresh_bulb: RigidObjectCfg | None = None
    # Seated in the socket. Retention is ``mdp.bulb_attachment``, which pins it at the seat until
    # it is rotated to the release angle and travels out of the channel.
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

    # -- Contact sensor on the grasping hand. Hand bodies only: this channel feeds the
    # fragility scoring, and a broader match would put the robot's own ground reaction
    # (~170 N standing, well past the 50 N threshold) into every episode's peak force.
    # PhysX logs "Filter pattern ... did not match the correct number of entries" here: it wants
    # one filter prim per sensor body. Benign -- force_matrix_w still allocates and populates.
    hand_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(right_hand_.*|right_wrist_.*|R_.*)",
        # Filtered to the manipulated object: the net force would also carry furniture the arm
        # rests against and the robot's own colliders. ``_sync_bulb_contact_filters`` fills this
        # from whichever bulbs the preset built -- do not set it per preset.
        filter_prim_paths_expr=[],
        history_length=1,
        track_air_time=False,
    )

    # -- The same channel for the LEFT hand, for recording only (issue #89).
    #
    # A SEPARATE sensor, not an extension of the one above, and the distinction is deliberate.
    # ``hand_contact`` feeds the fragility scoring through ``score.py``'s peak-force channel, and
    # it has eleven consumers. Widening its body match would silently change what
    # ``peak_contact_force`` measures and therefore what ``broken_rate`` reports, while #93 is
    # open about that score being wrong already. Whether the score should count both hands is
    # #93's decision to make, not a side effect of making the left hand visible.
    #
    # Visible it must be: the robot is two-armed and operators use both hands. Recording one arm
    # is why a bulb carried in the left hand read as a bulb nobody was holding, and why every
    # old-bulb contact in the 2026-08-21 bags reads 0.0 N -- not "no contact", but "not measured".
    left_hand_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(left_hand_.*|left_wrist_.*|L_.*)",
        filter_prim_paths_expr=[],
        history_length=1,
        track_air_time=False,
    )

    # ------------------------------------------------------------------ randomized dressing
    # Each cloned env spawns one randomly chosen fixture from FIXTURE_USDS. AssetBaseCfg keeps
    # it out of physics entirely and collisions are disabled. Heterogeneous per-env assets
    # require ``replicate_physics=False`` (see ``enable_dressing_randomization``).
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


def _sync_bulb_contact_filters(scene: G1ReplaceSceneCfg) -> None:
    """Rebuild ``hand_contact``'s filter list from whichever bulbs the preset actually built.

    Call this at the END of every preset. Rebuilding, rather than appending per preset, is what
    makes preset chaining safe: ``apply_remove_preset`` runs the tabletop preset first, which
    builds a ``fresh_bulb``, and then clears it. An appended filter would leave an expression
    pointing at a prim that no longer exists.

    This fails SILENTLY when it is wrong, which is why it is centralized: the sensor reports zero
    force for an unfiltered body, and zero force reads as "not touching".
    """
    paths = [bulb.prim_path for bulb in (scene.fresh_bulb, scene.old_bulb) if bulb is not None]
    scene.hand_contact.filter_prim_paths_expr = paths
    # The left hand filters the same bulbs. Its sensor exists to record, not to score (#89).
    if scene.left_hand_contact is not None:
        scene.left_hand_contact.filter_prim_paths_expr = list(paths)


def apply_workshop_preset(scene: G1ReplaceSceneCfg) -> None:
    """The default floor layout: climb ladder + socket-lamp and bulb on the floor.

    This was a documented no-op while the scene class carried a ``bulb`` field whose default
    WAS this layout. The class no longer does (issue #76 Step 1), so the layout is built here.
    """
    scene.fresh_bulb = _make_bulb_cfg("{ENV_REGEX_NS}/Bulb", BULB_POSITION)
    scene.old_bulb = None
    _sync_bulb_contact_filters(scene)


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
        # already a static collider -- immovable, and cheaper. Note rigid_props here would be
        # a no-op (modify_rigid_body_properties returns False without a RigidBodyAPI).
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_USD, func=_spawn_collidable_bench),
    )
    scene.robot.init_state.pos = TABLETOP_ROBOT_POSITION
    scene.robot.init_state.rot = _quat_z_deg(TABLETOP_ROBOT_YAW_DEG)
    scene.socket.init_state.pos = TABLETOP_SOCKET_POSITION
    scene.fresh_bulb = _make_bulb_cfg("{ENV_REGEX_NS}/Bulb", TABLETOP_BULB_POSITION)
    scene.old_bulb = None
    _sync_bulb_contact_filters(scene)


def apply_position_preset(scene: G1ReplaceSceneCfg) -> None:
    """Ladder-positioning start (FIATLUX-Carry-v0): a DYNAMIC, high-friction, graspable ladder
    standing upright out in front of the robot; the robot grasps a rail and carries it to
    TARGET_LADDER_POSITION, directly beneath the ceiling light fixture. Scored by reusing the
    Replace task's ladder terms (see carry_env_cfg).
    """
    scene.ladder.spawn = sim_utils.UsdFileCfg(
        usd_path=STEP_LADDER_RIGID_USD,
        # The `_collision_rigid` variant is already a single dynamic RigidBodyAPI + MassAPI, so
        # the spawner only tunes solver/mass props and binds a high-friction grip material
        # (UsdFileCfg has no physics_material field). cm-authored -> scale 0.01.
        func=_spawn_usd_as_rigid_body_frictional,
        scale=(0.01, 0.01, 0.01),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=1.0,
            sleep_threshold=SCORED_BODY_SLEEP_THRESHOLD,
            stabilization_threshold=0.001,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=LADDER_MASS_KG),
    )
    scene.ladder.init_state.pos = POSITION_LADDER_START_POS
    scene.ladder.init_state.rot = _quat_z_deg(POSITION_LADDER_START_YAW)
    scene.robot.init_state.pos = POSITION_ROBOT_POSITION
    scene.fixture = None
    # Same socket + bulb as the bench tasks, ceiling-mounted above the target, bulb-down. The
    # bulb stays kinematic: this task scores the ladder pose, so it needs no retention. Seated
    # is the fixture's own pose, both halves being authored assembled at identity.
    fixture_pos = (TARGET_LADDER_POSITION[0], TARGET_LADDER_POSITION[1], CEILING_FIXTURE_Z)
    fixture_quat = _quat_y_deg(180.0)
    scene.socket.init_state.pos = fixture_pos
    scene.socket.init_state.rot = fixture_quat
    # Seated in the socket, so it is the OLD bulb by placement. Kinematic: this task scores the
    # ladder pose, so the bulb is overhead context, not the manipuland, and needs no retention.
    scene.fresh_bulb = None
    scene.old_bulb = _make_bulb_cfg("{ENV_REGEX_NS}/OldBulb", fixture_pos, fixture_quat, kinematic=True)
    add_ceiling_pendant(scene, fixture_pos[0], fixture_pos[1], fixture_pos[2])
    _sync_bulb_contact_filters(scene)
    # hand_contact stays for the net-force obs + compliance penalty; no ladder force-matrix
    # filter.


def apply_at_height_preset(scene: G1ReplaceSceneCfg, robot_at: str = "base") -> None:
    """Elevated-fixture layout shared by climb / descend.

    The socket entity becomes a ceiling chandelier hung above the ladder; the floor lamp
    is gone. ``robot_at="base"`` starts the robot at the ladder's feet (climb),
    ``robot_at="top"`` starts it at the upper steps (descend). The random dressing
    ``fixture`` is dropped -- the task chandelier owns the ceiling.
    """
    scene.socket.spawn.usd_path = ELEVATED_SOCKET_USD
    scene.socket.init_state.pos = ELEVATED_SOCKET_POSITION
    scene.fresh_bulb = _make_bulb_cfg("{ENV_REGEX_NS}/Bulb", PARKED_BULB_POSITION)
    scene.old_bulb = None
    scene.robot.init_state.pos = CLIMB_ROBOT_POSITION if robot_at == "base" else TOP_ROBOT_POSITION
    scene.fixture = None
    add_ceiling_pendant(scene, ELEVATED_SOCKET_POSITION[0], ELEVATED_SOCKET_POSITION[1], ELEVATED_SOCKET_POSITION[2])
    _sync_bulb_contact_filters(scene)


def face_robot_at(scene: G1ReplaceSceneCfg, target: Vec2) -> None:
    """Point the robot's spawn yaw at a floor target (x, y).

    ``apply_replace_preset`` aims the robot at the table, which is the full task's first target. A
    subtask whose first target is something else must re-aim, or its own facing gate starts
    unsatisfiable and the target starts outside the ego camera's frustum.
    """
    x, y = scene.robot.init_state.pos[0], scene.robot.init_state.pos[1]
    scene.robot.init_state.rot = _quat_z_deg(math.degrees(math.atan2(target[1] - y, target[0] - x)))


def stand_robot_near(scene: G1ReplaceSceneCfg, target: Vec2, standoff: float) -> None:
    """Pull the robot's sampled spawn in to within ``standoff`` of ``target`` and face it.

    ``apply_replace_preset`` draws the robot's spawn from its own floor zone, independent of
    every other occupant -- correct for the leaves whose whole job is covering that gap (S01,
    S05, S07, S09), but wrong for a leaf that starts a grasp or a release (S06, S08): those are
    written assuming a predecessor already carried the robot into range, an
    assumption only the full chained curriculum enforces. Built standalone, as every one of these
    envs is for training, eval, and this render, the robot's own random zone can land metres from
    the object it is meant to already be holding or reaching for. No-ops if the sampled spawn is
    already within ``standoff``, so a draw that happens to land close is left alone.
    """
    rx, ry, rz = scene.robot.init_state.pos
    tx, ty = target
    dx, dy = tx - rx, ty - ry
    dist = math.hypot(dx, dy)
    if dist > standoff:
        ux, uy = dx / dist, dy / dist
        scene.robot.init_state.pos = (tx - ux * standoff, ty - uy * standoff, rz)
    face_robot_at(scene, target)


def stand_robot_at_offset(scene: G1ReplaceSceneCfg, target: Vec2, offset: Vec2) -> None:
    """Place the robot at ``target + offset`` and face it, preserving a validated APPROACH
    DIRECTION rather than just a distance.

    ``stand_robot_near`` pulls the robot to within a radius of the target from whatever
    direction it happens to already be on -- fine for a target with clearance on every side
    (a free-standing ladder), wrong for one that doesn't: pulled toward ``TABLETOP_BULB_
    POSITION`` by radius alone, a robot spawned behind the table can end up standing UNDER it,
    clipped into the tabletop collision mesh, with the bulb nowhere near either hand (S08,
    caught by inspecting the actual render, not assumed fixed by the radius change alone).

    ``offset`` should be a validated, authored relative position -- e.g. ``TABLETOP_ROBOT_
    POSITION[:2]`` minus ``TABLETOP_BULB_POSITION[:2]``, the same approach vector the
    non-subtask tabletop tasks already use -- not derived here, so this function cannot repeat
    the same mistake by construction.
    """
    tx, ty = target
    ox, oy = offset
    rz = scene.robot.init_state.pos[2]
    scene.robot.init_state.pos = (tx + ox, ty + oy, rz)
    face_robot_at(scene, target)


_ROOM_FLOOR_CENTER = (
    (ROOM_FLOOR_MIN[0] + ROOM_FLOOR_MAX[0]) / 2.0,
    (ROOM_FLOOR_MIN[1] + ROOM_FLOOR_MAX[1]) / 2.0,
)
_ROOM_VIEWER_MARGIN = 0.3  # m, keeps the eye off the wall plane itself


def frame_viewer_on(
    viewer: ViewerCfg, target: tuple[float, float, float], distance: float = 3.5, height: float = 2.5
) -> None:
    """Point the debug/recording viewer at ``target`` instead of a fixed world coordinate.

    The Replace layout randomizes every subtask's floor zones and, for the fixture-coupled
    ladder/mate/balance tier, the fixture's wall or ceiling mount too -- a camera hardcoded at
    the room's origin only happens to frame the action when a draw lands nearby. ``target`` must
    be a position already resolved by the preset (the robot's or ladder's ``init_state.pos``),
    so this has to run after ``apply_replace_preset``/``apply_at_height_preset``, not before.

    The eye sits ``distance`` out from ``target`` back towards the room's floor center, not at a
    fixed azimuth: a wall-mounted fixture can land within a couple of metres of the room's own
    wall, and a fixed-angle offset (the first version of this fix) walks the eye straight through
    that wall into the empty exterior -- every render came back a flat, featureless grey. Framing
    from the center side keeps the eye on the room's interior for any wall or ceiling mount, and
    the room-bounds clamp is what actually guarantees it for the pathological cases (target
    already near the center, distance overshooting the opposite wall).
    """
    tx, ty, tz = target
    dx, dy = _ROOM_FLOOR_CENTER[0] - tx, _ROOM_FLOOR_CENTER[1] - ty
    norm = math.hypot(dx, dy)
    ux, uy = (dx / norm, dy / norm) if norm > 1e-6 else (0.7071067811865476, 0.7071067811865476)
    ex = min(max(tx + ux * distance, ROOM_FLOOR_MIN[0] + _ROOM_VIEWER_MARGIN), ROOM_FLOOR_MAX[0] - _ROOM_VIEWER_MARGIN)
    ey = min(max(ty + uy * distance, ROOM_FLOOR_MIN[1] + _ROOM_VIEWER_MARGIN), ROOM_FLOOR_MAX[1] - _ROOM_VIEWER_MARGIN)
    viewer.eye = (ex, ey, tz + height)
    viewer.lookat = (tx, ty, tz + 1.0)  # chest height, not the target's (often floor-level) z


def frame_viewer_between(
    viewer: ViewerCfg, a: tuple[float, float, float], b: tuple[float, float, float], height: float = 2.7
) -> None:
    """:func:`frame_viewer_on` for a subtask defined by two points (a start and a destination).

    Framing on ``a`` alone (e.g. the robot) leaves the destination that gives the clip its
    meaning out of frame whenever the two are more than a couple of metres apart -- which they
    usually are, since covering that gap is the subtask. Frames the midpoint instead, with the
    eye pulled back enough to fit both points regardless of how far apart the random layout put
    them.
    """
    mx, my, mz = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, min(a[2], b[2])
    separation = math.hypot(a[0] - b[0], a[1] - b[1])
    frame_viewer_on(viewer, (mx, my, mz), distance=max(separation / 2.0 + 2.0, 3.5), height=height)


def park_old_bulb_in_crate(scene: G1ReplaceSceneCfg) -> None:
    """Move the old bulb from the fixture to the bottom of the disposal crate.

    ``apply_replace_preset`` starts the old bulb seated, which is the state the first half of the
    chain works from. Every subtask downstream of the disposal starts with it already thrown away
    and the fixture empty; both facts come from this one move.
    """
    bin_pos = scene.bin.init_state.pos
    scene.old_bulb.init_state.pos = (bin_pos[0], bin_pos[1], BIN_BULB_INTERIOR_Z - BULB_STAND_Z_OFFSET)
    scene.old_bulb.init_state.rot = (1.0, 0.0, 0.0, 0.0)


def seat_bulb_in_fixture(scene: G1ReplaceSceneCfg) -> None:
    """Start the fresh bulb seated in the fixture, whatever its mount orientation.

    Seated is the fixture's own pose: both halves are authored assembled at identity
    (``SOCKET_SEAT_OFFSET == BULB_PLUG_OFFSET``), so there is no offset arithmetic per mount kind.
    """
    scene.fresh_bulb.init_state.pos = scene.socket.init_state.pos
    scene.fresh_bulb.init_state.rot = scene.socket.init_state.rot


def add_ladder_contact_sensor(scene: G1ReplaceSceneCfg, bodies: list[str] | None = None) -> None:
    """Feet + palms filtered against the kinematic ladder (climb / descend tasks).

    Not a class field: presets without a ``Ladder`` prim (tabletop) could not resolve
    the filter expression. One multi-body sensor suffices — per-body ``force_matrix_w``
    against a *single* filter body works in this stack (proven by
    ``verify_interactions.py``'s whole-robot ``limb_ladder_contact`` sensor), so the
    per-link-sensor workaround from the upstream ContactSensor docstring is not needed.

    Bodies come from ``G1_LADDER_CONTACT_BODIES``, not a regex: the two hands disagree on the palm
    body's name, so any pattern spelling one variant's resolves to feet only on the other.

    ``bodies`` narrows that set. ``mdp.ladder_contact_fraction`` divides by the sensor's body
    count, so a hand that is occupied for the whole episode caps the term below 1.0 and charges
    the policy for holding its payload; a subtask whose hands are busy passes its feet alone.
    """
    scene.ladder_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(bodies or G1_LADDER_CONTACT_BODIES) + ")",
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

    The room's ceiling underside is ROOM_CEILING_Z (4.179 m), which is well out
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
    this task solvable. It must stay dynamic: a kinematic bulb cannot be moved by any
    action at all.
    """
    apply_tabletop_preset(scene)
    _add_parts_bin(scene)
    # The bench preset builds a fresh bulb at hand height. Remove's single bulb is SEATED, so it
    # is the old bulb, and the inherited fresh one has to go -- two bulbs would make
    # ``socket_empty`` read the wrong row.
    scene.fresh_bulb = None
    scene.old_bulb = _make_bulb_cfg("{ENV_REGEX_NS}/OldBulb", TABLETOP_SEATED_BULB_POSITION)
    _sync_bulb_contact_filters(scene)


def apply_install_preset(scene: G1ReplaceSceneCfg) -> None:
    """Bulb-installation start: Insert's bench, empty lamp socket, fresh bulb in the crate."""
    apply_tabletop_preset(scene)
    _add_parts_bin(scene)
    # Keeps the bench's fresh bulb; only its placement changes. The socket starts empty.
    scene.fresh_bulb.init_state.pos = BIN_BULB_POSITION
    _sync_bulb_contact_filters(scene)


##
# Replace preset (issue #20): randomized full-scene layout.
##

# Wall lookup for the fixture's random mount: name -> (fixed axis index (0=x, 1=y), the wall's
# INNER FACE on that axis, its inward-facing unit normal, yaw so local +X faces inward, and the
# span of real wall panel along the other axis).
#
# Do NOT substitute ROOM_FLOOR_MIN/MAX: those are inset FLOOR-placement bounds, 41 cm short of
# the side walls and 31 cm short of the back wall, and a fixture mounted on them hangs in
# mid-air.
#
# No "north" entry: Simple_Room HAS NO +y WALL -- the only geometry there is two corner blocks
# at |x| > 3.723, open between them.
_WALLS: dict[str, tuple[int, float, tuple[float, float], float, tuple[float, float]]] = {
    "west": (0, -4.410, (1.0, 0.0), 0.0, (-2.622, 4.078)),
    "east": (0, 4.410, (-1.0, 0.0), 180.0, (-2.622, 4.078)),
    "south": (1, -3.309, (0.0, 1.0), 90.0, (-3.723, 3.723)),
}


class LayoutInfeasible(RuntimeError):
    """No legal arrangement was found for a sampled layout. Retry with a fresh draw."""


def _clamp_to_floor(center: Vec2, half_size: float) -> Vec2:
    """Pull a zone center inside the floor bounds so its whole square fits.

    Only ever moves a zone FURTHER from the wall, which is always physically legal. Needed
    because the wall planes are the real inner faces while the zone bounds are the
    inset floor box: a standoff taken off the true wall can land a hair outside the floor
    box, which would otherwise reject every side-wall mount as infeasible.
    """
    return (
        min(max(center[0], ROOM_FLOOR_MIN[0] + half_size), ROOM_FLOOR_MAX[0] - half_size),
        min(max(center[1], ROOM_FLOOR_MIN[1] + half_size), ROOM_FLOOR_MAX[1] - half_size),
    )


def _bearing_into_room(origin: Vec2, bearing: Vec2, standoff: float) -> Vec2:
    """Flip whichever component of ``bearing`` would push ``origin + bearing * standoff`` out of
    the floor box, so the offset point stays inside it.

    Reflected rather than clamped: clamping shortens the standoff, which is the failure this
    offset prevents. At most one axis can violate, since the fixture is inset from both bounds.
    """
    out = [bearing[0], bearing[1]]
    for axis in (0, 1):
        target = origin[axis] + out[axis] * standoff
        if not ROOM_FLOOR_MIN[axis] <= target <= ROOM_FLOOR_MAX[axis]:
            out[axis] = -out[axis]
    return (out[0], out[1])


# OS entropy (None), except under FIATLUX_DEBUG_NO_RANDOMIZE, where every layout draw is fixed
# to seed 0 unless a caller overrides via set_layout_seed().
_layout_seed: int | None = 0 if os.environ.get("FIATLUX_DEBUG_NO_RANDOMIZE") else None


def set_layout_seed(seed: int | None) -> None:
    """Fix the entropy for every subsequent Replace layout draw.

    The layout sets scene-entity ``init_state``s, so it has to be drawn in ``__post_init__``
    -- which runs inside ``parse_env_cfg``, before the caller can assign ``cfg.seed`` and
    before Isaac Lab seeds anything. So the seed cannot come from the cfg: entry points must
    declare it here, BEFORE building the cfg, or the layout is drawn from OS entropy and the
    run is not reproducible.

    Deliberately not Python's global ``random`` stream, which this used to lean on: that
    couples the layout to every other consumer of ``random`` in the process, so any library
    reseeding it silently changes the scene.

    Args:
        seed: the layout's seed, or None to draw from OS entropy (an explicitly
            irreproducible layout).
    """
    global _layout_seed
    _layout_seed = seed


def _sample_fixture_mount(rng: random.Random) -> FixtureMount:
    """Randomly mount the fixture on the ceiling or a wall.

    Returns ``(mount_kind, position, orientation, anchor_bearing, ladder_anchor)``. The anchor is
    the floor point the ladder has to stand on for the fixture to be workable: for both mount
    kinds, the mount kind's own standoff out from the fixture along the bearing -- the wall's inward
    normal, or a sampled direction for a ceiling mount.

    The standoff applies to a ceiling mount too because ``stand_robot_on_ladder_top`` puts the
    pelvis over the ladder's root and the fixture hangs only 0.233 m above it, at chest height.
    An anchor directly beneath the fixture spawns the stance with the socket inside its torso
    (``imu_in_torso`` 0.039 m off the fixture axis, measured), and the robot hangs on the
    fixture's kinematic collider instead of falling.

    The mount is sampled so that the anchor's whole ladder zone fits inside the room, which
    is why the inset is a ladder zone rather than a decorative half-metre -- a fixture in the
    corner of the room has no floor under it to put a ladder on.

    ``ehjsdz`` is authored as an upright desk lamp (socket opening up, base on a horizontal
    surface), so each mount kind needs a reorienting rotation: ceiling flips it ~180 deg so
    the shade/socket point down like a pendant light; wall rotates it ~90 deg so it projects
    outward from the wall face.
    """
    inset = LADDER_ANCHOR_HALF_SIZE
    standoff = LADDER_FIXTURE_STANDOFF_CEILING
    if rng.random() < 0.5:
        x = rng.uniform(ROOM_FLOOR_MIN[0] + inset, ROOM_FLOOR_MAX[0] - inset)
        y = rng.uniform(ROOM_FLOOR_MIN[1] + inset, ROOM_FLOOR_MAX[1] - inset)
        # A wall dictates which side the ladder stands on; a ceiling does not, so the bearing is
        # drawn -- uniform over the circle, so the stance approaches from every direction.
        theta = rng.uniform(0.0, 2.0 * math.pi)
        normal = _bearing_into_room((x, y), (math.cos(theta), math.sin(theta)), standoff)
        anchor = _clamp_to_floor((x + normal[0] * standoff, y + normal[1] * standoff), LADDER_ANCHOR_HALF_SIZE)
        return "ceiling", (x, y, CEILING_FIXTURE_Z), _quat_y_deg(180.0), normal, anchor

    standoff = LADDER_FIXTURE_STANDOFF_WALL
    wall_name = rng.choice(list(_WALLS))
    axis, value, normal, yaw, (span_lo, span_hi) = _WALLS[wall_name]
    # Sample along the REAL panel span, inset so the anchor's ladder zone stays in the room.
    along = rng.uniform(span_lo + inset, span_hi - inset)
    pos = (value, along, WALL_MOUNT_Z) if axis == 0 else (along, value, WALL_MOUNT_Z)
    anchor = _clamp_to_floor((pos[0] + normal[0] * standoff, pos[1] + normal[1] * standoff), LADDER_ANCHOR_HALF_SIZE)
    # +90, not -90: local +Z (the shade/socket opening) must map to local +X so the per-wall
    # yaw points the shade into the room. -90 gives dot(inward_normal) = -1.0.
    quat = _quat_mul(_quat_z_deg(yaw), _quat_y_deg(90.0))
    return "wall", pos, quat, normal, anchor


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
    sampling it (the ladder in ``couple_ladder_to_fixture`` mode, and always the fixture's
    reserved ladder footprint) -- other zones are still sampled to avoid it. Runs once at
    cfg-build time (plain Python, no torch).

    Free zones are placed largest-first (a fixed zone still goes in first regardless of
    size): stress-tested at 5000 random layouts against this room/these zone sizes with
    ~0.1% placement failures, vs. ~1.3% with left-to-right order (small zones sampled first
    can strand a later, larger one with nowhere left to fit).

    Raises:
        LayoutInfeasible: if any zone cannot be placed, or a pinned zone does not fit inside
            the bounds. Never accept an overlapping layout: it renders and scores like any
            other, so the corruption is invisible. The caller resamples and retries.
    """
    n = len(half_sizes)
    centers: list[tuple[float, float] | None] = list(fixed) if fixed else [None] * n
    order = sorted((i for i in range(n) if centers[i] is None), key=lambda i: -half_sizes[i])

    for i, pinned in enumerate(centers):
        hs = half_sizes[i]
        if pinned is not None and not (
            bounds_min[0] + hs <= pinned[0] <= bounds_max[0] - hs
            and bounds_min[1] + hs <= pinned[1] <= bounds_max[1] - hs
        ):
            raise LayoutInfeasible(f"pinned zone {i} at {pinned} (half-size {hs}) does not fit in the room")

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
        for attempt in range(max_tries):
            candidate = (rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y))
            if not overlaps(i, candidate):
                centers[i] = candidate
                break
        else:
            raise LayoutInfeasible(f"no free placement for zone {i} (half-size {hs}) in {max_tries} tries")
    return cast(list[tuple[float, float]], centers)  # every slot filled: fixed, or by the loop


def _sample_replace_layout(
    rng: random.Random,
    couple_ladder_to_fixture: bool,
    max_tries: int = 64,
) -> tuple[FixtureMount, list[tuple[float, float]], float]:
    """Draw a *feasible* Replace layout: fixture mount plus the four floor zone centers.

    Fixture first, because it constrains the floor rather than the other way round: its
    ladder anchor is reserved as a zone no other occupant may take, so wherever the fixture
    lands there is somewhere legal to stand the ladder. Sampled independently, a ceiling
    fixture can land over the table and leave the task unsolvable while rendering normally.

    Returns ``(mount, [robot, table, ladder, disposal] centers, ladder_yaw_deg)``.

    Samples the SAME five zones in the SAME order regardless of ``couple_ladder_to_fixture``
    -- robot, table, a free ladder draw, disposal, and the fixture's anchor, always reserved
    as a fifth occupant-less zone so nothing else can land on it whichever way the ladder
    itself is resolved. Only the ladder's returned center depends on the flag (the anchor
    when coupled, its own free draw otherwise); everything else consumes the rng identically
    either way. Earlier this branched on the flag by changing which/how-many zones got
    sampled at all, which shifted every rng draw AFTER the branch -- robot, disposal, even the
    fixture's own later-drawn attributes would silently differ between a coupled and an
    uncoupled call at the same seed, so two subtasks that are supposed to share one stable
    room (issue #70 discussion) did not (caught by comparing ``bin`` position across subtasks
    at a fixed seed: identical for every uncoupled leaf, a completely different position for
    every coupled one).

    Raises:
        RuntimeError: if no feasible layout is found. Every constant involved is fixed at
            import time, so this is a statement about the room's geometry, not bad luck --
            it means the zones no longer fit and one of them has to shrink.
    """
    last: LayoutInfeasible | None = None
    for _ in range(max_tries):
        mount = _sample_fixture_mount(rng)
        _, _, _, anchor_bearing, ladder_anchor = mount
        ladder_yaw = rng.uniform(0.0, 360.0)
        if couple_ladder_to_fixture:
            # Turn the ladder back down its own standoff bearing, so the stance on its tread
            # faces the fixture. Both mount kinds, since both stand off.
            # Less the stance's own quarter turn, so it is the ROBOT that ends up pointing down
            # the bearing. Carries the step side, and so the climb approach, round with it.
            ladder_yaw = math.degrees(math.atan2(-anchor_bearing[1], -anchor_bearing[0])) - TOP_STANCE_YAW_OFFSET_DEG

        half_sizes = [
            ROBOT_ZONE_HALF_SIZE,
            TABLE_ZONE_HALF_SIZE,
            LADDER_ZONE_HALF_SIZE,
            DISPOSAL_ZONE_HALF_SIZE,
            LADDER_ANCHOR_HALF_SIZE,
        ]
        fixed: list[tuple[float, float] | None] = [None, None, None, None, ladder_anchor]

        try:
            centers = _sample_nonoverlapping_centers(
                rng,
                half_sizes=half_sizes,
                bounds_min=ROOM_FLOOR_MIN,
                bounds_max=ROOM_FLOOR_MAX,
                fixed=fixed,
            )
        except LayoutInfeasible as exc:
            last = exc
            continue
        ladder_center = ladder_anchor if couple_ladder_to_fixture else centers[2]
        return mount, [centers[0], centers[1], ladder_center, centers[3]], ladder_yaw
    raise RuntimeError(f"no feasible Replace layout in {max_tries} draws; last failure: {last}")


def apply_replace_preset(
    scene: G1ReplaceSceneCfg,
    rng: random.Random | None = None,
    couple_ladder_to_fixture: bool = False,
) -> None:
    """The full replacement-task layout: robot, ladder, table+fresh bulb, disposal crate, and
    the elevated socket/lamp ("fixture") with the OLD BULB seated in it -- each floor occupant
    randomized into its own non-overlapping "safe zone", and the fixture randomly ceiling- or
    wall-mounted. Randomized once per scene build (this function's own ``rng`` draw), not
    re-sampled every episode reset -- so one layout serves every env and every episode of a
    run, and varying it is a between-runs affair. ``rng`` defaults to :func:`set_layout_seed`'s
    declared seed; without one the layout is drawn from OS entropy.

    The fixture reuses ``SOCKET_USD`` (``ehjsdz``/``kfmkwd``, the validated bulblampF/M pair
    already used by the tabletop Insert task) rather than the decorative ``ELEVATED_SOCKET_USD``
    chandelier (used only by climb/descend, which never validated a socket metalink on it) --
    this scene needs a genuinely insertible bulb+socket at height.

    Unlike every other preset the ladder spawns *dynamic* (mass ``LADDER_MASS_KG``): a
    knocked-over ladder is a real, penalized event in this task. The old bulb starts seated
    and DYNAMIC at the fixture's own pose (both halves are authored assembled at identity,
    so no offset arithmetic is needed at any mount orientation). Because the fixture is
    inverted here, the bulb is held seated by ``mdp.bulb_attachment``'s axial retention spring
    (issue #167) until it is pulled far enough to release, same as any other mount orientation.

    Args:
        couple_ladder_to_fixture: place the ladder's zone reachably relative to wherever the
            fixture mounted (beneath a ceiling point, or standing off from a mounted wall)
            instead of sampling it fully independently. Off by default -- positioning the
            ladder is part of the task; coupling is a debug/curriculum aid only.
    """
    rng = rng or random.Random(_layout_seed if _layout_seed is not None else random.getrandbits(64))

    # Table: holds the fresh bulb. The elevated fixture is the insertion target.
    scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_USD, func=_spawn_collidable_bench),  # see apply_tabletop_preset
        init_state=AssetBaseCfg.InitialStateCfg(),
    )
    scene.fixture = None  # the task fixture owns the ceiling/wall in this scene

    # Fixture and floor zones are drawn together: the fixture's ladder anchor is a reserved
    # zone, so the layout is feasible by construction.
    mount, centers, ladder_yaw = _sample_replace_layout(rng, couple_ladder_to_fixture)
    mount_kind, fixture_pos, fixture_quat, _, _ = mount
    robot_center, table_center, ladder_center, disposal_center = centers
    scene.socket.init_state.pos = fixture_pos
    scene.socket.init_state.rot = fixture_quat
    if mount_kind == "ceiling":
        add_ceiling_pendant(scene, fixture_pos[0], fixture_pos[1], fixture_pos[2])

    scene.robot.init_state.pos = (robot_center[0], robot_center[1], ROBOT_POSITION[2])
    # Face the table, +/- a small jitter: the ego camera's 50 deg frustum must contain the work
    # area or the standard observation mode cannot see the task.
    facing = math.degrees(math.atan2(table_center[1] - robot_center[1], table_center[0] - robot_center[0]))
    scene.robot.init_state.rot = _quat_z_deg(facing + rng.uniform(-15.0, 15.0))
    scene.table.init_state.pos = (table_center[0], table_center[1], TABLE_POSITION[2])
    # Replace builds BOTH bulbs, and it chains through no preset that would build one for it --
    # it used to inherit the class default. The fresh bulb rides the table at the bench offset.
    bulb_local_offset = tuple(b - t for b, t in zip(TABLETOP_BULB_POSITION, TABLE_POSITION))
    scene.fresh_bulb = _make_bulb_cfg(
        "{ENV_REGEX_NS}/Bulb",
        (
            table_center[0] + bulb_local_offset[0],
            table_center[1] + bulb_local_offset[1],
            TABLE_POSITION[2] + bulb_local_offset[2],
        ),
    )
    scene.ladder.init_state.pos = (ladder_center[0], ladder_center[1], LADDER_POSITION[2])
    scene.ladder.init_state.rot = _quat_z_deg(ladder_yaw)
    # Dynamic ladder (this preset only): tipping/falling is a scored physical event.
    #
    # Spawn the ``_collision_rigid`` variant, the same one ``apply_position_preset`` uses, rather
    # than promoting the static ``_collision`` one at spawn. Only the rigid overlay carries the
    # authored ``centerOfMass`` / ``diagonalInertia`` / ``principalAxes``; referencing the static
    # file instead leaves PhysX deriving them from collider VOLUME, which models this ladder as a
    # solid block and counts the collision-only PlatformCollider box as material. Both dynamic
    # presets now agree on the asset, and the frictional spawner binds the grip material this
    # leg needs to hold a rail.
    scene.ladder.spawn.usd_path = STEP_LADDER_RIGID_USD
    scene.ladder.spawn.func = _spawn_usd_as_rigid_body_frictional
    scene.ladder.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        solver_position_iteration_count=16,
        # 1, not 8, and 1 is what every other dynamic prop in the family uses. Under TGS the
        # position loop already does most of the velocity correction, so the velocity pass is a
        # small correction on top; running many of them over-corrects and can add energy rather
        # than remove it. PhysX says so directly at spawn -- "Detected a rigid at .../Ladder with
        # more than 4 velocity iterations being added to a TGS scene" -- and this ladder was the
        # only body in the scene raising it.
        solver_velocity_iteration_count=1,
        max_depenetration_velocity=1.0,
        sleep_threshold=SCORED_BODY_SLEEP_THRESHOLD,
        stabilization_threshold=0.001,
    )
    scene.ladder.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=LADDER_MASS_KG)

    # Old bulb, seated: the fixture's own pose, both halves being authored assembled at
    # identity. Dynamic, and the fixture is inverted, so it is held by the seat constraint
    # (mdp/attach.py), not by gravity. Replace is the only preset that builds BOTH bulbs.
    scene.old_bulb = _make_bulb_cfg("{ENV_REGEX_NS}/OldBulb", fixture_pos, fixture_quat)

    # Disposal crate: the old bulb's destination, in its own sampled zone.
    _add_parts_bin(scene, position=(disposal_center[0], disposal_center[1], BIN_POSITION[2]))
    # Both bulbs are manipulated, so the hand-contact channel counts both.
    _sync_bulb_contact_filters(scene)
