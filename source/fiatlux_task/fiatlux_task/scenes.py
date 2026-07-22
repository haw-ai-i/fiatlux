# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared scene vocabulary for the Fiatlux benchmark scenes.

The Insert task scene (``g1_bulb_env_cfg``) and the ladder-family scene
(``ladder_scene_cfg``) are deliberately different *layouts* -- tabletop subtask vs
at-fixture task family -- but they describe the same world. This module holds the pieces
that must stay literally identical across them:

- :class:`DressedSceneCfg` -- the common room dressing (HDRI sky dome + Simple Room).

Per-scene knobs that *differ on purpose* (ground friction, key light, sensors, task
furniture) stay in the task scene cfgs.
"""

import re

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from .assets import ROOM_USD, SKY_HDRI


@clone
def _spawn_room_backdrop(prim_path, cfg, translation=None, orientation=None):
    """The Simple Room USD as walls + floor + ceiling: its own furniture is deactivated.

    The stock asset ships a low table (``table_low_327``) whose top surface IS the
    room's world origin plane: it spans x[-1.59,1.59] x y[-0.81,0.81] with its top
    at z=+0.01, overlapping the scene's ground-plane collider at z=0. The cfg-level
    ``collision_enabled=False`` does not reach its (instanced) collider prims, so
    robots and props inside that footprint stand on a 1 cm ledge with an active
    edge in the middle of the play area (probe-verified: feet straddling the edge
    destabilize a standing G1). ``SetActive(False)`` removes render and physics at
    once; the room stays a pure backdrop.

    Also aligns the room's floor to the scene's ground-plane collider at z=0. Simple_Room is
    authored *tabletop-at-origin*: its own floor sits ~0.77 m BELOW the USD origin, so once
    the opaque debug grid stopped being drawn the room rendered its real floor three-quarters
    of a metre under every prop, and everything appeared to hover. The shift is MEASURED off
    the asset's own floor prims rather than hardcoded, so swapping the room asset cannot
    silently reintroduce the gap.
    """
    from pxr import Gf, Usd, UsdGeom

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    for p in Usd.PrimRange(prim):
        # ``GroundPlane`` is the asset's own 2x2 m collider under the low table. After the
        # floor alignment below it lands coincident with the scene's ground plane, and two
        # colliders in the same plane chatter; the scene's ground keeps owning floor physics.
        if p.GetName().startswith("table_low") or p.GetName() == "GroundPlane":
            p.SetActive(False)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    floor_tops = []
    floor_named = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if not p.IsA(UsdGeom.Gprim) or "floor" not in p.GetName().lower():
            continue
        # Skip the asset's oversized decorative/light-helper prims. They are named Floor2..
        # Floor5 but span ~10 m vertically and are not surfaces (the same exclusion the
        # ROOM_FLOOR_MIN/MAX measurement in scene_cfg documents); Floor5's top sits above
        # the ceiling, so taking a naive max moves the whole room metres out of place.
        if any(re.fullmatch(r"Floor[2-9]", a) for a in str(p.GetPath()).split("/")):
            continue
        rng = cache.ComputeWorldBound(p).ComputeAlignedRange()
        if not rng.IsEmpty():
            floor_tops.append(rng.GetMax()[2])
            floor_named.append((p.GetName().lower(), rng.GetMax()[2]))
    # The walking surface is the topmost floor-ish face in the LOWER half of the room. Do not
    # trust the prim names: this asset calls its ceiling ``Towel_Room01_floor_top`` and its
    # actual floor ``floor_bottom``, so a naive max() aligns the room to its ceiling and drops
    # it several metres. Splitting on the room's own vertical midpoint is name-independent.
    room_rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mid_z = (room_rng.GetMin()[2] + room_rng.GetMax()[2]) / 2.0 if not room_rng.IsEmpty() else 0.0
    # Prefer the asset's actual walking-surface slab when present. Its neighbours are raised
    # rim/threshold trim standing ~0.19 m proud of the floor, which a plain max() would latch
    # onto. The value is still MEASURED at spawn -- only the choice of prim is anchored.
    named = [z for name, z in floor_named if "floor_bottom" in name]
    lower = named or [z for z in floor_tops if z < mid_z]
    if lower:
        # Adjust the translate op the spawner already authored -- AddTranslateOp would raise.
        xf = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        drop = max(lower)
        t_op = ops.get("xformOp:translate")
        if t_op is None:
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -drop))
        else:
            cur = t_op.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
            t_op.Set(Gf.Vec3d(cur[0], cur[1], cur[2] - drop))
    return prim


@configclass
class DressedSceneCfg(InteractiveSceneCfg):
    """Room dressing shared by every Fiatlux scene: HDRI sky + Simple Room backdrop."""

    # HDRI sky instead of a flat color, for realistic ambient lighting. Intensity is
    # randomized per reset by the task cfgs (see ``mdp.randomize_light_properties``).
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            texture_file=SKY_HDRI,
            texture_format="latlong",
            intensity=1000.0,
        ),
    )

    # Room: walls + floor + ceiling, furniture deactivated at spawn (see
    # _spawn_room_backdrop). PER-ENV and COLLIDING, so the visible walls ARE the physical
    # walls and the workspace is actually bounded -- as a collisionless shared backdrop a
    # falling ladder or a thrown bulb left the room and kept going. Stays an AssetBaseCfg:
    # a collider with no RigidBodyAPI is a static collider, which is both correct and
    # cheaper than a kinematic body. The scene's own ground plane keeps owning floor
    # physics (the asset's GroundPlane is deactivated). Env spacing must clear the room's
    # 9.04 x 8.26 m wall box -- see ROOM_ENV_SPACING.
    room: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROOM_USD,
            func=_spawn_room_backdrop,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
