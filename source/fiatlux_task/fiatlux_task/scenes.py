# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared scene vocabulary for the Fiatlux benchmark scenes.

The Insert task scene (``g1_bulb_env_cfg``) and the ladder-family scene
(``ladder_scene_cfg``) are deliberately different *layouts* -- tabletop subtask vs
at-fixture task family -- but they describe the same world. This module holds the pieces
that must stay literally identical across them:

- :func:`spawn_b1k_single_body` -- the BEHAVIOR-1K multi-body workaround (issue #14),
- :class:`DressedSceneCfg` -- the common room dressing (HDRI sky dome + Simple Room).

Per-scene knobs that *differ on purpose* (ground friction, key light, sensors, task
furniture) stay in the task scene cfgs.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from .assets import ROOM_USD, SKY_HDRI

# --- BEHAVIOR-1K single-body workaround (issue #14, Option A) ------------------
# BEHAVIOR-1K objects (lamp, ladder, ...) can carry ``meta__*`` helper links (light
# source, toggle button) joined to ``base_link``; Fiatlux scenes model them as single
# kinematic rigid bodies. This spawner deactivates the ``meta__*`` prims at spawn time so
# the object resolves to just ``base_link`` -- leaving reward / observation / reset code
# untouched. Decorated with Isaac Lab's ``clone`` (like the stock ``spawn_from_usd``) so
# the source prim is stripped *before* it is replicated to the other envs, keeping every
# env single-body. Interim fix; the eventual plan is to model the lamp as an articulation
# (issue #14, Option C).


@clone
def spawn_b1k_single_body(prim_path, cfg, translation=None, orientation=None):
    from pxr import Usd

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    meta_prims = [p for p in Usd.PrimRange(prim) if p.GetName().startswith("meta__")]
    for p in meta_prims:
        p.SetActive(False)
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

    # Room backdrop: visual only, collision explicitly disabled so it never conflicts
    # with the task scene's ground collider, which keeps owning floor physics. Shared
    # static geometry under /World -- per-env rooms would overlap at typical env spacings.
    room: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Room",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROOM_USD,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
