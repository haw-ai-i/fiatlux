# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The single source of truth for the Fiatlux ladder-family scene.

This ``InteractiveSceneCfg`` subclass assembles every element as a **named scene entity** so
each is addressable via ``SceneEntityCfg`` for later randomization: ``ground``, ``dome_light``,
``key_light``, ``robot``, ``ladder``, ``lamp``, ``bulb``.

All assets come from the ``gs://fiatlux`` bucket (see ``fiatlux_task.assets`` and
``assets/download_assets.sh``): the same Inspire-hand G1 and BEHAVIOR-1K bulb/lamp the
insertion task uses, plus the primary BEHAVIOR-1K climb ladder (``shfvtl``).

Note: ``InteractiveSceneCfg`` treats *every* dataclass field as a scene entity, so non-entity
tunables cannot live here as fields. Asset *placement* defaults are the module constants below
and are baked onto each entity's ``init_state`` (still overridable per-instance, e.g.
``scene.ladder.init_state.pos = ...``).
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from fiatlux_task.assets import BULB_USD, LADDER_USD, SOCKET_USD
from fiatlux_task.robots.g1 import G1_INSPIRE_CFG

# -- default placement (module constants, not scene fields; override via each entity's
#    init_state). BEHAVIOR-1K USDs are authored with the origin near the bbox center, so a
#    prop resting on the floor sits at roughly half its height. The z values below are
#    estimates pending the first verified spawn -- tune them with scripts/verify_scene.py. --
ROBOT_POSITION = (0.0, 0.0, 0.75)  # G1_INSPIRE_CFG's standing pelvis height
LADDER_POSITION = (1.0, 0.0, 0.85)  # shfvtl bbox_z = 1.67 m -> base on the floor
LAMP_POSITION = (-0.8, 0.0, 0.20)  # bbentu table lamp resting on the floor
BULB_POSITION = (-0.55, -0.20, 0.05)  # loose on the floor next to the lamp


def _quat_y_deg(angle_deg: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) quaternion for a rotation about +Y, in degrees."""
    half = math.radians(angle_deg) / 2.0
    return (math.cos(half), 0.0, math.sin(half), 0.0)


# --- BEHAVIOR-1K single-body workaround (issue #14, Option A) -----------------
# Same interim fix the insertion task uses for its lamp (see ``g1_bulb_env_cfg``):
# BEHAVIOR-1K objects can carry ``meta__*`` helper links (light source, toggle button)
# joined to ``base_link``; deactivating them at spawn time makes the object resolve to a
# single rigid body. Decorated with ``clone`` so the source prim is stripped *before* it
# is replicated to the other envs.
@clone
def _spawn_b1k_single_body(prim_path, cfg, translation=None, orientation=None):
    from pxr import Usd

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    meta_prims = [p for p in Usd.PrimRange(prim) if p.GetName().startswith("meta__")]
    for p in meta_prims:
        p.SetActive(False)
    return prim


@configclass
class G1LadderSceneCfg(InteractiveSceneCfg):
    """G1 + BEHAVIOR-1K ladder/lamp/bulb on a lit ground plane."""

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
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=800.0, color=(0.9, 0.9, 0.95)),
    )
    key_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=1500.0, color=(1.0, 0.98, 0.95), angle=0.53),
        # tilt the distant light downward; orientation is the future "light direction" knob
        init_state=AssetBaseCfg.InitialStateCfg(rot=_quat_y_deg(40.0)),
    )

    # ------------------------------------------------------------------ robot & props
    # Same reusable Inspire-hand G1 as the insertion task (USD, standing init pose,
    # actuator groups); here we only bind it into this scene's namespace and placement.
    robot: ArticulationCfg = G1_INSPIRE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=G1_INSPIRE_CFG.init_state.replace(pos=ROBOT_POSITION),
    )
    # Tall upright BEHAVIOR-1K ladder; kinematic so it stays put while climbed.
    ladder: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ladder",
        spawn=sim_utils.UsdFileCfg(
            usd_path=LADDER_USD,
            # Strip meta__ helper links so the object resolves to one rigid body.
            func=_spawn_b1k_single_body,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=LADDER_POSITION),
    )
    # Lamp / fixture: the same BEHAVIOR-1K lamp the insertion task uses as its socket,
    # placed at ground level here (at-height mounting is the Replace task's business).
    lamp: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Lamp",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SOCKET_USD,
            func=_spawn_b1k_single_body,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=LAMP_POSITION),
    )
    # Graspable bulb (dynamic), loose on the floor next to the lamp.
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
