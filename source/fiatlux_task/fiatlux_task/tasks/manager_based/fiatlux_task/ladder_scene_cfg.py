# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The single source of truth for the Fiatlux ladder-family scene.

This ``InteractiveSceneCfg`` subclass assembles every element as a **named scene entity** so
each is addressable via ``SceneEntityCfg`` for later randomization: ``ground``, ``dome_light``,
``key_light``, ``room``, ``robot``, ``ladder``, ``lamp``, ``bulb``, ``fixture``.

All assets come from the ``gs://fiatlux`` bucket (see ``fiatlux_task.assets`` and
``assets/download_assets.sh``): the same Inspire-hand G1 and BEHAVIOR-1K bulb/lamp the
insertion task uses, plus the primary BEHAVIOR-1K climb ladder (``shfvtl``).

The scene carries the same room dressing as the insertion task -- the Simple Room backdrop
and the PolyHaven HDRI sky -- plus one *randomly chosen* BEHAVIOR-1K ceiling fixture per env
(``fixture``; needs the opt-in ``download_assets.sh --scene-dressing`` asset group, and is
dropped automatically when those assets are absent).

Note: ``InteractiveSceneCfg`` treats *every* dataclass field as a scene entity, so non-entity
tunables cannot live here as fields. Asset *placement* defaults are the module constants below
and are baked onto each entity's ``init_state`` (still overridable per-instance, e.g.
``scene.ladder.init_state.pos = ...``).
"""

import glob
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from fiatlux_task.assets import BULB_USD, FIATLUX_ASSETS_DIR, LADDER_USD, SOCKET_USD
from fiatlux_task.robots.g1 import G1_INSPIRE_CFG
from fiatlux_task.scenes import DressedSceneCfg, spawn_b1k_single_body

# -- default placement (module constants, not scene fields; override via each entity's
#    init_state). BEHAVIOR-1K USDs are authored with the origin near the bbox center, so a
#    prop resting on the floor sits at roughly half its height. Tuned against
#    scripts/verify_scene.py --record orbit videos; adjust the same way. --
ROBOT_POSITION = (0.0, 0.0, 0.75)  # G1_INSPIRE_CFG's standing pelvis height
# x=1.5 puts the A-frame's near face ~arm's length in front of the robot (x=1.0 stood the
# robot inside the frame's footprint); z = bbox_z/2 rests the feet on the floor.
LADDER_POSITION = (1.5, 0.0, 0.85)
LAMP_POSITION = (-0.8, 0.0, 0.20)  # bbentu table lamp resting on the floor
BULB_POSITION = (-0.55, -0.20, 0.05)  # loose on the floor next to the lamp
FIXTURE_POSITION = (0.0, 0.0, 2.45)  # hangs overhead in the record camera's frame, clear of robot/ladder

# -- per-env random ceiling fixture pool (visual dressing) --
# Ceiling-mount BEHAVIOR-1K categories only: floor-standing fixtures would invade the task
# space. These are the opt-in ``download_assets.sh --scene-dressing`` asset group; when they
# are absent the pool is empty and ``G1LadderEnvCfg.__post_init__`` drops the ``fixture``
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


@configclass
class G1LadderSceneCfg(DressedSceneCfg):
    """G1 + BEHAVIOR-1K ladder/lamp/bulb in the shared dressed room (see ``DressedSceneCfg``)."""

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
            func=spawn_b1k_single_body,
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
            func=spawn_b1k_single_body,
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

    # ------------------------------------------------------------------ randomized dressing
    # Per-env random ceiling fixture: each cloned env spawns one randomly chosen BEHAVIOR-1K
    # ceiling-mount fixture from FIXTURE_USDS. AssetBaseCfg (not RigidObjectCfg) keeps it out
    # of physics entirely -- no meta__ single-body stripping needed -- and collisions are
    # disabled so task physics is untouched. Heterogeneous per-env assets require
    # ``replicate_physics=False`` (set where the scene is instantiated, see the base env cfg).
    fixture: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Fixture",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=FIXTURE_USDS,
            random_choice=True,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FIXTURE_POSITION),
    )
