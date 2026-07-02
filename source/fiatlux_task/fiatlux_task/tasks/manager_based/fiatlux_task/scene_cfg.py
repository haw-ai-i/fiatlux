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
from isaaclab.utils import configclass

from fiatlux_task.assets import (
    BULB_USD,
    CRATE_USD,
    ELEVATED_SOCKET_USD,
    FIATLUX_ASSETS_DIR,
    LADDER_USD,
    SOCKET_USD,
    TABLE_USD,
)
from fiatlux_task.robots.g1 import G1_INSPIRE_CFG
from fiatlux_task.scenes import DressedSceneCfg, spawn_b1k_single_body

# -- default (workshop) placement (module constants, not scene fields; override via each
#    entity's init_state). BEHAVIOR-1K USDs are authored with the origin near the bbox
#    center, so a prop resting on the floor sits at roughly half its height. Tuned against
#    scripts/verify_scene.py --record orbit videos; adjust the same way. --
ROBOT_POSITION = (0.0, 0.0, 0.75)  # G1_INSPIRE_CFG's standing pelvis height
# x=1.5 puts the A-frame's near face ~arm's length in front of the robot (x=1.0 stood the
# robot inside the frame's footprint); z = bbox_z/2 rests the feet on the floor.
LADDER_POSITION = (1.5, 0.0, 0.85)
SOCKET_POSITION = (-0.8, 0.0, 0.20)  # bbentu socket-lamp resting on the floor
BULB_POSITION = (-0.55, -0.20, 0.05)  # loose on the floor next to the lamp
FIXTURE_POSITION = (0.0, 0.0, 2.45)  # hangs overhead in the record camera's frame, clear of robot/ladder

# -- tabletop (manipulation bench) placement: the Insert layout --
TABLE_POSITION = (0.40, -0.10, 0.0)  # authored tabletop surface is ~1.0 m above the origin
TABLETOP_SOCKET_POSITION = (0.45, 0.0, 1.20)
TABLETOP_BULB_POSITION = (0.35, -0.20, 1.05)

# -- carry preset: ladder *stored* by the room wall, work area across the room --
CARRY_LADDER_POSITION = (-3.2, 1.8, 0.85)  # near the Simple Room wall (~4.5 m out)
CARRY_ROBOT_POSITION = (-2.4, 1.8, 0.75)  # standing next to the stored ladder
CARRY_LADDER_YAW_DEG = 90.0  # parallel to the wall

# -- at-height presets (climb / descend / remove / install): elevated fixture over the
#    ladder. The chandelier hangs above/behind the ladder's top; positions are tuned
#    against verify_scene --record orbit videos, same as the floor layout. --
ELEVATED_SOCKET_POSITION = (1.9, 0.0, 2.80)  # cage bottom clears the at-top robot's head
CLIMB_ROBOT_POSITION = (0.9, 0.0, 0.75)  # at the ladder's base, ready to ascend
TOP_ROBOT_POSITION = (1.15, 0.0, 1.80)  # pelvis at the upper steps (descend/remove/install)
PARKED_BULB_POSITION = (0.5, -0.6, 0.05)  # out of the way on the floor
SEATED_BULB_POSITION = (1.9, 0.0, 2.55)  # hanging in the fixture's seat, visible below the cage
BIN_POSITION = (0.9, -0.55, 0.0)  # parts crate at the ladder base (install preset)
BIN_BULB_POSITION = (0.9, -0.55, 0.15)  # fresh bulb resting in the crate

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
    # Tall upright BEHAVIOR-1K ladder; kinematic so it stays put while climbed.
    # Dropped by the tabletop preset.
    ladder: RigidObjectCfg | None = RigidObjectCfg(
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
    hand_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_.*",
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

    Drops the ladder; the robot stands at the table and never locomotes.
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
    scene.socket.init_state.pos = TABLETOP_SOCKET_POSITION
    scene.bulb.init_state.pos = TABLETOP_BULB_POSITION


def apply_carry_preset(scene: G1ReplaceSceneCfg) -> None:
    """Ladder-handling start: the ladder is *stored* by the room wall, robot beside it.

    The work area (the floor socket-lamp) stays across the room -- the task-phase goal is
    fetching the ladder and standing it up there.
    """
    scene.ladder.init_state.pos = CARRY_LADDER_POSITION
    scene.ladder.init_state.rot = _quat_z_deg(CARRY_LADDER_YAW_DEG)
    scene.robot.init_state.pos = CARRY_ROBOT_POSITION


def apply_at_height_preset(scene: G1ReplaceSceneCfg, robot_at: str = "base") -> None:
    """Elevated-fixture layout shared by climb / descend / remove / install.

    The socket entity becomes a ceiling chandelier hung above the ladder; the floor lamp
    is gone. ``robot_at="base"`` starts the robot at the ladder's feet (climb),
    ``robot_at="top"`` starts it at the upper steps (descend, and the working pose for
    remove/install). The random dressing ``fixture`` is dropped -- the task chandelier
    owns the ceiling.
    """
    scene.socket.spawn.usd_path = ELEVATED_SOCKET_USD
    scene.socket.init_state.pos = ELEVATED_SOCKET_POSITION
    scene.bulb.init_state.pos = PARKED_BULB_POSITION
    scene.robot.init_state.pos = CLIMB_ROBOT_POSITION if robot_at == "base" else TOP_ROBOT_POSITION
    scene.fixture = None


def apply_remove_preset(scene: G1ReplaceSceneCfg) -> None:
    """Bulb-removal start: at-height layout with the OLD BULB SEATED in the fixture.

    The bulb is kinematic here -- a stand-in for "screwed into the socket" until the
    task-phase attach joint exists (unification spec, Phase 4).
    """
    apply_at_height_preset(scene, robot_at="top")
    scene.bulb.init_state.pos = SEATED_BULB_POSITION
    scene.bulb.spawn.rigid_props.kinematic_enabled = True


def apply_install_preset(scene: G1ReplaceSceneCfg) -> None:
    """Bulb-installation start: at-height layout, empty fixture, fresh bulb in a crate.

    The parts crate (``bin``) sits at the ladder base with the dynamic bulb resting in it.
    """
    apply_at_height_preset(scene, robot_at="top")
    scene.bin = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bin",
        init_state=AssetBaseCfg.InitialStateCfg(pos=BIN_POSITION),
        spawn=sim_utils.UsdFileCfg(
            usd_path=CRATE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    scene.bulb.init_state.pos = BIN_BULB_POSITION
