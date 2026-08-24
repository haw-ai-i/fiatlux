# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable Unitree G1 articulation config and joint-name constants.

This mirrors Isaac Lab's ``isaaclab_assets`` pattern: the robot is defined once,
task-agnostically, and tasks reference it via ``G1_INSPIRE_CFG.replace(...)``.
The joint-name and end-effector constants live here too, since they describe the
G1 itself rather than any particular task. A Dex3-hand variant slots in alongside
``G1_INSPIRE_CFG`` when needed (different USD + hand joint names).

The cfg deliberately leaves ``prim_path`` unset (``MISSING``); each scene supplies
it via ``.replace(prim_path=...)`` so the same robot can be reused across tasks.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone

from ..assets import G1_DEX3_USD, G1_USD

# The Inspire hand is authored with collision meshes that interpenetrate their
# non-joint-connected neighbors at the default pose (PhysX adjacency filtering only exempts
# pairs sharing a joint), which saturates every contact reading on the hand. Filter exactly
# those pairs at spawn.
_G1_INSPIRE_FILTERED_PAIRS = {
    "{side}_hand_camera_base_link": ("{side}_wrist_pitch_link", "{side}_hand_base_link"),
    "{S}_thumb_proximal": ("{side}_hand_base_link",),
}

# The Dex3 hand has the same embedded-camera-housing defect, against the palm and the thumb
# base only; every wrist link is clear.
_G1_DEX3_FILTERED_PAIRS = {
    "{side}_hand_camera_base_link": ("{side}_hand_palm_link", "{side}_hand_thumb_0_link"),
}


def _make_filtered_hand_mount_spawner(pairs: dict[str, tuple[str, ...]]):
    """Build a spawner that filters ``pairs`` (formatted per side) after loading the USD."""

    @clone
    def _spawn(prim_path, cfg, translation=None, orientation=None):
        from pxr import UsdPhysics

        prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
        stage = prim.GetStage()
        for side in ("left", "right"):
            fmt = {"side": side, "S": side[0].upper()}
            for body, targets in pairs.items():
                api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/{body.format(**fmt)}"))
                rel = api.GetFilteredPairsRel()
                for target in targets:
                    rel.AddTarget(f"{prim_path}/{target.format(**fmt)}")
        return prim

    return _spawn


_spawn_g1_with_filtered_hand_mounts = _make_filtered_hand_mount_spawner(_G1_INSPIRE_FILTERED_PAIRS)
_spawn_g1_dex3_with_filtered_hand_mounts = _make_filtered_hand_mount_spawner(_G1_DEX3_FILTERED_PAIRS)


# ---------------------------------------------------------------------------
# Joint / body names (standard Unitree G1 naming)
# ---------------------------------------------------------------------------

# Right-arm joints used for the manipulation subtasks.
G1_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# Right Inspire-hand joints (12 DoF). Split four-fingers / thumb because they curl to different
# targets: the thumb's pitch joint tops out at 0.6 rad where the fingers reach 1.7. ORDER IS
# LOAD-BEARING -- it lays out the hand's slice of the action vector, so append, do not
# rearrange.
G1_FINGER_JOINTS = [
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
]
G1_THUMB_JOINTS = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
]
G1_HAND_JOINTS = G1_FINGER_JOINTS + G1_THUMB_JOINTS
# Left Inspire hand joints -- the mirror of the (right) G1_HAND_JOINTS above. Bimanual teleop envs
# scope a left grip to these; listing them here lets ``swap_robot_variant`` remap the left hand to
# Dex3 alongside the right (see ``_HAND_REMAPS``).
G1_LEFT_HAND_JOINTS = [j.replace("R_", "L_", 1) for j in G1_HAND_JOINTS]

# Inspire-hand open / power-grasp finger presets (rad). Open = fingers extended (0). Grasp curls
# the four fingers near their +1.7 limit and opposes the thumb (pitch caps at +0.6). Used by the
# teleop harness's binary grip; probed from the soft joint-position limits.
G1_HAND_OPEN = dict.fromkeys(G1_HAND_JOINTS, 0.0)
G1_HAND_GRASP = {
    "R_index_proximal_joint": 1.5,
    "R_index_intermediate_joint": 1.5,
    "R_middle_proximal_joint": 1.5,
    "R_middle_intermediate_joint": 1.5,
    "R_pinky_proximal_joint": 1.5,
    "R_pinky_intermediate_joint": 1.5,
    "R_ring_proximal_joint": 1.5,
    "R_ring_intermediate_joint": 1.5,
    "R_thumb_proximal_yaw_joint": 1.0,
    "R_thumb_proximal_pitch_joint": 0.5,
    "R_thumb_intermediate_joint": 0.6,
    "R_thumb_distal_joint": 0.9,
}
# End-effector body the wrist camera mounts on / eef pose is read from (exists in
# all G1 variants). The Inspire hand links hang off this via right_hand_palm_link.
G1_EE_BODY = "right_wrist_yaw_link"

# Climb-family limbs and joint groups.
G1_FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
# The Inspire palm body; its surface is the local -x side (see fiatlux_task/poses.py).
G1_PALM_BODIES = ["left_hand_base_link", "right_hand_base_link"]
G1_TORSO_BODY = "torso_link"
# Sensor-housing bodies authored on the USD (RealSense D435 + Livox Mid360, fixed to the torso
# -- G1 has no neck joint). See fiatlux_task/sensors.py.
G1_D435_BODY = "d435_link"
G1_MID360_BODY = "mid360_link"
# Joint-name patterns for reward scoping (match the actuator groups below).
G1_WAIST_JOINT_PATTERNS = ["waist_.*_joint"]
G1_FINGER_JOINT_PATTERNS = ["[LR]_.*_joint"]


# ---------------------------------------------------------------------------
# Actuator model
# ---------------------------------------------------------------------------

# Rotor inertia reflected through the gearbox, per Unitree motor type.
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

_LEG_STIFFNESS = {".*_hip_.*_joint": 150.0, ".*_knee_joint": 200.0, ".*_ankle_.*_joint": 40.0}
_LEG_DAMPING = {".*_hip_.*_joint": 2.0, ".*_knee_joint": 4.0, ".*_ankle_.*_joint": 2.0}
_LEG_EFFORT = {
    ".*_hip_yaw_joint": 88.0,
    ".*_hip_roll_joint": 139.0,
    ".*_hip_pitch_joint": 139.0,
    ".*_knee_joint": 139.0,
    ".*_ankle_.*_joint": 50.0,
}
_LEG_ARMATURE = {
    ".*_hip_yaw_joint": ARMATURE_7520_14,
    ".*_hip_roll_joint": ARMATURE_7520_22,
    ".*_hip_pitch_joint": ARMATURE_7520_22,
    ".*_knee_joint": ARMATURE_7520_22,
    ".*_ankle_.*_joint": 2.0 * ARMATURE_5020,
}
_ARM_STIFFNESS = {
    ".*_shoulder_pitch_joint": 100.0,
    ".*_shoulder_roll_joint": 100.0,
    ".*_shoulder_yaw_joint": 50.0,
    ".*_elbow_joint": 50.0,
    ".*_wrist_.*_joint": 20.0,
}
_ARM_DAMPING = {
    ".*_shoulder_.*_joint": 2.0,
    ".*_elbow_joint": 2.0,
    ".*_wrist_.*_joint": 1.0,
}
_ARM_EFFORT = {
    ".*_shoulder_.*_joint": 25.0,
    ".*_elbow_joint": 25.0,
    ".*_wrist_roll_joint": 25.0,
    ".*_wrist_pitch_joint": 5.0,
    ".*_wrist_yaw_joint": 5.0,
}
_ARM_ARMATURE = {
    ".*_shoulder_.*_joint": ARMATURE_5020,
    ".*_elbow_joint": ARMATURE_5020,
    ".*_wrist_roll_joint": ARMATURE_5020,
    ".*_wrist_pitch_joint": ARMATURE_4010,
    ".*_wrist_yaw_joint": ARMATURE_4010,
}

# ---------------------------------------------------------------------------
# Articulation config (legged / free base, Inspire hand)
# ---------------------------------------------------------------------------

# Spawns standing, but keeps its legs so the same asset can locomote and climb.
G1_INSPIRE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_USD,
        func=_spawn_g1_with_filtered_hand_mounts,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
        activate_contact_sensors=True,
    ),
    # Spawn standing (feet on the floor at the bent-knee pose; settled height 0.787 m).
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.79),
        # Only the bent leg joints are listed; every other joint defaults to 0.0. A ``".*"``
        # catch-all would also match these and trip Isaac Lab's one-regex-per-joint resolver.
        joint_pos={
            ".*_hip_pitch_joint": -0.05,
            ".*_knee_joint": 0.2,
            ".*_ankle_pitch_joint": -0.15,
        },
    ),
    # Disjoint actuator groups covering every joint. NOTE the arm regex is anchored to
    # shoulder/elbow/wrist so it does not also grab the *leg* joints (which share the
    # left_/right_ prefix).
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"],
            effort_limit_sim=_LEG_EFFORT,
            stiffness=_LEG_STIFFNESS,
            damping=_LEG_DAMPING,
            armature=_LEG_ARMATURE,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim={"waist_yaw_joint": 88.0, "waist_(roll|pitch)_joint": 50.0},
            stiffness=250.0,
            damping=5.0,
            armature={
                "waist_yaw_joint": ARMATURE_7520_14,
                "waist_(roll|pitch)_joint": 2.0 * ARMATURE_5020,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_(shoulder|elbow|wrist).*_joint"],
            effort_limit_sim=_ARM_EFFORT,
            stiffness=_ARM_STIFFNESS,
            damping=_ARM_DAMPING,
            armature=_ARM_ARMATURE,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_.*_joint"],
            # Real Inspire fingers produce ~1-2 N.m. A high limit lets a wedged finger's
            # saturated PD torque catapult the whole robot off furniture.
            effort_limit_sim=2.0,
            stiffness=1000.0,
            damping=15.0,
        ),
    },
)


# ---------------------------------------------------------------------------
# Articulation config (legged / free base, Dex3 hand)
# ---------------------------------------------------------------------------

# Dex3 finger joints per hand, in GR00T's REAL_G1 hand-channel order
# (GR00T-WholeBodyControl ``g1_supplemental_info.py`` ``joint_groups``).
_DEX3_HAND_ORDER = [
    "hand_index_0_joint",
    "hand_index_1_joint",
    "hand_middle_0_joint",
    "hand_middle_1_joint",
    "hand_thumb_0_joint",
    "hand_thumb_1_joint",
    "hand_thumb_2_joint",
]
G1_DEX3_LEFT_HAND_JOINTS = [f"left_{j}" for j in _DEX3_HAND_ORDER]
G1_DEX3_RIGHT_HAND_JOINTS = [f"right_{j}" for j in _DEX3_HAND_ORDER]
G1_DEX3_FINGER_JOINT_PATTERNS = [".*_hand_(thumb|index|middle)_._joint"]
# Dex3-1 open (fingers extended = 0) / power-grasp presets (rad), probed from the joint-position
# limits. The two hands are MIRRORED: right index/middle curl toward + (limits [0, +1.6]), left toward
# - (limits [-1.6, 0]); the thumb opposes. Used by the teleop harness's binary grip. Tune magnitudes if
# the grasp over/under-closes.
G1_DEX3_HAND_OPEN = dict.fromkeys(G1_DEX3_RIGHT_HAND_JOINTS, 0.0)
G1_DEX3_HAND_GRASP = {
    # Fingers curl to ~95% of their limits (1.57/1.75) so the fingertips come BACK toward the thumb --
    # the right thumb can't reach far forward (model limit), so closing the gap means bringing the
    # fingers to it. This tightens the pinch at the index for both hands.
    "right_hand_index_0_joint": 1.5,
    "right_hand_index_1_joint": 1.7,
    "right_hand_middle_0_joint": 1.5,
    "right_hand_middle_1_joint": 1.7,
    # Right thumb flipped to close from the OPPOSITE direction (operator request): negate the yaw
    # (thumb_0) and pitch (thumb_1) so the thumb opposes from the other side; keep the distal curl.
    "right_hand_thumb_0_joint": -0.8,
    "right_hand_thumb_1_joint": -0.2,
    "right_hand_thumb_2_joint": -1.2,
}
G1_DEX3_LEFT_HAND_OPEN = dict.fromkeys(G1_DEX3_LEFT_HAND_JOINTS, 0.0)
G1_DEX3_LEFT_HAND_GRASP = {
    "left_hand_index_0_joint": -1.5,
    "left_hand_index_1_joint": -1.7,
    "left_hand_middle_0_joint": -1.5,
    "left_hand_middle_1_joint": -1.7,
    "left_hand_thumb_0_joint": -0.8,
    "left_hand_thumb_1_joint": 0.4,
    "left_hand_thumb_2_joint": 1.2,
}

G1_DEX3_PALM_BODIES = ["left_hand_palm_link", "right_hand_palm_link"]

# Distal link of each digit that closes on a grasped object, per variant. Their centroid
# against the palm's locates the hand's cup without needing to know which local axis the palm
# surface is; the two hands disagree on that.
G1_GRASP_DISTAL_BODIES: dict[str, list[str]] = {
    "inspire": ["R_index_intermediate", "R_middle_intermediate", "R_ring_intermediate", "R_thumb_distal"],
    "dex3": ["right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link"],
}
# Right-hand joints and palm body per variant, for scripted poses that must name them.
G1_RIGHT_HAND_JOINTS_BY_VARIANT: dict[str, list[str]] = {
    "inspire": G1_HAND_JOINTS,
    "dex3": G1_DEX3_RIGHT_HAND_JOINTS,
}
G1_PALM_BODY_BY_VARIANT: dict[str, str] = {
    "inspire": G1_PALM_BODIES[1],
    "dex3": G1_DEX3_PALM_BODIES[1],
}

# The palm body's own origin is NOT near the visible palm surface (several cm off the mesh,
# toward the wrist). For placing something ON the palm, anchor position on the centroid of these
# finger-BASE (proximal) bodies and use the palm body only for orientation.
# Dex3: unverified placeholder.
G1_FINGER_BASE_BODIES_BY_VARIANT: dict[str, list[str]] = {
    "inspire": ["R_index_proximal", "R_middle_proximal", "R_ring_proximal"],
    "dex3": ["right_hand_index_0_link", "right_hand_middle_0_link"],
}

# Palm-link local axes as ``(axis_index, sign)`` -- (outward normal, along fingers, across
# palm). Dex3: +y is the face the digits close onto, +x runs out toward the tips, +z spans the
# palm. Inspire's face is its local -x.
G1_PALM_LOCAL_AXES: dict[str, tuple[tuple[int, float], ...]] = {
    "dex3": ((1, 1.0), (0, 1.0), (2, 1.0)),
    "inspire": ((0, -1.0), (1, 1.0), (2, 1.0)),
}

# Limbs the ladder-contact sensor watches: both feet plus every variant's palm. The sensor is
# built before ``swap_robot_variant`` may change the hand, so it must name all variants; names
# belonging to the absent one never resolve.
G1_LADDER_CONTACT_BODIES: list[str] = [*G1_FOOT_BODIES, *G1_PALM_BODIES, *G1_DEX3_PALM_BODIES]

G1_DEX3_CFG = G1_INSPIRE_CFG.replace(
    spawn=G1_INSPIRE_CFG.spawn.replace(usd_path=G1_DEX3_USD, func=_spawn_g1_dex3_with_filtered_hand_mounts),
    actuators={
        **{k: v for k, v in G1_INSPIRE_CFG.actuators.items() if k != "hands"},
        # Unitree Dex3 driver gains; torque limits come from the URDF/USD.
        "hands": ImplicitActuatorCfg(
            joint_names_expr=G1_DEX3_FINGER_JOINT_PATTERNS,
            stiffness=1.5,
            damping=0.1,
        ),
    },
)

G1_VARIANTS = {"inspire": G1_INSPIRE_CFG, "dex3": G1_DEX3_CFG}

# Every task in this repo is *authored* against the Inspire hand (G1_INSPIRE_CFG is the
# scene's default robot everywhere); swapping is always FROM that fixed baseline TO the
# target variant. Each entry maps an exact Inspire joint-name list a task might reference
# (a reward/termination finger-deviation pattern, an action term's controlled joints, or
# an observation term's scoped joints) to its dex3 equivalent. Insert's action/observation
# scope arm+hand together (``G1_ARM_JOINTS + G1_HAND_JOINTS``), hence that combined entry.
_HAND_REMAPS: dict[str, dict[tuple[str, ...], list[str]]] = {
    "dex3": {
        tuple(G1_FINGER_JOINT_PATTERNS): list(G1_DEX3_FINGER_JOINT_PATTERNS),
        tuple(G1_HAND_JOINTS): list(G1_DEX3_RIGHT_HAND_JOINTS),
        tuple(G1_LEFT_HAND_JOINTS): list(G1_DEX3_LEFT_HAND_JOINTS),
        tuple(G1_ARM_JOINTS + G1_HAND_JOINTS): list(G1_ARM_JOINTS + G1_DEX3_RIGHT_HAND_JOINTS),
    },
}
# Symmetric "back to inspire" entries, kept for completeness / testability even though no
# script currently calls ``swap_robot_variant(cfg, "inspire")`` (eval.py / record_run.py
# only swap when ``--robot != "inspire"``).
_HAND_REMAPS["inspire"] = {tuple(v): list(k) for k, v in _HAND_REMAPS["dex3"].items()}

# Substrings that flag a joint-name list as hand-specific for *some* variant, so an
# unrecognized list containing one can be told apart from a variant-agnostic list (arm,
# waist, leg joints -- identical names on every G1 variant) that never needed remapping.
_HAND_NAME_MARKERS = ("R_", "L_", "_hand_")
# Same markers, split by which variant they name -- Inspire's fingers are ``R_``/``L_``
# prefixed, Dex3's contain ``_hand_``. Neither marker occurs in any arm/waist/leg joint name.
_HAND_MARKERS_BY_VARIANT: dict[str, tuple[str, ...]] = {"inspire": ("R_", "L_"), "dex3": ("_hand_",)}


def _drop_foreign_hand_joint_pos(joint_pos: dict[str, float], variant: str) -> dict[str, float]:
    """Strip ``init_state.joint_pos`` entries authored for a hand variant other than ``variant``.

    ``swap_robot_variant`` carries the old ``init_state`` over unchanged; a finger-pose entry a
    task authored for the Inspire hand (``R_index_proximal_joint``, ...) would otherwise resolve
    against zero joints on a swapped-in Dex3 robot, and ``resolve_matching_names_values`` is
    strict by default -- that is a crash on env creation, not a silent no-op.
    """
    foreign_markers = [m for v, ms in _HAND_MARKERS_BY_VARIANT.items() if v != variant for m in ms]
    return {k: v for k, v in joint_pos.items() if not any(m in k for m in foreign_markers)}


def swap_robot_variant(env_cfg, variant: str) -> None:
    """Swap the scene's G1 hand variant in a parsed env cfg, keeping its placement.

    Rewrites every joint-name reference this function recognizes -- reward/termination/
    event *and* action *and* observation terms scoped to the Inspire hand's finger
    pattern or literal hand-joint list -- to the target variant's equivalent. A
    wildcard action term (``joint_names=[".*"]``) needs no rewriting; it resolves
    against whichever robot is attached.

    Raises ``ValueError`` if it finds a joint-name list that looks hand-specific (matches
    neither variant's known joint names, but contains a hand-name marker) and isn't in
    ``_HAND_REMAPS``, rather than leaving it pointed at joints the swapped-in robot
    does not have.
    """
    if variant not in G1_VARIANTS:
        raise ValueError(f"unknown G1 variant {variant!r}; choose from {sorted(G1_VARIANTS)}")
    robot = env_cfg.scene.robot
    init_state = robot.init_state.replace(
        joint_pos=_drop_foreign_hand_joint_pos(dict(robot.init_state.joint_pos), variant)
    )
    env_cfg.scene.robot = G1_VARIANTS[variant].replace(prim_path=robot.prim_path, init_state=init_state)

    remap_table = _HAND_REMAPS[variant]

    def remap(names) -> list[str] | None:
        names = list(names or [])
        if not names or names == [".*"]:
            return None  # empty / wildcard: variant-agnostic, nothing to do
        mapped = remap_table.get(tuple(names))
        if mapped is not None:
            return mapped
        if any(any(marker in n for marker in _HAND_NAME_MARKERS) for n in names):
            raise ValueError(
                f"swap_robot_variant({variant!r}): don't know how to remap "
                f"joint_names={names!r} -- it looks hand-specific but isn't in "
                "robots.g1._HAND_REMAPS. Add it there rather than swapping the robot "
                "and leaving this term pointed at joints the new hand doesn't have."
            )
        return None  # arm / waist / leg joints: identical names on every variant

    def remap_asset_cfg(term) -> None:
        asset_cfg = getattr(term, "params", {}).get("asset_cfg") if hasattr(term, "params") else None
        if asset_cfg is not None and getattr(asset_cfg, "name", "robot") == "robot":
            remapped = remap(asset_cfg.joint_names)
            if remapped is not None:
                asset_cfg.joint_names = remapped

    for manager_name in ("rewards", "terminations", "events"):
        manager = getattr(env_cfg, manager_name, None)
        if manager is None:
            continue
        for term_name in dir(manager):
            if term_name.startswith("_"):
                continue
            remap_asset_cfg(getattr(manager, term_name))

    actions = getattr(env_cfg, "actions", None)
    if actions is not None:
        for term_name in dir(actions):
            if term_name.startswith("_"):
                continue
            term = getattr(actions, term_name)
            if getattr(term, "asset_name", None) == "robot" and hasattr(term, "joint_names"):
                remapped = remap(term.joint_names)
                if remapped is not None:
                    term.joint_names = remapped

    observations = getattr(env_cfg, "observations", None)
    if observations is not None:
        for group_name in dir(observations):
            if group_name.startswith("_"):
                continue
            group = getattr(observations, group_name)
            if not isinstance(group, ObsGroup):
                continue
            for term_name in dir(group):
                if term_name.startswith("_"):
                    continue
                remap_asset_cfg(getattr(group, term_name))
