"""LadderMan's G1 with FIATLUX's actuator model.

Deployed to Koa as ``LadderMan/active_adaptation/assets/g1_fiatlux.py`` and registered in that
package's ``__init__.py`` as ``ROBOTS["g1_fiatlux"]``; selected with ``task.robot.name=g1_fiatlux``.
Kept here because this file is ours, it is the thing under test, and a copy that lives only on
scratch is not reviewable.

Why
---
A teacher trained under LadderMan's actuators does not transfer into our envs. Measured
2026-11-17 on checkpoint_3900 over pinned layouts: it dies in the MOUNT, 7 of 9 episodes ending
before the reference asks for a rung's height. Two inference-time fixes were tried and both
failed -- closing the 285 mm spawn standoff (9 runs, no effect on height) and running our robot
on LadderMan's gains (3 runs, three times WORSE: 114 +/- 10 steps vs 336 +/- 199).

So instead of converting at inference, move the TEACHER to our physics and train there. This is
the direction that costs GPU time rather than benchmark integrity: the benchmark's robot is the
thing being measured and does not bend to suit a policy.

What differs, measured against active_adaptation/assets/g1.py
-------------------------------------------------------------
LadderMan derives gains from a 10 Hz natural frequency (kp = armature * w^2,
kd = 2 * zeta * armature * w, zeta=2); fiatlux uses hand-tuned constants. Ratios fiatlux/LadderMan:
waist_roll/pitch 8.8x, shoulder_pitch/roll 7.0x, waist_yaw 6.2x, hip_yaw/hip_pitch 3.7x,
shoulder_yaw/elbow 3.5x, knee 2.0x, hip_roll 1.5x, ankle 1.4x, wrist_roll 1.4x, wrist_pitch/yaw
1.2x.

Armature and effort agree on 27 of 29 joints. The whole disagreement is hip_pitch, where
LadderMan models the 7520 at 14:1 (armature 0.010178, effort 88) and fiatlux at 22:1
(armature 0.025102, effort 139) -- 2.47x the reflected rotor inertia, and the primary lifting
joint in a mount.

Deliberately NOT matched
------------------------
* ``velocity_limit_sim`` keeps LadderMan's values. fiatlux sets none, so it inherits the USD's,
  and swapping a declared limit for an unknown one is an uncontrolled change, not a match. The
  limits here (20-37 rad/s) are far above anything a climb demands.
* Link masses and the hand. fiatlux's robot carries a Dex3/Inspire hand -- 55 bodies, 38.61 kg --
  where this URDF has 30 bodies and a cylinder end-effector. That is a residual no actuator
  config can close, and it is why this run is a test of a hypothesis rather than a guaranteed fix.
"""

import copy

from isaaclab.actuators import ImplicitActuatorCfg

from .g1 import (
    ARMATURE_4010,
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    G1_CYLINDER_CFG,
)

# fiatlux writes these as 2.0 * ARMATURE_5020 rather than naming a motor.
ANKLE_ARMATURE = 2.0 * ARMATURE_5020
WAIST_RP_ARMATURE = 2.0 * ARMATURE_5020

# deepcopy so spawn, init_state, soft_joint_pos_limit_factor and anything added upstream later
# are inherited rather than duplicated -- only the actuators are ours.
G1_FIATLUX_CFG = copy.deepcopy(G1_CYLINDER_CFG)

G1_FIATLUX_CFG.actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim={
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_pitch_joint": 139.0,   # LadderMan: 88.0
            ".*_knee_joint": 139.0,
        },
        velocity_limit_sim={
            ".*_hip_yaw_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_hip_pitch_joint": 32.0,
            ".*_knee_joint": 20.0,
        },
        stiffness={
            ".*_hip_yaw_joint": 150.0,
            ".*_hip_roll_joint": 150.0,
            ".*_hip_pitch_joint": 150.0,
            ".*_knee_joint": 200.0,
        },
        damping={
            ".*_hip_yaw_joint": 2.0,
            ".*_hip_roll_joint": 2.0,
            ".*_hip_pitch_joint": 2.0,
            ".*_knee_joint": 4.0,
        },
        armature={
            ".*_hip_yaw_joint": ARMATURE_7520_14,
            ".*_hip_roll_joint": ARMATURE_7520_22,
            ".*_hip_pitch_joint": ARMATURE_7520_22,   # LadderMan: ARMATURE_7520_14
            ".*_knee_joint": ARMATURE_7520_22,
        },
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        stiffness=40.0,
        damping=2.0,
        armature=ANKLE_ARMATURE,
    ),
    "waist": ImplicitActuatorCfg(
        joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        stiffness=250.0,
        damping=5.0,
        armature=WAIST_RP_ARMATURE,
    ),
    "waist_yaw": ImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint"],
        effort_limit_sim=88.0,
        velocity_limit_sim=32.0,
        stiffness=250.0,
        damping=5.0,
        armature=ARMATURE_7520_14,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        effort_limit_sim={
            ".*_shoulder_pitch_joint": 25.0,
            ".*_shoulder_roll_joint": 25.0,
            ".*_shoulder_yaw_joint": 25.0,
            ".*_elbow_joint": 25.0,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
            ".*_wrist_yaw_joint": 5.0,
        },
        velocity_limit_sim={
            ".*_shoulder_pitch_joint": 37.0,
            ".*_shoulder_roll_joint": 37.0,
            ".*_shoulder_yaw_joint": 37.0,
            ".*_elbow_joint": 37.0,
            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_pitch_joint": 22.0,
            ".*_wrist_yaw_joint": 22.0,
        },
        stiffness={
            ".*_shoulder_pitch_joint": 100.0,
            ".*_shoulder_roll_joint": 100.0,
            ".*_shoulder_yaw_joint": 50.0,
            ".*_elbow_joint": 50.0,
            ".*_wrist_roll_joint": 20.0,
            ".*_wrist_pitch_joint": 20.0,
            ".*_wrist_yaw_joint": 20.0,
        },
        damping={
            ".*_shoulder_pitch_joint": 2.0,
            ".*_shoulder_roll_joint": 2.0,
            ".*_shoulder_yaw_joint": 2.0,
            ".*_elbow_joint": 2.0,
            ".*_wrist_roll_joint": 1.0,
            ".*_wrist_pitch_joint": 1.0,
            ".*_wrist_yaw_joint": 1.0,
        },
        armature={
            ".*_shoulder_pitch_joint": ARMATURE_5020,
            ".*_shoulder_roll_joint": ARMATURE_5020,
            ".*_shoulder_yaw_joint": ARMATURE_5020,
            ".*_elbow_joint": ARMATURE_5020,
            ".*_wrist_roll_joint": ARMATURE_5020,
            ".*_wrist_pitch_joint": ARMATURE_4010,
            ".*_wrist_yaw_joint": ARMATURE_4010,
        },
    ),
}


# ---- the open Dex3 hand: one collision shape per hand part ------------------------------------------
# The URDF importer puts every collision mesh of a link under one ``collisions`` prim carrying
# PhysxMeshMergeCollisionAPI, so a link with several meshes becomes ONE convex hull. For the rigid open
# Dex3 (wrist + palm + two fingers + a thumb hanging 100 mm below the palm) that hull is a wedge filling
# the space between thumb and fingertips: PhysX reported a single shape with a local box of
# 235 x 143 x 87 mm. A palm laid flat on a tread then starts 37 mm INSIDE the tread and is thrown off it
# on the first step of every episode. The benchmark's hand has one hull per link. This restores that:
# after conversion, the hand links' collision meshes each get their own convex hull.
def _split_hand_colliders(usd_path: str) -> int:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    done = 0
    for prim in list(stage.Traverse()):
        if prim.GetName() != "collisions" or not prim.GetParent().GetName().endswith("_wrist_yaw_link"):
            continue
        prim.SetInstanceable(False)
        for api in (PhysxSchema.PhysxMeshMergeCollisionAPI, UsdPhysics.MeshCollisionAPI, UsdPhysics.CollisionAPI):
            prim.RemoveAPI(api)
        n = 0
        for m in Usd.PrimRange(prim):
            if m.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(m)
                UsdPhysics.MeshCollisionAPI.Apply(m).CreateApproximationAttr("convexHull")
                n += 1
        print(f"[g1_fiatlux] {prim.GetParent().GetName()}: {n} hand parts, one convex hull each")
        done += 1
    if done:
        stage.GetRootLayer().Save()
    return done


def _make_spawn_split_hands():
    from isaaclab.sim import converters
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.utils import clone

    @clone
    def spawn_from_urdf_split_hands(prim_path, cfg, translation=None, orientation=None, **kwargs):
        loader = converters.UrdfConverter(cfg)
        if "dex3open" in str(cfg.asset_path):
            assert _split_hand_colliders(loader.usd_path) == 2, "expected two hand links to split"
        return _spawn_from_usd_file(prim_path, loader.usd_path, cfg, translation, orientation)

    return spawn_from_urdf_split_hands


G1_FIATLUX_CFG.spawn.func = _make_spawn_split_hands()
