# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-LadderGallery-Teleop-v0`` -- a walk-through gallery of every ladder design.

A teleop scene (same SONIC loco + arm-IK + hand + first-person follow-camera machinery as
:class:`CarryTeleopEnvCfg`) whose props are **one of every Omniverse ladder design**, laid out on an
open floor in a grid so you can walk the robot down the aisle and grab each one. Every gallery ladder
spawns with its **file** collision -- the tightened ``convexDecomposition`` authored by
``scripts/omniverse/omniverse_ladder_collision.py`` -- so this is the place to feel the asset-level
collision fix across the whole set. Self-standing designs are dynamic (grabbable); the few narrow /
leaning designs that topple on their own are spawned kinematic so they stay upright for inspection.

Not an RL task -- it reuses the Carry teleop's disabled terminations and is driven by
``scripts/teleop/sonic_teleop.py``. The single ``scene.ladder`` (kept so the inherited Carry obs/rewards
still resolve) is repositioned just in front of the robot and keeps its SDF collider, as an SDF-vs-
convexDecomposition reference next to the gallery.
"""

import glob
import os

from fiatlux_task.assets import FIATLUX_ASSETS_DIR
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import (
    SCORED_BODY_SLEEP_THRESHOLD,
    _quat_z_deg,
    _spawn_usd_as_rigid_body_frictional,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .carry_teleop_env_cfg import CarryTeleopEnvCfg

# Gallery grid: 7 columns x 4 rows = 28 slots (one per ladder design folder), on open floor in
# front of the robot (which stays at the Carry spawn so the arm-teleop root transform is unchanged).
_GALLERY_COLS_X = [2.5 + 2.5 * c for c in range(7)]  # 2.5 .. 17.5 m ahead of the robot
_GALLERY_ROWS_Y = [-3.75, -1.25, 1.25, 3.75]  # central aisle at y~0; the robot walks +x down it


@configclass
class LadderGalleryTeleopEnvCfg(CarryTeleopEnvCfg):
    """Every ladder design in one open-floor teleop scene (see module docstring)."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Lazy import: pxr is only available inside the running Isaac process (where the cfg is built).
        from pxr import Usd, UsdGeom

        # Open-floor gallery: drop the Simple Room walls (28 designs need more room than it has) and
        # make the ground-plane collider visible so there is a floor to see. dome_light + key_light
        # are separate entities and stay.
        self.scene.room = None
        self.scene.ground.spawn.func = sim_utils.spawn_ground_plane
        # High-friction floor so the bases of the leaning ladders don't slip out from under them.
        self.scene.ground.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.3, restitution=0.0
        )

        # Drop the Carry teleop's three A-frame test ladders; the gallery supersedes them.
        for _i in range(3):
            if hasattr(self.scene, f"test_ladder_{_i}"):
                delattr(self.scene, f"test_ladder_{_i}")

        # Keep the single reference ladder (the inherited Carry obs/rewards reference
        # SceneEntityCfg("ladder"), so removing it would break the managers) but move it just in
        # front of the robot beside the gallery. Its collision (SDF + 6 mm) comes from the asset.
        self.scene.ladder.init_state.pos = (0.9, 0.0, 0.0)
        self.scene.ladder.init_state.rot = _quat_z_deg(0.0)

        # One representative *_collision_rigid.usd per design folder, tiled across the grid. Each opens
        # its USD to read metersPerUnit (scale = mpu gives real-world size for both the cm- and the
        # metre-authored assets) and the collider base offset (so it seats on the floor).
        folders = sorted(
            {os.path.dirname(f) for f in glob.glob(
                os.path.join(FIATLUX_ASSETS_DIR, "omniverse_ladder", "**", "*_collision_rigid.usd"), recursive=True
            )}
        )
        n_cols = len(_GALLERY_COLS_X)
        slot = 0
        for _dir in folders:
            hits = sorted(glob.glob(os.path.join(_dir, "*_collision_rigid.usd")))
            if not hits or slot >= n_cols * len(_GALLERY_ROWS_Y):
                continue
            path = hits[0]
            stage = Usd.Stage.Open(path)
            mpu = UsdGeom.GetStageMetersPerUnit(stage)
            bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            base_z = bbox.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange().GetMin()[2] * mpu
            _x = _GALLERY_COLS_X[slot % n_cols]
            _y = _GALLERY_ROWS_Y[slot // n_cols]
            slot += 1

            # Every design is dynamic (grabbable), spawned upright on the floor with a small settle
            # drop. The designs that aren't free-standing just settle however they land (still
            # grabbable) -- a reliable "leaned against a wall" display needs per-design mesh-origin
            # geometry this loop does not do, so we keep it simple and robust here.
            setattr(
                self.scene,
                f"gallery_ladder_{slot:02d}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/GalleryLadder{slot:02d}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=path,
                        func=_spawn_usd_as_rigid_body_frictional,  # collision (SDF + 6 mm) comes from the asset
                        scale=(mpu, mpu, mpu),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=False,
                            solver_position_iteration_count=16,
                            solver_velocity_iteration_count=1,
                            max_depenetration_velocity=1.0,
                            sleep_threshold=SCORED_BODY_SLEEP_THRESHOLD,
                            stabilization_threshold=0.001,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(_x, _y, -base_z + 0.02)),
                ),
            )
