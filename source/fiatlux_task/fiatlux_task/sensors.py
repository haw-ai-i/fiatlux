# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable, task-agnostic G1 sensor configs (cameras, lidar).

Mirrors ``robots/g1.py``: a sensor is defined once against a mount body, and tasks
attach it into their scene via the ``add_*`` helpers in ``scene_cfg.py`` (the same
composable-preset-function pattern as ``add_ladder_contact_sensor``) instead of
re-authoring a ``TiledCameraCfg`` / lidar cfg inline per task.

``EGO_CAMERA_NAME`` is a fixed contract, not a knob: GR00T's PolicyServer adapter
(``fiatlux_task.groot.GrootPolicy``) looks the camera up by that scene key
(``env.scene["ego_camera"]``) and reads raw ``rgb`` frames for the REAL_G1
embodiment's ``ego_view`` input. Renaming it here would silently break the GR00T baseline.

The ego camera and lidar mount on the G1 USD's own sensor-housing bodies (``d435_link`` /
``mid360_link`` -- a RealSense D435 + Livox Mid360, matching NVIDIA's head-mounted data
collection rig), zero local offset, ``convention="world"``. Isaac Lab's offset convention
determines the meaning of "identity" (forward axis differs per convention), so a mount
change here needs re-verifying by rendering the frame, not just checking it doesn't crash.
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns

from .robots.g1 import G1_D435_BODY, G1_MID360_BODY

EGO_CAMERA_NAME = "ego_camera"
MID360_LIDAR_NAME = "mid360_lidar"


def ego_camera_cfg() -> TiledCameraCfg:
    """Head-mounted RGB camera (on ``d435_link``): room-scale exteroception, GR00T's ego view.

    Scene key stays :data:`EGO_CAMERA_NAME` -- see the module docstring's GR00T note.
    256x256 matches the GR00T-N1.7-3B checkpoint's own image preprocessing target
    (``shortest_image_edge=256`` in its ``processor_config.json``). ``focal_length`` is
    tuned (aperture fixed) for a 69.4 deg horizontal FOV, matching the real Intel
    RealSense D435 RGB sensor's published spec -- a narrower FOV understates how much
    of the scene the real camera would see around an occluding hand.
    """
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + G1_D435_BODY + "/" + EGO_CAMERA_NAME,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.13,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        height=256,
        width=256,
        data_types=["rgb"],
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )


def mid360_lidar_cfg(include_ladder: bool) -> MultiMeshRayCasterCfg:
    """Head-mounted spinning lidar (on ``mid360_link``; 32-channel, +-180 deg horizontal,
    +-25 deg vertical).

    Uses ``MultiMeshRayCasterCfg`` rather than the plain single-static-mesh
    ``RayCasterCfg``: the plain raycaster in this Isaac Lab version supports exactly one
    mesh prim, read once and reused for every env clone (fine for the shared ground
    plane, wrong for a per-env ladder). ``MultiMeshRayCasterCfg`` resolves
    ``{ENV_REGEX_NS}``-templated targets per env and can track a moving mesh's
    transform, which the ladder needs wherever it is dynamic (``FIATLUX-Replace-v0``)
    or simply per-env-cloned (everywhere else).

    Args:
        include_ladder: also ray-cast against ``{ENV_REGEX_NS}/Ladder``. False for
            presets with no ``Ladder`` scene entity (the tabletop preset).
    """
    targets: list = ["/World/ground"]
    if include_ladder:
        targets.append(
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Ladder", track_mesh_transforms=True)
        )
    return MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + G1_MID360_BODY,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="base",
        mesh_prim_paths=targets,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=32,
            vertical_fov_range=(-25.0, 25.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=2.0,
        ),
        max_distance=10.0,
    )
