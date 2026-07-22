# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-time events for the Fiatlux tasks.

Bulb and socket pose randomization use Isaac Lab's built-in
``reset_root_state_uniform``; only light randomization needs custom functions
because the scene lights are single shared USD prims.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import omni.usd
from pxr import Gf, UsdLux

from isaaclab.envs.mdp import randomize_rigid_body_material
from isaaclab.managers import EventTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_light_properties(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    intensity_range: tuple[float, float] | None = None,
    color: tuple[float, float, float] | None = None,
) -> None:
    """Randomize a scene light's intensity (and optionally set a color) on reset.

    The one light-randomization term for every Fiatlux scene: lights are addressed as named
    scene entities (``dome_light``, ``key_light``, ...), so both the Insert task and the
    ladder family use this with a ``SceneEntityCfg``. Isaac Lab has no built-in
    light-randomization event term. Lights are single shared prims, so one global sample is
    applied regardless of ``env_ids``. Leave ``color`` unset when the light carries an HDRI
    ``texture_file`` -- the color attribute only tints the texture, which reads as a render
    bug rather than useful domain randomization.

    TODO(task phase):
        * sample per-env values and support per-env lights (the scene lights are global prims);
        * randomize light *orientation* (the distant-light direction) as well;
        * sample ``color`` from a range rather than taking a single value.
    """
    # Read the prim path from the scene *cfg*: lights are plain AssetBaseCfg entities, and
    # the scene stores those as bare XFormPrim objects with no `.cfg` attribute at runtime.
    prim_path = getattr(env.scene.cfg, asset_cfg.name).prim_path
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    light = UsdLux.LightAPI(prim)
    if intensity_range is not None:
        intensity = torch.empty(1).uniform_(intensity_range[0], intensity_range[1]).item()
        light.GetIntensityAttr().Set(intensity)
    if color is not None:
        light.GetColorAttr().Set(Gf.Vec3f(*color))


# Hand collider shapes on both G1 variants: Inspire (``left_hand_base_link``,
# ``L_index_proximal``, ``R_thumb_distal``, ...) and Dex3 (``left_hand_palm_link``,
# ``right_hand_thumb_2_link``, ...). Verified against both USDs -- one expression matches every
# hand body on either and nothing else, so ``swap_robot_variant`` has nothing to remap here.
G1_HAND_BODY_EXPR = "(left|right)_hand.*|[LR]_(index|middle|pinky|ring|thumb).*"


def hand_grip_material_event() -> EventTermCfg:
    """Grip friction for the G1's hands, as a startup event term.

    Every grasp in the benchmark is made with these shapes, and without this they run on the
    PhysX default 0.5/0.5 -- bare steel on glass, which is not what a robot hand is.

    Deliberately an EVENT rather than a material bound in the robot spawner:
    ``bind_physics_material`` is ``apply_nested``-decorated and ``apply_nested`` SKIPS
    INSTANCED PRIMS, while every G1 link's ``collisions`` child is authored instanceable. The
    USD route therefore binds nothing and leaves the whole robot at 0.5/0.5 while reading as
    applied. This term writes through the PhysX view, which has no notion of instancing.

    Range: NVIDIA's own manipulation environments bracket it -- Factory/AutoMate fix 1.0/1.0
    for insertion, Dexsuite randomizes the hand over [0.5, 1.0]. Centered on 1.0, with a band
    wide enough to be real domain randomization. Returns a fresh cfg per call so each task's
    ``EventCfg`` owns its own instance.
    """
    return EventTermCfg(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_HAND_BODY_EXPR),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.7, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
