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

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_dome_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (1500.0, 3500.0),
    color_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0),
    ),
    intensity_only: bool = False,
) -> None:
    """Randomize the dome light's intensity and, optionally, color on reset.

    The light is a single shared prim, so the randomization is global across
    all environments regardless of ``env_ids``. Set ``intensity_only=True``
    when the dome uses an HDRI ``texture_file``: the color attribute only
    tints the sky texture there, so jittering it looks like a rendering bug
    rather than useful domain randomization.
    """
    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath("/World/light")
    if not light_prim.IsValid():
        return
    light = UsdLux.DomeLight(light_prim)

    intensity = torch.empty(1).uniform_(intensity_range[0], intensity_range[1]).item()
    light.GetIntensityAttr().Set(intensity)

    if intensity_only:
        return

    color_min, color_max = color_range
    r = torch.empty(1).uniform_(color_min[0], color_max[0]).item()
    g = torch.empty(1).uniform_(color_min[1], color_max[1]).item()
    b = torch.empty(1).uniform_(color_min[2], color_max[2]).item()
    light.GetColorAttr().Set(Gf.Vec3f(r, g, b))


def randomize_light_properties(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    intensity_range: tuple[float, float] | None = None,
    color: tuple[float, float, float] | None = None,
) -> None:
    """Randomize a scene light's intensity (and optionally set a color) on reset.

    Generic counterpart of :func:`randomize_dome_light` for the ladder-family scene, which
    addresses its lights as named scene entities (``dome_light``, ``key_light``). Isaac Lab
    has no built-in light-randomization event term. Lights are single shared prims, so one
    global sample is applied regardless of ``env_ids``. Leave ``color`` unset when the light
    carries an HDRI ``texture_file`` -- the color attribute only tints the texture, which
    reads as a render bug rather than useful domain randomization.

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
