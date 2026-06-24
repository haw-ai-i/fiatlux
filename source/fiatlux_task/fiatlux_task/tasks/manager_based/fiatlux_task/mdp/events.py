# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-time events for the Fiatlux G1 bulb task.

Bulb and socket pose randomization use Isaac Lab's built-in
``reset_root_state_uniform``; only the dome-light randomization needs a custom
function because the light is a single shared USD prim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import omni.usd
import torch
from pxr import Gf, UsdLux

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
) -> None:
    """Randomize the dome light's intensity and color on reset.

    The light is a single shared prim, so the randomization is global across
    all environments regardless of ``env_ids``.
    """
    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath("/World/light")
    if not light_prim.IsValid():
        return
    light = UsdLux.DomeLight(light_prim)

    intensity = torch.empty(1).uniform_(intensity_range[0], intensity_range[1]).item()
    light.GetIntensityAttr().Set(intensity)

    color_min, color_max = color_range
    r = torch.empty(1).uniform_(color_min[0], color_max[0]).item()
    g = torch.empty(1).uniform_(color_min[1], color_max[1]).item()
    b = torch.empty(1).uniform_(color_min[2], color_max[2]).item()
    light.GetColorAttr().Set(Gf.Vec3f(r, g, b))
