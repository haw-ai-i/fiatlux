# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event (reset / randomization) terms for the Fiatlux ladder task family.

Built-in event terms (``reset_scene_to_default``, ``reset_root_state_uniform``,
``reset_joints_by_offset``, ``randomize_rigid_body_scale``, ``randomize_rigid_body_material``,
``randomize_visual_color`` ...) are re-exported through the package ``__init__`` and used directly.
This module only adds the **custom** term Isaac Lab does not ship: light randomization.
"""

from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedEnv


def randomize_light_properties(
    env: "ManagerBasedEnv",
    env_ids: "torch.Tensor | None",
    asset_cfg: SceneEntityCfg,
    intensity_range: tuple[float, float] | None = None,
    color: tuple[float, float, float] | None = None,
) -> None:
    """STUB -- randomize a light's intensity/color. Disabled by default (see ``EventCfg``).

    Isaac Lab has no built-in light-randomization event term, so this is a clearly-marked
    placeholder wired with the correct event-term signature. It does a best-effort write of the
    light's ``intensity``/``color`` USD attributes so enabling it will not crash.

    TODO(task phase):
        * sample per-env values and support per-env lights (the scene lights are currently global);
        * randomize light *orientation* (the distant-light direction) as well;
        * sample ``color`` from a range rather than taking a single value.
    """
    # NOTE: scene lights are global prims here, so this applies one sampled value to the prim.
    import random

    prim_path = env.scene[asset_cfg.name].cfg.prim_path
    prim = prim_utils.get_prim_at_path(prim_path)
    if not prim.IsValid():
        return
    if intensity_range is not None:
        lo, hi = intensity_range
        attr = prim.GetAttribute("inputs:intensity")
        if attr.IsValid():
            attr.Set(random.uniform(lo, hi))
    if color is not None:
        attr = prim.GetAttribute("inputs:color")
        if attr.IsValid():
            attr.Set(tuple(color))
