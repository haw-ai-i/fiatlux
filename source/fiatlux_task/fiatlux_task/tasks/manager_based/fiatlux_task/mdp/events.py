# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-time and prestartup events for the Fiatlux tasks.

Bulb and socket pose randomization use Isaac Lab's built-in
``reset_root_state_uniform``; the custom terms here cover what the built-ins
cannot: shared global prims (lights, the room), multiplicative scale on assets
with a baked spawn scale, and in-place material tinting that keeps the curated
MDL bindings.
"""

from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING

import torch

import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
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
    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    rotation_range_deg: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Randomize a scene light's intensity, color, and orientation on reset.

    The one light-randomization term for every Fiatlux scene: lights are addressed as named
    scene entities (``dome_light``, ``key_light``, ...), so both the Insert task and the
    ladder family use this with a ``SceneEntityCfg``. Isaac Lab has no built-in
    light-randomization event term. Lights are single shared prims, so one global sample is
    applied regardless of ``env_ids``. Leave ``color``/``color_range`` unset when the light
    carries an HDRI ``texture_file`` -- the color attribute only tints the texture, which
    reads as a render bug rather than useful domain randomization.

    Orientation (``rotation_range_deg``, keys ``roll``/``pitch``/``yaw`` [deg]) samples a
    perturbation composed onto the light's AUTHORED orientation (``init_state.rot`` from the
    scene cfg), so the cone is stable across resets instead of drifting. For the HDRI dome,
    randomize yaw only (sun azimuth); tilting the sky's horizon reads as a render bug. All
    three angles are always drawn (absent keys as (0, 0)) so the per-reset RNG draw count
    does not depend on the dict contents.

    TODO(task phase):
        * sample per-env values and support per-env lights (the scene lights are global prims).
    """
    # Read the prim path from the scene *cfg*: lights are plain AssetBaseCfg entities, and
    # the scene stores those as bare XFormPrim objects with no `.cfg` attribute at runtime.
    entity_cfg = getattr(env.scene.cfg, asset_cfg.name)
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(entity_cfg.prim_path)
    if not prim.IsValid():
        return
    light = UsdLux.LightAPI(prim)
    if intensity_range is not None:
        intensity = torch.empty(1).uniform_(intensity_range[0], intensity_range[1]).item()
        light.GetIntensityAttr().Set(intensity)
    if color_range is not None:
        sampled = [torch.empty(1).uniform_(lo, hi).item() for lo, hi in color_range]
        light.GetColorAttr().Set(Gf.Vec3f(*sampled))
    elif color is not None:
        light.GetColorAttr().Set(Gf.Vec3f(*color))
    if rotation_range_deg is not None:
        # Constant draw count: always sample all three angles (absent keys are (0, 0)).
        angles_rad = torch.stack(
            [
                torch.empty(1).uniform_(*rotation_range_deg.get(key, (0.0, 0.0))).deg2rad()
                for key in ("roll", "pitch", "yaw")
            ]
        ).squeeze(-1)
        q_delta = math_utils.quat_from_euler_xyz(angles_rad[0:1], angles_rad[1:2], angles_rad[2:3])
        q_authored = torch.tensor([entity_cfg.init_state.rot], dtype=torch.float32)
        q_new = math_utils.quat_mul(q_delta, q_authored)[0].tolist()
        orient_attr = prim.GetAttribute("xformOp:orient")
        current = orient_attr.Get() if orient_attr else None
        if current is None:
            return
        orient_attr.Set(type(current)(q_new[0], q_new[1], q_new[2], q_new[3]))


# Hand collider shapes on both G1 variants: Inspire (``left_hand_base_link``,
# ``L_index_proximal``, ``R_thumb_distal``, ...) and Dex3 (``left_hand_palm_link``,
# ``right_hand_thumb_2_link``, ...). Verified against both USDs -- one expression matches every
# hand body on either and nothing else, so ``swap_robot_variant`` has nothing to remap here.
G1_HAND_BODY_EXPR = "(left|right)_hand.*|[LR]_(index|middle|pinky|ring|thumb).*"


# Nominal grip friction (static, dynamic) and the half-width the randomized band spans
# around it. One source for both, so pinning the term cannot drift outside its own range.
HAND_GRIP_FRICTION = (1.0, 0.9)
HAND_GRIP_FRICTION_SPREAD = 0.2


def hand_grip_material_event(randomize: bool = True) -> EventTermCfg:
    """Grip friction for the G1's hands, as a startup event term.

    Every grasp in the benchmark is made with these shapes, and without this they run on the
    PhysX default 0.5/0.5 -- bare steel on glass, which is not what a robot hand is.

    Deliberately an EVENT rather than a material bound in the robot spawner:
    ``bind_physics_material`` is ``apply_nested``-decorated and ``apply_nested`` SKIPS
    INSTANCED PRIMS, while every G1 link's ``collisions`` child is authored instanceable. The
    USD route therefore binds nothing and leaves the whole robot at 0.5/0.5 while reading as
    applied. This term writes through the PhysX view, which has no notion of instancing.

    Range: NVIDIA's own manipulation environments bracket it -- Factory/AutoMate fix 1.0/1.0
    for insertion, Dexsuite randomizes the hand over [0.5, 1.0]. Centered on
    ``HAND_GRIP_FRICTION``, with a band wide enough to be real domain randomization. Returns
    a fresh cfg per call so each task's ``EventCfg`` owns its own instance.

    Args:
        randomize: when False the band collapses onto ``HAND_GRIP_FRICTION``. The term stays
            -- the hands keep grip friction instead of falling back to the PhysX 0.5/0.5
            default -- but every run measures the same contact. This is what
            ``disable_randomization`` and the interaction scenarios want; anything that
            measures a grasp force is otherwise seed-dependent.
    """
    static, dynamic = HAND_GRIP_FRICTION
    spread = HAND_GRIP_FRICTION_SPREAD if randomize else 0.0
    return EventTermCfg(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_HAND_BODY_EXPR),
            "static_friction_range": (static - spread, static + spread),
            "dynamic_friction_range": (dynamic - spread, dynamic + spread),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64 if randomize else 1,
            "make_consistent": True,
        },
    )


def randomize_prop_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
) -> None:
    """Randomize a prop's scale MULTIPLICATIVELY on top of its authored spawn scale.

    The multiplicative sibling of Isaac Lab's :func:`~isaaclab.envs.mdp.events.randomize_rigid_body_scale`,
    which overwrites ``xformOp:scale`` absolutely and would blow up assets that carry a
    baked spawn scale (the cm-authored SimReady ladder and crate spawn at 0.01). Ranges are
    factors around 1.0; the sampled factor multiplies the authored ``spawn.scale`` (or 1.0
    when unset). Works for ``AssetBaseCfg`` entities too (path is read from the scene cfg,
    not the runtime physics asset).

    Same contract as the built-in: ``mode="prestartup"`` only, which requires
    ``InteractiveSceneCfg.replicate_physics=False`` (the event manager enforces this). One
    independent sample per env prim. RL cfgs keep replicated physics, so this term is a
    scaffold default and a documented RL opt-in.
    """
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing prop scale while simulation is running leads to unpredictable behaviors."
            " Please use the event term with 'prestartup' mode."
        )
    entity_cfg = getattr(env.scene.cfg, asset_cfg.name)
    base_scale = getattr(entity_cfg.spawn, "scale", None) or (1.0, 1.0, 1.0)

    prim_path_expr = entity_cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
    if not prim_paths:
        return

    # sample one factor triplet per prim (dict form = per-axis ranges, tuple = isotropic)
    if isinstance(scale_range, dict):
        range_list = [scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(prim_paths), 3), device="cpu")
    else:
        rand_samples = math_utils.sample_uniform(*scale_range, (len(prim_paths), 1), device="cpu")
        rand_samples = rand_samples.repeat(1, 3)
    rand_samples = (rand_samples * torch.tensor(base_scale, device="cpu")).tolist()

    # write through Sdf for speed, creating the scale op if the prim lacks one
    # (same mechanics as the built-in term).
    stage = omni.usd.get_context().get_stage()
    with Sdf.ChangeBlock():
        for prim_path, sample in zip(prim_paths, rand_samples):
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)
            scale_spec.default = Gf.Vec3f(*sample)
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])


_TINT_INPUT_NAMES = ("diffuse_tint", "diffuse_color_constant", "diffuseColor")
"""Shader/material color inputs the tint term probes, most-specific first.

``diffuse_tint``/``diffuse_color_constant`` cover OmniPBR and the SimReady MDL families
(ladder, packing table); ``diffuseColor`` covers ``UsdPreviewSurface``. B1K's
``OmniGibsonVRayMtl`` authors none of these, so B1K assets are naturally skipped even if
targeted (and they are deliberately not targeted -- the team keeps those materials original).
"""

_BASE_COLOR_KEY = "fiatlux:base_color"


def randomize_material_tint(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfgs: list[SceneEntityCfg],
    hue_range_deg: tuple[float, float] = (0.0, 360.0),
    saturation_range: tuple[float, float] = (0.0, 0.15),
    value_range: tuple[float, float] = (0.7, 1.2),
) -> None:
    """Tint the albedo of existing bound materials in place (HSV multiplier), on reset.

    Visual domain randomization without Replicator: Isaac Lab 2.3.2's built-in
    ``randomize_visual_color``/``randomize_visual_texture_material`` rebind an OmniPBR
    material over whatever is authored and require ``replicate_physics=False``, so they
    cannot touch the shared ``/World/Room`` under replicated-physics training cfgs and
    would replace the curated SimReady/B1K materials. This term instead multiplies a
    sampled HSV tint into the color inputs the materials already author
    (:data:`_TINT_INPUT_NAMES`, written at both Material and Shader level when both are
    authored -- Material-level inputs override shader params). ``diffuse_tint`` is created
    when missing, but only on shaders positively identified as OmniPBR (the parameter
    exists there with default (1, 1, 1)); every other MDL family is probe-by-authored-name.

    One global sample per reset (``env_ids`` accepted, one stage-level pass); the sample is
    drawn BEFORE any prim lookups so the RNG stream does not depend on which optional
    entities a preset keeps. Entities absent from the scene cfg (or set ``None`` by a
    preset) are skipped silently -- pass the cfgs as a plain list, which the event manager
    does not auto-resolve. The original color is cached in attribute customData on first
    touch, and every reset recomputes ``original * tint``, so repeated resets never
    compound. Texture swapping is out of scope (no texture assets exist).
    """
    # Draw first: fixed 3-draw cost per reset regardless of preset/entity availability.
    hue = torch.empty(1).uniform_(*hue_range_deg).item()
    saturation = torch.empty(1).uniform_(*saturation_range).item()
    value = torch.empty(1).uniform_(*value_range).item()
    tint = Gf.Vec3f(*colorsys.hsv_to_rgb((hue % 360.0) / 360.0, min(max(saturation, 0.0), 1.0), max(value, 0.0)))

    stage = omni.usd.get_context().get_stage()
    for asset_cfg in asset_cfgs:
        entity_cfg = getattr(env.scene.cfg, asset_cfg.name, None)
        if entity_cfg is None:
            continue
        prim_path_expr = entity_cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        for prim_path in sim_utils.find_matching_prim_paths(prim_path_expr):
            root = stage.GetPrimAtPath(prim_path)
            if not root.IsValid():
                continue
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if prim.IsInstanceProxy() or not prim.IsA(UsdShade.Material):
                    continue
                _tint_material(prim, tint)


def _tint_material(material_prim: Usd.Prim, tint: Gf.Vec3f) -> None:
    """Apply ``tint`` to every recognized color input of one material (and its shaders)."""
    targets = []
    for prim in Usd.PrimRange(material_prim):
        if prim != material_prim and not prim.IsA(UsdShade.Shader):
            continue
        authored = [prim.GetAttribute(f"inputs:{name}") for name in _TINT_INPUT_NAMES]
        authored = [attr for attr in authored if attr and attr.HasAuthoredValue()]
        if not authored and prim.IsA(UsdShade.Shader) and _is_omnipbr(prim):
            # OmniPBR always exposes diffuse_tint (default white); create it so
            # texture-only materials (the Simple Room walls) participate in the tint.
            shader_input = UsdShade.Shader(prim).CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f)
            authored = [shader_input.GetAttr()]
        targets.extend(authored)
    for attr in targets:
        base = attr.GetCustomDataByKey(_BASE_COLOR_KEY)
        if base is None:
            current = attr.Get()
            base = tuple(current) if current is not None else (1.0, 1.0, 1.0)
            attr.SetCustomDataByKey(_BASE_COLOR_KEY, Gf.Vec3f(*base))
        attr.Set(Gf.Vec3f(base[0] * tint[0], base[1] * tint[1], base[2] * tint[2]))


def _is_omnipbr(shader_prim: Usd.Prim) -> bool:
    """True if the shader's MDL source sub-identifier is exactly ``OmniPBR``."""
    shader = UsdShade.Shader(shader_prim)
    if shader.GetImplementationSource() != UsdShade.Tokens.sourceAsset:
        return False
    sub_identifier = shader.GetSourceAssetSubIdentifier("mdl")
    return sub_identifier == "OmniPBR"
