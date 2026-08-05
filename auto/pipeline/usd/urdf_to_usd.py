#!/usr/bin/env python3
"""SimArt URDF (+ prediction JSON) -> Isaac-Sim-ready USD.  Pure ``pxr``, no GPU.

This is the piece that exists nowhere else.  Hunyuan3D-2.1 makes a mesh, SimArt
decomposes it into parts and writes a URDF -- and then the trail stops: SimArt
has no USD code at all (a repo-wide grep for ``pxr|Usd|UsdPhysics`` returns
nothing), and Isaac Sim's own URDF importer cannot run on this box because an
A100 has no RT cores.  So the stage is authored by hand.

Conventions follow the ones the surrounding fiatlux benchmark already uses --
see ``scripts/omniverse/omniverse_ladder_collision.py`` (every mesh gets a
convex-decomposition collider bound to a high-friction physics material) and
``scripts/omniverse/omniverse_bulb_rigid.py`` (Z-up, metres, verified extents).

Three things here are subtle enough to be worth stating up front.

**Where the numbers come from.** Topology and part-mesh paths come from the
URDF; ``scale``/``density``/raw limits come from the prediction JSON.  Both are
needed: the URDF drops density and scale entirely, and its revolute limits use a
conversion that disagrees with the model's own prompt (see ``LIMIT_*`` below).

**Part meshes are in object-GLOBAL coordinates.** ``trimesh.submesh`` only
re-indexes vertices, so every ``<pid>.obj`` shares one frame.  SimArt compensates
with ``<joint origin="+center">`` plus ``<visual origin="-center">``, which
cancels only when the parent transform is identity -- i.e. only at tree depth 1.
We instead place each link's frame at its own *absolute* anchor and offset that
link's points by ``-anchor``, which is correct at any depth and silently fixes
SimArt's nested-part bug.

**The part-OBJ frame is already Z-up, so the default is no rotation.**  This one
is easy to get backwards.  SimArt's README asks for "+Z up" input, but its own
reference assets are not: ``chair_00.glb`` and ``fridge_00.glb`` both carry their
height along **Y**, i.e. glTF's usual Y-up.  ``inference/infer.py:428`` then
applies ``R_x(+90)`` to that input before segmenting (matching
``utils/mesh_utils.py:126``), which maps +Y onto +Z -- so the frame the part
OBJs, joint anchors and axes live in is **Z-up already**.  Hunyuan3D likewise
exports Y-up glTF, so the whole chain lines up and ``--frame-rot none`` is
correct.  ``undo-simart`` is kept for input that really was Z-up before SimArt
saw it.  Getting this wrong lays the object on its side with joint axes to
match, so ``verify_usd.py`` re-checks the extent against a plausible range.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict

import numpy as np
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

# --------------------------------------------------------------------------
# House constants (verbatim from scripts/omniverse/omniverse_ladder_collision.py)
STATIC_FRICTION, DYNAMIC_FRICTION, RESTITUTION = 1.2, 1.0, 0.0

# R_x(-90): (x, y, z) -> (x, z, -y).  Only used by --frame-rot undo-simart, for
# input that was genuinely Z-up before SimArt applied its own R_x(+90).  For the
# normal Y-up glTF chain the part OBJs are already Z-up -- see the module
# docstring -- and the default --frame-rot none applies no rotation at all.
R_X_NEG90 = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0]])

# utils/urdf_utils.py:441 computes (val/100)*2*pi radians, but the model's own
# system prompt at inference/infer.py:101 says "for 'revolute', 100 represents
# 180 degrees".  That is a factor of exactly 2, and the prismatic branch right
# below it uses a bare val/100 with no 2*pi -- so it reads as a slip, not a
# second convention.  Default to the prompt; `urdf` reproduces SimArt for parity.
LIMIT_DEG_PER_UNIT = {"spec": 180.0 / 100.0, "urdf": 360.0 / 100.0}

DENSITY_DEFAULT = 1500.0            # kg/m^3, a generic moulded-plastic/composite
DENSITY_CLAMP = (20.0, 12000.0)     # styrofoam .. denser than lead
SCALE_CLAMP = (0.02, 10.0)          # metres, largest bbox extent

_UNITS = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
          "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
          "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
          "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
          "ft": 0.3048, "foot": 0.3048, "feet": 0.3048}

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

_FRAME_ROT = {
    "none": np.eye(3),                                                  # default
    "undo-simart": R_X_NEG90,
    "rx-90": R_X_NEG90,
    "rx90": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
}


def _v3f(a):
    """Gf.Vec3f from anything array-like.

    Gf's Boost bindings accept Python floats but not numpy scalars, so the
    explicit float() conversion is required, not cosmetic.
    """
    return Gf.Vec3f(float(a[0]), float(a[1]), float(a[2]))


def _v3d(a):
    return Gf.Vec3d(float(a[0]), float(a[1]), float(a[2]))


# ==========================================================================
# Report
# ==========================================================================
@dataclass
class LinkInfo:
    pid: str
    prim: str
    n_points: int
    n_faces: int
    density: float
    volume_m3: float
    mass_kg: float
    bbox_min: list
    bbox_max: list
    material: str = ""


@dataclass
class JointInfo:
    name: str
    type: str
    parent: str
    child: str
    axis_world: list
    usd_axis: str
    lower: float
    upper: float
    units: str
    lower_urdf_convention: float = 0.0
    upper_urdf_convention: float = 0.0


@dataclass
class Report:
    output: str = ""
    scale_m: float = 0.0
    scale_provenance: str = ""
    frame_rot: str = ""
    floor_offset: list = field(default_factory=list)
    bbox_world_min: list = field(default_factory=list)
    bbox_world_max: list = field(default_factory=list)
    total_mass_kg: float = 0.0
    links: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    textures: dict = field(default_factory=dict)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [WARN] {msg}", file=sys.stderr)


# ==========================================================================
# Parsing helpers
# ==========================================================================
def parse_scale_m(raw, report: Report, unit="auto"):
    """Turn the MLLM's free-text ``object_captions.scale`` into metres.

    The geometry SimArt segments is normalised to a max bbox extent of exactly
    1.0, so this value *is* the scale factor -- no ratio needed.

    The catch: the system prompt (``inference/infer.py:103``) only says
    "'object_captions' must include 'name' and 'scale'" and never states a unit,
    so the model emits bare numbers -- ``"scale": 30`` for a cardboard box.
    Interpreting that as 30 m would give a six-storey box.  With no unit token
    present we therefore apply a magnitude heuristic: <= 10 reads as metres
    (a "1.8" ladder), 10..500 as centimetres (a "30" box), above that as
    millimetres.  It is a heuristic, so the choice is always logged, and
    ``--scale-m`` / ``--scale-unit`` override it outright.
    """
    if raw is None:
        return None, "absent"
    s = str(raw).strip().lower()
    nums = [float(x) for x in re.findall(_NUM, s)]
    if not nums:
        return None, f"unparseable {raw!r}"
    v = max(nums)                            # "1.8 x 0.6 x 0.5 m" -> the max extent
    tok_found = None
    for tok in sorted(_UNITS, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(tok)}(?![a-z])", s):
            tok_found = tok
            break

    if unit != "auto":
        mul, how = _UNITS[unit], f"--scale-unit {unit}"
    elif tok_found:
        mul, how = _UNITS[tok_found], f"unit '{tok_found}' in the string"
    elif v <= 10.0:
        mul, how = 1.0, "bare number <= 10 read as metres"
    elif v <= 500.0:
        mul, how = 0.01, "bare number in (10, 500] read as centimetres"
    else:
        mul, how = 0.001, "bare number > 500 read as millimetres"

    v *= mul
    if not (SCALE_CLAMP[0] <= v <= SCALE_CLAMP[1]):
        return None, f"out of range: {v:.4g} m from {raw!r} ({how})"
    return v, f"{raw!r} -> {v:.4g} m ({how})"


def parse_density(raw, report: Report, pid: str):
    """Parse a free-text density into kg/m^3."""
    if raw is None:
        return DENSITY_DEFAULT
    s = str(raw).strip().lower()
    nums = re.findall(_NUM, s)
    if not nums:
        report.warn(f"part {pid}: unparseable density {raw!r} -> {DENSITY_DEFAULT} kg/m^3")
        return DENSITY_DEFAULT
    v = float(nums[0])
    if "g/cm" in s or "g / cm" in s:
        v *= 1000.0
    elif "kg/m" not in s and 0.05 <= v <= 25.0:
        v *= 1000.0                          # a bare small number is g/cm^3
    lo, hi = DENSITY_CLAMP
    if not (lo <= v <= hi):
        report.warn(f"part {pid}: density {v:.4g} clamped into [{lo}, {hi}] kg/m^3")
        v = float(np.clip(v, lo, hi))
    return float(v)


def _floats(s, n=3, default=(0.0, 0.0, 0.0)):
    try:
        v = [float(x) for x in str(s).split()]
        return v[:n] if len(v) >= n else list(default)
    except Exception:
        return list(default)


# ==========================================================================
# OBJ reading
# ==========================================================================
def read_obj(path):
    """Minimal Wavefront reader: ``v`` / ``vt`` / ``f`` only.

    Returns ``(P, VT, PF, TF)``.  ``PF``/``TF`` are the position and texcoord
    index arrays, kept separate because Hunyuan3D's own exporter genuinely uses
    independent tables (``mesh_utils.py:109-113``), while trimesh's part exports
    use matching ones.  n-gons are fan-triangulated.
    """
    V, T, PF, TF = [], [], [], []
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("v "):
                V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith("vt "):
                p = ln.split()
                T.append([float(p[1]), float(p[2]) if len(p) > 2 else 0.0])
            elif ln.startswith("f "):
                pi, ti = [], []
                for tok in ln.split()[1:]:
                    f = tok.split("/")
                    pi.append(int(f[0]))
                    ti.append(int(f[1]) if len(f) > 1 and f[1] else 0)
                for k in range(1, len(pi) - 1):
                    PF.append([pi[0], pi[k], pi[k + 1]])
                    TF.append([ti[0], ti[k], ti[k + 1]])
    if not V or not PF:
        return None, None, None, None
    P = np.asarray(V, np.float64)
    VT = np.asarray(T, np.float64) if T else None

    def rebase(a, n):                        # OBJ is 1-based; negatives are relative
        a = np.asarray(a, np.int64)
        return np.where(a < 0, n + a, a - 1)

    PF = rebase(PF, len(P))
    TF = rebase(TF, len(VT)) if VT is not None else None
    return P, VT, PF, TF


def sanitize(P, VT, PF, TF):
    """Drop non-finite vertices, degenerate faces and out-of-range indices."""
    if P is None:
        return None, None, None, None
    finite = np.isfinite(P).all(axis=1)
    ok = finite[PF].all(axis=1) & (PF >= 0).all(axis=1) & (PF < len(P)).all(axis=1)
    ok &= (PF[:, 0] != PF[:, 1]) & (PF[:, 1] != PF[:, 2]) & (PF[:, 0] != PF[:, 2])
    if TF is not None and VT is not None:
        ok &= (TF >= 0).all(axis=1) & (TF < len(VT)).all(axis=1)
    PF = PF[ok]
    TF = TF[ok] if TF is not None else None
    if len(PF) < 4:
        return None, None, None, None
    return P, VT, PF, TF


def vertex_normals(P, PF):
    """Area-weighted vertex normals (the part OBJs carry no ``vn``)."""
    tri = P[PF]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])   # unnormalised == area-weighted
    n = np.zeros_like(P)
    for k in range(3):
        np.add.at(n, PF[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln < 1e-20] = 1.0
    return n / ln


def mesh_volume(P, PF):
    """Signed tetrahedron sum; exact for a closed mesh, indicative otherwise."""
    a, b, c = P[PF[:, 0]], P[PF[:, 1]], P[PF[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


# ==========================================================================
# USD authoring helpers
# ==========================================================================
def author_physics_material(stage, root_path):
    """High-friction physics material, verbatim the fiatlux values."""
    scope = root_path.AppendChild("PhysicsMaterials")
    UsdGeom.Scope.Define(stage, scope)
    mat = UsdShade.Material.Define(stage, scope.AppendChild("HighFriction"))
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(STATIC_FRICTION)
    api.CreateDynamicFrictionAttr(DYNAMIC_FRICTION)
    api.CreateRestitutionAttr(RESTITUTION)
    return mat


def author_pbr_material(stage, root_path, textures, report: Report):
    """UsdPreviewSurface + UsdUVTexture + UsdPrimvarReader_float2.

    One material shared by every part.  SimArt's per-part OBJ export runs
    ``PBRMaterial.to_simple()``, which throws metallic/roughness away and
    duplicates the albedo atlas into each part dir -- so those .mtl files are
    ignored and Hunyuan3D's original maps are used instead.
    """
    looks = root_path.AppendChild("Looks")
    UsdGeom.Scope.Define(stage, looks)
    mat = UsdShade.Material.Define(stage, looks.AppendChild("SimArtPBR"))
    surf = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("PreviewSurface"))
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")

    if not textures:
        # Never leave a mesh unbound: UsdUtils' MaterialBindingAPIAppliedChecker
        # flags it and Isaac renders default grey.
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.72, 0.72, 0.74))
        surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        return mat

    reader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("stReader"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def tex(name, rel, channel, colorspace):
        t = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild(name))
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(rel))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        # Getting this wrong is the classic washed-out/chalky asset: albedo is
        # sRGB, the data maps are raw.
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
        vt = Sdf.ValueTypeNames.Float3 if channel == "rgb" else Sdf.ValueTypeNames.Float
        return t.CreateOutput(channel, vt)

    if "albedo" in textures:
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex("diffuseTex", textures["albedo"], "rgb", "sRGB"))
    # Hunyuan writes these as grayscale JPEGs (cv2.COLOR_RGB2GRAY), so read `r`.
    if "metallic" in textures:
        surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
            tex("metallicTex", textures["metallic"], "r", "raw"))
    else:
        surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if "roughness" in textures:
        surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
            tex("roughnessTex", textures["roughness"], "r", "raw"))
    else:
        surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    if "normal" in textures:
        nt = tex("normalTex", textures["normal"], "rgb", "raw")
        nsh = stage.GetPrimAtPath(mat.GetPath().AppendChild("normalTex"))
        UsdShade.Shader(nsh).CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        UsdShade.Shader(nsh).CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        surf.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(nt)
    return mat


def author_mesh(stage, path, P, VT, PF, TF, pbr_mat, phys_mat, approximation, flip_v=False):
    """Author one UsdGeom.Mesh with geometry, UVs, collider and bindings."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(PF.reshape(-1).astype(np.int32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(PF), 3, np.int32)))
    # Mandatory, not cosmetic: the default catmullClark makes Hydra render a
    # smoothed limit surface that no longer matches what PhysX cooked.
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(Vt.Vec3fArray([_v3f(P.min(0)), _v3f(P.max(0))]))

    n = vertex_normals(P, PF)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    if VT is not None and TF is not None and len(VT):
        uv = VT.copy()
        if flip_v:
            uv[:, 1] = 1.0 - uv[:, 1]
        pv = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        if np.array_equal(TF, PF) and len(VT) == len(P):
            st = pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            st.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
        else:
            # faceVarying + indices is the only encoding that survives UV seams
            # without duplicating positions.
            st = pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            st.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
            st.SetIndices(Vt.IntArray.FromNumpy(TF.reshape(-1).astype(np.int32)))

    prim = mesh.GetPrim()
    if approximation != "none":
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    if pbr_mat:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(pbr_mat)
    return mesh


def usd_axis_and_rot(axis_world):
    """Encode an arbitrary axis as a USD axis token plus a joint-frame rotation.

    ``physics:axis`` only accepts "X"/"Y"/"Z", so the direction is carried by
    baking into localRot0/localRot1 the rotation taking that canonical axis onto
    the real one.  Choosing the *dominant* component keeps that rotation <= 45
    deg, which puts the antiparallel degeneracy of Gf.Rotation out of reach and
    leaves localRot ~= identity for already-aligned hinges.
    """
    a = np.asarray(axis_world, float)
    n = float(np.linalg.norm(a))
    degenerate = not np.isfinite(n) or n < 1e-9
    if degenerate:
        a, n = np.array([0.0, 0.0, 1.0]), 1.0
    a = a / n
    k = int(np.argmax(np.abs(a)))

    # Flip to a non-negative dominant component before building the rotation.
    # Without this, an axis like (0,0,-1) -- which arises routinely, since the
    # frame rotation maps SimArt's +Y onto -Z -- lands exactly antiparallel to
    # its canonical axis, and Gf.Rotation then picks an arbitrary perpendicular
    # for a 180 deg turn.  A hinge is a line, not a ray, so flipping costs
    # nothing physically; the caller negates and swaps the limits to keep the
    # signed range identical.
    flipped = a[k] < 0.0
    if flipped:
        a = -a

    ref = np.zeros(3)
    ref[k] = 1.0
    rot = Gf.Rotation(_v3d(ref), _v3d(a))    # now always <= 45 deg
    return ("X", "Y", "Z")[k], Gf.Quatf(rot.GetQuat()), a, degenerate, flipped


# ==========================================================================
# URDF / JSON loading
# ==========================================================================
def load_urdf(path):
    """Topology, part-mesh paths, joint anchors/axes -- straight from the XML."""
    root = ET.parse(path).getroot()
    base_dir = os.path.dirname(os.path.abspath(path))
    links, joints = {}, {}
    for le in root.findall("link"):
        name = le.get("name")
        mesh_file = None
        for blk in ("visual", "collision"):
            el = le.find(f"{blk}/geometry/mesh")
            if el is not None and el.get("filename"):
                mesh_file = el.get("filename")
                break
        if mesh_file:
            cand = mesh_file if os.path.isabs(mesh_file) else os.path.join(base_dir, mesh_file)
            if not os.path.isfile(cand):     # infer.py writes paths relative to its cwd
                cand = os.path.join(base_dir, os.path.basename(os.path.dirname(mesh_file)),
                                    os.path.basename(mesh_file))
        else:
            cand = None
        links[name] = {"mesh": cand, "raw_mesh": mesh_file}
    for je in root.findall("joint"):
        p = je.find("parent")
        c = je.find("child")
        if p is None or c is None:
            continue
        o = je.find("origin")
        a = je.find("axis")
        joints[c.get("link")] = {
            "name": je.get("name"),
            "type": (je.get("type") or "fixed").lower(),
            "parent": p.get("link"),
            "child": c.get("link"),
            "origin": _floats(o.get("xyz") if o is not None else None),
            "axis": _floats(a.get("xyz") if a is not None else None, default=(0.0, 0.0, 1.0)),
        }
    return links, joints, root.get("name") or "object"


def load_prediction(path):
    """The MLLM payload.  ``infer.py:444-456`` wraps it, so unwrap if needed."""
    with open(path) as fh:
        doc = json.load(fh)
    out = doc.get("output", doc)
    if isinstance(out, str):
        out = json.loads(out)
    return out or {}


def pid_of(link_name):
    return link_name[2:] if link_name.startswith("l_") else link_name


def resolve_graph(links, joints, report: Report):
    """Make the parent graph a well-formed tree: no cycles, no dangling parents."""
    present = set(links)
    for child, j in list(joints.items()):
        if j["parent"] not in present:
            report.warn(f"joint {j['name']}: parent {j['parent']} not present -> reparenting to base")
            j["parent"] = None
        elif j["parent"] == child:
            report.warn(f"joint {j['name']}: self-parent -> reparenting to base")
            j["parent"] = None

    roots = [n for n in links if n not in joints or joints[n]["parent"] is None]
    base = None
    if "l_0" in links:
        base = "l_0"
    elif roots:
        base = sorted(roots)[0]
    else:
        base = sorted(links)[0]
    for child, j in joints.items():
        if j["parent"] is None:
            j["parent"] = base

    # Break cycles by reparenting the offending edge to the base.
    for start in list(links):
        seen, cur = set(), start
        while cur in joints:
            if cur in seen:
                report.warn(f"cycle detected at {cur} -> reparenting to {base}")
                joints[cur]["parent"] = base
                break
            seen.add(cur)
            cur = joints[cur]["parent"]
            if cur == base or cur not in links:
                break
    joints.pop(base, None)
    return base


# ==========================================================================
# Main conversion
# ==========================================================================
def convert(args) -> Report:
    report = Report(output=args.out, frame_rot=args.frame_rot)
    links, joints, robot_name = load_urdf(args.urdf)
    pred = load_prediction(args.json) if args.json and os.path.isfile(args.json) else {}
    caps = pred.get("parts_captions", {}) or {}
    obj_caps = pred.get("object_captions", {}) or {}

    # ---- load part geometry, dropping empties and contracting the graph -----
    geo = {}
    for name, meta in list(links.items()):
        path = meta["mesh"]
        if not path or not os.path.isfile(path):
            report.warn(f"{name}: part mesh missing ({meta['raw_mesh']}) -> link dropped")
            links.pop(name)
            continue
        P, VT, PF, TF = sanitize(*read_obj(path))
        if P is None:
            report.warn(f"{name}: part mesh empty/degenerate -> link dropped")
            links.pop(name)
            continue
        geo[name] = (P, VT, PF, TF)

    if not geo:
        raise SystemExit("ERROR: no usable part meshes; nothing to convert")

    # Contract the graph across dropped links so children are not orphaned.
    for child, j in list(joints.items()):
        if child not in links:
            joints.pop(child)
            continue
        hops = 0
        while j["parent"] not in links and j["parent"] in joints and hops < 64:
            j["parent"] = joints[j["parent"]]["parent"]
            hops += 1
    base = resolve_graph(links, joints, report)

    # ---- frame, scale, floor alignment -------------------------------------
    R = _FRAME_ROT[args.frame_rot]

    allP = np.vstack([g[0] for g in geo.values()])
    extent = float(np.max(allP.max(0) - allP.min(0)))
    centre = (allP.max(0) + allP.min(0)) / 2.0
    renorm_shift, renorm_div = np.zeros(3), 1.0
    if not (0.99 <= extent <= 1.01) or np.abs(centre).max() > 0.01:
        # infer.py:427 segments the ORIGINAL glb, not the _scaled one it wrote.
        # If someone fed it an unnormalised mesh, the JSON anchors (which live in
        # [-0.5, 0.5]) and these vertices are in different spaces.
        report.warn(f"geometry not unit-normalised (extent={extent:.4f}, centre={centre.round(4).tolist()}) "
                    "-> renormalising; check that stage 2 ran")
        renorm_shift, renorm_div = centre, extent

    scale_m, prov = parse_scale_m(obj_caps.get("scale"), report, args.scale_unit)
    if args.scale_m:
        scale_m, prov = args.scale_m, f"--scale-m {args.scale_m}"
    elif scale_m is None:
        scale_m, prov = 1.0, f"fallback 1.0 m ({prov})"
        report.warn(f"could not determine real-world scale: {prov}")
    report.scale_m, report.scale_provenance = float(scale_m), prov

    def to_world(v):
        """Normalised SimArt frame -> metres, Z-up (positions)."""
        v = (np.asarray(v, float) - renorm_shift) / renorm_div
        return (v @ R.T) * scale_m

    def dir_to_world(v):
        """Directions: rotation only, no translation or scale."""
        return np.asarray(v, float) @ R.T

    worldP = {n: to_world(geo[n][0]) for n in geo}
    stacked = np.vstack(list(worldP.values()))
    if args.floor_align:
        t_align = np.array([-(stacked[:, 0].max() + stacked[:, 0].min()) / 2.0,
                            -(stacked[:, 1].max() + stacked[:, 1].min()) / 2.0,
                            -stacked[:, 2].min()])
    else:
        t_align = np.zeros(3)
    for n in worldP:
        worldP[n] = worldP[n] + t_align
    stacked = np.vstack(list(worldP.values()))
    report.floor_offset = t_align.round(6).tolist()
    report.bbox_world_min = stacked.min(0).round(6).tolist()
    report.bbox_world_max = stacked.max(0).round(6).tolist()

    # Link frame origins: the joint's ABSOLUTE anchor (see module docstring).
    origins = {n: np.zeros(3) for n in links}
    for child, j in joints.items():
        origins[child] = to_world(j["origin"]) + t_align

    # ---- stage -------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)
    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root_name = Tf.MakeValidIdentifier(args.name or robot_name)
    root = UsdGeom.Xform.Define(stage, Sdf.Path(f"/{root_name}"))
    root_path = root.GetPath()
    stage.SetDefaultPrim(root.GetPrim())

    textures = stage_textures(args, out_dir, report)
    pbr = author_pbr_material(stage, root_path, textures, report)
    phys = author_physics_material(stage, root_path)

    single = len(geo) == 1
    if single:
        report.warn("articulation prediction has a single part -- emitting a plain rigid body")
    else:
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    # ---- links -------------------------------------------------------------
    # --total-mass: the predicted per-part density is often the weakest number in
    # the JSON (the ladder comes back "Metal" at 7800 kg/m^3, i.e. solid steel,
    # for a ~13 kg fibreglass object).  Real-world mass is something the user
    # usually does know, so allow pinning it and back out a uniform density.
    uniform_rho = None
    if args.total_mass:
        vols = {n: mesh_volume(worldP[n] - origins[n], geo[n][2]) for n in links}
        tot = sum(vols.values())
        if tot > 1e-9:
            uniform_rho = float(np.clip(args.total_mass / tot, *DENSITY_CLAMP))
            print(f"  --total-mass {args.total_mass} kg over {tot:.5f} m^3 "
                  f"-> uniform density {uniform_rho:.1f} kg/m^3")
        else:
            report.warn("--total-mass ignored: total mesh volume is ~0")

    prim_of = {}
    for name in sorted(links, key=lambda n: (n != base, n)):
        P, VT, PF, TF = geo[name]
        pid = pid_of(name)
        cap = caps.get(str(pid), {}) if isinstance(caps, dict) else {}
        pts = worldP[name] - origins[name]

        lp = root_path.AppendChild(Tf.MakeValidIdentifier(name))
        link = UsdGeom.Xform.Define(stage, lp)
        if np.abs(origins[name]).max() > 0:
            link.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(_v3d(origins[name]))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())

        rho = uniform_rho if uniform_rho else parse_density(cap.get("density"), report, pid)
        # Author density, not mass: PhysX then derives BOTH mass and a correct
        # inertia tensor from the cooked convex decomposition.  Segmented parts
        # are open shells whose analytic volume is unreliable, and a wrong
        # authored inertia is far worse than a computed one.
        UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateDensityAttr(rho)

        author_mesh(stage, lp.AppendChild("mesh"), pts, VT, PF, TF, pbr, phys, args.collision)
        prim_of[name] = lp

        vol = mesh_volume(pts, PF)
        report.links.append(LinkInfo(
            pid=str(pid), prim=str(lp), n_points=len(pts), n_faces=len(PF),
            density=rho, volume_m3=round(vol, 8), mass_kg=round(rho * vol, 5),
            bbox_min=pts.min(0).round(5).tolist(), bbox_max=pts.max(0).round(5).tolist(),
            material=str(cap.get("material", ""))))
    report.total_mass_kg = round(sum(l.mass_kg for l in report.links), 4)

    # ---- joints ------------------------------------------------------------
    if not single:
        jscope = root_path.AppendChild("Joints")
        UsdGeom.Scope.Define(stage, jscope)
        for child in sorted(joints, key=lambda c: c):
            if child not in prim_of:
                continue
            j = joints[child]
            parent = j["parent"]
            if parent not in prim_of:
                continue
            pid = pid_of(child)
            cap = caps.get(str(pid), {}) if isinstance(caps, dict) else {}
            jtype = j["type"]
            if jtype == "floating":
                jtype = {"fixed": "fixed", "spherical": "spherical",
                         "free-body": None}[args.floating_joint]
                report.warn(f"{j['name']}: URDF 'floating' has no USD equivalent "
                            f"-> {args.floating_joint}")
                if jtype is None:
                    continue

            anchor = to_world(j["origin"]) + t_align
            axis_w = dir_to_world(j["axis"])
            tok, quat, axis_n, degen, flipped = usd_axis_and_rot(axis_w)
            if degen:
                report.warn(f"{j['name']}: degenerate axis {j['axis']} -> +Z")

            jp = jscope.AppendChild(Tf.MakeValidIdentifier(j["name"]))
            if jtype == "revolute":
                jt = UsdPhysics.RevoluteJoint.Define(stage, jp)
            elif jtype == "prismatic":
                jt = UsdPhysics.PrismaticJoint.Define(stage, jp)
            elif jtype == "spherical":
                jt = UsdPhysics.SphericalJoint.Define(stage, jp)
            else:
                jt = UsdPhysics.FixedJoint.Define(stage, jp)

            jt.CreateBody0Rel().SetTargets([prim_of[parent]])
            jt.CreateBody1Rel().SetTargets([prim_of[child]])
            jt.CreateLocalPos0Attr(_v3f(anchor - origins[parent]))
            jt.CreateLocalPos1Attr(_v3f(anchor - origins[child]))
            # Link Xforms are pure translations, so setting both localRots equal
            # puts the two joint frames at the same world rotation, and the
            # joint's local axis e_k maps to the intended world direction.
            if jtype in ("revolute", "prismatic", "spherical"):
                jt.CreateLocalRot0Attr(quat)
                jt.CreateLocalRot1Attr(quat)
                jt.CreateAxisAttr(tok)
            jt.CreateCollisionEnabledAttr(False)

            lo = hi = 0.0
            units = "-"
            lo_urdf = hi_urdf = 0.0
            if jtype in ("revolute", "prismatic"):
                raw = cap.get("limits", [-100, 100])
                try:
                    rlo, rhi = float(raw[0]), float(raw[1])
                except Exception:
                    rlo, rhi = -100.0, 100.0
                if jtype == "revolute":
                    units = "deg"
                    lo = rlo * LIMIT_DEG_PER_UNIT[args.limit_convention]
                    hi = rhi * LIMIT_DEG_PER_UNIT[args.limit_convention]
                    lo_urdf = rlo * LIMIT_DEG_PER_UNIT["urdf"]
                    hi_urdf = rhi * LIMIT_DEG_PER_UNIT["urdf"]
                else:
                    # URDF writes val/100 in NORMALISED units; convert to metres.
                    units = "m"
                    lo, hi = rlo / 100.0 * scale_m, rhi / 100.0 * scale_m
                    lo_urdf, hi_urdf = lo, hi
                    span = float(np.ptp(worldP[child] @ axis_n))
                    lo, hi = max(lo, -span), min(hi, span)
                if flipped:
                    # The axis was reversed to keep it canonical; mirror the
                    # range so the reachable configurations are unchanged.
                    lo, hi = -hi, -lo
                    lo_urdf, hi_urdf = -hi_urdf, -lo_urdf
                if lo > hi:
                    lo, hi = hi, lo
                if jtype == "revolute" and (hi - lo) >= 359.999:
                    jt.CreateLowerLimitAttr(1.0)      # lower > upper == unlimited
                    jt.CreateUpperLimitAttr(-1.0)
                elif abs(hi - lo) < 1e-6:
                    report.warn(f"{j['name']}: zero-width limit -> leaving unlimited")
                else:
                    jt.CreateLowerLimitAttr(float(lo))
                    jt.CreateUpperLimitAttr(float(hi))

                if args.drive != "none":
                    inst = UsdPhysics.Tokens.angular if jtype == "revolute" else UsdPhysics.Tokens.linear
                    d = UsdPhysics.DriveAPI.Apply(jt.GetPrim(), inst)
                    d.CreateTypeAttr(UsdPhysics.Tokens.force)
                    d.CreateTargetPositionAttr(0.0)
                    # NB: angular drive gains are per DEGREE, not per radian.
                    d.CreateStiffnessAttr(args.drive_stiffness if args.drive == "position" else 0.0)
                    d.CreateDampingAttr(args.drive_damping)

            report.joints.append(JointInfo(
                name=j["name"], type=jtype, parent=parent, child=child,
                axis_world=axis_n.round(6).tolist(), usd_axis=tok,
                lower=round(lo, 4), upper=round(hi, 4), units=units,
                lower_urdf_convention=round(lo_urdf, 4),
                upper_urdf_convention=round(hi_urdf, 4)))

        if args.fixed_base:
            fj = UsdPhysics.FixedJoint.Define(stage, jscope.AppendChild("root_joint"))
            fj.CreateBody1Rel().SetTargets([prim_of[base]])
            fj.CreateLocalPos0Attr(_v3f(origins[base]))
            fj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

    # Keep the bake auditable/reversible.
    root.GetPrim().SetCustomDataByKey("simart:scale_m", float(scale_m))
    root.GetPrim().SetCustomDataByKey("simart:frame_rot", args.frame_rot)
    root.GetPrim().SetCustomDataByKey("simart:floor_offset", _v3d(t_align))
    root.GetPrim().SetCustomDataByKey("simart:limit_convention", args.limit_convention)
    root.GetPrim().SetCustomDataByKey("simart:source_urdf", os.path.abspath(args.urdf))

    stage.GetRootLayer().Save()
    report.textures = textures
    return report


def stage_textures(args, out_dir, report: Report):
    """Copy the PBR maps next to the USD and return relative asset paths."""
    explicit = {"albedo": args.albedo, "metallic": args.metallic,
                "roughness": args.roughness, "normal": args.normal_map}
    found = {k: v for k, v in explicit.items() if v and os.path.isfile(v)}
    if not found and args.textures:
        # Hunyuan3D writes <stem>.jpg / <stem>_metallic.jpg / <stem>_roughness.jpg
        base = args.textures
        if os.path.isfile(base) and base.lower().endswith(".obj"):
            stem = base[:-4]
            cand = {"albedo": stem + ".jpg", "metallic": stem + "_metallic.jpg",
                    "roughness": stem + "_roughness.jpg", "normal": stem + "_normal.jpg"}
        elif os.path.isdir(base):
            cand = {}
            for k, pat in (("albedo", "albedo"), ("metallic", "metallic"),
                           ("roughness", "roughness"), ("normal", "normal")):
                for f in sorted(os.listdir(base)):
                    if pat in f.lower() and f.lower().endswith((".jpg", ".jpeg", ".png")):
                        cand[k] = os.path.join(base, f)
                        break
        else:
            cand = {}
        found = {k: v for k, v in cand.items() if v and os.path.isfile(v)}
    if not found:
        report.warn("no PBR textures found -> binding an untextured UsdPreviewSurface")
        return {}
    tex_dir = os.path.join(out_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)
    rel = {}
    for k, src in found.items():
        dst = os.path.join(tex_dir, k + os.path.splitext(src)[1].lower())
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
        rel[k] = "./textures/" + os.path.basename(dst)
    return rel


# ==========================================================================
# --rigid: single-body fallback straight from the Hunyuan3D mesh
# ==========================================================================
def convert_rigid(args) -> Report:
    """Two-tier static/rigid USD from one mesh, bypassing SimArt entirely.

    Mirrors the repo's ``_collision`` / ``_collision_rigid`` split: a geometry +
    collider layer, and a ~1 KB overlay that sublayers it and adds the rigid
    body.  This exercises the same mesh/UV/material/collision code as the
    articulated path, so if it looks right, only articulation is in question.
    """
    report = Report(output=args.out, frame_rot=args.frame_rot)
    src = args.mesh
    flip_v = False
    if src.lower().endswith(".obj"):
        P, VT, PF, TF = sanitize(*read_obj(src))
    else:
        import trimesh
        m = trimesh.load(src, force="mesh", process=False)
        P, PF = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
        VT = np.asarray(m.visual.uv, float) if getattr(m.visual, "uv", None) is not None else None
        TF = PF if VT is not None else None
        # Raw glTF TEXCOORD_0 is upper-left; USD st is lower-left.
        flip_v = True
    if P is None:
        raise SystemExit(f"ERROR: could not read a usable mesh from {src}")

    R = _FRAME_ROT[args.frame_rot]
    pts = P @ R.T
    ext = float(np.max(pts.max(0) - pts.min(0)))
    scale_m = args.scale_m or 1.0
    if ext > 0:
        pts = pts / ext * scale_m            # normalise then scale to real size
    report.scale_m, report.scale_provenance = scale_m, f"--scale-m {scale_m}" if args.scale_m else "1.0 default"
    if args.floor_align:
        pts -= np.array([(pts[:, 0].max() + pts[:, 0].min()) / 2,
                         (pts[:, 1].max() + pts[:, 1].min()) / 2, pts[:, 2].min()])
    report.bbox_world_min = pts.min(0).round(6).tolist()
    report.bbox_world_max = pts.max(0).round(6).tolist()

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    ext_out = os.path.splitext(args.out)[1] or ".usda"
    static_path = os.path.join(out_dir, f"{stem}_static{ext_out}")

    for p in (static_path, args.out):
        if os.path.exists(p):
            os.remove(p)

    stage = Usd.Stage.CreateNew(static_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root_name = Tf.MakeValidIdentifier(args.name or stem)
    root = UsdGeom.Xform.Define(stage, Sdf.Path(f"/{root_name}"))
    stage.SetDefaultPrim(root.GetPrim())
    textures = stage_textures(args, out_dir, report)
    pbr = author_pbr_material(stage, root.GetPath(), textures, report)
    phys = author_physics_material(stage, root.GetPath())
    author_mesh(stage, root.GetPath().AppendChild("mesh"), pts, VT, PF, TF,
                pbr, phys, args.collision, flip_v=flip_v)
    stage.GetRootLayer().Save()

    rho = args.density or DENSITY_DEFAULT
    vol = mesh_volume(pts, PF)
    report.links.append(LinkInfo(pid="0", prim=str(root.GetPath()) + "/mesh",
                                 n_points=len(pts), n_faces=len(PF), density=rho,
                                 volume_m3=round(vol, 8), mass_kg=round(rho * vol, 5),
                                 bbox_min=pts.min(0).round(5).tolist(),
                                 bbox_max=pts.max(0).round(5).tolist()))
    report.total_mass_kg = report.links[0].mass_kg

    # Overlay: sublayer the static tier and add the rigid body on top.
    over = Usd.Stage.CreateNew(args.out)
    over.GetRootLayer().subLayerPaths.append(f"./{os.path.basename(static_path)}")
    UsdGeom.SetStageUpAxis(over, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(over, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(over, 1.0)
    op = over.OverridePrim(f"/{root_name}")
    UsdPhysics.RigidBodyAPI.Apply(op)
    UsdPhysics.MassAPI.Apply(op).CreateDensityAttr(rho)
    over.SetDefaultPrim(op)
    over.GetRootLayer().Save()
    report.textures = textures
    return report


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Convert a SimArt URDF (+ prediction JSON) into an Isaac-Sim-ready USD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--urdf", help="SimArt URDF (topology + part mesh paths)")
    ap.add_argument("--json", help="SimArt prediction JSON (scale, density, limits)")
    ap.add_argument("--out", required=True, help="output .usd/.usda")
    ap.add_argument("--name", default=None, help="root prim name")
    ap.add_argument("--scale-m", type=float, default=None,
                    help="real-world max bbox extent in metres (overrides the JSON)")
    ap.add_argument("--scale-unit", default="auto",
                    choices=["auto"] + sorted(_UNITS),
                    help="unit for a bare numeric 'scale' in the JSON; 'auto' uses a "
                         "magnitude heuristic (<=10 m, <=500 cm, else mm)")
    ap.add_argument("--frame-rot", default="none",
                    choices=["none", "undo-simart", "rx90", "rx-90"],
                    help="rotation applied to points, joint anchors and axes; "
                         "'none' is right for the Y-up glTF chain (see module docstring)")
    ap.add_argument("--limit-convention", default="spec", choices=["spec", "urdf"],
                    help="'spec': 100 units == 180 deg (the model's prompt); "
                         "'urdf': 100 == 360 deg (what SimArt's writer emits)")
    ap.add_argument("--floating-joint", default="fixed",
                    choices=["fixed", "spherical", "free-body"])
    ap.add_argument("--collision", default="convexDecomposition",
                    choices=["convexDecomposition", "convexHull", "boundingCube", "none"])
    ap.add_argument("--drive", default="none", choices=["none", "damping", "position"])
    ap.add_argument("--drive-stiffness", type=float, default=100.0)
    ap.add_argument("--drive-damping", type=float, default=0.1)
    ap.add_argument("--fixed-base", action="store_true", help="pin the base link to the world")
    ap.add_argument("--floor-align", dest="floor_align", action="store_true", default=True)
    ap.add_argument("--no-floor-align", dest="floor_align", action="store_false")
    ap.add_argument("--textures", default=None,
                    help="Hunyuan3D textured .obj, or a directory of PBR maps")
    ap.add_argument("--albedo"), ap.add_argument("--metallic")
    ap.add_argument("--roughness"), ap.add_argument("--normal-map", dest="normal_map")
    ap.add_argument("--rigid", action="store_true", help="single-body mode, bypassing SimArt")
    ap.add_argument("--mesh", help="[--rigid] source .obj/.glb")
    ap.add_argument("--density", type=float, default=None, help="[--rigid] kg/m^3")
    ap.add_argument("--total-mass", type=float, default=None,
                    help="pin the object's real mass in kg; overrides the predicted "
                         "per-part densities with a uniform one derived from mesh volume")
    args = ap.parse_args()

    if args.rigid:
        if not args.mesh:
            ap.error("--rigid requires --mesh")
        report = convert_rigid(args)
    else:
        if not args.urdf:
            ap.error("--urdf is required (or use --rigid --mesh)")
        if not args.json:
            cand = os.path.splitext(args.urdf)[0] + ".json"
            args.json = cand if os.path.isfile(cand) else None
        report = convert(args)

    with open(args.out + ".report.json", "w") as fh:
        json.dump(asdict(report), fh, indent=2)

    print(f"\n  wrote {report.output}")
    print(f"  scale     {report.scale_m:.4g} m   ({report.scale_provenance})")
    print(f"  bbox      min={report.bbox_world_min}  max={report.bbox_world_max}")
    print(f"  links     {len(report.links)}   total mass {report.total_mass_kg:.3f} kg")
    for l in report.links:
        print(f"    {l.prim:<40} {l.n_faces:>7} faces  rho={l.density:>7.1f}  "
              f"m={l.mass_kg:>8.3f} kg  {l.material}")
    print(f"  joints    {len(report.joints)}")
    for j in report.joints:
        extra = ""
        if j.units == "deg" and (j.lower, j.upper) != (j.lower_urdf_convention, j.upper_urdf_convention):
            extra = f"   [urdf convention would be {j.lower_urdf_convention:g}..{j.upper_urdf_convention:g}]"
        print(f"    {j.name:<12} {j.type:<10} {j.parent} -> {j.child}  axis={j.usd_axis} "
              f"{np.round(j.axis_world, 3).tolist()}  limit=[{j.lower:g}, {j.upper:g}] {j.units}{extra}")
    if report.warnings:
        print(f"  warnings  {len(report.warnings)}")
    print(f"  report    {args.out}.report.json")


if __name__ == "__main__":
    main()
