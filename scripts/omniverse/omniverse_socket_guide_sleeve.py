#!/usr/bin/env python3
"""Author a guide sleeve for the Omniverse LightBulb socket (issue #171), no GPU.

Measured on the shipped meshes (``scripts/measure_bore_geometry.py``): the only thing that guides
the bulb's plug is the socket's mouth ring, radius 20.2 mm and 6 mm long (seat-frame axial +28 to
+34 mm). Below it the cavity opens to r 22-26 mm and the 17.0 mm plug body floats free. With 3.2 mm
of radial clearance over 6 mm the seated plug can lean until the glass shoulder rocks on the rim
and the plug side hits the ring's lower edge, atan(3.2 / 8) ~ 22 deg, which is where every wedged
S11 take parked (the attachment holds it there; it never sits straight, and the seat gate's tilt
limit never clears).

This adds a TUBE collider inside the cavity, from axial +8 mm (where the plug body reaches full
radius) up to the ring at +28 mm, at inner radius ``--r_in``, so the plug is guided over ~26 mm
instead of 6: tilt play = asin(2 (r_in - 17.0 mm) / 26 mm), 7.5 deg at the default 18.7 mm, inside
the seat gate's tilt tolerance and the tilt spring's recovery basin. The tube carries a low-friction
physics material (``--mu``, default 0.1: a smooth sleeve, the variant the S11 scoring takes used;
``--stock-friction`` leaves the socket's own high-friction material on it).

Two ADDITIVE layers are written next to the stock socket files, tagged ``--tag``:

    LightBulb_socket_<tag>.usda           sublayers LightBulb_socket.usda; defines one extra exact
                                          triangle collider under /World/Geom/Base
    LightBulb_socket_z_static_<tag>.usda  the Z-up/metre wrapper, same conversion as
                                          LightBulb_socket_z_static.usda, referencing the layer above

Nothing in the stock files changes. ``SOCKET_USD`` in ``fiatlux_task/assets.py`` points at the default
output (tag ``sleeve``), so run this once after pulling the stock socket; delete the two outputs and
point it back at ``LightBulb_socket_z_static.usda`` to return to stock.

The tube is built in the socket's seat frame (``SOCKET_SEAT_OFFSET`` / ``SOCKET_SEAT_AXIS`` from
``fiatlux_task.assets``, root frame, Z-up, metres) and mapped into the ``Geom/Base`` prim's own
Y-up centimetre frame through the stock wrapper's composed transform, so it lands exactly where
the bulb seats regardless of how the asset is authored. Needs only ``pxr``; a python without it
(the Isaac env) falls back to bootstrapping Kit headless.

Usage: python scripts/omniverse/omniverse_socket_guide_sleeve.py [<omniverse_bulb dir>]
       [--r_in 0.0187] [--r_out 0.0215] [--a0 0.008] [--a1 0.0281] [--segments 48]
       [--mu 0.1 | --stock-friction] [--tag sleeve]
"""

import argparse
import math
import os
import sys

try:
    from pxr import Gf, Usd, UsdGeom
except ModuleNotFoundError:  # pxr only resolves once Kit has been bootstrapped
    from isaaclab.app import AppLauncher

    _APP = AppLauncher(headless=True).app
    from pxr import Gf, Usd, UsdGeom

try:
    from fiatlux_task.assets import SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
except ModuleNotFoundError:  # a bare-pxr python without the package on its path: same values
    SOCKET_SEAT_OFFSET = (0.0, 0.0, 0.036259)
    SOCKET_SEAT_AXIS = (0.0, 0.0, 1.0)

STOCK_LAYER = "LightBulb_socket.usda"
STOCK_WRAPPER = "LightBulb_socket_z_static.usda"
BASE_PRIM = "/Socket/Convert/LightBulb/Geom/Base"  # the bore mesh's parent in the stock wrapper
TUBE_PRIM = "GuideSleeve"
MATERIAL_PRIM = "GuideSleeveMaterial"

PLUG_BODY_RADIUS = 0.0170  # m, the bulb's plug body (LightBulb_bulb_z_rigid.usda, after the 0.84 shrink)
RING_TOP = 0.0342  # m, seat-frame axial of the mouth ring's top edge (end of the guided length)
_TOL = 1e-6  # m, verification tolerance on the re-read geometry


def _basis(axis):
    """Two unit vectors spanning the plane normal to ``axis`` (a Gf.Vec3d)."""
    u = Gf.Cross(axis, Gf.Vec3d(0.0, 0.0, 1.0))
    if u.GetLength() < 1e-3:
        u = Gf.Cross(axis, Gf.Vec3d(0.0, 1.0, 0.0))
    u = u.GetNormalized()
    return u, Gf.Cross(axis, u)


def build_tube(a0, a1, r_in, r_out, segments):
    """Tube points (seat frame, metres) and outward-wound triangles: (points, faces)."""
    seat = Gf.Vec3d(*SOCKET_SEAT_OFFSET)
    axis = Gf.Vec3d(*SOCKET_SEAT_AXIS).GetNormalized()
    u, v = _basis(axis)
    n = segments
    ring = [(math.cos(2.0 * math.pi * k / n), math.sin(2.0 * math.pi * k / n)) for k in range(n)]

    def circle(r, a):
        return [seat + a * axis + r * (c * u + s * v) for c, s in ring]

    # index blocks: inner-bottom 0..n-1, inner-top n..2n-1, outer-bottom 2n..3n-1, outer-top 3n..4n-1
    pts = circle(r_in, a0) + circle(r_in, a1) + circle(r_out, a0) + circle(r_out, a1)
    faces = []

    def quad(a, b, c, d, want):
        """Two triangles for the quad a-b-c-d, wound so the face normal points along ``want``."""
        for tri in ((a, b, c), (a, c, d)):
            p0, p1, p2 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            normal = Gf.Cross(p1 - p0, p2 - p0)
            faces.append(tri if Gf.Dot(normal, want) >= 0.0 else (tri[0], tri[2], tri[1]))

    for k in range(n):
        j = (k + 1) % n
        c = (ring[k][0] + ring[j][0]) / 2.0
        s = (ring[k][1] + ring[j][1]) / 2.0
        radial = c * u + s * v
        quad(k, j, n + j, n + k, -radial)  # inner wall: normal toward the axis
        quad(2 * n + k, 2 * n + j, 3 * n + j, 3 * n + k, radial)  # outer wall: normal outward
        quad(k, j, 2 * n + j, 2 * n + k, -axis)  # bottom cap
        quad(n + k, n + j, 3 * n + j, 3 * n + k, axis)  # top cap
    return pts, faces


def base_to_root(src_dir):
    """Transform of the stock socket's ``Geom/Base`` prim into the socket root frame (Z-up, m)."""
    wrapper = os.path.join(src_dir, STOCK_WRAPPER)
    if not os.path.isfile(wrapper):
        raise FileNotFoundError(wrapper)
    stage = Usd.Stage.Open(wrapper)
    base = stage.GetPrimAtPath(BASE_PRIM)
    if not base or not base.IsValid():
        raise AssertionError(f"{STOCK_WRAPPER}: no {BASE_PRIM} -- did the socket reference compose?")
    return UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def tilt_play_deg(a0, r_in):
    guided = RING_TOP - a0
    return math.degrees(math.asin(min(1.0, 2.0 * (r_in - PLUG_BODY_RADIUS) / guided))), guided


def author(src_dir, a0, a1, r_in, r_out, segments, mu, tag):
    pts_root, faces = build_tube(a0, a1, r_in, r_out, segments)
    to_local = base_to_root(src_dir).GetInverse()
    pts_local = [to_local.Transform(p) for p in pts_root]
    play, guided = tilt_play_deg(a0, r_in)

    pts = ", ".join(f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})" for p in pts_local)
    fvi = ", ".join(str(i) for tri in faces for i in tri)
    fvc = ", ".join("3" for _ in faces)
    mm = lambda x: f"{x * 1000.0:.1f}"  # noqa: E731

    header = ""
    material = ""
    schemas = '["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]'
    binding = ""
    if mu is not None:
        header = f"# LOW-FRICTION VARIANT (mu {mu:g} on the tube, a smooth guide sleeve).\n"
        material = f"""            def Material "{MATERIAL_PRIM}" (
                prepend apiSchemas = ["PhysicsMaterialAPI"]
            )
            {{
                float physics:dynamicFriction = {mu:g}
                float physics:staticFriction = {mu:g}
                float physics:restitution = 0.0
            }}
"""
        schemas = '["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]'
        binding = f"                rel material:binding:physics = </World/Geom/Base/{MATERIAL_PRIM}>\n"

    layer = os.path.join(src_dir, f"LightBulb_socket_{tag}.usda")
    with open(layer, "w") as f:
        f.write(f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 0.01
    subLayers = [
        @./{STOCK_LAYER}@
    ]
    upAxis = "Y"
)

{header}# Guide sleeve for the bulb socket (issue #171, S11: the seated bulb tilts to 22 deg and jams
# because only a 6 mm ring guides the plug). ADDITIVE: sublayers the stock socket and DEFINES one
# extra exact-triangle collider, a tube from axial +{mm(a0)} to +{mm(a1)} mm (seat frame) at
# inner radius {mm(r_in)} mm (plug body {mm(PLUG_BODY_RADIUS)} mm), so the plug is guided over
# ~{guided * 1000.0:.0f} mm and can tilt only ~{play:.1f} deg before it binds. Delete this file (and the
# z-up wrapper) to return to stock.
over "World"
{{
    over "Geom"
    {{
        over "Base"
        {{
{material}            def Mesh "{TUBE_PRIM}" (
                prepend apiSchemas = {schemas}
            )
            {{
{binding}                uniform bool doubleSided = 1
                int[] faceVertexCounts = [{fvc}]
                int[] faceVertexIndices = [{fvi}]
                uniform token physics:approximation = "none"
                point3f[] points = [{pts}]
                uniform token subdivisionScheme = "none"
            }}
        }}
    }}
}}
""")

    wrapper = os.path.join(src_dir, f"LightBulb_socket_z_static_{tag}.usda")
    with open(wrapper, "w") as f:
        f.write(f"""#usda 1.0
(
    defaultPrim = "Socket"
    metersPerUnit = 1
    upAxis = "Z"
)

# z-up wrapper for the guide-sleeve socket (issue #171). Same frame conversion as
# {STOCK_WRAPPER}, referencing the guide-sleeve override layer instead of the stock one.
def Xform "Socket"
{{
    def Xform "Convert"
    {{
        quatd xformOp:orient = (0.7071067811865476, 0.7071067811865476, 0, 0)
        double3 xformOp:scale = (0.01, 0.01, 0.01)
        uniform token[] xformOpOrder = ["xformOp:orient", "xformOp:scale"]

        def Xform "LightBulb" (
            prepend references = @./LightBulb_socket_{tag}.usda@</World>
        )
        {{
            over "ActionGraph" (
                active = false
            )
            {{
            }}

            over "Camera" (
                active = false
            )
            {{
            }}
        }}
    }}
}}
""")
    print(
        f"tube: axial {mm(a0)}..{mm(a1)} mm, r_in {mm(r_in)} r_out {mm(r_out)} mm, {len(pts_local)} pts "
        f"{len(faces)} tris; guided length incl. ring {mm(guided)} mm -> tilt play ~{play:.1f} deg"
        + (f"; mu {mu:g}" if mu is not None else "")
    )
    return layer, wrapper


def verify(wrapper, a0, a1, r_in, r_out, faces_expected, mu):
    """Reopen the wrapper and check the composed tube sits where it was asked, in the root frame."""
    stage = Usd.Stage.Open(wrapper)
    tube = stage.GetPrimAtPath(f"{BASE_PRIM}/{TUBE_PRIM}")
    if not tube or not tube.IsValid() or tube.GetTypeName() != "Mesh":
        raise AssertionError(f"{os.path.basename(wrapper)}: {TUBE_PRIM} did not compose under {BASE_PRIM}")
    to_root = UsdGeom.Xformable(tube).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    seat = Gf.Vec3d(*SOCKET_SEAT_OFFSET)
    axis = Gf.Vec3d(*SOCKET_SEAT_AXIS).GetNormalized()
    problems = []
    axials, radii = [], []
    for p in UsdGeom.Mesh(tube).GetPointsAttr().Get():
        rel = to_root.Transform(Gf.Vec3d(p)) - seat
        a = Gf.Dot(rel, axis)
        axials.append(a)
        radii.append((rel - a * axis).GetLength())
    if abs(min(axials) - a0) > _TOL or abs(max(axials) - a1) > _TOL:
        problems.append(f"axial [{min(axials):.6f}, {max(axials):.6f}] != [{a0:.6f}, {a1:.6f}]")
    if any(min(abs(r - r_in), abs(r - r_out)) > _TOL for r in radii):
        problems.append("a point is off both tube radii")
    counts = UsdGeom.Mesh(tube).GetFaceVertexCountsAttr().Get()
    if len(counts) != faces_expected or any(c != 3 for c in counts):
        problems.append(f"{len(counts)} faces, expected {faces_expected} triangles")
    if tube.GetAttribute("physics:approximation").Get() != "none":
        problems.append("collider is not an exact triangle mesh")
    if mu is not None:
        rel = tube.GetRelationship("material:binding:physics")
        targets = rel.GetTargets() if rel else []
        if len(targets) != 1 or targets[0].name != MATERIAL_PRIM:
            problems.append("physics material is not bound")
        else:
            mat = stage.GetPrimAtPath(targets[0])
            got = mat.GetAttribute("physics:dynamicFriction").Get() if mat else None
            if got is None or abs(got - mu) > 1e-6:
                problems.append(f"bound material friction {got} != {mu}")
    if problems:
        raise AssertionError(f"{os.path.basename(wrapper)}: " + "; ".join(problems))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src_dir", nargs="?", help="the omniverse_bulb asset dir (default: repo assets/omniverse_bulb)")
    parser.add_argument(
        "--a0", type=float, default=0.0080, help="tube bottom, seat-frame axial (m); plug body reaches full radius here"
    )
    parser.add_argument(
        "--a1", type=float, default=0.0281, help="tube top, seat-frame axial (m); meets the stock ring at +28.1"
    )
    parser.add_argument("--r_in", type=float, default=0.0187, help="tube inner radius (m); plug body r = 17.0 mm")
    parser.add_argument(
        "--r_out", type=float, default=0.0215, help="tube outer radius (m); the cavity is r >= 22.3 mm above +24"
    )
    parser.add_argument("--segments", type=int, default=48, help="facets around the tube")
    parser.add_argument(
        "--mu", type=float, default=0.1, help="static/dynamic friction of the physics material bound to the tube"
    )
    parser.add_argument(
        "--stock-friction", action="store_true", help="bind no material: the tube keeps the socket's own high friction"
    )
    parser.add_argument(
        "--tag", type=str, default="sleeve", help="output file tag: LightBulb_socket_<tag>.usda + z_static wrapper"
    )
    args = parser.parse_args()

    src_dir = args.src_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "omniverse_bulb"
    )
    if not os.path.isdir(src_dir):
        print(f"not a directory: {src_dir}\n")
        print(__doc__)
        sys.exit(1)
    if not (PLUG_BODY_RADIUS < args.r_in < args.r_out) or not (0.0 <= args.a0 < args.a1 <= RING_TOP):
        print(f"bad tube: need {PLUG_BODY_RADIUS} < r_in < r_out and 0 <= a0 < a1 <= {RING_TOP}")
        sys.exit(1)

    mu = None if args.stock_friction else args.mu
    layer, wrapper = author(src_dir, args.a0, args.a1, args.r_in, args.r_out, args.segments, mu, args.tag)
    verify(wrapper, args.a0, args.a1, args.r_in, args.r_out, 8 * args.segments, mu)
    for out in (layer, wrapper):
        print(f"  [ok] {os.path.basename(out)}")


if __name__ == "__main__":
    main()
