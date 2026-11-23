"""Give LadderMan's G1 the benchmark robot's BODY collision shapes where the two differ.

The open-Dex3 URDF (make_dex3_urdf.py) matched the hand and the mass. The body was assumed equal.
Compared link by link against the benchmark robot as the live S02 task spawns it, it is not:

    link                    trainer                       benchmark
    shoulder_pitch (L/R)    none                          cylinder r 30 mm, 50 mm long
    shoulder_roll  (L/R)    none                          cylinder r 30 mm, 30 mm long
    shoulder_yaw   (L/R)    capsule r 35 mm, 200 mm long  convex hull 81 x 75 x 145 mm
    hip_pitch      (L/R)    none                          convex hull 83 x 99 x 117 mm
    pelvis                  sphere r 70 mm                none (the contour link carries it)
    pelvis_contour_link     none                          convex hull 120 x 136 x 124 mm
    logo_link               none                          convex hull 149 x 149 x 29 mm

Every other body link (hip roll / yaw, knee, torso mesh, head, elbow, wrists) has the same shape in
both to under 3 mm. Feet and hands are separate matters and are not touched here.

Why it matters. Self-collision is on in both. With no shoulder collider the trainer lets the upper
arm tuck through space that is solid in the benchmark; a teacher trained that way holds both
shoulder_roll joints about 20 deg inside the clip, and in the benchmark (S02, free ladder) those
links then press on the torso with 350-435 N, their motors at 100 % torque, in every run.

This writes a sibling URDF in which those links carry the benchmark's shapes: each one the convex
hull of the benchmark collider's points (`fx_body_collision_points.npz`, link frame, measured with
the benchmark's own stage), as an STL. Cylinders are written as hulls too, because LadderMan's
import turns URDF cylinders into capsules, which are longer. Inertials, joints, names and visuals
are untouched, so checkpoints, clips and the action space carry over.

    python make_fxbody_urdf.py --urdf <.../g1_29dof_dex3open-complexfeet-complex.urdf> --out <dir>
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

ap = argparse.ArgumentParser()
ap.add_argument("--urdf", required=True)
ap.add_argument("--out", required=True, help="directory of the URDF (its meshes/ gets the new STLs)")
ap.add_argument("--name", default="g1_29dof_dex3open-complexfeet-fxbody")
a = ap.parse_args()
HERE = pathlib.Path(__file__).parent
PTS = np.load(HERE / "fx_body_collision_points.npz")
out = pathlib.Path(a.out)
(out / "meshes").mkdir(parents=True, exist_ok=True)
DROP = {"pelvis"}  # trainer-only shapes the benchmark does not have


def write_stl(path, V):
    h = ConvexHull(V)
    c = V[h.vertices].mean(0)
    tris = []
    for s, eq in zip(h.simplices, h.equations):
        p = V[s]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.dot(n, eq[:3]) < 0:
            p = p[[0, 2, 1]]
            n = -n
        tris.append((n / (np.linalg.norm(n) + 1e-12), p))
    with open(path, "wb") as f:
        f.write(b"fiatlux benchmark collider hull".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for n, p in tris:
            f.write(struct.pack("<12fH", *n, *p.reshape(-1), 0))
    return len(tris)


tree = ET.parse(a.urdf)
root = tree.getroot()
links = {l.get("name"): l for l in root.findall("link")}
by_link: dict[str, list[np.ndarray]] = {}
for k in PTS.files:
    by_link.setdefault(k.split("|")[0], []).append(PTS[k].astype(float))
for name in DROP:
    for c in links[name].findall("collision"):
        links[name].remove(c)
    print(f"[drop ] {name}: trainer-only collision removed")
for name, geoms in sorted(by_link.items()):
    if name not in links:
        raise SystemExit(f"{name} is not a link of {a.urdf}")
    old = links[name].findall("collision")
    for c in old:
        links[name].remove(c)
    for i, V in enumerate(geoms):
        fn = f"fxbody_{name}_{i}.STL"
        nt = write_stl(out / "meshes" / fn, V)
        c = ET.SubElement(links[name], "collision")
        ET.SubElement(c, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(c, "geometry"), "mesh", filename=f"meshes/{fn}")
        ext = np.round((V.max(0) - V.min(0)) * 1000).astype(int).tolist()
        print(
            f"[link ] {name}: {'replaced ' + str(len(old)) + ' shape(s)' if old else 'added'} -> hull {ext} mm, {nt} triangles"
        )
root.set("name", a.name)
tree.write(out / f"{a.name}.urdf", xml_declaration=True, encoding="utf-8")
print(f"[out  ] {out / (a.name + '.urdf')}")
