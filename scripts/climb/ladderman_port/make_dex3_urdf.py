"""Give LadderMan's G1 the benchmark robot's hand and mass.

LadderMan trains on `g1_29dof_rubberhand-complexfeet-complex.urdf`: every link's mass is
identical to the benchmark's Dex3 G1 EXCEPT the hand -- a 0.17 kg one-piece rubber mitt, 132 mm
long, where the benchmark carries a 0.70 kg Dex3 that is 210 mm long with the fingers open and
whose thumb hangs 100 mm below the palm. A teacher trained with the mitt meets the Dex3's shape
for the first time in the benchmark, on holds with 5-18 mm of margin.

This writes a sibling URDF in which each hand is the OPEN Dex3 as one rigid piece:

* collision (and visual): the convex hull of each Dex3 hand link -- palm, two fingers of two
  links, thumb of three -- at the pose it has with every finger joint at zero. The wrist camera
  housing is left out, as it is in the benchmark (removed at spawn, #201).
* inertial: the Dex3 hand links' masses, centres of mass and inertia tensors summed into the
  one link.

The hand stays a fixed, jointless part of the wrist, so the action space and every joint name
are unchanged and LadderMan's own code runs as is.

It also reproduces, on request (--phantom), a defect of the benchmark robot: four sensor frames
(imu_in_pelvis, imu_in_torso, mid360_link, d435_link) have no authored mass and no collider, so
PhysX gives each its 1 kg default -- 4 kg that the real robot does not have, 3 of them on the
torso. Measured in the sim: 38.40 kg against 34.4 kg authored. A teacher has to be trained
against the robot it will be run on, so the default here is to match the benchmark AS IT IS; if
the benchmark is corrected, regenerate with --no-phantom.

    python make_dex3_urdf.py --urdf <ladderman g1 urdf> --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

HERE = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--urdf", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", default="g1_29dof_dex3open-complexfeet-complex")
ap.add_argument("--phantom", action=argparse.BooleanOptionalAction, default=True)
a = ap.parse_args()

USD = json.loads((HERE / "fx_dex3_usd_mass.json").read_text())
SIM = json.loads((HERE / "fx_dex3_sim_mass.json").read_text())
PTS = np.load(HERE / "dex3_hand_collision_points.npz")
out = pathlib.Path(a.out)
(out / "meshes").mkdir(parents=True, exist_ok=True)
HAND_LINKS = ["hand_palm_link", "hand_index_0_link", "hand_index_1_link", "hand_middle_0_link",
              "hand_middle_1_link", "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link"]


def offsets(side):
    """Each hand link's origin in the PALM frame, fingers at zero (every joint frame unrotated)."""
    sy = 1.0 if side == "left" else -1.0
    o = {"hand_palm_link": np.zeros(3)}
    for f, z in (("index", 0.0285), ("middle", -0.0285)):
        o[f"hand_{f}_0_link"] = np.array([0.0777, 0.0016 * sy, z])
        o[f"hand_{f}_1_link"] = o[f"hand_{f}_0_link"] + np.array([0.0458, 0.0, 0.0])
    o["hand_thumb_0_link"] = np.array([0.0255, 0.0, 0.0])
    o["hand_thumb_1_link"] = o["hand_thumb_0_link"] + np.array([-0.0025, -0.0193 * sy, 0.0])
    o["hand_thumb_2_link"] = o["hand_thumb_1_link"] + np.array([0.0, -0.0458 * sy, 0.0])
    return o


def reduce_hull(P, n=60):
    """<= n hull vertices (PhysX cooks at most 64): support points in n spread directions."""
    k = np.arange(n) + 0.5
    ph, th = np.arccos(1 - 2 * k / n), np.pi * (1 + 5 ** 0.5) * k
    D = np.concatenate([np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1),
                        np.eye(3), -np.eye(3)])
    V = P[np.unique((P @ D.T).argmax(0))]
    return V[ConvexHull(V).vertices]


def write_stl(path, V):
    h = ConvexHull(V)
    c = V.mean(0)
    with open(path, "wb") as f:
        f.write(b"dex3 open hand hull".ljust(80, b" "))
        f.write(struct.pack("<I", len(h.simplices)))
        for s, eq in zip(h.simplices, h.equations):
            A, B, C = V[s]
            n = np.cross(B - A, C - A)
            if n @ (A - c) < 0:
                B, C = C, B
                n = -n
            n = n / (np.linalg.norm(n) + 1e-12)
            f.write(struct.pack("<12fH", *n, *A, *B, *C, 0))


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def hand_inertial(side):
    off = offsets(side)
    M, mc, parts = 0.0, np.zeros(3), []
    for ln in HAND_LINKS:
        u = USD[f"{side}_{ln}"]
        m = float(u["mass"])
        c = off[ln] + np.array(u["com"], float)
        R = quat_R(u["axes"])
        I = R @ np.diag(u["inertia"]) @ R.T
        M += m; mc += m * c; parts.append((m, c, I))
    com = mc / M
    I = np.zeros((3, 3))
    for m, c, Ic in parts:
        d = c - com
        I += Ic + m * ((d @ d) * np.eye(3) - np.outer(d, d))
    return M, com, I


def set_inertial(link, mass, com, I):
    old = link.find("inertial")
    if old is not None:
        link.remove(old)
    el = ET.SubElement(link, "inertial")
    ET.SubElement(el, "origin", xyz=" ".join(f"{v:.6g}" for v in com), rpy="0 0 0")
    ET.SubElement(el, "mass", value=f"{mass:.6g}")
    ET.SubElement(el, "inertia", ixx=f"{I[0,0]:.6g}", ixy=f"{I[0,1]:.6g}", ixz=f"{I[0,2]:.6g}",
                  iyy=f"{I[1,1]:.6g}", iyz=f"{I[1,2]:.6g}", izz=f"{I[2,2]:.6g}")


tree = ET.parse(a.urdf)
root = tree.getroot()
links = {l.get("name"): l for l in root.findall("link")}
total0 = sum(float(l.find("inertial").find("mass").get("value")) for l in links.values() if l.find("inertial") is not None)

for side in ("left", "right"):
    hl = links[f"{side}_rubber_hand"]
    for tag in ("visual", "collision"):
        for el in hl.findall(tag):
            hl.remove(el)
    off = offsets(side)
    for ln in HAND_LINKS:
        V = reduce_hull(PTS[f"{side}_{ln}"].astype(float))
        fn = f"dex3open_{side}_{ln}.STL"
        write_stl(out / "meshes" / fn, V)
        for tag in ("visual", "collision"):
            el = ET.SubElement(hl, tag)
            ET.SubElement(el, "origin", xyz=" ".join(f"{v:.6g}" for v in off[ln]), rpy="0 0 0")
            g = ET.SubElement(el, "geometry")
            ET.SubElement(g, "mesh", filename=f"meshes/{fn}")
    M, com, I = hand_inertial(side)
    set_inertial(hl, M, com, I)
    print(f"[hand] {side}: {len(HAND_LINKS)} hulls, mass {M:.4f} kg (was 0.17), com {com.round(4)}")

if a.phantom:
    # PhysX's default for a body with neither mass nor collider, read back from the benchmark sim
    for name in ("imu_in_pelvis", "imu_in_torso", "mid360_link", "d435_link"):
        s = SIM[name]
        I = np.diag(s["inertia"][0::4]) if len(s["inertia"]) == 9 else np.diag(s["inertia"][:3])
        if name not in links:                              # d435_link: not in LadderMan's URDF
            par = USD[name]["parent"]
            T = np.linalg.inv(np.array(USD[par]["M"]).T) @ np.array(USD[name]["M"]).T     # child in parent
            l = ET.SubElement(root, "link", name=name)
            j = ET.SubElement(root, "joint", name=name.replace("_link", "") + "_phantom_joint", type="fixed")
            ET.SubElement(j, "origin", xyz=" ".join(f"{v:.6g}" for v in T[:3, 3]), rpy="0 0 0")
            ET.SubElement(j, "parent", link=par)
            ET.SubElement(j, "child", link=name)
            links[name] = l
            print(f"[phantom] added {name} on {par} at {T[:3, 3].round(4)}")
        set_inertial(links[name], float(s["mass"]), np.zeros(3), I)
        print(f"[phantom] {name}: {s['mass']:.3f} kg, inertia diag {np.diag(I).round(6)}")

total1 = sum(float(l.find("inertial").find("mass").get("value")) for l in links.values() if l.find("inertial") is not None)
ET.indent(tree, space="  ")
dst = out / f"{a.name}.urdf"
tree.write(dst, xml_declaration=True, encoding="utf-8")
sim_total = sum(v["mass"] for v in SIM.values())
print(f"[mass] URDF {total0:.3f} -> {total1:.3f} kg; benchmark robot in the sim {sim_total:.3f} kg; difference {total1 - sim_total:+.3f}")
print(f"[out ] {dst}")
