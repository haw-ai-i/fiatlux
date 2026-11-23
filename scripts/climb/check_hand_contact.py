"""Does the clip's hand ever actually touch THIS ladder? Offline, no Isaac.

The failure this exists to catch: a reference authored against LadderMan's procedural ladder can
command a hand onto a rung that is real THERE and empty air HERE. The previous top-out clip did
exactly that for 478 of its 939 steps and nothing in the pipeline noticed -- training success
stayed high, because in the training env those bars exist.

So this measures the hands against the BENCHMARK ASSET's own mesh: for every step where a hand is
supposed to be holding something, the distance from the wrist contact point to the nearest point
on the real ladder. A good clip holds that near zero; the old one will not.

Point-to-surface distance is approximated by barycentric sampling of every triangle, which is
exact to the sample spacing -- a prismatic member has no vertices mid-span, so sampling vertices
alone (the mistake that first reported a 6.7 mm rail) reads a long rail as absent.

    python scripts/climb/check_hand_contact.py --clip <dir> [--clip <dir> ...]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(
    0,
    "/home/yujin-chen/robotica_project/fiatlux/fiatlux/.venv/lib/python3.11/"
    "site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311",
)
from scipy.spatial import cKDTree  # noqa: E402

from pxr import Gf, Usd, UsdGeom  # noqa: E402

USD = (
    "/home/yujin-chen/robotica_project/fiatlux/fiatlux/assets/omniverse_ladder/AlumStep_D/"
    "AluminumStepLadder_D01_PR_NVD_01_collision.usd"
)
# the clip is authored with rung 0 at this point in the LADDER's frame; the asset's own origin is
# the ladder root, and rung 0 is our tread 1, whose top is 0.236
CLIP_BASE = np.array([-0.58, 0.0, 0.196])

ap = argparse.ArgumentParser()
ap.add_argument("--clip", action="append", required=True)
ap.add_argument("--stride", type=int, default=2, help="sample every Nth 50 Hz step")
ap.add_argument("--samples", type=int, default=6, help="barycentric samples per triangle")
a = ap.parse_args()

stage = Usd.Stage.Open(USD)
tris = []
for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    m = UsdGeom.Mesh(prim)
    if not m:
        continue
    P, FI, FC = (m.GetPointsAttr().Get(), m.GetFaceVertexIndicesAttr().Get(), m.GetFaceVertexCountsAttr().Get())
    if not P or not FI or not FC:
        continue
    x = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    V = np.array([[*x.Transform(Gf.Vec3d(q[0], q[1], q[2]))] for q in P], float) * 0.01
    FI = np.asarray(FI)
    k = 0
    for c in FC:
        f = FI[k : k + c]
        k += c
        for j in range(1, c - 1):
            tris.append(V[[f[0], f[j], f[j + 1]]])
T = np.asarray(tris)
rng = np.random.default_rng(0)
w = rng.dirichlet(np.ones(3), size=(len(T), a.samples))
cloud = np.einsum("tsk,tkd->tsd", w, T).reshape(-1, 3)
cloud = np.vstack([cloud, T.reshape(-1, 3)])
tree = cKDTree(cloud)
print(f"ladder mesh: {len(T)} triangles -> {len(cloud)} surface samples\n")

# the ladder asset's own frame has rung 0 (tread 1 top) at z=0.236; the clip's BASE is the rung
# CENTRE, 40 mm below that. Express clip points in the asset frame.
# clip -> asset frame. This is NOT a pure translation. The asset's x axis is the ladder's WIDTH
# and its y axis is fore-aft; the clip frame has those swapped, so the mapping carries a 90 deg
# yaw. The translation-only mapping this replaces put rung k at 6 / 41 / 115 / 189 / 285 mm from
# the mesh -- growing by the ladder's own 84 mm run once per rung, the signature of exactly this
# error -- and so reported every clip's hands as ~400 mm from a ladder they were touching.
# Yaw and offset were fitted by minimising rung-centre-to-mesh distance: 1.4 mm mean.
YAW_DEG = 90.0
T_CLIP2ASSET = np.array([+0.2442, +0.1491, 0.0])
_c, _s = np.cos(np.radians(YAW_DEG)), np.sin(np.radians(YAW_DEG))
R_CLIP2ASSET = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]])


def to_asset(p):
    """Clip-frame point(s) -> asset frame."""
    return np.asarray(p) @ R_CLIP2ASSET.T + T_CLIP2ASSET


# SELF-TEST. The rung centres were measured FROM this mesh, so they must land back on it. With
# no control the script cannot tell a bad clip from a bad frame -- which is the mistake that
# made its first set of numbers worthless. Refuse to report rather than report confidently wrong.
_rung = np.array([[-0.580 + 0.0839 * k, 0.0, 0.196 + 0.2347 * k] for k in range(5)])
_d, _ = tree.query(to_asset(_rung))
print("frame self-test, rung centre -> mesh: " + "  ".join(f"{x * 1000:.1f}" for x in _d) + " mm")
if _d.max() > 0.010:
    sys.exit("FRAME SELF-TEST FAILED: clip->asset transform is wrong, numbers would be meaningless")

print(f"{'clip':<34}{'held steps':>11}{'median':>9}{'p90':>9}{'max':>9}{'>50mm':>8}")
for cdir in a.clip:
    p = pathlib.Path(cdir)
    d = dict(np.load(p / "motion.npz"))
    meta = json.loads((p / "meta.json").read_text())
    bn = list(meta["body_names"])
    fps = float(meta["fps"])
    oc = d["object_contact"]
    n50 = oc.shape[0]
    lw, rw = bn.index("left_wrist_yaw_link"), bn.index("right_wrist_yaw_link")
    ds = []
    for i in range(0, n50, a.stride):
        f = min(int(i / 50 * fps), d["body_pos_w"].shape[0] - 1)
        for col, bidx in ((0, lw), (1, rw)):
            if int(oc[i, col]) == 0:
                continue
            q = to_asset(d["body_pos_w"][f, bidx])
            ds.append(tree.query(q)[0])
    ds = np.asarray(ds)
    if not len(ds):
        print(f"{p.name:<34}{'none':>11}")
        continue
    print(
        f"{p.name:<34}{len(ds):>11}{np.median(ds) * 1000:>8.0f}m{np.percentile(ds, 90) * 1000:>8.0f}m"
        f"{ds.max() * 1000:>8.0f}m{100 * np.mean(ds > 0.05):>7.0f}%"
    )
