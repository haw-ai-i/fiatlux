"""Convert a loader-format motion.npz into the dof/root_rot/root_trans_offset clip format.

measure_balance.py and replay_clip_kinematic.py read the older "clip" layout; the LadderMan
loader reads motion.npz. Same motion, different packaging. Joints are reordered BY NAME into
retarget_clearance.JOINT_ORDER rather than assumed to already match -- a silent permutation here
would move every limb and still produce a plausible-looking balance report.

    python scripts/climb/motion_to_clip.py --motion <dir> --out <file.npz>
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "teleop" / "fullbody"))
from retarget_clearance import JOINT_ORDER  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--motion", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

p = pathlib.Path(a.motion)
d = dict(np.load(p / "motion.npz"))
meta = json.loads((p / "meta.json").read_text())
names = list(meta["joint_names"])
bodies = list(meta["body_names"])
fps = float(meta["fps"])

missing = [n for n in JOINT_ORDER if n not in names]
if missing:
    raise SystemExit(f"clip is missing joints the balance checker needs: {missing}")
src = [names.index(n) for n in JOINT_ORDER]
if src != list(range(len(JOINT_ORDER))):
    print(f"[conv] reordering joints (clip order != JOINT_ORDER at {sum(i != j for i, j in enumerate(src))} slots)")

pel = bodies.index("pelvis")
np.savez(a.out,
         dof=d["joint_pos"][:, src].astype(np.float32),
         root_rot=d["body_quat_w"][:, pel].astype(np.float32),      # wxyz, as the readers expect
         root_trans_offset=d["body_pos_w"][:, pel].astype(np.float32),
         fps=np.float32(fps))
print(f"[conv] {d['joint_pos'].shape[0]} frames @ {fps:g} fps -> {a.out}")
