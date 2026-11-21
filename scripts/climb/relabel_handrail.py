"""A hand on the handrail's front edge is not a hold until both hands are there.

From tread 2 the handrail can only be reached on the finger pads, hand pitched up against the
rim's rounded near edge. Measured in the trainer (64 robots, iteration 900): a lone left hand
there pushes on the rim sideways, median (-18, +2) N, so LadderMan's hold rule -- upward force
over 5 N within 60 deg of vertical -- reads it as "not holding" for the 2 s the right hand is in
transit, and 17 of 28 robots are terminated at steps 750-800 with the feet carrying 100% of the
weight and the pelvis within 1 cm of the reference. Once both hands are on, they do bear load.

This rewrites only the contact LABELS: the first hand's handrail label starts when the second
hand's does, plus --settle seconds. The motion is untouched.

    python relabel_handrail.py --clip <in dir> --out <out dir> [--settle 0.4]
"""
import argparse, json, pathlib, shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--label", type=int, default=6, help="the handrail's contact label")
ap.add_argument("--settle", type=float, default=0.4, help="s after the second hand arrives")
a = ap.parse_args()
src, dst = pathlib.Path(a.clip), pathlib.Path(a.out)
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "motion.npz":
        shutil.copy(f, dst / f.name)
d = dict(np.load(src / "motion.npz"))
oc = d["object_contact"].copy()
first = [int(np.argmax(oc[:, h] == a.label)) for h in (0, 1)]
late = max(first) + int(round(a.settle * 50))
for h in (0, 1):
    cut = (np.arange(len(oc)) < late) & (oc[:, h] == a.label)
    print(f"{'LR'[h]}: handrail label was from step {first[h]}; now from step {late} ({int(cut.sum())} steps cleared)")
    oc[cut, h] = 0
d["object_contact"] = oc
np.savez(dst / "motion.npz", **d)
hj = dst / "holds.json"
if hj.exists():
    h = json.loads(hj.read_text()); h["relabel_handrail"] = dict(settle=a.settle, from_step=late); hj.write_text(json.dumps(h, indent=1))
print("motion arrays unchanged:", all(np.array_equal(d[k], np.load(src / "motion.npz")[k]) for k in d if k != "object_contact"))
