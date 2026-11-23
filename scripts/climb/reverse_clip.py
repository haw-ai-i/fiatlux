"""Play a climb clip backwards: the first draft of a DESCENT (S04 / S12).

Every pose of a checked climb is a pose the robot can hold on this ladder, with the same foot and hand
placements, so the reversed clip starts from poses that are already known to clear the ladder. What
reversal does NOT preserve is the dynamics: lowering onto a tread loads the legs differently from
pushing up off it, and a foot that was "about to land" is now "just lifted". The reversed clip gets
the same suite as a climb before anything is trained on it.

    python scripts/climb/reverse_clip.py --clip <climb> --out <descent> --keep 265 1906

`--keep FROM TO` keeps the climb's 50 Hz steps FROM..TO (before reversing), so `--keep 265 1906` on
rebuild 87 drops its walk-in: the descent then ends in the stance at the ladder's base instead of
walking backwards to the climb's spawn spot, which is outside the descent gate's 0.6 m radius.

Positions and orientations are reversed in time; every velocity is reversed AND negated; the per-step
hold labels (`object_contact`) are reversed. Holds and anchors are properties of the ladder and are
copied unchanged.
"""

import argparse
import json
import pathlib
import shutil

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--out", required=True)
ap.add_argument(
    "--keep",
    type=int,
    nargs=2,
    default=None,
    metavar=("FROM", "TO"),
    help="50 Hz steps of the climb to keep before reversing (default: all)",
)
a = ap.parse_args()

src, dst = pathlib.Path(a.clip), pathlib.Path(a.out)
d = np.load(src / "motion.npz")
meta = json.loads((src / "meta.json").read_text())
fps = float(meta["fps"])
per = fps / 50.0
nf = d["joint_pos"].shape[0]
ns = d["object_contact"].shape[0]
s0, s1 = a.keep if a.keep is not None else (0, ns)
s1 = min(s1, ns)
f0, f1 = int(round(s0 * per)), min(nf, int(round(s1 * per)))
out = {}
for k in d.files:
    v = d[k]
    if k == "object_contact":
        out[k] = v[s0:s1][::-1].copy()
    elif k in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        out[k] = -v[f0:f1][::-1]
    else:
        out[k] = v[f0:f1][::-1].copy()
# the stored velocities must be the reversed clip's own derivatives: check against a finite difference
q = out["joint_pos"]
fd = np.gradient(q, 1.0 / fps, axis=0)
err = float(np.abs(fd[2:-2] - out["joint_vel"][2:-2]).max())
p = out["body_pos_w"]
fdp = np.gradient(p, 1.0 / fps, axis=0)
errp = float(np.abs(fdp[2:-2] - out["body_lin_vel_w"][2:-2]).max())
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "motion.npz":
        shutil.copy(f, dst / f.name)
np.savez(dst / "motion.npz", **out)
bn = meta["body_names"]
print(f"[rev  ] kept climb steps {s0}-{s1} ({f1 - f0} frames), reversed: {out['object_contact'].shape[0]} steps")
print(f"[rev  ] starts at pelvis {np.round(p[0, 0], 3).tolist()}, ends at pelvis {np.round(p[-1, 0], 3).tolist()}")
print(
    f"[rev  ] stored joint velocity against the reversed clip's own finite difference: worst {err:.3f} rad/s; "
    f"body velocity worst {errp:.3f} m/s"
)
print(
    f"[rev  ] peak joint speed {np.abs(out['joint_vel']).max():.2f} rad/s (the climb's {np.abs(d['joint_vel'][f0:f1]).max():.2f})"
)
print(f"[out  ] {dst}")
