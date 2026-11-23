"""Build a live S02 env, compute all four LadderMan observation groups, and check the widths.

This is the first point where the layout contract (ladderman_obs.py, checked against the
checkpoint) meets real tensors. A width mismatch here means a term was computed with the wrong
shape; the layout check alone cannot see that.
"""

import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--motion", default="/home/yujin-chen/psi0_ft/climb/topout")
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import fiatlux_task.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import ladderman_obs  # noqa: E402
import torch  # noqa: E402
from ladderman_runtime import LadderManObs  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(a.task, num_envs=1)
env = gym.make(a.task, cfg=cfg).unwrapped
env.reset()

obs = LadderManObs(env, a.motion)
print(f"[FRAME ] rung0 world {[round(v, 4) for v in obs.rung0_w[0].tolist()]}")
print(
    f"[FRAME ] climb yaw {float(obs.climb_yaw[0]):+.4f} rad "
    f"({float(obs.climb_yaw[0]) * 57.2958:+.1f} deg); dz {obs.dz * 1000:+.1f} mm"
)
print(f"[MOTION] {obs.motion_len} steps at 50 Hz = {obs.motion_len / 50:.2f} s")
print(f"[MAP   ] 29 joints, {len(obs.track_bodies)} tracking bodies, {len(obs.track_joints)} tracking joints")

obs.update()
g = obs.groups()
bad = []
for name, t in g.items():
    want = ladderman_obs.width(name)
    got = t.shape[-1]
    flag = "OK " if want == got else "BAD"
    if want != got:
        bad.append(f"{name}: computed {got}, contract {want}")
    finite = bool(torch.isfinite(t).all())
    print(
        f"[{flag}] {name:8s} computed {got:5d}  contract {want:5d}  finite={finite}  "
        f"range [{t.min():+.3f}, {t.max():+.3f}]"
    )

print()
print("per-term ranges at reset (robot staged at the ladder, reference at phase 0):")
for name, t in g.items():
    for term, lo, hi in ladderman_obs.offsets(name):
        v = t[:, lo:hi]
        print(
            f"  {name:8s} {term:32s} [{float(v.min()):+9.3f}, {float(v.max()):+9.3f}]"
            f"  |mean| {float(v.abs().mean()):7.3f}"
        )

if bad:
    raise SystemExit("WIDTH MISMATCH:\n  " + "\n  ".join(bad))
print("[OK] all four groups computed at the contracted widths")
app.close()
