"""Re-simulate the ladder from the moment an operator let go of it in a recorded S01 take.

A recorded carry cannot be replayed (the operator and the walking controller are not in the bag),
but the ladder's own state is: position, orientation and both velocities at every step. This
takes that state at the last step a hand gripped the ladder, puts the ladder there with nothing
touching it (robot pinned at its spawn), and lets it settle -- once with the ladder's
stabilization_threshold as shipped, once with another value -- and compares each against what the
bag itself recorded next. The ladder is moved to this layout's own ladder spot, keeping its height,
tilt and velocities.

    python scripts/climb/ladder_release_replay.py --bags <take dirs or run.h5 files...> --threshold 0.001 --headless
"""
import argparse
import glob
import pathlib

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="FIATLUX-S01-MoveLadder-v0")
ap.add_argument("--bags", nargs="+", required=True)
ap.add_argument("--threshold", type=float, default=None, help="ladder stabilization_threshold (default: as shipped)")
ap.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

import fiatlux_task.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.robots.g1 import swap_robot_variant  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

set_layout_seed(0)
cfg = parse_env_cfg(a.task, num_envs=1)
swap_robot_variant(cfg, "dex3")
cfg.scene.robot.spawn.articulation_props.fix_root_link = True
if a.threshold is not None:
    cfg.scene.ladder.spawn.rigid_props.stabilization_threshold = a.threshold
print(f"[replay] ladder stabilization_threshold {cfg.scene.ladder.spawn.rigid_props.stabilization_threshold}", flush=True)
env = gym.make(a.task, cfg=cfg).unwrapped
env.reset()
lad = env.scene["ladder"]; dev = env.device
home = lad.data.root_pos_w[0].clone()
zero = torch.zeros((1, env.action_space.shape[-1]), device=dev)


def tilt_q(q):
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))))


files = []
for b in a.bags:
    files += sorted(glob.glob(b)) if any(c in b for c in "*?") else [b]
for f in files:
    f = pathlib.Path(f); f = f / "run.h5" if f.is_dir() else f
    g = h5py.File(f, "r")["data/demo_0"]; n = len(g["ladder_pos"])
    P = np.asarray(g["ladder_pos"]).reshape(n, 3); Q = np.asarray(g["ladder_quat"]).reshape(n, 4)
    LV = np.asarray(g["ladder_lin_vel"]).reshape(n, 3); AV = np.asarray(g["ladder_ang_vel"]).reshape(n, 3)
    grip = np.asarray(g["grip_force"]).reshape(n, -1).max(1)
    held = np.flatnonzero(grip > 5.0)
    rel = int(held.max()) + 1 if len(held) else 0
    if rel >= n - 5:
        print(f"[replay] {f.parent.name}: the ladder is still gripped at the end of the bag; skipped", flush=True)
        continue
    bag_t = [tilt_q(Q[i]) for i in range(rel, n)]
    bag_sp = np.linalg.norm(LV[rel:], axis=1) * 1000
    st = torch.zeros(1, 13, device=dev)
    st[0, :3] = torch.tensor([float(home[0]), float(home[1]), float(P[rel, 2])], device=dev)
    st[0, 3:7] = torch.tensor(Q[rel], device=dev, dtype=torch.float32)
    st[0, 7:10] = torch.tensor(LV[rel], device=dev, dtype=torch.float32); st[0, 10:13] = torch.tensor(AV[rel], device=dev, dtype=torch.float32)
    env.reset()
    lad.write_root_state_to_sim(st)
    ts, sp = [], []
    for k in range(a.steps):
        env.step(zero)
        ts.append(tilt_q(lad.data.root_quat_w[0].cpu().numpy())); sp.append(float(lad.data.root_lin_vel_w[0].norm()) * 1000)
    ts, sp = np.array(ts), np.array(sp)
    quiet = lambda s: next((i for i in range(len(s) - 25) if s[i:i + 25].max() < 10.0), None)      # first 0.5 s under 10 mm/s
    qs, qb = quiet(sp), quiet(bag_sp)
    m = min(len(bag_t), a.steps)
    print(f"[replay] {str(f.parent).split('vr/')[-1]}: released at step {rel} of {n} (tilt {bag_t[0]:.2f} deg, speed {bag_sp[0]:.0f} mm/s). "
          f"BAG next {len(bag_t)} steps: tilt max {max(bag_t):.2f}, end {bag_t[-1]:.2f} deg, quiet after {('%.2f s' % (qb * 0.02)) if qb is not None else 'not within the bag'}. "
          f"REPLAY {a.steps} steps: tilt max {ts.max():.2f}, end {ts[-1]:.2f} deg, quiet after {('%.2f s' % (qs * 0.02)) if qs is not None else 'never'}"
          + (f"; over the bag's own {m} steps the replay's max tilt is {ts[:m].max():.2f}" if m > 5 else "")
          + ("; FELL OVER" if ts[-1] > 45 else ""), flush=True)
env.close()
app.close()
