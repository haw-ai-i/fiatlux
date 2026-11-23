"""Measure the G1 arm chain so a reference can be re-authored with analytic IK, offline.

Re-aiming the hands at the ladder's rails means solving shoulder/elbow angles for a wrist
position. Doing that inside Isaac per frame is thousands of FK round-trips; doing it offline
needs only the chain's fixed geometry, which is what this dumps -- once.

Reports, in the PELVIS frame at the default pose: the shoulder origin, and the upper-arm and
forearm lengths taken between link origins (shoulder_pitch -> elbow -> wrist_yaw).
"""

import argparse
import json
import pathlib

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--out", default="/home/yujin-chen/psi0_ft/climb/arm_chain.json")
ap.add_argument("--layout_seed", type=int, default=0)
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

import fiatlux_task.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

set_layout_seed(a.layout_seed)
env = gym.make(a.task, cfg=parse_env_cfg(a.task, num_envs=1)).unwrapped
env.reset()
rob = env.scene["robot"]
bn = list(rob.body_names)
P = rob.data.body_pos_w[0].cpu().numpy().astype(float)
Q = rob.data.body_quat_w[0].cpu().numpy().astype(float)
root = P[bn.index("pelvis")]

out = {"body_names": bn, "n_bodies": len(bn), "n_joints": len(rob.joint_names)}
for side in ("left", "right"):
    sh = f"{side}_shoulder_pitch_link"
    el = f"{side}_elbow_link"
    wr = f"{side}_wrist_yaw_link"
    missing = [n for n in (sh, el, wr) if n not in bn]
    if missing:
        print(f"[arm] MISSING {missing}")
        continue
    ps, pe, pw = P[bn.index(sh)], P[bn.index(el)], P[bn.index(wr)]
    out[side] = {
        "shoulder_in_pelvis": (ps - root).tolist(),
        "upper_arm_len": float(np.linalg.norm(pe - ps)),
        "forearm_len": float(np.linalg.norm(pw - pe)),
        "shoulder_to_wrist_default": float(np.linalg.norm(pw - ps)),
        "wrist_in_pelvis": (pw - root).tolist(),
    }
    print(
        f"[arm] {side:5s} shoulder {np.round(ps - root, 4)}  upper {out[side]['upper_arm_len']:.4f} m  "
        f"fore {out[side]['forearm_len']:.4f} m  reach(default) {out[side]['shoulder_to_wrist_default']:.4f} m"
    )

# the palm/fingertip beyond the wrist, which is what actually meets the rail
for cand in ("left_hand_palm_link", "L_hand_base_link", "left_wrist_yaw_link"):
    if cand in bn:
        out["palm_probe"] = {
            "name": cand,
            "offset_from_wrist": (P[bn.index(cand)] - P[bn.index("left_wrist_yaw_link")]).tolist(),
        }
        break
out["total_mass_kg"] = float(rob.root_physx_view.get_masses()[0].sum())
print(f"[arm] bodies {len(bn)}  joints {len(rob.joint_names)}  mass {out['total_mass_kg']:.2f} kg")
pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
print(f"[arm] -> {a.out}")
env.close()
app.close()
