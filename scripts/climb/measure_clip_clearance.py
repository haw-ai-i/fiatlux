"""Measure, IN THE SIM, how far the clip's hands and feet are from the REAL ladder.

Why a new script
----------------
Every clearance number so far came from an offline clip->asset transform fitted by minimising
distance to the mesh. That fit is wrong: it slid the clip 0.244 m sideways onto the ladder's
stringer, where 37160 mesh vertices sit within 10 mm, so the rung-centre self-test scored ~1 mm
while sitting at the tread's outer EDGE instead of its centre (tread spans asset x -0.247..+0.247,
centre 0.000).

So stop fitting anything. Place the clip exactly the way the policy sees it -- through
LadderManObs, the same path render_reference.py uses -- and read the ladder's ACTUAL world pose
out of the scene. Distances are then computed against the ladder's own mesh transformed by that
pose, with no fitted quantity anywhere in the chain.

render_reference.py stays runnable; this is a variant, not an edit.
"""
import argparse
import json
import pathlib
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description="In-sim clearance of a reference clip to the ladder.")
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--motion", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--robot", default="dex3", choices=["dex3", "inspire"],
                help="G1 hand variant. The scene default is INSPIRE everywhere -- every task "
                     "in the repo is authored against it and swapped explicitly -- so without "
                     "this the climb work silently ran on the wrong hand. Dex3 is what this "
                     "project uses, so it is the default here.")
ap.add_argument("--layout_seed", type=int, default=0)
ap.add_argument("--stride", type=int, default=4)
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.headless = True
a.enable_cameras = True   # the S02 scene carries a tiled camera; without this the env refuses to build
app = AppLauncher(a).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import fiatlux_task.tasks  # noqa: E402,F401
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ladderman_runtime import LadderManObs  # noqa: E402

set_layout_seed(a.layout_seed)
cfg = parse_env_cfg(a.task, num_envs=1)
if a.robot != "inspire":
    from fiatlux_task.robots.g1 import swap_robot_variant
    swap_robot_variant(cfg, a.robot)
print(f"[robot] {a.robot}: {cfg.scene.robot.spawn.usd_path.split('/')[-1]}")
env = gym.make(a.task, cfg=cfg).unwrapped
env.reset()
base = env
dev = base.device
robot = base.scene["robot"]
ob = LadderManObs(base, a.motion)

lad = base.scene["ladder"].data
lad_p = lad.root_pos_w[0].cpu().numpy().astype(float)
lad_q = lad.root_quat_w[0].cpu().numpy().astype(float)
print(f"[ladder] world pos {lad_p.round(4)}  quat(wxyz) {lad_q.round(4)}")

names = ["left_wrist_yaw_link", "right_wrist_yaw_link",
         "left_ankle_roll_link", "right_ankle_roll_link", "pelvis"]
bidx = [robot.body_names.index(n) for n in names]

T = int(ob.motion_len)
rec = []
for t in range(0, T, a.stride):
    tt = torch.tensor([t], device=dev, dtype=torch.long)
    root_p = ob.ref_body_pos[tt][:, ob.root_mo]
    root_q = ob.ref_body_quat[tt][:, ob.root_mo]
    robot.write_root_pose_to_sim(torch.cat([root_p, root_q], dim=-1))
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
    q = ob.ref_joint_pos[tt]
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ob.j_fx)
    base.sim.forward()
    rec.append(robot.data.body_pos_w[0, bidx].cpu().numpy().astype(float))
    if t % 400 == 0:
        print(f"[frame] {t}/{T}", flush=True)

P = np.stack(rec)
o = pathlib.Path(a.out)
o.parent.mkdir(parents=True, exist_ok=True)
np.savez(o, body_pos_w=P, names=np.array(names), ladder_pos=lad_p, ladder_quat=lad_q,
         stride=a.stride, motion_len=T)
print(f"[done] {P.shape[0]} frames -> {o}")
env.close()
