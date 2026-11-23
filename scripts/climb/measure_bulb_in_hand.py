"""Where does the benchmark put the bulb in the right hand at an S10 start?

Resets FIATLUX-S10-ClimbWithBulb-v0 (Dex3), lets its own settle event seat the bulb, and prints the
bulb's pose in the right wrist_yaw_link frame -- the frame the climb clips are built in -- with the
finger angles. Offline numbers baked in grasp_poses.py are a target; the event re-seats it live.

    python scripts/climb/measure_bulb_in_hand.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="FIATLUX-S10-ClimbWithBulb-v0")
ap.add_argument("--steps", type=int, default=3)
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
app = AppLauncher(a).app
import fiatlux_task.tasks  # noqa: E402,F401
import gymnasium as gym
import numpy as np
import torch  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

set_layout_seed(0)
cfg = parse_env_cfg(a.task, num_envs=1)
from fiatlux_task.robots.g1 import swap_robot_variant  # noqa: E402

swap_robot_variant(cfg, "dex3")
env = gym.make(a.task, cfg=cfg)
base = env.unwrapped
base.reset()
robot, bulb = base.scene["robot"], base.scene["fresh_bulb"]


def report(tag):
    i = robot.body_names.index("right_wrist_yaw_link")
    wp, wq = robot.data.body_link_pos_w[0, i], robot.data.body_link_quat_w[0, i]
    bp, bq = bulb.data.root_link_pos_w[0], bulb.data.root_link_quat_w[0]
    loc = quat_apply_inverse(wq[None], (bp - wp)[None])[0]
    lq = quat_mul(quat_conjugate(wq[None]), bq[None])[0]
    ax = quat_apply(lq[None], torch.tensor([[0.0, 0.0, 1.0]], device=wp.device))[0]
    pn = quat_apply(wq[None], torch.tensor([[0.0, 1.0, 0.0]], device=wp.device))[0]
    print(
        f"[bulb] {tag}: root in wrist_yaw frame pos {np.round(loc.cpu().numpy(), 4).tolist()} quat(wxyz) {np.round(lq.cpu().numpy(), 4).tolist()}",
        flush=True,
    )
    print(
        f"[bulb] {tag}: bulb axis in wrist frame {np.round(ax.cpu().numpy(), 3).tolist()}; palm normal (link +y) in world {np.round(pn.cpu().numpy(), 2).tolist()}; "
        f"bulb height above ground {float(bp[2]):.3f}",
        flush=True,
    )
    for nm, dz in (("cap bottom", 0.036), ("cap centre", 0.054), ("glass centre", 0.131), ("glass top", 0.171)):
        print(f"[bulb]    {nm}: {np.round((loc + ax * dz).cpu().numpy(), 3).tolist()}", flush=True)


report("after reset")
fj = [
    k
    for k, n in enumerate(robot.joint_names)
    if n.startswith("right_hand_")
    or n.startswith("right_wrist")
    or n.startswith("right_shoulder")
    or n.startswith("right_elbow")
]
print(
    "[bulb] right arm/hand joints: "
    + ", ".join(
        f"{robot.joint_names[k].replace('right_', '').replace('_joint', '')} {float(robot.data.joint_pos[0, k]):+.2f}"
        for k in fj
    ),
    flush=True,
)
act = torch.zeros((1, base.action_manager.total_action_dim), device=base.device)
with torch.inference_mode():
    for s in range(a.steps):
        base.step(act)
report(f"after {a.steps} zero-action steps")
env.close()
app.close()
