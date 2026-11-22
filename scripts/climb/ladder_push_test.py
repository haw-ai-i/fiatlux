"""Push the benchmark's free-standing ladder the way a landing hand does, with no robot touching it.

In S02 the ladder tips 1.7-3.3 deg (median 2.5, 29 recorded runs) when the hands land on tread 4;
the same ladder asset in the LadderMan trainer tips 0.0 deg under the same teacher. This applies a
known wrench to the ladder alone -- the measured landing impact, 84 N down and 46 N along the climb
direction at tread 4's nose for two control steps, then 20 N down and 10 N forward held -- and
prints the tilt, so the two simulators can be compared on the ladder and floor alone. The twin for
the trainer is ladderman_port/play_probe_push.py.

    python scripts/climb/ladder_push_test.py --headless
"""
import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--layout_seed", type=int, default=0)
ap.add_argument("--impulse", type=float, nargs=3, default=[84.0, 46.0, 2], metavar=("DOWN", "FORWARD", "STEPS"), help="the sharp push, N and control steps")
ap.add_argument("--hold", type=float, nargs=2, default=[20.0, 10.0], metavar=("DOWN", "FORWARD"), help="held afterwards, N")
ap.add_argument("--free_robot", action="store_true", help="leave the robot's root free (with no policy it falls onto the ladder in about 0.6 s and that is what tips it)")
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

import fiatlux_task.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.robots.g1 import swap_robot_variant  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

set_layout_seed(a.layout_seed)
cfg = parse_env_cfg(a.task, num_envs=1)
swap_robot_variant(cfg, "dex3")
if not a.free_robot:
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True      # keep the robot out of it
env = gym.make(a.task, cfg=cfg).unwrapped
env.reset()
lad = env.scene["ladder"]
dev = env.device
px = env.sim.cfg.physx
print(f"[push] sim: dt {env.physics_dt} x {env.cfg.decimation}; solver_type {px.solver_type}, min pos/vel iterations {px.min_position_iteration_count}/{px.min_velocity_iteration_count}, "
      f"stabilization {px.enable_stabilization}, bounce threshold {px.bounce_threshold_velocity}")
v = lad.root_physx_view
print(f"[push] ladder mass {float(v.get_masses().sum()):.2f} kg, com {[round(float(x), 3) for x in v.get_coms()[0].flatten()[:3]]}, inertia {[round(float(x), 3) for x in v.get_inertias()[0].flatten()[[0, 4, 8]]]}")
try:
    mp = v.get_material_properties()[0]
    print(f"[push] ladder shape materials (static, dynamic, restitution): {[[round(float(x), 2) for x in r] for r in mp.reshape(-1, 3)[:4].tolist()]}")
except Exception as e:  # noqa: BLE001
    print("[push] material read failed:", e)
R_COM = torch.tensor([0.0, -0.1694 - (-0.012), 0.944 - 0.826])         # tread 4's nose from the com, ladder frame


def wrench(down, fwd):
    F = torch.tensor([0.0, fwd, -down])                                 # ladder frame: +y is the climb direction
    return F, torch.linalg.cross(R_COM, F)


def tilt():
    g = lad.data.projected_gravity_b[0]
    return float(torch.rad2deg(torch.atan2(g[1], -g[2]))), float(torch.rad2deg(torch.atan2(g[0], -g[2])))


zero = torch.zeros((1, env.action_space.shape[-1]), device=dev)
p0 = lad.data.root_pos_w[0].clone()
for k in range(75):
    if 10 <= k < 10 + int(a.impulse[2]):
        F, T = wrench(a.impulse[0], a.impulse[1])
    elif 10 + int(a.impulse[2]) <= k < 60:
        F, T = wrench(a.hold[0], a.hold[1])
    else:
        F, T = wrench(0.0, 0.0)
    lad.set_external_force_and_torque(F.to(dev)[None, None], T.to(dev)[None, None])
    env.step(zero)
    if k % 3 == 0 or k in (10, 11, 12):
        tx, ty = tilt()
        d = (lad.data.root_pos_w[0] - p0) * 1000
        print(f"[push] step {k:3d}  push down/forward {float(-F[2]):.0f}/{float(F[1]):.0f} N  tilt about x {tx:+.2f} about y {ty:+.2f} deg  origin moved z {float(d[2]):+.1f} mm, |xy| {float(d[:2].norm()):.1f} mm", flush=True)
env.close()
app.close()
