"""What the descent success gate (S04 / S12) says along a descent clip, on staged poses in the live task.

The gate is `all_of(descended_from_ladder, <bulb conjunct>, robot_standing, ladder_near_vertical)`.
`descended_from_ladder` wants the pelvis under the floor-stance height, within 0.6 m of the ladder's
ROOT in the horizontal plane, and slower than 1.5 m/s. The climb's mount stance is 0.614 m out and
S02's spawn spot 0.85 m, so where a descent clip ENDS decides whether it can score at all.

This stages poses of the clip kinematically (robot written, ladder made kinematic, nothing moves) and
prints every conjunct, the pelvis's distance from the ladder's root, and the first clip step at which
`descended_from_ladder` is true. The bulb conjunct is reported as it reads but not staged: only the
robot is written, so it says nothing about the bulb.

    python scripts/climb/verify_descent_gate.py --task FIATLUX-S12-ClimbDown-v0 --clip <descent> --headless
"""

import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="FIATLUX-S12-ClimbDown-v0")
ap.add_argument("--clip", required=True)
ap.add_argument(
    "--steps", type=int, nargs="*", default=None, help="clip steps to tabulate (default: start, quarter points, end)"
)
ap.add_argument("--layout_seed", type=int, default=0)
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
import fiatlux_task.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.robots.g1 import swap_robot_variant  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task import mdp  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402
from ladderman_runtime import LadderManObs  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

set_layout_seed(a.layout_seed)
cfg = parse_env_cfg(a.task, num_envs=1)
swap_robot_variant(cfg, "dex3")
cfg.scene.ladder.spawn.rigid_props.kinematic_enabled = True
env = gym.make(a.task, cfg=cfg).unwrapped
env.reset()
dev = env.device
robot = env.scene["robot"]
term = env.cfg.terminations.success
conj = mdp.conjuncts_of(term.func, term.params)
names = [getattr(f, "__name__", "?") for f, _ in conj]
print(
    f"[gate] {a.task}, layout seed {a.layout_seed}: term {getattr(term.func, '__name__', term.func)}, "
    f"hold {term.params.get('seconds')} s, linger {term.params.get('linger_seconds')} s"
)
print("[gate] conjuncts: " + ", ".join(names))
for (f, p), n in zip(conj, names):
    if n == "descended_from_ladder":
        print(f"[gate] descended_from_ladder parameters: {p}")
o = LadderManObs(env, a.clip)
lad = env.scene["ladder"]


def stage(step):
    t = torch.tensor([min(step, o.motion_len - 1)], device=dev, dtype=torch.long)
    robot.write_root_link_pose_to_sim(
        torch.cat([o.ref_body_pos[t][:, o.root_mo], o.ref_body_quat[t][:, o.root_mo]], -1)
    )
    robot.write_root_com_velocity_to_sim(torch.zeros(1, 6, device=dev))
    q = robot.data.joint_pos.clone() * 0.0
    q[:, o.j_fx] = o.ref_joint_pos[t]
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.0)


def table():
    return {n: bool(f(env, **(p or {}))[0]) for (f, p), n in zip(conj, names)}


def where():
    d = float((robot.data.root_pos_w[0, :2] - lad.data.root_pos_w[0, :2]).norm())
    return d, float(robot.data.root_pos_w[0, 2])


n = o.motion_len
steps = a.steps or [0, n // 4, n // 2, 3 * n // 4, n - 1]
for s in steps:
    stage(s)
    g = table()
    d, z = where()
    print(
        f"[gate] clip step {s:5d}: pelvis {d:.3f} m from the ladder's root, height {z:.3f} m | "
        + " ".join(f"{k}={v}" for k, v in g.items())
    )
first = None
for s in range(0, n, 5):
    stage(s)
    if table().get("descended_from_ladder"):
        first = s
        break
stage(n - 1)
g = table()
d, z = where()
others = {k: v for k, v in g.items() if k != "descended_from_ladder"}
print(f"[gate] descended_from_ladder first true at clip step {first} of {n} (searched every 5 steps)")
print(
    f"[{'PASS' if g.get('descended_from_ladder') else 'FAIL'}] last frame: descended_from_ladder={g.get('descended_from_ladder')}, "
    f"pelvis {d:.3f} m from the root (limit 0.6), height {z:.3f} m; the other conjuncts read {others}"
)
env.close()
app.close()
