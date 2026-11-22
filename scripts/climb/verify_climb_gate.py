"""Truth table for the climb success gate (S02 / S10), on staged poses in the live task.

The gate is `sustained(all_of(...))`: pelvis up at the ladder's top, BOTH feet up at the platform
and still, every corner of both soles over the platform, robot standing, ladder upright; held
0.5 s; and the episode ends 2 s after that. This stages poses of a reference climb in the real
scene (robot written kinematically, ladder made kinematic so nothing moves) and checks each
condition says what the geometry says, then that the termination fires on the step it should.

    python scripts/climb/verify_climb_gate.py --old <clip ending near the edge> --new <clip with the tail> --headless

`--old` is a climb whose last foot hovers over the platform before landing and whose heels
overhang the open edge until its last steps (rebuild 72/74); `--new` ends in the next stage's
start stance (rebuild 76). Steps are 50 Hz clip steps; `--old_steps` names the five moments.
"""
import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--old", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--old_steps", type=int, nargs=5, default=[1250, 1346, 1364, 1400, 1478],
                help="tread-4 stance, pelvis first over the height bound, last foot hovering, planted with heels over the edge, final")
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
from fiatlux_task.tasks.manager_based.fiatlux_task.subtask_env_cfg import SUCCESS_LINGER_SECONDS  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from ladderman_runtime import LadderManObs  # noqa: E402

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
print(f"[gate] {a.task}, layout seed {a.layout_seed}: term {getattr(term.func, '__name__', term.func)}, "
      f"hold {term.params.get('seconds')} s, linger {term.params.get('linger_seconds')} s")
print("[gate] conjuncts: " + ", ".join(names))
ob = {"old": LadderManObs(env, a.old), "new": LadderManObs(env, a.new)}
fails = []


def stage(which, step, root_vel=(0.0, 0.0, 0.0)):
    o = ob[which]
    t = torch.tensor([min(step, o.motion_len - 1)], device=dev, dtype=torch.long)
    robot.write_root_link_pose_to_sim(torch.cat([o.ref_body_pos[t][:, o.root_mo], o.ref_body_quat[t][:, o.root_mo]], -1))
    robot.write_root_com_velocity_to_sim(torch.tensor([[*root_vel, 0.0, 0.0, 0.0]], device=dev))
    q = robot.data.joint_pos.clone() * 0.0
    q[:, o.j_fx] = o.ref_joint_pos[t]
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.0)


def table():
    return {n: bool(f(env, **(p or {}))[0]) for (f, p), n in zip(conj, names)}


def check(label, got, want):
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    print(f"[{'PASS' if not bad else 'FAIL'}] {label}: " + " ".join(f"{k}={got[k]}" for k in names)
          + ("" if not bad else f"   <-- expected {bad}"))
    if bad:
        fails.append(label)


s_tread4, s_height, s_hover, s_overhang, s_end = a.old_steps
T, F = True, False
print("\n[gate] (1) each condition against the geometry")
stage("old", s_tread4)
check(f"old clip step {s_tread4}: standing on tread 4", table(),
      {"climbed_to_ladder_top": F, "feet_on_ladder_top": F, "soles_within_ladder_top": F})
stage("old", s_height)
check(f"old clip step {s_height}: pelvis over the height bound, last foot still below the platform", table(),
      {"climbed_to_ladder_top": T, "feet_on_ladder_top": F})
stage("old", s_hover)
check(f"old clip step {s_hover}: last foot hovering over the platform", table(),
      {"climbed_to_ladder_top": T, "feet_on_ladder_top": F})
stage("old", s_overhang)
check(f"old clip step {s_overhang}: both feet planted, heels over the open edge", table(),
      {"climbed_to_ladder_top": T, "feet_on_ladder_top": T, "soles_within_ladder_top": F})
stage("old", s_end)
check(f"old clip step {s_end}: final stance, soles just inside", table(), {n: T for n in names})
stage("new", 10 ** 9)
check("new clip, last frame: the next stage's start stance", table(), {n: T for n in names})
stage("new", 10 ** 9, root_vel=(0.0, 0.0, 0.5))
check("same stance moving at 0.5 m/s: feet are not still", table(), {"feet_on_ladder_top": F})

print("\n[gate] (2) the termination: hold, then linger")
dt = env.step_dt
hold, linger = round(float(term.params["seconds"]) / dt), round(float(term.params.get("linger_seconds", 0.0)) / dt)
tm = env.termination_manager
tm.reset(torch.arange(1, device=dev))


def run(script):
    """script: list of (clip, step, n calls). Returns the call index at which `success` first reads True."""
    k, fired = 0, None
    for which, step, n in script:
        for _ in range(n):
            stage(which, step)
            tm.compute()
            if fired is None and bool(tm.get_term("success")[0]):
                fired = k
            k += 1
    return fired


def check_fire(label, got, want):
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: success first true at call {got} (expected {want})")
    if not ok:
        fails.append(label)


check_fire(f"final stance held: fires after hold {hold} + linger {linger} steps",
           run([("new", 10 ** 9, hold + linger + 20)]), max(hold, 1) + linger - 1)
tm.reset(torch.arange(1, device=dev))
check_fire("gate met for the hold, then the robot leaves the stance: still fires at the same step (latched)",
           run([("new", 10 ** 9, max(hold, 1)), ("old", s_tread4, linger + 20)]), max(hold, 1) + linger - 1)
tm.reset(torch.arange(1, device=dev))
check_fire("gate met for less than the hold, then lost: never fires",
           run([("new", 10 ** 9, max(hold, 1) - 1), ("old", s_tread4, linger + 60)]), None)
tm.reset(torch.arange(1, device=dev))
check_fire("heels over the edge the whole time: never fires",
           run([("old", s_overhang, hold + linger + 60)]), None)
tm.reset(torch.arange(1, device=dev))
check_fire("old clip's own final stance: fires", run([("old", s_end, hold + linger + 20)]), max(hold, 1) + linger - 1)
assert abs(float(term.params.get("linger_seconds", 0.0)) - SUCCESS_LINGER_SECONDS) < 1e-9

print(f"\n[gate] {'ALL PASS' if not fails else 'FAILED: ' + '; '.join(fails)}")
sys.stdout.flush()
env.close()
app.close()
