# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise ``FIATLUX-S01-MoveLadder-v0``'s start state and success gate, with PASS/FAIL per item.

``verify_scene.py`` covers the scene (assets, colliders, settling, penetration); this covers the
start state and the success gate.

The gate is read through the env's own termination manager, so what is measured is what a rollout
scores. The ladder's root pose is written once per case and left to settle -- re-writing it every
step holds ``object_at_rest`` false. The robot's root is pinned each step, so the cases measure
the ladder conjuncts rather than whether an unactuated G1 stays upright.

The "at the fixture" pose is the one the rest of the chain spawns
(``apply_replace_preset(couple_ladder_to_fixture=True)``, read off S02 at the same layout seed):
a gate that does not fire there means S01 and its successors disagree about where the ladder goes.

Examples
--------
    uv run python scripts/verify_move_ladder.py --headless --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify S01-MoveLadder's start state and success gate.")
parser.add_argument("--seed", type=int, default=0, help="Layout seed (the room layout is drawn once, at cfg build).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# The frozen policy observation group carries an RGB term, so the env needs camera rendering.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import fiatlux_task.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import rewards  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import (  # noqa: E402
    LADDER_READY_XY_RADIUS,
    set_layout_seed,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.subtasks.s01_move_ladder_env_cfg import (  # noqa: E402
    LADDER_MOVED_CONJUNCTS,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.subtasks.s02_climb_ladder_env_cfg import (  # noqa: E402
    S02ClimbLadderEnvCfg,
)

from isaaclab.managers import SceneEntityCfg  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "FIATLUX-S01-MoveLadder-v0"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} -- {detail}")


# The successor's ladder pose at the same seed: the state this subtask must produce.
set_layout_seed(args_cli.seed)
chain_ladder = S02ClimbLadderEnvCfg().scene.ladder.init_state
set_layout_seed(args_cli.seed)
env_cfg = parse_env_cfg(TASK, num_envs=1)
env = gym.make(TASK, cfg=env_cfg).unwrapped

ladder = env.scene["ladder"]
robot = env.scene["robot"]
zero_action = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
sustain_s = env_cfg.success_params["seconds"]
sustain_steps = round(sustain_s / env.step_dt)


def conjuncts() -> dict[str, bool]:
    return {fn.__name__: bool(fn(env, **params)[0]) for fn, params in LADDER_MOVED_CONJUNCTS}


def ladder_fixture_xy() -> float:
    delta = rewards._ladder_top_point_w(env) - rewards._seat_point_w(env)
    return float(torch.norm(delta[:, :2], dim=1)[0])


def stage_ladder(*, dz: float = 0.0, tilt: float = 0.0, at_fixture: bool = True) -> None:
    """Write the ladder's root pose once: at the chain's placement, raised, and/or rolled."""
    state = ladder.data.root_state_w.clone()
    if at_fixture:
        state[0, :2] = torch.tensor(chain_ladder.pos[:2], device=env.device) + env.scene.env_origins[0, :2]
        state[0, 3:7] = torch.tensor(chain_ladder.rot, device=env.device)
    state[0, 2] = env.scene.env_origins[0, 2] + dz
    if tilt:
        half = torch.tensor(tilt / 2.0)
        roll = torch.tensor([torch.cos(half), torch.sin(half), 0.0, 0.0], device=env.device)
        w0, x0, y0, z0 = state[0, 3:7]
        w1, x1, y1, z1 = roll
        state[0, 3:7] = torch.stack(
            [
                w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
                w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
                w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
                w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
            ]
        )
    ladder.write_root_pose_to_sim(state[:, :7])
    ladder.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))


def run_case(steps: int, **stage) -> tuple[dict[str, bool], float, int, int]:
    """Stage the ladder, step, and return the last pre-step gate reading and what fired."""
    env.reset()
    pinned_root = robot.data.root_state_w[:, :7].clone()
    stage_ladder(**stage)
    last, last_xy, success_at, tipped_at = conjuncts(), ladder_fixture_xy(), -1, -1
    for i in range(steps):
        # Read before the step: env.step auto-resets on termination, restoring the drawn layout.
        last, last_xy = conjuncts(), ladder_fixture_xy()
        robot.write_root_pose_to_sim(pinned_root)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
        env.step(zero_action)
        if bool(env.termination_manager.get_term("success")[0]):
            success_at = i + 1
            break
        if bool(env.termination_manager.get_term("ladder_tipped")[0]):
            tipped_at = i + 1
            break
    return last, last_xy, success_at, tipped_at


print(f"\n[verify] {TASK}, layout seed {args_cli.seed}, sustain {sustain_s} s = {sustain_steps} steps")

print("\n[verify] (1) Start state")
env.reset()
start = conjuncts()
facing = float(rewards.base_facing_error(env, SceneEntityCfg("ladder"))[0])
check(
    "start:hands_free",
    "grip_contact" not in env.scene.sensors and "release_contact" not in env.scene.sensors,
    f"sensors={sorted(env.scene.sensors)} -- no grip/release channel, so no gate can read HOW the ladder moves",
)
check("start:robot_faces_ladder", facing < 0.5, f"facing error = {facing:.3f} rad")
check(
    "start:ladder_on_its_feet",
    start["ladder_feet_down"] and start["object_at_rest"],
    f"feet_down={start['ladder_feet_down']} at_rest={start['object_at_rest']}",
)
check(
    "start:ladder_not_yet_ready",
    not start["ladder_ready"],
    f"ladder top is {ladder_fixture_xy():.3f} m from the fixture (gate radius {LADDER_READY_XY_RADIUS:.3f} m)",
)
check("start:gate_open", not all(start.values()), "the episode does not begin already solved")

print("\n[verify] (2) Gate fires on the deliverable")
c, xy, success_at, _ = run_case(sustain_steps + 30, at_fixture=True)
check(
    "gate:fires_at_chain_placement",
    all(c.values()) and success_at == sustain_steps,
    f"xy={xy:.3f} m, success at step {success_at} (sustain = {sustain_steps}); "
    + " ".join(f"{k}={v}" for k, v in c.items()),
)

print("\n[verify] (3) Gate rejects what is not the deliverable")
c, xy, success_at, _ = run_case(10, dz=0.5, at_fixture=True)
check(
    "gate:rejects_held_in_the_air",
    success_at < 0 and not c["ladder_feet_down"],
    f"ladder over the fixture but 0.5 m up: feet_down={c['ladder_feet_down']} at_rest={c['object_at_rest']}",
)
c, xy, success_at, tipped_at = run_case(10, tilt=1.0, at_fixture=True)
check(
    "gate:rejects_tipped",
    success_at < 0 and tipped_at > 0,
    f"1.0 rad > LADDER_TILT_LIMIT: ladder_tipped terminated at step {tipped_at}",
)
c, xy, success_at, _ = run_case(sustain_steps + 30, at_fixture=False)
check(
    "gate:rejects_left_where_drawn",
    success_at < 0 and not c["ladder_ready"],
    f"upright and at rest, but {xy:.3f} m from the fixture",
)

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n[verify] {passed}/{len(results)} checks passed -- OVERALL: {'PASS' if passed == len(results) else 'FAIL'}")

env.close()
simulation_app.close()
