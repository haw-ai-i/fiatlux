# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that the on-ladder working stance stands CLEAR of the fixture, on every layout draw.

``stand_robot_on_ladder_top`` puts the pelvis over the ladder's root, and the fixture hangs only
0.233 m above it -- chest height, not overhead. A layout that stands the ladder directly beneath
the fixture spawns the robot with the socket inside its torso, and because the socket is a
kinematic rigid body with an exact triangle-mesh collider the robot hangs on it instead of
falling.

Two passes, which fail in opposite directions:

* ``geometry`` -- cfg-level, no simulation, so it can sweep many seeds. Asserts the stance is
  further from the fixture than ``LADDER_FIXTURE_MIN_STANDOFF`` on every draw. Cheap enough to
  run over enough seeds that a 50%-of-draws bug cannot hide.
* ``physics`` -- instantiates ONE layout and asserts no torso body is inside the fixture and that
  the robot is not being carried by it: under zero action a free-base biped must fall, and one
  hanging on the fixture does not. One env per run, because building a second
  ``ManagerBasedRLEnv`` in the same process after ``close()`` hangs -- covering both mount kinds
  is two invocations (``--physics_seed``), not two passes.

``--measure`` re-derives the two constants the guard is built from -- the robot's torso
half-extent at fixture height and the socket's AABB -- instead of checking anything.

Examples
--------
    uv run python scripts/verify_ladder_stance.py --headless
    uv run python scripts/verify_ladder_stance.py --headless --seeds 64
    uv run python scripts/verify_ladder_stance.py --headless --physics_seed 0   # a wall draw
    uv run python scripts/verify_ladder_stance.py --headless --measure
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the on-ladder stance clears the fixture.")
parser.add_argument("--seeds", type=int, default=32, help="Layout seeds to sweep in the geometry pass.")
parser.add_argument("--no_physics", action="store_true", help="Skip the (slow) instantiated pass.")
parser.add_argument(
    "--physics_seed",
    type=int,
    default=None,
    help="Layout seed for the physics pass. Defaults to the sweep's first CEILING draw, the kind "
    "the bug lived on; pass a wall seed to cover the other kind. ONE env per run: building a "
    "second ManagerBasedRLEnv in the same process after close() hangs, so the two mount kinds "
    "are two invocations, not two passes.",
)
parser.add_argument("--measure", action="store_true", help="Re-derive the clearance constants and exit.")
parser.add_argument("--steps", type=int, default=150, help="Zero-action steps in the physics pass.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# The frozen policy observation group carries an RGB term, so the env needs camera rendering.
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import math  # noqa: E402

import fiatlux_task.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import (  # noqa: E402
    FIXTURE_HALF_EXTENT,
    G1_STANCE_TORSO_HALF_EXTENT,
    LADDER_FIXTURE_MIN_STANDOFF,
    LADDER_FIXTURE_STANDOFF,
    set_layout_seed,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.subtasks.s03_remove_old_bulb_env_cfg import (  # noqa: E402
    S03RemoveOldBulbEnvCfg,
)

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "FIATLUX-S03-RemoveOldBulb-v0"
# Bodies that are meant to reach the fixture; everything else must clear it. ``hand`` covers
# ``*_hand_base_link``, the wrist end of the palm, which reads as a torso body without it.
ARM_TOKENS = ("thumb", "index", "middle", "ring", "pinky", "wrist", "elbow", "hand")
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} -- {detail}")


def stance_geometry(seed: int) -> tuple[str, float]:
    """Mount kind and the floor-plane distance from the fixture to the on-ladder stance."""
    set_layout_seed(seed)
    scene = S03RemoveOldBulbEnvCfg().scene
    fx, fy, _ = scene.socket.init_state.pos
    rx, ry, _ = scene.robot.init_state.pos
    kind = "ceiling" if getattr(scene, "pendant", None) is not None else "wall"
    return kind, math.hypot(fx - rx, fy - ry)


def build(seed: int):
    set_layout_seed(seed)
    cfg = parse_env_cfg(TASK, num_envs=1)
    # LADDER_FIXTURE_MIN_STANDOFF already accounts for the jitter; drop it so the measurement is
    # of the authored stance.
    cfg.events.reset_robot_root = None
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    return env


def torso_clearance(env) -> tuple[float, str]:
    """Smallest floor-plane distance from the fixture axis to a non-arm body near its height."""
    robot = env.scene["robot"]
    fixture = env.scene["socket"].data.root_pos_w[0]
    pos = robot.data.body_pos_w[0]
    near = (pos[:, 2] - fixture[2]).abs() < 0.15
    horiz = torch.norm(pos[:, :2] - fixture[:2], dim=1)
    worst, worst_name = math.inf, "none in band"
    for i, name in enumerate(robot.body_names):
        if not near[i] or any(t in name.lower() for t in ARM_TOKENS):
            continue
        if float(horiz[i]) < worst:
            worst, worst_name = float(horiz[i]), name
    return worst, worst_name


if args_cli.measure:
    env = build(1)
    robot = env.scene["robot"]
    fixture = env.scene["socket"].data.root_pos_w[0]
    print(f"\nfixture z {float(fixture[2]):.3f}, pelvis z {float(robot.data.root_pos_w[0, 2]):.3f}")
    worst, name = torso_clearance(env)
    # The stance stands LADDER_FIXTURE_STANDOFF from the fixture axis, so the body nearest the
    # fixture reaches (standoff - clearance) out from the pelvis axis toward it.
    print(
        f"G1_STANCE_TORSO_HALF_EXTENT: the non-arm body nearest the fixture is {name}, "
        f"{worst:.3f} m from its axis = {LADDER_FIXTURE_STANDOFF - worst:.3f} m out from the pelvis"
    )
    from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache

    aabb = compute_aabb(create_bbox_cache(), "/World/envs/env_0/Socket", include_children=True)
    print(
        f"FIXTURE_HALF_EXTENT: socket AABB half-extents xy = "
        f"{abs(aabb[3] - aabb[0]) / 2:.3f} x {abs(aabb[4] - aabb[1]) / 2:.3f}"
    )
    env.close()
    simulation_app.close()
    raise SystemExit(0)

print(
    f"\n[verify] on-ladder stance vs fixture; standoff {LADDER_FIXTURE_STANDOFF} m, "
    f"required > {LADDER_FIXTURE_MIN_STANDOFF:.3f} m "
    f"(torso {G1_STANCE_TORSO_HALF_EXTENT} + fixture {FIXTURE_HALF_EXTENT} + reset jitter)"
)

print(f"\n[verify] (1) Geometry, {args_cli.seeds} layout seeds")
draws = [stance_geometry(s) for s in range(args_cli.seeds)]
kinds = {k for k, _ in draws}
worst_seed, (worst_kind, worst_dxy) = min(enumerate(draws), key=lambda kv: kv[1][1])
check(
    "geometry:both_mount_kinds_drawn",
    kinds == {"ceiling", "wall"},
    f"{sum(k == 'ceiling' for k, _ in draws)} ceiling / {sum(k == 'wall' for k, _ in draws)} wall "
    f"in {args_cli.seeds} seeds -- a one-sided sweep would not exercise the bug",
)
check(
    "geometry:stance_clears_fixture",
    worst_dxy > LADDER_FIXTURE_MIN_STANDOFF,
    f"tightest draw is seed {worst_seed} ({worst_kind}) at {worst_dxy:.3f} m",
)

if not args_cli.no_physics:
    seed = args_cli.physics_seed
    if seed is None:
        seed = next(i for i, (k, _) in enumerate(draws) if k == "ceiling")
    kind = draws[seed][0] if seed < len(draws) else stance_geometry(seed)[0]
    print(f"\n[verify] (2) Physics, seed {seed} ({kind} mount)")
    env = build(seed)
    robot = env.scene["robot"]
    clear, name = torso_clearance(env)
    check(
        f"physics:{kind}:torso_clears_fixture",
        clear > FIXTURE_HALF_EXTENT,
        f"nearest non-arm body {name} at {clear:.3f} m from the fixture axis "
        f"(its own half-extent is {FIXTURE_HALF_EXTENT} m)",
    )
    zero = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
    z0 = float(robot.data.root_pos_w[0, 2])
    lowest = z0
    for _ in range(args_cli.steps):
        env.step(zero)
        lowest = min(lowest, float(robot.data.root_pos_w[0, 2]))
    # A free-base biped under zero action collapses. One resting on the fixture does not, which
    # is the signature of the bug: the pelvis barely leaves its spawn height.
    check(
        f"physics:{kind}:not_hanging_on_fixture",
        z0 - lowest > 0.25,
        f"pelvis fell {z0 - lowest:.3f} m from {z0:.3f} under zero action in {args_cli.steps} steps",
    )
    env.close()

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n[verify] {passed}/{len(results)} checks passed -- OVERALL: {'PASS' if passed == len(results) else 'FAIL'}")
simulation_app.close()
