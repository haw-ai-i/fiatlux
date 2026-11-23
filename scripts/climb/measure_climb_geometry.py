# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the HORIZONTAL geometry the four ladder subtasks stage and gate, against a reference clip.

A climb reference is authored at one standoff from the ladder and covers one horizontal span. The
benchmark stages the robot at its own standoff and scores arrival inside its own radius. If those
disagree, a teacher that tracks the reference perfectly still starts in the wrong place or stops
short, and no amount of training fixes it. Nothing here needs physics -- the staging helpers are
pure arithmetic on the scene cfg -- so this is a cfg-only pass over layout seeds.

Reported per subtask:

* ``stage->top``   the horizontal distance from the staged pelvis to the platform stance, i.e. the
  span a climb reference has to cover.
* ``stage r``      staged pelvis distance from the ladder ROOT, which is what the descend gate
  measures against ``LADDER_SUCCESS_XY_RADIUS``.
* ``budget``       for a descent, how far the pelvis may travel from the platform stance and still
  land inside the gate radius.

Examples
--------
    uv run python scripts/climb/measure_climb_geometry.py --headless
    uv run python scripts/climb/measure_climb_geometry.py --headless --seeds 16
    uv run python scripts/climb/measure_climb_geometry.py --headless --clip /path/to/motion.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure staged vs gated ladder geometry.")
parser.add_argument("--seeds", type=int, default=8, help="Layout seeds to sweep.")
parser.add_argument("--clip_up", type=str, default=None, help="Built ASCENT motion.npz to score.")
parser.add_argument("--clip_down", type=str, default=None, help="Built DESCENT motion.npz to score.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import math  # noqa: E402

import numpy as np  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp.place_terms import (  # noqa: E402
    LADDER_STEP_FACE_LOCAL,
    step_face_dir_from_yaw,
    yaw_from_quat,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import (  # noqa: E402
    STEP_LADDER_TOP_OFFSET,
    TOP_STANCE_PELVIS_OFFSET,
    set_layout_seed,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.subtask_tiers.balance import (  # noqa: E402
    LADDER_FLOOR_STANCE_HEIGHT,
    LADDER_SUCCESS_XY_RADIUS,
    LADDER_TOP_HEIGHT_SLACK,
    MOUNT_STANCE_STANDOFF,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.subtasks.s02_climb_ladder_env_cfg import (  # noqa: E402
    S02ClimbLadderEnvCfg,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.subtasks.s04_descend_with_bulb_env_cfg import (  # noqa: E402
    S04DescendWithBulbEnvCfg,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.subtasks.s10_climb_with_bulb_env_cfg import (  # noqa: E402
    S10ClimbWithBulbEnvCfg,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.subtasks.s12_climb_down_env_cfg import (  # noqa: E402
    S12ClimbDownEnvCfg,
)

CFGS = {
    "S02": (S02ClimbLadderEnvCfg, "up"),
    "S10": (S10ClimbWithBulbEnvCfg, "up"),
    "S04": (S04DescendWithBulbEnvCfg, "down"),
    "S12": (S12ClimbDownEnvCfg, "down"),
}


def platform_stance_xy(ladder_pos, ladder_rot) -> tuple[float, float]:
    """Where ``stand_robot_on_ladder_top`` puts the pelvis, in world xy."""
    yaw = yaw_from_quat(ladder_rot)
    lx, ly = local_x, local_y = STEP_LADDER_TOP_OFFSET[0], STEP_LADDER_TOP_OFFSET[1]
    dx = local_x * math.cos(yaw) - local_y * math.sin(yaw)
    dy = local_x * math.sin(yaw) + local_y * math.cos(yaw)
    del lx, ly
    return ladder_pos[0] + dx, ladder_pos[1] + dy


def measure(name: str, cls, direction: str, seed: int) -> dict:
    set_layout_seed(seed)
    scene = cls().scene
    lp, lr = scene.ladder.init_state.pos, scene.ladder.init_state.rot
    rp = scene.robot.init_state.pos
    px, py = platform_stance_xy(lp, lr)
    face = step_face_dir_from_yaw(yaw_from_quat(lr))
    stage_r = math.hypot(rp[0] - lp[0], rp[1] - lp[1])
    # Signed offset of the platform stance along the step-face direction: negative means the
    # stance sits on the BACK side, so a descent crosses the root and the travel partly cancels.
    plat_along_face = (px - lp[0]) * face[0] + (py - lp[1]) * face[1]
    return {
        "name": name,
        "dir": direction,
        "seed": seed,
        "stage_r": stage_r,
        "stage_to_top": math.hypot(rp[0] - px, rp[1] - py),
        "plat_r": math.hypot(px - lp[0], py - lp[1]),
        "plat_along_face": plat_along_face,
        "pelvis_z": rp[2],
        # A descent may travel this far from the platform stance along the step face and still
        # land inside the gate radius. The stance sits at signed offset ``plat_along_face`` from
        # the root, the descent adds ``d`` along +face, and the gate wants |plat+d| < radius --
        # so the bound is radius MINUS the (negative) offset, i.e. a back-side stance BUYS room.
        "budget": LADDER_SUCCESS_XY_RADIUS - plat_along_face,
    }


print(f"[CONST] LADDER_STEP_FACE_LOCAL   {LADDER_STEP_FACE_LOCAL}")
print(f"[CONST] STEP_LADDER_TOP_OFFSET   {STEP_LADDER_TOP_OFFSET}")
print(f"[CONST] MOUNT_STANCE_STANDOFF    {MOUNT_STANCE_STANDOFF:.4f} m")
print(f"[CONST] LADDER_SUCCESS_XY_RADIUS {LADDER_SUCCESS_XY_RADIUS:.4f} m")
print(f"[CONST] TOP_STANCE_PELVIS_OFFSET {TOP_STANCE_PELVIS_OFFSET:.4f} m")
print(f"[CONST] LADDER_TOP_HEIGHT_SLACK  {LADDER_TOP_HEIGHT_SLACK:.4f} m")
print(f"[CONST] LADDER_FLOOR_STANCE_HEIGHT {LADDER_FLOOR_STANCE_HEIGHT:.4f} m")
# Where the ladder's own origin sits inside its mesh, which is what turns a root-relative
# standoff into a distance from the STEPS -- the only standoff a climb reference can be
# compared against.
try:
    from fiatlux_task.assets import STEP_LADDER_USD  # noqa: E402

    from pxr import Usd, UsdGeom  # noqa: E402

    _st = Usd.Stage.Open(STEP_LADDER_USD)
    # Vendor USDs are authored in centimetres; everything else here is metres.
    _mpu = UsdGeom.GetStageMetersPerUnit(_st)
    _bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    _r = _bb.ComputeWorldBound(_st.GetDefaultPrim()).ComputeAlignedRange()
    _mn = [v * _mpu for v in _r.GetMin()]
    _mx = [v * _mpu for v in _r.GetMax()]
    print(f"[LADDER] metersPerUnit {_mpu}; bbox min {[round(v, 4) for v in _mn]} max {[round(v, 4) for v in _mx]} (m)")
    # The steps face LADDER_STEP_FACE_LOCAL; its extreme along that axis is the step face.
    _ax = [i for i, v in enumerate(LADDER_STEP_FACE_LOCAL) if v][0]
    _sgn = LADDER_STEP_FACE_LOCAL[_ax]
    STEP_FACE_OFFSET = abs(_mn[_ax] if _sgn < 0 else _mx[_ax])
    print(
        f"[LADDER] step face sits {STEP_FACE_OFFSET:.4f} m from the root along local "
        f"axis {_ax} (sign {_sgn:+.0f}); depth {_mx[_ax] - _mn[_ax]:.4f} m"
    )
except Exception as exc:  # pragma: no cover -- diagnostic only
    STEP_FACE_OFFSET = None
    print(f"[LADDER] bbox unavailable ({exc})")
print()

rows: list[dict] = []
for name, (cls, direction) in CFGS.items():
    for seed in range(args_cli.seeds):
        rows.append(measure(name, cls, direction, seed))

hdr = f"{'task':>5} {'dir':>4} {'stage r':>9} {'stage->top':>11} {'plat r':>8} {'plat/face':>10} {'budget':>8}"
print(hdr)
print("-" * len(hdr))
for name in CFGS:
    sel = [r for r in rows if r["name"] == name]
    f = lambda k: (min(r[k] for r in sel), max(r[k] for r in sel))  # noqa: E731
    sr, st, pr, pf, bd = f("stage_r"), f("stage_to_top"), f("plat_r"), f("plat_along_face"), f("budget")
    rng = lambda t: f"{t[0]:.3f}" if abs(t[1] - t[0]) < 5e-4 else f"{t[0]:.3f}-{t[1]:.3f}"  # noqa: E731
    print(f"{name:>5} {sel[0]['dir']:>4} {rng(sr):>9} {rng(st):>11} {rng(pr):>8} {rng(pf):>10} {rng(bd):>8}")

print()
# The staged climb start, judged by the DESCENT gate -- the two must agree for the chain to close.
up = [r for r in rows if r["dir"] == "up"]
worst = max(r["stage_r"] for r in up)
ok = worst < LADDER_SUCCESS_XY_RADIUS
print(f"[CHAIN] a descent that returns to the CLIMB START stands {worst:.3f} m from the ladder root;")
print(
    f"        the descend gate admits {LADDER_SUCCESS_XY_RADIUS:.3f} m -> {'INSIDE' if ok else 'OUTSIDE'}"
    f" by {abs(worst - LADDER_SUCCESS_XY_RADIUS) * 1000:.0f} mm"
)


def span(path: str) -> tuple[float, float, float]:
    pel = np.load(path)["body_pos_w"][:, 0, :]
    return (float(np.hypot(pel[-1, 0] - pel[0, 0], pel[-1, 1] - pel[0, 1])), float(pel[0, 2]), float(pel[-1, 2]))


if STEP_FACE_OFFSET is not None:
    print()
    for name in [n for n in CFGS if CFGS[n][1] == "up"]:
        sr = max(r["stage_r"] for r in rows if r["name"] == name)
        print(
            f"[FACE ] {name} staged pelvis is {sr:.3f} m from the root = "
            f"{sr - STEP_FACE_OFFSET:.3f} m clear of the STEP FACE"
        )
    _pf = min(r["plat_along_face"] for r in rows)
    print(
        f"[FACE ] the platform stance sits {STEP_FACE_OFFSET - _pf:.3f} m BEHIND the step "
        f"face, so a top-out must carry the pelvis that far back from the face"
    )

if args_cli.clip_up:
    travel, z0, z1 = span(args_cli.clip_up)
    print()
    print(f"[UP   ] {args_cli.clip_up}")
    print(f"[UP   ] horizontal span {travel:.3f} m, pelvis {z0:.3f} -> {z1:.3f} m")
    for name in [n for n in CFGS if CFGS[n][1] == "up"]:
        need = max(r["stage_to_top"] for r in rows if r["name"] == name)
        mark = "COVERS" if travel >= need else "SHORT"
        print(
            f"[UP   ]   {name} staged start is {need:.3f} m from the platform stance "
            f"-> {mark} by {abs(travel - need) * 1000:.0f} mm"
        )

if args_cli.clip_down:
    travel, z0, z1 = span(args_cli.clip_down)
    print()
    print(f"[DOWN ] {args_cli.clip_down}")
    print(
        f"[DOWN ] horizontal span {travel:.3f} m, pelvis {z0:.3f} -> {z1:.3f} m "
        f"(gate wants < {LADDER_FLOOR_STANCE_HEIGHT:.3f})"
    )
    for name in [n for n in CFGS if CFGS[n][1] == "down"]:
        bd = min(r["budget"] for r in rows if r["name"] == name)
        mark = "FITS" if travel <= bd else "OVER"
        print(
            f"[DOWN ]   {name} may travel {bd:.3f} m from the platform stance "
            f"-> {mark} by {abs(bd - travel) * 1000:.0f} mm"
        )

simulation_app.close()
