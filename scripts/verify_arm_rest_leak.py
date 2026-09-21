# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Is the FIATLUX_ARM_REST_LEFT leak into non-rail tasks fixed? (code-review finding, fix
verification -- sonic_teleop.py ~L609)

The original bug: sonic_teleop.py wrote the FIATLUX_ARM_REST_LEFT-derived left-arm pose into
``robot.data.default_joint_pos`` unconditionally (``if _ARM_REST_L.strip():``), never consulting
``_rail_on`` even though it was already computed earlier in ``main()``. ``default_joint_pos`` is
shared state that the post-settle "restore the staged arm pose" step and the mid-session reset
re-home both read for EVERY task -- so a non-rail task's left arm silently ended up restored to
the ladder-cap-recorded pose instead of its own task-authored one.

The fix: gate the write on ``_rail_on`` too (``if _rail_on and _ARM_REST_L.strip():``).

This script checks the fix two ways:

1. Pure-logic: extract the literal gate condition on that ``if`` line out of sonic_teleop.py by
   regex (so it can't silently drift from what's actually gating the write) and eval it for
   ``_rail_on`` True and False -- the write must now depend on ``_rail_on``, not fire
   unconditionally.
2. End-to-end: build a real non-rail task (FIATLUX-TestLightbulbMechanism-Teleop-v0, whose ladder
   scene attribute is non-None but parked away from the tabletop bench, so ``_rail_on`` is really
   False for it -- the exact "ladder scene exists but this spawn isn't on it" case the bug leaked
   into), evaluate the extracted gate with that task's real ``_rail_on``, and confirm the write is
   skipped -- so the left arm's ``default_joint_pos`` (and the post-settle ``_staged_arm`` slice
   that reads it) stays exactly as the task authored it.

Run via ./pyrun (repo root), not a bare .venv/bin/python -- see verify_common.py's docstring.

Example
-------
    ./pyrun scripts/verify_arm_rest_leak.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the FIATLUX_ARM_REST_LEFT non-rail-task leak is fixed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything else follows."""

import math
import re
from pathlib import Path

import fiatlux_teleop  # noqa: F401  -- registers the FIATLUX-*-Teleop-v0 Gym environments
import gymnasium as gym
import verify_common

from isaaclab_tasks.utils import parse_env_cfg

SRC = Path(__file__).resolve().parent / "teleop" / "sonic_teleop.py"
TASK = "FIATLUX-TestLightbulbMechanism-Teleop-v0"
ARM_REST_LEFT_DEFAULT = "-4.8,9.4,0.5,15.6,-4.3,-9.6,4.8"
LEFT_ARM_SUFFIXES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)


def extract_gate_condition() -> str:
    text = SRC.read_text()
    m = re.search(r'_ARM_REST_L = os\.environ\.get\("FIATLUX_ARM_REST_LEFT".*?\)\n {4}if ([^\n]+):\n', text)
    if not m:
        raise RuntimeError(f"could not locate the FIATLUX_ARM_REST_LEFT write's 'if ...:' gate in {SRC} -- moved?")
    return m.group(1)


def main() -> int:
    gate_expr = extract_gate_condition()
    print(f"[verify] extracted write gate from {SRC.name}: if {gate_expr}:", flush=True)

    def gate(rail_on: bool) -> bool:
        return bool(eval(gate_expr, {"__builtins__": {}}, {"_rail_on": rail_on, "_ARM_REST_L": ARM_REST_LEFT_DEFAULT}))

    logic_ok = (gate(False) is False) and (gate(True) is True)
    print(
        f"[verify] gate(_rail_on=False)={gate(False)} (want False)  gate(_rail_on=True)={gate(True)} (want True) "
        f"[{'OK' if logic_ok else 'MISMATCH'}]",
        flush=True,
    )

    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    verify_common.assert_right_checkout(cfg, "robot")

    # Mirror sonic_teleop.py's own _rail_on computation (~L531-535): this task's ladder scene
    # attribute is NOT None (Replace's obs/reward terms need it, so TestLightbulbMechanism keeps
    # it, just parked away from the tabletop bench) -- _rail_on is decided by whether the robot is
    # actually staged ON the ladder (pelvis >= 1 m above the ladder root), not by the attribute's
    # mere presence. This bench spawns the robot at the tabletop, not the ladder, so _rail_on is
    # really False here -- exactly the "ladder scene exists but this spawn isn't on it" case that
    # the bug leaked into.
    ladder = getattr(cfg.scene, "ladder", None)
    rail_on = ladder is not None and (float(cfg.scene.robot.init_state.pos[2]) - float(ladder.init_state.pos[2]) >= 1.0)
    assert not rail_on, f"{TASK} spawns the robot on the ladder -- picked a bad non-rail task for this repro"
    verify_common.strip_visual_obs(cfg)

    # Drop the disposal bin/crate prop (isaac_packing_table, several separate prop USD files) --
    # pure set-dressing, irrelevant to the left-arm joint check this script makes, but a slow first-
    # cook (uncached collision approximation for several small meshes) on a cold box. Its only
    # consumer is the privileged 'disposal_pose' observation term; drop both together.
    if getattr(cfg.scene, "bin", None) is not None:
        cfg.scene.bin = None
        if getattr(cfg.observations, "privileged", None) is not None:
            cfg.observations.privileged.disposal_pose = None
            # replace_score_distances() also reads scene["bin"] internally (old_bulb_disposal_distance)
            cfg.observations.privileged.score_distances = None

    env = gym.make(TASK, cfg=cfg).unwrapped
    robot = env.scene["robot"]

    names = [f"left_{n}_joint" for n in LEFT_ARM_SUFFIXES]
    missing = [n for n in names if n not in robot.joint_names]
    if missing:
        print(f"FATAL joint names not found on this asset: {missing}", flush=True)
        return 1

    idx = [robot.joint_names.index(n) for n in names]
    task_authored = robot.data.default_joint_pos[:, idx].clone()

    # Exactly sonic_teleop.py's write, but gated by the REAL extracted condition (not a hardcoded
    # True) evaluated against this task's REAL rail_on=False -- so if the fix regressed back to
    # firing unconditionally, this reproduces the leak again instead of silently passing.
    if gate(rail_on):
        vals = [math.radians(float(x)) for x in ARM_REST_LEFT_DEFAULT.split(",")]
        for n, v in zip(names, vals):
            robot.data.default_joint_pos[:, robot.joint_names.index(n)] = v
    after_write = robot.data.default_joint_pos[:, idx].clone()

    # Exactly sonic_teleop.py's post-settle restore, for a non-rail task: _rail is None so
    # _rail_braced is False, so _arm_idx admits left_* joints too.
    rail = None
    rail_braced = rail is not None
    arm_idx = [
        i
        for i, n in enumerate(robot.joint_names)
        if any(k in n for k in ("shoulder", "elbow", "wrist")) and (not rail_braced or n.startswith("right_"))
    ]
    left_in_arm_idx = [i for i in arm_idx if i in idx]
    staged_arm_left = robot.data.default_joint_pos[:, left_in_arm_idx].clone()

    print(f"[verify] task-authored left-arm default_joint_pos (rad): {task_authored.tolist()}", flush=True)
    print(f"[verify] default_joint_pos after the (gated) write     : {after_write.tolist()}", flush=True)
    print(f"[verify] _staged_arm left-arm slice (rad)              : {staged_arm_left.tolist()}", flush=True)

    no_leak = bool((after_write - task_authored).abs().max().item() < 1e-6) and bool(
        (staged_arm_left - task_authored).abs().max().item() < 1e-6
    )

    if logic_ok and no_leak:
        print(
            "[verify] CONFIRMED FIXED: the write gate now depends on _rail_on, and this non-rail "
            "task's left arm keeps its own task-authored pose end-to-end (no leak).",
            flush=True,
        )
        return 0
    print("[verify] FIX NOT CONFIRMED -- see MISMATCH/values above.", flush=True)
    return 1


if __name__ == "__main__":
    verify_common.run_verify_main(main, simulation_app)
