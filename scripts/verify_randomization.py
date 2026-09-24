# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the domain-randomization axes (prop scale, material tint, light orientation).

Headless, PhysX-only: every check is a USD-attribute read, so it runs where the RTX
renderer is unavailable. Checks ``CarryEnvCfg`` (RL defaults, ``replicate_physics=True``):
NO scale DR (ladder stays at its authored 0.01), but room tint + light orientation active.

Run directly (no flags needed): the parent process fans out three subprocesses (one Kit
per process: a second ManagerBasedEnv in one process hangs at scene creation) and asserts
same-seed determinism: two runs with the same seed must emit byte-identical
``[SIGNATURE]`` lines, and a different seed must not.

Exit code is non-zero on FAIL (``os._exit`` before Kit shutdown, which would otherwise
force 0 and swallow unflushed stdout).
"""

import argparse
import sys

parser = argparse.ArgumentParser(description="Verify Fiatlux domain randomization.")
parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=2)

# ---------------------------------------------------------------------------
# Parent mode: fan out subprocesses BEFORE booting Kit (unless --child).
# ---------------------------------------------------------------------------


def _run_child(seed: int, extra: list[str]) -> tuple[int, str]:
    import subprocess

    cmd = [sys.executable, __file__, "--child", "--seed", str(seed), "--headless", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr[-2000:] if proc.returncode else "")
    signature = ""
    for line in proc.stdout.splitlines():
        if line.startswith("[SIGNATURE] "):
            signature = line[len("[SIGNATURE] ") :]
    return proc.returncode, signature


def _parent_main(args) -> int:
    failures = []
    print(f"[verify_randomization] run 1 (seed {args.seed})")
    rc1, sig1 = _run_child(args.seed, [])
    print(f"[verify_randomization] run 2 (seed {args.seed})")
    rc2, sig2 = _run_child(args.seed, [])
    print(f"[verify_randomization] run 3 (seed {args.seed + 1})")
    rc3, sig3 = _run_child(args.seed + 1, [])

    def record(name: str, ok: bool, detail: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    record("runs_pass", rc1 == 0 and rc2 == 0 and rc3 == 0, f"exit codes {rc1}/{rc2}/{rc3}")
    record(
        "determinism:same_seed_same_draws",
        bool(sig1) and sig1 == sig2,
        f"signatures {'match' if sig1 == sig2 else 'DIFFER'} ({len(sig1)} chars)",
    )
    record("determinism:different_seed_differs", bool(sig3) and sig1 != sig3, "seed+1 signature differs")
    print("ALL CHECKS PASSED" if not failures else "FAILED: " + ", ".join(failures))
    return 1 if failures else 0


_args_cli, _ = parser.parse_known_args()
if not _args_cli.child:
    raise SystemExit(_parent_main(_args_cli))

# ---------------------------------------------------------------------------
# Child mode: boot Kit, build ONE env, assert on USD attributes.
# ---------------------------------------------------------------------------

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os
import random

import omni.usd
from pxr import Usd, UsdShade

BASE_COLOR_KEY = "fiatlux:base_color"

failures: list[str] = []


def record(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def get_scale(stage, path: str):
    attr = stage.GetPrimAtPath(path).GetAttribute("xformOp:scale")
    value = attr.Get() if attr else None
    return tuple(round(float(v), 6) for v in value) if value is not None else None


def get_orient(stage, path: str):
    attr = stage.GetPrimAtPath(path).GetAttribute("xformOp:orient")
    quat = attr.Get() if attr else None
    if quat is None:
        return None
    imag = quat.GetImaginary()
    return tuple(round(float(v), 6) for v in (quat.GetReal(), imag[0], imag[1], imag[2]))


def tinted_attrs(stage, root_path: str):
    """All material color attrs under root_path carrying the anti-compounding cache."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []
    found = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not (prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader)):
            continue
        for attr in prim.GetAttributes():
            if attr.GetCustomDataByKey(BASE_COLOR_KEY) is not None:
                found.append(attr)
    return found


def child_rl(seed: int, num_envs: int) -> dict:
    from fiatlux_task.tasks.manager_based.fiatlux_task.carry_env_cfg import CarryEnvCfg

    from isaaclab.envs import ManagerBasedRLEnv

    random.seed(seed)
    env_cfg = CarryEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env = ManagerBasedRLEnv(cfg=env_cfg)
    stage = omni.usd.get_context().get_stage()

    record(
        "rl:replicate_physics_on",
        env.scene.cfg.replicate_physics is True,
        f"replicate_physics={env.scene.cfg.replicate_physics}",
    )
    ladder = [get_scale(stage, f"/World/envs/env_{i}/Ladder") for i in range(num_envs)]
    record(
        "rl:no_scale_dr_by_default",
        all(s == (0.01, 0.01, 0.01) for s in ladder),
        f"ladder scales {ladder} (authored 0.01, unrandomized)",
    )
    env.reset()
    room_tinted = tinted_attrs(stage, "/World/Room")
    record("rl:room_tint_active", len(room_tinted) > 0, "shared room tinted")
    record(
        "rl:per_env_ladder_untinted",
        len(tinted_attrs(stage, "/World/envs/env_0/Ladder")) == 0,
        "per-env materials untouched under replicated physics",
    )
    before = (get_orient(stage, "/World/KeyLight"), get_orient(stage, "/World/DomeLight"))
    env.reset()
    after = (get_orient(stage, "/World/KeyLight"), get_orient(stage, "/World/DomeLight"))
    record("rl:light_orient_changes", before != after, "key/dome orient vary across resets")

    return {
        "orients": [before, after],
        "room_tint": [tuple(round(float(v), 6) for v in a.Get()) for a in room_tinted[:3]],
    }


def main() -> int:
    signature = child_rl(args_cli.seed, args_cli.num_envs)
    print(f"[SIGNATURE] {json.dumps(signature, sort_keys=True)}")
    print("ALL CHECKS PASSED" if not failures else "FAILED: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # Kit's own shutdown otherwise forces exit code 0 and can swallow unflushed stdout.
    os._exit(code)
