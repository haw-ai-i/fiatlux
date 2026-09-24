# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the domain-randomization axes (prop scale, material tint, light orientation).

Headless, PhysX-only: every check is a USD-attribute read, so it runs where the RTX
renderer is unavailable. Two scenarios:

* ``rl`` (default) -- ``CarryEnvCfg`` (RL defaults, ``replicate_physics=True``): NO scale DR
  (ladder stays at its authored 0.01), but room tint (bounded, anti-compounding across resets)
  and light orientation (dome yaw-only, key light within its cone) active.
* ``scale_dr`` -- ``ReplaceEnvCfg`` with the documented opt-in from replace_env_cfg.py's own
  comment (``scene.replicate_physics = False`` + a prestartup ``mdp.randomize_prop_scale``/
  ``mdp.randomize_rigid_body_scale`` EventTerm on ladder/socket/fresh_bulb): per-env scale
  lands in range and varies across envs.

Run directly (no flags needed): the parent process fans out subprocesses (one Kit per
process: a second ManagerBasedEnv in one process hangs at scene creation). The ``rl``
scenario additionally asserts same-seed determinism: two runs with the same seed must emit
byte-identical ``[SIGNATURE]`` lines, and a different seed must not.

Exit code is non-zero on FAIL (``os._exit`` before Kit shutdown, which would otherwise
force 0 and swallow unflushed stdout).
"""

import argparse
import sys

parser = argparse.ArgumentParser(description="Verify Fiatlux domain randomization.")
parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--scenario", choices=["rl", "scale_dr"], default="rl", help=argparse.SUPPRESS)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=2)

# ---------------------------------------------------------------------------
# Parent mode: fan out subprocesses BEFORE booting Kit (unless --child).
# ---------------------------------------------------------------------------


def _run_child(seed: int, extra: list[str], scenario: str = "rl") -> tuple[int, str]:
    import subprocess

    cmd = [sys.executable, __file__, "--child", "--scenario", scenario, "--seed", str(seed), "--headless", *extra]
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
    print(f"[verify_randomization] scale_dr run (seed {args.seed})")
    rc4, _ = _run_child(args.seed, ["--enable_cameras"], scenario="scale_dr")

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
    record("scale_dr_run_passes", rc4 == 0, f"exit code {rc4}")
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
import math
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


def quat_delta_deg(q1, q2) -> float:
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


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

    key_authored = tuple(env.scene.cfg.key_light.init_state.rot)
    orients = []
    tint_bound_ok = True
    for _ in range(5):
        env.reset()
        orients.append((get_orient(stage, "/World/KeyLight"), get_orient(stage, "/World/DomeLight")))
        for attr in room_tinted:
            base = attr.GetCustomDataByKey(BASE_COLOR_KEY)
            value = attr.Get()
            if any(v > b * 1.2 + 1e-4 or v < 0.0 for v, b in zip(value, base)):
                tint_bound_ok = False
    record("rl:tint_never_compounds", tint_bound_ok, "5 resets stay within original*[0,1.2]")
    key_orients = {o[0] for o in orients}
    dome_orients = {o[1] for o in orients}
    record("rl:light_orient_changes", len(key_orients) > 1 and len(dome_orients) > 1, "key/dome orient vary across resets")
    record(
        "rl:dome_yaw_only",
        all(abs(o[1]) < 1e-5 and abs(o[2]) < 1e-5 for o in dome_orients if o is not None),
        "dome quat has no x/y components",
    )
    record(
        "rl:key_light_within_cone",
        all(quat_delta_deg(o, key_authored) <= 46.0 for o in key_orients if o is not None),
        f"max delta {max(quat_delta_deg(o, key_authored) for o in key_orients):.1f} deg <= 46",
    )

    return {
        "orients": orients,
        "room_tint": [tuple(round(float(v), 6) for v in a.Get()) for a in room_tinted[:3]],
    }


def in_range(values, lo: float, hi: float, tol: float = 1e-6) -> bool:
    return all(lo - tol <= v <= hi + tol for v in values)


def child_scale_dr(seed: int, num_envs: int) -> None:
    """Replace's documented ``replicate_physics=False`` + ``mdp.randomize_prop_scale`` opt-in
    (see replace_env_cfg.py's EventCfg comment): per-env prestartup scale lands in range and
    varies across envs. Wires the exact EventTerms that comment recommends, since no currently
    registered task turns this opt-in on by default.
    """
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg

    from fiatlux_task.tasks.manager_based.fiatlux_task import mdp
    from fiatlux_task.tasks.manager_based.fiatlux_task.replace_env_cfg import ReplaceEnvCfg

    random.seed(seed)
    env_cfg = ReplaceEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.replicate_physics = False
    env_cfg.seed = seed
    env_cfg.events.randomize_ladder_scale = EventTerm(
        func=mdp.randomize_prop_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("ladder"),
            "scale_range": {"x": (0.95, 1.05), "y": (0.95, 1.05), "z": (0.95, 1.1)},
        },
    )
    env_cfg.events.randomize_socket_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("socket"), "scale_range": (0.9, 1.1)},
    )
    env_cfg.events.randomize_bulb_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={"asset_cfg": SceneEntityCfg("fresh_bulb"), "scale_range": (0.9, 1.1)},
    )
    env = ManagerBasedRLEnv(cfg=env_cfg)
    stage = omni.usd.get_context().get_stage()

    record(
        "scale_dr:replicate_physics_off",
        env.scene.cfg.replicate_physics is False,
        f"replicate_physics={env.scene.cfg.replicate_physics} (scale DR legal)",
    )
    ladder = [get_scale(stage, f"/World/envs/env_{i}/Ladder") for i in range(num_envs)]
    record(
        "scale_dr:ladder_scale_in_range",
        all(
            s is not None and in_range(s[:2], 0.01 * 0.95, 0.01 * 1.05) and in_range(s[2:], 0.01 * 0.95, 0.01 * 1.1)
            for s in ladder
        ),
        f"env scales {ladder} (multiplicative on the baked 0.01)",
    )
    record("scale_dr:ladder_scale_varies", len(set(ladder)) > 1, f"{len(set(ladder))} distinct of {num_envs}")
    for name in ("Socket", "Bulb"):
        scales = [get_scale(stage, f"/World/envs/env_{i}/{name}") for i in range(num_envs)]
        record(
            f"scale_dr:{name.lower()}_scale_in_range",
            all(s is not None and in_range(s, 0.9, 1.1) and len(set(s)) == 1 for s in scales),
            f"isotropic env scales {scales}",
        )


def main() -> int:
    if args_cli.scenario == "scale_dr":
        child_scale_dr(args_cli.seed, args_cli.num_envs)
        print("ALL CHECKS PASSED" if not failures else "FAILED: " + ", ".join(failures))
        return 1 if failures else 0
    signature = child_rl(args_cli.seed, args_cli.num_envs)
    print(f"[SIGNATURE] {json.dumps(signature, sort_keys=True)}")
    print("ALL CHECKS PASSED" if not failures else "FAILED: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # Kit's own shutdown otherwise forces exit code 0 and can swallow unflushed stdout.
    os._exit(code)
