# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the bulb-to-socket collision filter survives env cloning (issue #77 task 6).

``_spawn_bulb_socket_filtered`` applies ``UsdPhysics.FilteredPairsAPI`` on a bulb, pointing at
the sibling socket in the SAME env. The spawner runs once and the cloner replicates it, so the
relationship target has to be remapped per env. If it is not, every env filters against env 0's
socket and only env 0 behaves.

Reads the composed stage and reports, per env, which socket each bulb filters against. Then it
measures the socket contact force, which must collapse from the ~1067 N recorded before the
filter (see ``scripts/step0_contact.py``).

Run with more than one env or it proves nothing about cloning:

    python scripts/verify_pair_filter.py --headless --num_envs 4
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check the bulb/socket collision filter after cloning.")
parser.add_argument("--seed", type=int, default=0, help="Env seed (deterministic).")
parser.add_argument("--num_envs", type=int, default=4, help="Envs to clone (use >1).")
parser.add_argument("--steps", type=int, default=40, help="Steps to sample contact over.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything else follows."""

import importlib

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import torch
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import BULB_MASS_KG, set_layout_seed

from isaaclab.sensors import ContactSensorCfg

from isaaclab_tasks.utils import parse_env_cfg

BULB_WEIGHT_N = BULB_MASS_KG * 9.81


def build_cfg():
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    for camera in ("ego_camera", "torso_camera", "wrist_camera"):
        if getattr(cfg.scene, camera, None) is not None:
            setattr(cfg.scene, camera, None)
    for group_name in ("policy", "privileged"):
        group = getattr(cfg.observations, group_name, None)
        for term in ("ego_rgb", "torso_rgb", "wrist_rgb"):
            if group is not None and getattr(group, term, None) is not None:
                setattr(group, term, None)
    for term in ("success", "old_bulb_dropped", "fresh_bulb_dropped"):
        if getattr(cfg.terminations, term, None) is not None:
            setattr(cfg.terminations, term, None)
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    cfg.scene.socket_bulb_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Socket",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/OldBulb"],
        history_length=1,
        track_air_time=False,
    )
    return cfg


def main() -> int:
    from pxr import UsdPhysics

    cfg = build_cfg()
    spec = gym.spec("FIATLUX-Replace-v0")
    module_name, class_name = spec.entry_point.split(":")
    env = getattr(importlib.import_module(module_name), class_name)(cfg=cfg)
    env.reset(seed=args_cli.seed)

    stage = env.sim.stage
    ok = True
    print(f"SETUP num_envs={args_cli.num_envs}", flush=True)
    for i in range(args_cli.num_envs):
        for name in ("Bulb", "OldBulb"):
            path = f"/World/envs/env_{i}/{name}"
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                print(f"FILTER env_{i} {name:<8} PRIM MISSING", flush=True)
                continue
            if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                print(f"FILTER env_{i} {name:<8} NO FilteredPairsAPI", flush=True)
                ok = False
                continue
            rel = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
            targets = [str(t) for t in (rel.GetTargets() or [])]
            want = f"/World/envs/env_{i}/Socket"
            good = targets == [want]
            ok = ok and good
            print(f"FILTER env_{i} {name:<8} -> {targets} {'OK' if good else 'WRONG (want ' + want + ')'}", flush=True)

    print(f"FILTER_REMAPPED_PER_ENV {ok}", flush=True)

    # Contact force must collapse now that the pair is filtered.
    sensor = env.scene.sensors["socket_bulb_contact"]
    zero_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    mags = []
    for _ in range(args_cli.steps):
        env.step(zero_action)
        f = sensor.data.force_matrix_w
        if f is not None:
            mags.append(float(torch.norm(f.sum(dim=2), dim=-1).max().item()))
    if mags:
        t = torch.tensor(mags)
        print(
            f"CONTACT_FORCE_N steps={len(mags)} min={t.min():.4f} median={t.median():.4f} "
            f"max={t.max():.4f}  ratio_to_weight={t.median() / BULB_WEIGHT_N:.2f}x",
            flush=True,
        )
        cleared = bool(t.median() < BULB_WEIGHT_N)
        print(f"CONTACT_CLEARED {cleared}  (was ~1067 N median before the filter)", flush=True)
        ok = ok and cleared

    print(f"VERDICT {'PASS' if ok else 'FAIL'}", flush=True)
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    import os
    import sys

    exit_code = 1
    try:
        exit_code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code:
            os._exit(exit_code)
        simulation_app.close()
