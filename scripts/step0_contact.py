# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""#77 task 6 step 0: quantify the bulb-to-socket contact at the seat pose.

Every candidate fix in task 6 assumes the projection forces the bulb into the socket, and that
the contact solver answers hard enough to swamp an applied twist. Nobody measured it. This does.

Two measurements, on `FIATLUX-Replace-v0` with the layout pinned:

1. CONTACT FORCE while the state machine holds the old bulb at the seat. A resting 35 g bulb
   would push about 0.34 N. Anything far above that is the projection driving an overlap.
2. DEPENETRATION. Set the old bulb's phase to FREE so the projection stops writing it, place it
   at the exact seat pose with zero velocity, and step. If the colliders overlap, PhysX shoves
   it out and the displacement measures the overlap. If they do not, it stays put.

If both come back small, the "two owners fighting" diagnosis is wrong and task 6 steps 1 to 3
are aimed at the wrong thing.

What it found, seed 0, RTX 5090, `isaaclab 2.3.2.post1`:

    CONTACT_FORCE_N   min 368.19   median 1066.93   max 1999.20   = 3107x the bulb weight
    DEPENETRATION_MM  0.829 on the first step, about 1.8 mm total over 20 steps

The two disagree, and the disagreement is the result.

The contact force is enormous, so the fight is real. But the bulb barely moves once the
projection stops writing it, so there is NO static overlap: the seat pose is geometrically
right and the collider is not oversized. Measurement 2 removes the cause when it sets `FREE`,
so it answers "is the seat pose wrong" rather than "is there a fight". The answer is no.

So the ~1000 N comes from writing the pose every step to a body that is already in contact.
Teleporting a touching body is what generates the impulses. Nothing is misplaced or too fat.

That re-ranks the task 6 candidates. An applied 0.1 N.m at the ~0.02 m plug radius is about 5 N
of tangential force. Against a 1067 N normal load, friction cannot be tuned out of the way --
even at mu = 0.2 it is about 200 N. Filtering the collision pair removes the contact outright,
and a compliant channel removes the hard per-step write, which is the actual source. Tuning
`contact_offset` or `rest_offset` targets a static overlap that does not exist.

CAUTION: contact force reported for a body under pose writes is partly a solver artifact. Treat
the RATIO as the finding, not the absolute newtons. The ratio is robust.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure bulb-to-socket contact at the seat pose.")
parser.add_argument("--seed", type=int, default=0, help="Env seed (deterministic).")
parser.add_argument("--steps", type=int, default=60, help="Steps to sample contact over.")
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
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import BULB_MASS_KG, set_layout_seed

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg

BULB_WEIGHT_N = BULB_MASS_KG * 9.81


def build_cfg():
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=1)
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
    # The measurement channel: socket contact, filtered to the old bulb alone, so the lamp's
    # own scenery contacts cannot enter it.
    cfg.scene.socket_bulb_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Socket",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/OldBulb"],
        history_length=1,
        track_air_time=False,
    )
    return cfg


def main() -> int:
    cfg = build_cfg()
    spec = gym.spec("FIATLUX-Replace-v0")
    module_name, class_name = spec.entry_point.split(":")
    env = getattr(importlib.import_module(module_name), class_name)(cfg=cfg)
    env.reset(seed=args_cli.seed)

    socket = env.scene["socket"]
    old_bulb = env.scene["old_bulb"]
    manager = task_attach.attachment_manager(env)
    if manager is None:
        print("FATAL no bulb_attachment term", flush=True)
        return 1
    sensor = env.scene.sensors["socket_bulb_contact"]

    device = env.device
    zero_action = torch.zeros((1, env.action_manager.total_action_dim), device=device)
    seat_axis = torch.tensor(SOCKET_SEAT_AXIS, device=device)
    seat_offset = torch.tensor(SOCKET_SEAT_OFFSET, device=device)
    plug_offset = torch.tensor(BULB_PLUG_OFFSET, device=device)

    print(f"SETUP seed={args_cli.seed} bulb_mass={BULB_MASS_KG} kg  weight={BULB_WEIGHT_N:.3f} N", flush=True)

    def seat_pose():
        q = socket.data.root_quat_w[0]
        seat = socket.data.root_pos_w[0] + quat_apply(q.unsqueeze(0), seat_offset.unsqueeze(0))[0]
        pos = seat - quat_apply(q.unsqueeze(0), plug_offset.unsqueeze(0))[0]
        return pos, q

    # -- 1. contact force while the projection holds the bulb ---------------------------
    mags = []
    for _ in range(args_cli.steps):
        env.step(zero_action)
        f = sensor.data.force_matrix_w  # (N, B, M, 3)
        if f is None:
            continue
        mags.append(float(torch.norm(f.sum(dim=2), dim=-1).max().item()))

    if mags:
        t = torch.tensor(mags)
        print(
            f"CONTACT_FORCE_N steps={len(mags)} min={t.min():.4f} median={t.median():.4f} "
            f"max={t.max():.4f}  ratio_to_weight={t.median() / BULB_WEIGHT_N:.1f}x",
            flush=True,
        )
        print(f"CONTACT_VERDICT {'OVERLAP DRIVEN' if t.median() > 5 * BULB_WEIGHT_N else 'NO LARGE CONTACT'}", flush=True)
    else:
        print("CONTACT_FORCE_N no readings", flush=True)

    # -- 2. depenetration: stop projecting, place at the seat, watch it get pushed out ---
    pos, quat = seat_pose()
    manager._phase[task_attach._OLD, :] = task_attach._FREE  # projection now ignores this bulb
    old_bulb.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))
    old_bulb.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    start = old_bulb.data.root_pos_w[0].clone()
    trace = []
    for i in range(20):
        env.step(zero_action)
        d = float(torch.norm(old_bulb.data.root_pos_w[0] - start).item())
        v = float(torch.norm(old_bulb.data.root_lin_vel_w[0]).item())
        if i in (0, 1, 2, 4, 9, 19):
            trace.append((i, round(d * 1000, 3), round(v, 3)))
    first = trace[0][1]
    print(f"DEPENETRATION_MM trace(step, mm, m/s)={trace}", flush=True)
    print(f"FIRST_STEP_MM {first:.3f}", flush=True)
    # Free fall under gravity alone is ~2 mm in the first step; a shove is much larger.
    print(f"DEPEN_VERDICT {'OVERLAP' if first > 5.0 else 'NO CLEAR OVERLAP (gravity-scale motion)'}", flush=True)

    env.close()
    return 0


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
