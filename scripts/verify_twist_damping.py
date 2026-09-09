# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Does the new twist damping actually arrest a real spin, under real contact? (issue #171)

Directly reproduces the real-teleop finding that motivated it: a ceiling-seated bulb (seed 3,
dex3, S03) was found spinning about the seat axis at 1-19 rad/s for a sustained ~2.9s -- present
almost from the moment it seated, never decaying under the OLD tilt-damping term (which shared
tilt's tiny max_torque budget with twist, so arresting even 10 rad/s needed several times more
torque than that budget allowed) -- before abruptly destabilizing into a real ejection, no
operator, no hand contact the entire time.

This spawns the old bulb already seated at a forced ceiling mount, then INJECTS an initial spin
about the seat axis matching the reported range, and steps forward under zero action for several
real seconds -- no hand, no disturbance, nothing overwriting pose/velocity after the initial
kick -- reading the manager's own state (phase, axial, twist rate) straight off the real,
contact-resolved simulation. Fixed: twist rate decays toward 0 and phase stays SEATED the whole
run. Still broken: twist rate persists/stays large and/or the bulb eventually destabilizes
(axial or tilt blowing up, phase flipping to FREE) the way the original bug did.

Run via ./pyrun (repo root), not a bare .venv/bin/python: on a machine where .venv is shared
with a different checkout, a bare invocation can silently resolve fiatlux_task to the WRONG
checkout's source -- pyrun fixes that by pointing PYTHONPATH at this repo's own source/ first.

Example
-------
    ./pyrun scripts/verify_twist_damping.py --headless --spin_rate 15.0
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify twist damping arrests a real spin under real contact.")
parser.add_argument("--seed", type=int, default=0, help="Env seed (deterministic).")
parser.add_argument("--seconds", type=float, default=5.0, help="Real simulated seconds to watch.")
parser.add_argument(
    "--spin_rate", type=float, default=15.0, help="Initial spin about the seat axis, rad/s (reported range: 1-19)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything else follows."""

import importlib
import inspect
import sys

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import torch
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import CEILING_FIXTURE_Z, _quat_y_deg, set_layout_seed

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg


def build_cfg():
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=1)
    assert hasattr(cfg.scene, "fresh_bulb"), (
        f"cfg.scene ({type(cfg.scene)} from {sys.modules[type(cfg.scene).__module__].__file__}) has no "
        "fresh_bulb -- fiatlux_task likely resolved to the wrong checkout again; check sys.path/pyrun"
    )
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

    # Force a CEILING mount, matching the reported failure (seed 3, S03, on-ladder wall/ceiling
    # preset) -- same technique as verify_ceiling_hold.py.
    ceiling_quat = _quat_y_deg(180.0)
    x, y, _ = cfg.scene.socket.init_state.pos
    cfg.scene.socket.init_state.pos = (x, y, CEILING_FIXTURE_Z)
    cfg.scene.socket.init_state.rot = ceiling_quat
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
        print("FATAL no bulb_attachment term on FIATLUX-Replace-v0", flush=True)
        return 1

    device = env.device
    zero_action = torch.zeros((1, env.action_manager.total_action_dim), device=device)
    seat_axis = torch.tensor(SOCKET_SEAT_AXIS, device=device)
    seat_offset = torch.tensor(SOCKET_SEAT_OFFSET, device=device)
    plug_offset = torch.tensor(BULB_PLUG_OFFSET, device=device)
    # Read off attach.py's own signature rather than duplicated: the default changed from
    # 0.02 to 0.008 (issue #171 -- the bore is only ~25mm deep, so 20mm was most of the way
    # out), and a stale copy here would silently grade a fallen-out bulb as still held.
    release_threshold = inspect.signature(task_attach.bulb_attachment.__call__).parameters["release_threshold"].default

    def seated_pose() -> tuple[torch.Tensor, torch.Tensor]:
        socket_quat = socket.data.root_quat_w[0]
        seat = socket.data.root_pos_w[0] + quat_apply(socket_quat.unsqueeze(0), seat_offset.unsqueeze(0))[0]
        pos = seat - quat_apply(socket_quat.unsqueeze(0), plug_offset.unsqueeze(0))[0]
        return pos, socket_quat

    def axis_w() -> torch.Tensor:
        return quat_apply(socket.data.root_quat_w, seat_axis.unsqueeze(0))

    def axial() -> float:
        seat = socket.data.root_pos_w + quat_apply(socket.data.root_quat_w, seat_offset.unsqueeze(0))
        plug = old_bulb.data.root_pos_w + quat_apply(old_bulb.data.root_quat_w, plug_offset.unsqueeze(0))
        return float(((plug - seat) * axis_w()).sum(dim=1).item())

    def twist_rate() -> float:
        return float((old_bulb.data.root_ang_vel_w * axis_w()).sum(dim=1).item())

    def phase() -> int:
        return int(manager._phase[task_attach._OLD, 0].item())

    # Spawn the old bulb EXACTLY at the seat, aligned with the (now ceiling) socket -- matching
    # how a real ceiling-mounted Replace scene starts (old_bulb pre-seated) -- then inject the
    # spin. Written once; nothing overwrites pose/velocity again after this.
    pos, quat = seated_pose()
    old_bulb.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))
    old_bulb.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    env.step(zero_action)  # resolves spawn phase -> SEATED, since it's within tolerance of the seat
    if phase() != task_attach._SEATED:
        print("FATAL did not seat at spawn -- the setup itself is wrong, not testing damping", flush=True)
        return 1

    spin_vel = torch.zeros((1, 6), device=device)
    spin_vel[:, 3:] = args_cli.spin_rate * axis_w()  # angular velocity about the seat axis
    old_bulb.write_root_velocity_to_sim(spin_vel)
    print(
        f"SETUP mount=ceiling spin_rate0={twist_rate():.2f} axial0={axial():.4f} phase0={phase()} "
        "(expect phase0=1/SEATED)",
        flush=True,
    )

    steps = int(round(args_cli.seconds / env.step_dt))
    trace = []
    max_abs_axial = 0.0
    dropped_at = None
    for i in range(steps):
        env.step(zero_action)  # no hand, no disturbance -- gravity + real contact + retention only
        a, w, p = axial(), twist_rate(), phase()
        max_abs_axial = max(max_abs_axial, abs(a))
        if p != task_attach._SEATED and dropped_at is None:
            dropped_at = i * env.step_dt
        if i % max(1, steps // 30) == 0 or i == steps - 1:
            trace.append((round(i * env.step_dt, 3), round(a, 4), round(w, 3), p))

    final_twist = twist_rate()
    held = dropped_at is None and max_abs_axial <= release_threshold
    damped = abs(final_twist) < 0.1 * abs(args_cli.spin_rate)  # decayed to <10% of the initial kick
    print(
        f"RESULT held={held} damped={damped} final_twist_rate={final_twist:.3f} "
        f"initial_spin_rate={args_cli.spin_rate:.2f} max_abs_axial={max_abs_axial:.4f} "
        f"release_threshold={release_threshold:.4f} dropped_at={dropped_at} final_phase={phase()} "
        f"trace={trace}",
        flush=True,
    )
    env.close()
    return 0 if (held and damped) else 1


if __name__ == "__main__":
    import os
    import traceback

    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code:
            os._exit(exit_code)
        simulation_app.close()
