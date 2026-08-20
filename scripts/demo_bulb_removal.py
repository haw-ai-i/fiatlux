# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record ONE clean bayonet removal: unscrew, then eject. Nothing else (issue #77).

`verify_attach.py --video` records the whole 25-check suite, which makes a poor demonstration.
It drives the old bulb's rotation six times -- down, up, down, up, down, out -- because two of
those runs are adversarial checks that prove reversing mid-unlock RE-LOCKS the bulb
(`reversed_unlock_relocks`, `unlock_step_pins_axial`). It also teleports bulbs between checks
with `place_bulb`, which reads as jumpy. Both are correct for a test and wrong for a video.

This performs only the two motions that are the removal:

  1. rotate the seated bulb from the lock angle to zero  (verify_attach.py:675)
  2. travel out along the seat axis until it ejects      (verify_attach.py:694)

then carries it to the disposal crate. One continuous take, no reversals, no teleports.

NO ROBOT IS INVOLVED. The bulb is driven by direct pose writes with robot actions at zero, the
same as the harness. A humanoid cannot reach these fixtures standing anyway -- `G1_OVERHEAD_REACH`
is 1.374 m against a wall mount at 2.2 m -- so a robot in shot would stand short of a lamp it
cannot touch. Filming a robot that actually turns a bulb needs the bench task and the state
machine ported to it, which is #76 Step 2.

The rotation itself is invisible: the bulb is a surface of revolution, so turning it about its
own axis changes almost nothing on screen. The burnt-in overlay and the theta plot are what show
the release actually happening.

Examples
--------
    python scripts/demo_bulb_removal.py --headless --video /tmp/removal.mp4
    python scripts/demo_bulb_removal.py --headless --seed 1 --video /tmp/ceiling.mp4
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record one clean bayonet bulb removal.")
parser.add_argument("--seed", type=int, default=0, help="Layout seed. 0/2/3/5/6 wall, 1/4/7 ceiling.")
parser.add_argument("--video", type=str, required=True, help="Output MP4 path.")
parser.add_argument("--unscrew-frames", type=int, default=70, help="Frames for the release rotation.")
parser.add_argument("--eject-frames", type=int, default=45, help="Frames for the axial travel out.")
parser.add_argument("--carry-frames", type=int, default=45, help="Frames to carry the freed bulb to the crate.")
parser.add_argument("--radius-scale", type=float, default=0.62, help="Tighten the fixture orbit by this factor.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything else follows."""

import importlib

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import torch
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed
from fiatlux_task.viz import VideoRecorder, fixture_orbit, make_video_camera_cfg, orbit_pose

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_tasks.utils import parse_env_cfg

PHASE_NAMES = {task_attach._FREE: "FREE", task_attach._AXIAL: "AXIAL", task_attach._ROTATING: "ROTATING"}


def build_cfg():
    """Replace, stripped to the bayonet rig, with the layout pinned."""
    # The mount is drawn inside parse_env_cfg, before cfg.seed exists, so declare it first or
    # the fixture lands somewhere different every run.
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=1)
    cfg.seed = args_cli.seed
    for event in ("randomize_sky_intensity", "randomize_key_light", "randomize_material_tint"):
        if getattr(cfg.events, event, None) is not None:
            setattr(cfg.events, event, None)
    for term in ("success", "old_bulb_dropped", "fresh_bulb_dropped"):
        if getattr(cfg.terminations, term, None) is not None:
            setattr(cfg.terminations, term, None)
    for camera in ("ego_camera", "torso_camera", "wrist_camera"):
        if getattr(cfg.scene, camera, None) is not None:
            setattr(cfg.scene, camera, None)
    for group_name in ("policy", "privileged"):
        group = getattr(cfg.observations, group_name, None)
        for term in ("ego_rgb", "torso_rgb", "wrist_rgb"):
            if group is not None and getattr(group, term, None) is not None:
                setattr(group, term, None)
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    # Same isolation the harness uses: the state machine owns the bulb here, and contact
    # artifacts would only add noise to a demonstration.
    for bulb_cfg in (cfg.scene.bulb, cfg.scene.old_bulb):
        if getattr(bulb_cfg.spawn, "rigid_props", None) is not None:
            bulb_cfg.spawn.rigid_props.disable_gravity = True
        bulb_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    cfg.scene.video_cam = make_video_camera_cfg()
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
    zeros6 = torch.zeros((1, 6), device=device)
    seat_axis = torch.tensor(SOCKET_SEAT_AXIS, device=device)
    seat_offset = torch.tensor(SOCKET_SEAT_OFFSET, device=device)
    plug_offset = torch.tensor(BULB_PLUG_OFFSET, device=device)
    depth = float(manager._depth[0].item())
    angle = float(manager._angle[0].item())
    sign = manager.rotation_sign

    video = VideoRecorder(env, env.scene["video_cam"], args_cli.video, fps=20)
    orbit = dict(fixture_orbit(env.cfg))
    orbit["radius"] *= args_cli.radius_scale
    total = args_cli.unscrew_frames + args_cli.eject_frames + args_cli.carry_frames
    print(f"SETUP seed={args_cli.seed} depth={depth:.4f} angle={angle:.4f} sign={sign:+.0f}", flush=True)
    print(f"ORBIT {orbit}", flush=True)

    trace: list[tuple[int, float]] = []

    def seat_geometry():
        quat = socket.data.root_quat_w[0]
        seat = socket.data.root_pos_w[0] + quat_apply(quat.unsqueeze(0), seat_offset.unsqueeze(0))[0]
        axis_w = quat_apply(quat.unsqueeze(0), seat_axis.unsqueeze(0))[0]
        return seat, quat, axis_w

    def place(axial: float, rotation: float) -> None:
        """Put the bulb at `axial` along the seat axis, twisted `rotation` in LOCK-POSITIVE units."""
        seat, socket_quat, axis_w = seat_geometry()
        half = torch.tensor(0.5 * sign * rotation, device=device)
        spin = torch.cat([torch.cos(half).unsqueeze(0), axis_w * torch.sin(half)])
        bulb_quat = quat_mul(spin.unsqueeze(0), socket_quat.unsqueeze(0))[0]
        plug_point = seat + axial * axis_w
        pos = plug_point - quat_apply(bulb_quat.unsqueeze(0), plug_offset.unsqueeze(0))[0]
        old_bulb.write_root_pose_to_sim(torch.cat([pos, bulb_quat]).unsqueeze(0))
        old_bulb.write_root_velocity_to_sim(zeros6)

    def capture() -> None:
        phase = int(manager._phase[task_attach._OLD, 0].item())
        theta = float(manager._theta[task_attach._OLD, 0].item())
        trace.append((phase, theta))
        eye, lookat = orbit_pose(len(video), total, **orbit)
        video.set_pose(eye, lookat)
        caption = f"old bulb  {PHASE_NAMES.get(phase, phase)}\ntheta = {theta:5.3f} rad"
        video.capture(overlay=caption)

    # Warm the renderer before the first capture, or frame 0 comes back empty.
    eye, lookat = orbit_pose(0, total, **orbit)
    video.set_pose(eye, lookat)
    for _ in range(4):
        env.step(zero_action)

    # 1. UNSCREW. Rotate from the lock angle to zero, continuously. No reversal: the harness
    #    reverses twice on purpose to prove the lock re-engages, which is a test, not a removal.
    print("UNSCREW start", flush=True)
    for i in range(args_cli.unscrew_frames):
        place(0.0, angle * (1.0 - (i + 1) / args_cli.unscrew_frames))
        env.step(zero_action)
        capture()
    released = int(manager._phase[task_attach._OLD, 0].item()) != task_attach._ROTATING
    theta_now = float(manager._theta[task_attach._OLD, 0].item())
    print(f"UNSCREW done  released={released}  theta={theta_now:.4f}", flush=True)

    # 2. EJECT. Travel out along the seat axis past the insertion depth, at which point the
    #    state machine drops the bulb to FREE and physics owns it again.
    print("EJECT start", flush=True)
    for i in range(args_cli.eject_frames):
        place(1.2 * depth * (i + 1) / args_cli.eject_frames, 0.0)
        env.step(zero_action)
        capture()
    freed = int(manager._phase[task_attach._OLD, 0].item()) == task_attach._FREE
    print(f"EJECT done  free={freed}", flush=True)

    # 3. Carry it to the crate, where the task wants it.
    start = old_bulb.data.root_pos_w[0].clone()
    quat = old_bulb.data.root_quat_w[0].clone()
    crate = env.scene["bin"].data.root_pos_w[0].clone()
    crate[2] += 0.05
    for i in range(args_cli.carry_frames):
        frac = (i + 1) / args_cli.carry_frames
        pos = start * (1.0 - frac) + crate * frac
        old_bulb.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))
        old_bulb.write_root_velocity_to_sim(zeros6)
        env.step(zero_action)
        capture()

    print(f"wrote video {video.write()} ({len(video)} frames)", flush=True)
    plot = _write_plot(args_cli.video, trace, angle)
    if plot:
        print(f"wrote plot {plot}", flush=True)
    print(f"RESULT released={released} freed={freed}", flush=True)
    env.close()
    return 0 if (released and freed) else 1


def _write_plot(video_path: str, trace, rotation_angle: float):
    """theta against frame, with the phase shaded behind it."""
    if not trace:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    phases = [row[0] for row in trace]
    thetas = [row[1] for row in trace]
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=140)
    colors = {task_attach._FREE: "#e8f5e9", task_attach._AXIAL: "#fff4e5", task_attach._ROTATING: "#e8eefc"}
    start = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            # Bin EDGES, so a phase lasting one frame still shades a full cell.
            ax.axvspan(start - 0.5, i - 0.5, color=colors.get(phases[start], "#ffffff"), zorder=0)
            start = i
    ax.plot(range(len(thetas)), thetas, color="#1f4fd8", linewidth=2.2)
    ax.axhline(rotation_angle, color="#666666", linewidth=0.9, linestyle=":")
    ax.axhline(0.0, color="#666666", linewidth=0.9, linestyle=":")
    ax.set_xlabel("frame")
    ax.set_ylabel("theta (rad)")
    ax.set_title("One removal: unscrew then eject (blue ROTATING, orange AXIAL, green FREE)")
    ax.margins(x=0)
    fig.tight_layout()
    out = video_path.rsplit(".", 1)[0] + "_theta.png"
    fig.savefig(out)
    plt.close(fig)
    return out


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
