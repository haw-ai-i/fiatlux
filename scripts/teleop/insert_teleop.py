# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperation for the Fiatlux benchmark (issue #51).

Drives a Fiatlux *teleop* env (default ``FIATLUX-Insert-Teleop-v0``) with an SE(3) device (keyboard
or 3Dconnexion SpaceMouse). The device's 6-DoF motion is integrated into an **absolute** end-effector
target pose that the env's IK holds against gravity (relative IK would re-anchor to the sagged pose
and drift the hand down); the gripper toggle drives the binary Inspire-hand grip.

    python scripts/insert_teleop.py --task FIATLUX-Insert-Teleop-v0                          # keyboard
    python scripts/insert_teleop.py --task FIATLUX-Insert-Teleop-v0 --teleop_device spacemouse

Keyboard controls: W/S A/D Q/E move the hand; Z/X T/G C/V rotate it; K toggles grip; R resets;
arrow keys slide the base (keyboard only).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperation for the Fiatlux benchmark.")
parser.add_argument("--task", type=str, default="FIATLUX-Insert-Teleop-v0", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (teleop uses 1).")
parser.add_argument(
    "--teleop_device", type=str, default="keyboard", help="Teleop device: keyboard | spacemouse | gamepad."
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Device sensitivity multiplier.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Insert-family envs carry a wrist camera sensor; enable cameras so the obs group builds.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import logging

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX gym ids
import fiatlux_teleop  # noqa: F401,E402  -- registers the FIATLUX-*-Teleop gym ids
import gymnasium as gym
import numpy as np
import torch
from fiatlux_task.robots.g1 import G1_EE_BODY

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import normalize, quat_from_angle_axis, quat_mul, subtract_frame_transforms

from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)

# Max distance (m) the integrated IK target may lead the actual hand, so an out-of-reach command
# can't run the target away and leave control feeling stuck (see the leash in the drive loop).
LEASH = 0.10


def _make_device():
    """Build a raw SE(3) device (returns 6-DoF pose deltas + a gripper flag we integrate)."""
    s = args_cli.sensitivity
    dev = args_cli.teleop_device.lower()
    if dev == "keyboard":
        return Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.01 * s, rot_sensitivity=0.03 * s))
    if dev == "spacemouse":
        return Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.05 * s, rot_sensitivity=0.05 * s))
    if dev == "gamepad":
        return Se3Gamepad(Se3GamepadCfg(pos_sensitivity=0.05 * s, rot_sensitivity=0.05 * s))
    raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")


class _BaseMover:
    """Hold-to-move keyboard control for the robot's fixed base.

    Arrow keys slide the whole robot across the floor (the bolted base is repositioned via
    ``write_root_pose_to_sim`` each step) so the operator can walk up to the target and reach it,
    without a locomotion/balance policy. The arm's IK target is in the body frame, so the hand
    rides along with the base. Tracks key press/release on the same keyboard the SE(3) device uses.
    """

    def __init__(self, max_speed: float = 0.010, smooth: float = 0.12):
        import carb
        import omni

        self._carb = carb
        self._input = carb.input.acquire_input_interface()
        self._kb = omni.appwindow.get_default_app_window().get_keyboard()
        self._pressed: set = set()
        self._sub = self._input.subscribe_to_keyboard_events(self._kb, self._on_event)
        self.max_speed = max_speed  # m per step at full glide
        self.smooth = smooth  # 0..1 easing per step toward the target velocity (lower = smoother)
        self._vx = 0.0
        self._vy = 0.0
        k = carb.input.KeyboardInput
        # world-frame slide: UP toward the bench (-y), DOWN back (+y), LEFT -x, RIGHT +x
        self._map = {k.UP: (0.0, -1.0), k.DOWN: (0.0, 1.0), k.LEFT: (-1.0, 0.0), k.RIGHT: (1.0, 0.0)}

    def _on_event(self, event, *args):
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed.add(event.input)
        elif event.type == self._carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(event.input)
        return True

    def delta_xy(self):
        """Per-step base displacement, velocity-smoothed so the robot eases in/out (no jerk)."""
        tx = ty = 0.0
        for key in self._pressed:
            if key in self._map:
                mx, my = self._map[key]
                tx += mx * self.max_speed
                ty += my * self.max_speed
        # ease the current velocity toward the key-target velocity (ramp up when held, coast to a
        # stop when released) instead of snapping -- this is what removes the "draggy" feel.
        self._vx += (tx - self._vx) * self.smooth
        self._vy += (ty - self._vy) * self.smooth
        if abs(self._vx) < 1e-5:
            self._vx = 0.0
        if abs(self._vy) < 1e-5:
            self._vy = 0.0
        return self._vx, self._vy


def _read_delta(cmd, device):
    """Normalize a device command to (dpos[3], drot[3] rotvec, gripper scalar; -1 close / +1 open)."""
    if isinstance(cmd, tuple):
        dpose, gripper = cmd
        dpose = torch.as_tensor(np.asarray(dpose).ravel(), dtype=torch.float32, device=device)
        grip = torch.tensor(-1.0 if gripper else 1.0, device=device)
    else:
        cmd = torch.as_tensor(cmd, dtype=torch.float32, device=device).flatten()
        dpose = cmd[:6]
        grip = cmd[6] if cmd.numel() > 6 else torch.tensor(1.0, device=device)
    return dpose[:3], dpose[3:6], grip


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(f"Teleoperation needs a ManagerBasedRLEnv task; '{args_cli.task}' is not one.")

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    ee_idx = robot.body_names.index(G1_EE_BODY)
    # arm IK (pose = 7) + optional binary grip (1)
    has_grip = env.action_manager.total_action_dim >= 8

    def capture_target():
        """Current EE pose in the robot base frame -> the initial absolute IK target."""
        p_b, q_b = subtract_frame_transforms(
            robot.data.root_pos_w[:1], robot.data.root_quat_w[:1],
            robot.data.body_pos_w[:1, ee_idx], robot.data.body_quat_w[:1, ee_idx],
        )
        return p_b[0].clone(), q_b[0].clone()

    should_reset = {"flag": False}

    def _trigger_reset() -> None:
        should_reset["flag"] = True

    teleop = _make_device()
    for key in ("R", "RESET"):
        with contextlib.suppress(ValueError, TypeError):
            teleop.add_callback(key, _trigger_reset)
    base_mover = _BaseMover() if args_cli.teleop_device.lower() == "keyboard" else None
    print(f"[teleop] device: {teleop}")

    zero_vel = torch.zeros((env.num_envs, 6), device=env.device)

    def capture_base():
        return robot.data.root_pos_w[0].clone(), robot.data.root_quat_w[0].clone()

    env.reset()
    teleop.reset()
    tgt_pos, tgt_quat = capture_target()
    base_pos, base_quat = capture_base()
    print(
        "[teleop] started. Hand: W/S A/D Q/E move, Z/X T/G C/V rotate, K grip. "
        "Base: arrow keys walk the robot. R resets."
    )

    while simulation_app.is_running():
        with torch.inference_mode():
            # Base: arrow keys slide the whole robot; hold its pose each step so the fixed base stays
            # where the operator parked it (the arm's body-frame target rides along automatically).
            if base_mover is not None:
                dx, dy = base_mover.delta_xy()
                if dx or dy:
                    base_pos = base_pos + torch.tensor([dx, dy, 0.0], device=env.device)
                robot.write_root_pose_to_sim(torch.cat([base_pos, base_quat]).unsqueeze(0))
                robot.write_root_velocity_to_sim(zero_vel)

            dpos, drot, grip = _read_delta(teleop.advance(), env.device)
            # integrate the delta into the absolute target (position + orientation).
            tgt_pos = tgt_pos + dpos
            # Leash the target to the actual reachable hand: if a command pushes the target past the
            # arm's reach it would otherwise run away, and you'd have to unwind the overshoot before
            # the hand moves again ("moves but stuck"). Clamp it within LEASH of the current EE so
            # reversing direction responds immediately.
            cur_pos, _ = capture_target()
            offset = tgt_pos - cur_pos
            dist = torch.norm(offset)
            if dist > LEASH:
                tgt_pos = cur_pos + offset * (LEASH / dist)
            ang = torch.norm(drot)
            if ang > 1e-6:
                dq = quat_from_angle_axis(ang.unsqueeze(0), (drot / ang).unsqueeze(0))
                tgt_quat = normalize(quat_mul(dq, tgt_quat.unsqueeze(0)))[0]
            parts = [tgt_pos, tgt_quat]
            if has_grip:
                parts.append(grip.reshape(1))
            action = torch.cat(parts).unsqueeze(0).repeat(env.num_envs, 1)

            env.step(action)

            # Reset ONLY when the human presses 'R'. We deliberately ignore the env's
            # terminated/truncated (e.g. episode time-out) so the scene doesn't reset itself out
            # from under the operator mid-manipulation.
            if should_reset["flag"]:
                env.reset()
                teleop.reset()
                tgt_pos, tgt_quat = capture_target()
                base_pos, base_quat = capture_base()
                should_reset["flag"] = False

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
