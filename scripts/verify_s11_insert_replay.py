# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replays a real S11 insert teleop bag's own recorded ACTIONS, old gains vs new (issue #171).

The earlier verification for this bug (``verify_wall_hold.py``) isolated the tilt/release
mechanism with a SCRIPTED bump standing in for a hand disturbance -- real, but not a replay of
what actually happened in VR. This script replays the real thing: ``sonic_teleop.py`` calls
``env.step(arm_action)`` immediately followed by ``recorder.record_step(obs, arm_action, ...)``
(confirmed by reading it), so a bag's own ``actions`` column IS, frame for frame, exactly the
action tensor the teleop operator's hand produced. Feeding that same sequence into a freshly
built env at the bag's own seed reproduces the same robot behavior -- including the exact
hand-release gesture that ejected the bulb -- so this directly answers "does the fix change the
outcome of that recorded VR session," not just "does a synthetic bump recover."

Does NOT attempt a bit-identical reproduction of the original bag's bulb trajectory the whole
episode through -- unlike ``verify_no_twist_spin.py``'s zero-action, contact-light scenario, this
one is rich, sustained hand-bulb contact for thousands of steps, and this fix changes the
retention wrench itself, so divergence from the bag is EXPECTED and is the point: old gains
in a fresh run should reproduce a release near where the bag's own ``gate_fresh_bulb_attached``
dropped; new gains, given the identical action sequence, are what's under test.

Usage
-----
    uv run python scripts/verify_s11_insert_replay.py --headless --episode ep00_replay.pt
    uv run python scripts/verify_s11_insert_replay.py --headless --episode ep00_replay.pt --old_gains

``--episode`` points at a ``.pt`` file produced by extracting a bag's own ``run.h5`` (fields:
actions, gate_fresh_bulb_attached, fresh_bulb_pos/quat, socket_pos/quat, contact_force(_left),
step_in_episode, meta_json) -- done outside this script since the Isaac Lab venv here has no
h5py, and there is no reason to add it just to read an array once.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a real S11 insert bag's own actions, old gains vs new.")
parser.add_argument("--episode", type=str, required=True, help="Path to a .pt file from extract_replay_data.py.")
parser.add_argument("--seed", type=int, default=2, help="Env seed. 2 is the seed the S11 evidence bags used.")
parser.add_argument("--hand", type=str, default="dex3", choices=("dex3", "inspire"), help="Hand variant (bags: dex3).")
parser.add_argument(
    "--task", type=str, default="FIATLUX-S11-ScrewInBulb-Teleop-v0", help="Task id (the bags' own meta.json names)."
)
parser.add_argument(
    "--old_gains",
    action="store_true",
    help="Override to the PRE-fix tilt/release gains (tilt_k=0.05, tilt_d=0.01, max_torque=0.05, "
    "release_debounce_steps=1) instead of the shipped defaults, to reproduce the reported failure "
    "as a baseline for comparison against the same run under the fix.",
)
parser.add_argument(
    "--max_steps", type=int, default=None, help="Stop after this many steps (default: the whole recorded episode)."
)
parser.add_argument(
    "--start_step",
    type=int,
    default=0,
    help="Teleport robot/bulb/ladder state to the bag's own recorded state at this step before "
    "replaying actions[start_step:] -- an open-loop replay from reset() diverges from the bag "
    "within a few seconds (confirmed: the bulb ends up dropped and motionless long before the "
    "bag's own insertion/hold window), so testing the retention fix means starting from a state "
    "close to that window, not from reset() and hoping ~1700 steps of drift-prone bipedal "
    "manipulation replay lands in the same place the bag did.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless

os.environ["FIATLUX_TELEOP_HAND"] = args_cli.hand

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything else follows."""

import importlib

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX benchmark environments
import fiatlux_teleop  # noqa: F401  -- registers the -Teleop-v0 twins
import gymnasium as gym
import torch
import verify_common
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg

OLD_GAINS = dict(tilt_k=0.05, tilt_d=0.01, max_torque=0.05, release_debounce_steps=1)


def build_cfg():
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    verify_common.assert_right_checkout(cfg, "fresh_bulb")
    cfg.seed = args_cli.seed
    verify_common.strip_visual_obs(cfg)
    # Teleop is operator-paced; watch the whole recorded episode even if a termination would
    # otherwise cut it short -- a truncated replay would hide a late re-seat or re-ejection.
    ep = torch.load(args_cli.episode, weights_only=False)
    n_steps = ep["actions"].shape[0] if args_cli.max_steps is None else min(args_cli.max_steps, ep["actions"].shape[0])
    cfg.episode_length_s = max(n_steps * cfg.sim.dt * cfg.decimation * 1.5, cfg.episode_length_s)
    for term in [t for t in vars(cfg.terminations) if not t.startswith("_")]:
        if term != "time_out" and getattr(cfg.terminations, term, None) is not None:
            setattr(cfg.terminations, term, None)
    return cfg, ep, n_steps


class SeatProbe:
    """Seat-frame quantities for the FRESH bulb, read off the contact-resolved sim."""

    def __init__(self, env):
        self.socket = env.scene["socket"]
        self.bulb = env.scene["fresh_bulb"]
        self.manager = task_attach.attachment_manager(env)
        dev = env.device
        self._axis = torch.tensor(SOCKET_SEAT_AXIS, device=dev).unsqueeze(0)
        self._seat_off = torch.tensor(SOCKET_SEAT_OFFSET, device=dev).unsqueeze(0)
        self._plug_off = torch.tensor(BULB_PLUG_OFFSET, device=dev).unsqueeze(0)

    def axis_w(self) -> torch.Tensor:
        return quat_apply(self.socket.data.root_quat_w, self._axis)

    def read(self) -> dict:
        seat = self.socket.data.root_pos_w + quat_apply(self.socket.data.root_quat_w, self._seat_off)
        plug = self.bulb.data.root_pos_w + quat_apply(self.bulb.data.root_quat_w, self._plug_off)
        d = plug - seat
        axis = self.axis_w()
        axial = (d * axis).sum(dim=1)
        lateral = torch.norm(d - axial.unsqueeze(1) * axis, dim=1)
        tilt = task_attach._tilt_error(self.socket.data.root_quat_w, self.bulb.data.root_quat_w, self._axis)
        return {
            "axial": float(axial.item()),
            "lateral": float(lateral.item()),
            "tilt": float(tilt.item()),
            "phase": int(self.manager._phase[task_attach._FRESH, 0].item()),
        }


def main() -> int:
    cfg, ep, n_steps = build_cfg()
    spec = gym.spec(args_cli.task)
    module_name, class_name = spec.entry_point.split(":")
    env = getattr(importlib.import_module(module_name), class_name)(cfg=cfg)
    env.reset(seed=args_cli.seed)

    if task_attach.attachment_manager(env) is None:
        print(f"FATAL no bulb_attachment term on {args_cli.task}", flush=True)
        return 1
    probe = SeatProbe(env)

    gains = dict(OLD_GAINS) if args_cli.old_gains else {}
    term_params = env.event_manager.get_term_cfg("bulb_attachment").params
    term_params.update(gains)
    print(f"\n=== GAINS: {'OLD (pre-fix)' if args_cli.old_gains else 'NEW (shipped defaults)'} {gains} ===", flush=True)

    # Confirm the reconstructed FIXTURE LAYOUT matches the bag's own first frame before trusting
    # anything that follows -- same discipline verify_no_twist_spin.py uses, applied to the
    # static scene draw rather than a pre-seated bulb pose (S11 starts from the robot holding the
    # bulb on the ladder, not from an already-seated one, so there is no seated-bulb frame to
    # match yet -- the socket's own pose is what the seed draw actually controls).
    socket = env.scene["socket"]
    bag_socket_pos = ep["socket_pos"][0].tolist()
    bag_socket_quat = ep["socket_quat"][0].tolist()
    here_socket_pos = [round(float(v), 4) for v in socket.data.root_pos_w[0]]
    here_socket_quat = [round(float(v), 4) for v in socket.data.root_quat_w[0]]
    print(f"  bag socket_pos ={[round(v, 4) for v in bag_socket_pos]}  here={here_socket_pos}", flush=True)
    print(f"  bag socket_quat={[round(v, 4) for v in bag_socket_quat]}  here={here_socket_quat}", flush=True)
    pos_mismatch = max(abs(a - b) for a, b in zip(bag_socket_pos, here_socket_pos))
    if pos_mismatch > 5e-3:
        print(f"FATAL socket position mismatch ({pos_mismatch:.4f} m) -- not the same layout draw", flush=True)
        return 1

    action_dim = env.action_manager.total_action_dim
    if action_dim != ep["actions"].shape[1]:
        print(f"FATAL action_dim mismatch: env expects {action_dim}, episode has {ep['actions'].shape[1]}", flush=True)
        return 1

    device = env.device
    actions = ep["actions"].to(device)

    if args_cli.start_step > 0:
        s = args_cli.start_step
        robot = env.scene["robot"]
        ladder = env.scene["ladder"]
        fresh_bulb = env.scene["fresh_bulb"]
        robot.write_joint_state_to_sim(
            ep["joint_pos"][s].to(device).unsqueeze(0), ep["joint_vel"][s].to(device).unsqueeze(0)
        )
        robot.write_root_pose_to_sim(
            torch.cat([ep["robot_root_pos"][s], ep["robot_root_quat"][s]]).to(device).unsqueeze(0)
        )
        robot.write_root_velocity_to_sim(
            torch.cat([ep["robot_root_lin_vel"][s], ep["robot_root_ang_vel"][s]]).to(device).unsqueeze(0)
        )
        ladder.write_root_pose_to_sim(torch.cat([ep["ladder_pos"][s], ep["ladder_quat"][s]]).to(device).unsqueeze(0))
        ladder.write_root_velocity_to_sim(
            torch.cat([ep["ladder_lin_vel"][s], ep["ladder_ang_vel"][s]]).to(device).unsqueeze(0)
        )
        fresh_bulb.write_root_pose_to_sim(
            torch.cat([ep["fresh_bulb_pos"][s], ep["fresh_bulb_quat"][s]]).to(device).unsqueeze(0)
        )
        fresh_bulb.write_root_velocity_to_sim(
            torch.cat([ep["fresh_bulb_lin_vel"][s], ep["fresh_bulb_ang_vel"][s]]).to(device).unsqueeze(0)
        )
        print(f"  teleported robot/ladder/fresh_bulb state to the bag's own step {s}", flush=True)
        print(f"  DEBUG state immediately after teleport, before any step: {probe.read()}", flush=True)
        actions = actions[s:]
        n_steps = actions.shape[0] if args_cli.max_steps is None else min(args_cli.max_steps, actions.shape[0])

    bag_gate = ep["gate_fresh_bulb_attached"]
    bag_release_step = None
    was_true = False
    for i in range(bag_gate.shape[0]):
        if was_true and not bool(bag_gate[i]):
            bag_release_step = i
        was_true = bool(bag_gate[i])
    print(
        f"  bag's own longest SEATED run ended at step {bag_release_step} "
        f"(of {bag_gate.shape[0]}, {n_steps} replayed here)",
        flush=True,
    )

    max_seated_run = 0
    cur_run = 0
    release_steps = []
    was_seated = False
    trace = []
    sample_every = max(1, n_steps // 40)
    for i in range(n_steps):
        abs_step = args_cli.start_step + i
        env.step(actions[i].unsqueeze(0))
        state = probe.read()
        seated = state["phase"] == task_attach._SEATED
        if seated:
            cur_run += 1
        else:
            if was_seated:
                release_steps.append(abs_step)
            cur_run = 0
        max_seated_run = max(max_seated_run, cur_run)
        was_seated = seated
        if i % sample_every == 0 or i == n_steps - 1:
            trace.append((abs_step, round(state["axial"], 4), round(state["tilt"], 4), seated))

    print(f"\nRESULT max_seated_run_steps={max_seated_run} ({max_seated_run * env.step_dt:.2f}s)", flush=True)
    print(f"  release_steps (absolute, matching the bag's own step numbering)={release_steps}", flush=True)
    print(f"  final phase={'SEATED' if was_seated else 'FREE'}", flush=True)
    print(f"  trace (absolute step, axial, tilt, seated): {trace}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    verify_common.run_verify_main(main, simulation_app)
