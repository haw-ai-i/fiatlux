# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that asset *interactions* are modeled correctly -- no policies involved.

``verify_scene.py`` checks that the scene is solid; this tool checks the physical
interactions the benchmark actually grades, using hardcoded scenarios instead of
trained policies: the robot root is fixed (``fix_root_link``) in calibrated poses
(``fiatlux_task.poses``), joints follow scripted position targets, and objects are
placed/launched via their initial states. Each scenario prints PASS/FAIL checks and
the process exits non-zero if any fail.

Scenarios (``--scenario``):

- ``socket``   : bulb seated in the lamp rests stably (deep resting contact -- the
                 classic PhysX blowup locus), the ``bulb_seated`` success pose is
                 physically attainable, and the stock robot carries no
                 self-collision force noise.
- ``hand``     : palm-down press on the bulb lying on the bench: hover clear,
                 sustained gentle contact under the fragility threshold, bulb stays
                 put, and releasing does not launch it.
- ``fragility``: the sensor->recorder->scorer break/drop chain fires exactly when it
                 should: a gentle press scores unbroken, a hard wedge scores broken,
                 a free fall scores dropped (three recorded, offline-scored episodes).
- ``ladder``   : the free-rooted G1 leaning onto the step ladder holds body-weight-
                 scale contact, stays put, and nothing explodes.
- ``all``      : everything above, one subprocess per scenario.

Examples
--------
    uv run python scripts/verify_interactions.py --headless --scenario socket
    uv run python scripts/verify_interactions.py --headless --scenario all
    # print poses/contacts for calibrating fiatlux_task.poses constants:
    uv run python scripts/verify_interactions.py --headless --scenario hand --probe
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify Fiatlux asset interactions (policy-free scenarios).")
parser.add_argument(
    "--scenario",
    type=str,
    default="all",
    choices=["all", "socket", "hand", "fragility", "ladder"],
    help="Which interaction scenario(s) to run.",
)
parser.add_argument("--seed", type=int, default=0, help="Env seed (scenarios are deterministic).")
parser.add_argument(
    "--probe",
    action="store_true",
    help="Calibration mode: print body names, palm/ladder poses and live contact readings "
    "instead of grading, so fiatlux_task.poses constants can be tuned.",
)
parser.add_argument(
    "--record-bag",
    type=str,
    default=None,
    help="Optional output dir: record the fragility scenario runs as trajectory bags "
    "(golden-run artifacts; also exercised internally for scoring).",
)
parser.add_argument(
    "--video",
    type=str,
    default=None,
    help="Debug: write an MP4 of the scenario to this path (implies camera rendering).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Every scenario is headless-friendly; default to headless like eval/record do.
args_cli.headless = True if args_cli.headless is None else args_cli.headless

# ``--scenario all`` fans out one subprocess per scenario BEFORE booting Kit here:
# a second ManagerBasedEnv in one Kit process hangs at scene creation (observed on
# Isaac Sim 5.1), so every scenario gets a fresh process and one env.
if args_cli.scenario == "all":
    import subprocess

    failed = []
    for name in ("socket", "hand", "fragility", "ladder"):
        cmd = [sys.executable, os.path.abspath(__file__), "--scenario", name, "--seed", str(args_cli.seed)]
        if args_cli.headless:
            cmd.append("--headless")
        if args_cli.probe:
            cmd.append("--probe")
        if args_cli.record_bag:
            cmd += ["--record-bag", args_cli.record_bag]
        if args_cli.video:
            cmd += ["--video", os.path.join(args_cli.video, f"{name}.mp4")]
        print(f"\n[verify] ===== spawning scenario process: {name} =====", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
    if failed:
        print(f"\n[verify] OVERALL: FAIL (failing scenarios: {', '.join(failed)})")
    else:
        print("\n[verify] OVERALL: PASS (all scenarios)")
    sys.exit(1 if failed else 0)

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.util
import tempfile

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import torch
from fiatlux_task.poses import (
    ARM_PRESS_CRUSH,
    ARM_PRESS_DOWN,
    ARM_PRESS_HOVER,
    BULB_LYING_QUAT,
    HAND_FLAT,
    LADDER_STANCE_JOINTS,
    LADDER_STANCE_ROOT_POS,
    LADDER_STANCE_ROOT_ROT,
)
from fiatlux_task.recording import TrajectoryRecorder
from fiatlux_task.robots.g1 import G1_ARM_JOINTS, G1_HAND_JOINTS
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import rewards as task_rewards
from prettytable import PrettyTable

from isaaclab.sensors import ContactSensorCfg

from isaaclab_tasks.utils import parse_env_cfg

# scripts/score.py is a sibling script, not a package module.
_SCORE_SPEC = importlib.util.spec_from_file_location(
    "fiatlux_score", os.path.join(os.path.dirname(os.path.abspath(__file__)), "score.py")
)
score = importlib.util.module_from_spec(_SCORE_SPEC)
sys.modules["fiatlux_score"] = score  # dataclass resolution needs the module registered
_SCORE_SPEC.loader.exec_module(score)

# Sanity bounds shared with verify_scene.py: anything past these is an explosion,
# not a plausible interaction outcome.
MAX_SPEED = 25.0  # m/s
MAX_ROOT_Z = 2.6  # m
MIN_ROOT_Z = -0.05  # m

STEPS_PER_SECOND = 30  # sim.dt=1/120 * decimation=4 (g1_bulb_env_cfg)


# --------------------------------------------------------------------------- #
# Result collection (same PASS/FAIL surface as verify_scene.py)                #
# --------------------------------------------------------------------------- #
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}{' -- ' + detail if detail else ''}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


# --------------------------------------------------------------------------- #
# Env construction                                                             #
# --------------------------------------------------------------------------- #
def build_insert_cfg(num_envs: int = 1):
    """Insert-task cfg stripped for deterministic, camera-free scenario runs."""
    cfg = parse_env_cfg("FIATLUX-Insert-v0", device=args_cli.device, num_envs=num_envs)
    cfg.seed = args_cli.seed
    # No camera sensor -> no --enable_cameras, no resnet18 feature download.
    cfg.scene.wrist_camera = None
    cfg.observations.policy.wrist_rgb = None
    # Deterministic resets: zero every randomization range, keep the reset terms so
    # each reset returns entities exactly to their (scenario-crafted) init_state.
    cfg.events.randomize_light = None
    cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_socket.params["pose_range"] = {}
    cfg.events.reset_bulb.params["pose_range"] = {}
    # only time_out may end an episode: scenarios hold poses that trip the task
    # terminations every step (the scorer detects drops from bulb height anyway)
    cfg.terminations.success = None
    cfg.terminations.bulb_dropped = None
    return cfg


def make_env(task_id: str, cfg):
    """Instantiate the env directly from its registered entry point (verify_scene pattern)."""
    spec = gym.spec(task_id)
    module_name, class_name = spec.entry_point.split(":")
    env_class = getattr(importlib.import_module(module_name), class_name)
    env = env_class(cfg=cfg)
    env.reset(seed=args_cli.seed)
    return env


VIDEO = None  # debug VideoRecorder; captures every env_step when --video is set


def attach_video(env, lookat, eye):
    global VIDEO
    if args_cli.video:
        from fiatlux_task.viz import VideoRecorder

        VIDEO = VideoRecorder(env, env.scene["video_cam"], args_cli.video)
        VIDEO.set_pose(eye, lookat)


def env_step(env, actions):
    """Step either an RL (5-tuple) or base (2-tuple) manager env."""
    out = env.step(actions)
    if VIDEO is not None:
        VIDEO.capture()
    return out  # callers only use the env's scene state, not the returns


# --------------------------------------------------------------------------- #
# Scripted actions                                                             #
# --------------------------------------------------------------------------- #
def action_slots(env):
    """Map action-vector slots -> joint names, replicating JointPositionAction's
    resolution (asset-order ``find_joints``, arm term first as declared in ActionsCfg)."""
    robot = env.scene["robot"]
    slots = []
    for group in (G1_ARM_JOINTS, G1_HAND_JOINTS):
        ids, names = robot.find_joints(group)
        slots.extend(zip(ids, names))
    return slots


def targets_to_actions(env, targets: dict[str, float]) -> torch.Tensor:
    """Invert the action transform (target = default + 0.5 * action) for named joints;
    unnamed joints get zero action (hold default)."""
    robot = env.scene["robot"]
    slots = action_slots(env)
    act = torch.zeros((env.num_envs, len(slots)), device=env.device)
    for k, (joint_id, name) in enumerate(slots):
        if name in targets:
            default = robot.data.default_joint_pos[:, joint_id]
            act[:, k] = 2.0 * (targets[name] - default)
    return act


# --------------------------------------------------------------------------- #
# State monitoring                                                             #
# --------------------------------------------------------------------------- #
class Monitor:
    """Per-run accumulator for explosion/NaN detection and scenario metrics."""

    def __init__(self, env):
        self.env = env
        self.nan = False
        self.max_speed = 0.0
        self.min_z = float("inf")
        self.max_z = float("-inf")

    def step(self):
        robot = self.env.scene["robot"]
        bulb = self.env.scene["bulb"]
        if torch.isnan(robot.data.root_pos_w).any() or torch.isnan(bulb.data.root_pos_w).any():
            self.nan = True
            return
        self.max_speed = max(
            self.max_speed,
            robot.data.root_lin_vel_w.norm(dim=-1).max().item(),
            bulb.data.root_lin_vel_w.norm(dim=-1).max().item(),
        )
        z = robot.data.root_pos_w[:, 2]
        self.min_z = min(self.min_z, z.min().item())
        self.max_z = max(self.max_z, z.max().item())

    def bounded(self) -> tuple[bool, str]:
        ok = (
            not self.nan
            and self.max_speed < MAX_SPEED
            and self.max_z < MAX_ROOT_Z
            and self.min_z > MIN_ROOT_Z
        )
        detail = (
            "NaN in states"
            if self.nan
            else f"peak speed {self.max_speed:.1f} m/s, root z in [{self.min_z:.2f}, {self.max_z:.2f}] m"
        )
        return ok, detail


def run_steps(env, actions, n, monitor=None, per_step=None):
    """Step ``n`` env steps under fixed or callable actions."""
    for i in range(n):
        act = actions(i) if callable(actions) else actions
        env_step(env, act)
        if monitor is not None:
            monitor.step()
            if monitor.nan:
                break
        if per_step is not None:
            per_step(i)


def pairwise_force(sensor) -> torch.Tensor:
    """Per-body force magnitudes against the sensor's filtered prims: (B,) tensor (env 0)."""
    fm = sensor.data.force_matrix_w  # (N, B, M, 3)
    return fm[0].sum(dim=1).norm(dim=-1)  # (B,)


# --------------------------------------------------------------------------- #
# Scenario: bulb in socket                                                     #
# --------------------------------------------------------------------------- #
def scenario_socket():
    from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import (
        TABLETOP_SEATED_BULB_POSITION,
        TABLETOP_SOCKET_POSITION,
    )

    print("\n[verify] === scenario: bulb-in-socket ===")

    # --- part 1: physically-seated rest pose is stable -----------------------------
    cfg = build_insert_cfg()
    cfg.scene.bulb.init_state.pos = TABLETOP_SEATED_BULB_POSITION
    cfg.scene.bulb.spawn.activate_contact_sensors = True
    # B1K objects nest their single rigid body under <entity>/base_link.
    cfg.scene.bulb_socket_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Bulb/base_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Socket/base_link"],
        history_length=1,
    )
    env = make_env("FIATLUX-Insert-v0", cfg)
    try:
        bulb = env.scene["bulb"]
        zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        mon = Monitor(env)
        start = bulb.data.root_pos_w.clone()
        forces: list[float] = []

        def track(_i):
            forces.append(pairwise_force(env.scene.sensors["bulb_socket_contact"]).sum().item())

        run_steps(env, zero, 60, mon)  # settle
        settled = bulb.data.root_pos_w.clone()
        run_steps(env, zero, 240, mon, per_step=track)  # soak ~8 s

        moved = (bulb.data.root_pos_w - settled).norm().item()
        drop = (start - settled)[0, 2].item()
        speed = bulb.data.root_lin_vel_w.norm().item()
        ok, detail = mon.bounded()
        record("socket:no_explosion", ok, detail)
        record(
            "socket:rest_stable",
            (not mon.nan) and moved < 0.01 and speed < 0.05,
            f"soak drift {moved * 1000:.1f} mm, final speed {speed:.3f} m/s (settle drop {drop * 1000:.0f} mm)",
        )
        mean_f = sum(forces) / max(len(forces), 1)
        record(
            "socket:contact_sane",
            0.0 <= mean_f < 20.0 and max(forces, default=0.0) < 100.0,
            f"mean bulb-socket force {mean_f:.2f} N, peak {max(forces, default=0.0):.2f} N",
        )
        rest_err = task_rewards._bulb_socket_pos_error(env).item()
        info(f"pos_error at physical rest = {rest_err * 100:.1f} cm (success needs < 1.5 cm)")

        # Stock-robot health: with nothing near the hand, the hand/wrist bodies must
        # carry no contact force. Sustained kN-scale readings here are the robot's own
        # links self-colliding (authored collider overlap + enabled_self_collisions);
        # they saturate the recorded ``contact_force`` channel that fragility scoring
        # reads, marking every episode "broken" regardless of policy behavior.
        sensor = env.scene.sensors["hand_contact"]
        norms = sensor.data.net_forces_w[0].norm(dim=-1)
        hand_noise = [
            f"{name}={norms[i].item():.0f}N"
            for i, name in enumerate(sensor.body_names)
            if norms[i].item() > 5.0 and ("hand" in name or "wrist" in name)
        ]
        record(
            "robot:no_self_collision_noise",
            not hand_noise,
            "; ".join(hand_noise) if hand_noise else "hand/wrist bodies carry no contact force",
        )

        # --- part 2: the pose the success predicate demands is physically attainable ---
        # bulb_seated() requires |bulb_origin - socket_origin| < 1.5 cm; teleport the
        # bulb exactly there and check physics tolerates it (rather than ejecting it).
        target = torch.tensor(TABLETOP_SOCKET_POSITION, device=env.device).unsqueeze(0)
        pose = torch.cat([target, bulb.data.root_quat_w], dim=-1)
        bulb.write_root_pose_to_sim(pose)
        bulb.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
        mon = Monitor(env)
        run_steps(env, zero, 120, mon)
        err = task_rewards._bulb_socket_pos_error(env).item()
        seated = bool(task_rewards.bulb_seated(env).item())
        speed = bulb.data.root_lin_vel_w.norm().item()
        ok, detail = mon.bounded()
        record("socket:success_pose_no_explosion", ok, detail)
        record(
            "socket:success_pose_attainable",
            (not mon.nan) and seated and speed < 0.05,
            f"after 4 s at the success pose: pos_error {err * 100:.1f} cm, "
            f"bulb_seated={seated}, speed {speed:.3f} m/s",
        )
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Scenario: bulb in hand                                                       #
# --------------------------------------------------------------------------- #
def build_hand_cfg():
    """Fixed-root G1, palm-down HOVER pose over the bench table as its default.

    Self-collisions stay OFF: the stock G1 hand-mount bodies interpenetrate by
    authoring and self-collide at ~9 kN sustained (graded by the socket scenario's
    ``robot:no_self_collision_noise`` check), which makes the hand launch anything
    it touches. These checks grade the hand<->bulb interaction, not that defect.
    """
    cfg = build_insert_cfg()
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    # outside the table footprint (slab spans x[-0.82,1.62], y[-0.48,0.28]; the
    # stock spot is INSIDE it), facing -y, palm working just inboard of the +y
    # slab edge so the forearm clears it (the edge otherwise becomes a fulcrum
    # under the wrist); z raised so the feet hang clear of the ground
    cfg.scene.robot.init_state.pos = (0.60, 0.58, 0.85)
    cfg.scene.robot.init_state.rot = (0.7071068, 0.0, 0.0, -0.7071068)  # yaw -90 deg
    # kinematic lamp parked clear of the palm's workspace
    cfg.scene.socket.init_state.pos = (-0.45, -0.30, 1.20)
    # must outlast all phases: a time_out mid-scenario re-teleports the bulb
    cfg.episode_length_s = 60.0
    if args_cli.video:
        from fiatlux_task.viz import make_video_camera_cfg

        cfg.scene.video_cam = make_video_camera_cfg()
    cfg.scene.robot.init_state.joint_pos = {
        **cfg.scene.robot.init_state.joint_pos,
        **ARM_PRESS_HOVER,
        **HAND_FLAT,
    }
    cfg.scene.bulb.spawn.activate_contact_sensors = True
    # hand bodies only: the stock right_.* sensor also nets leg/foot forces, which
    # would dominate the recorded contact_force channel the fragility scorer reads
    cfg.scene.hand_contact.prim_path = "{ENV_REGEX_NS}/Robot/(right_hand_.*|right_wrist_.*|R_.*)"
    cfg.scene.hand_bulb_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(right_hand_.*|right_wrist_yaw_link|R_.*)",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Bulb/base_link"],
        history_length=1,
    )
    return cfg


def palm_body_index(env) -> int:
    robot = env.scene["robot"]
    for name in ("right_hand_base_link", "right_wrist_yaw_link"):
        try:
            ids, _ = robot.find_bodies(name)
            if ids:
                return ids[0]
        except ValueError:
            continue
    raise RuntimeError(f"no palm body found among {robot.body_names}")


def place_bulb_under_palm(env, settle_steps: int = 30):
    """Lay the bulb on the tabletop exactly where the PRESSED palm will arrive.

    The press pose is dipped once to measure the pressed palm point (the shoulder
    pitch moves the palm along an arc, not straight down), the arm returns to
    hover, and the bulb is released lying just above its rest height so it
    settles on the (kinematic) table without depenetration kicks.
    """
    robot = env.scene["robot"]
    bulb = env.scene["bulb"]
    zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    press = targets_to_actions(env, {**ARM_PRESS_DOWN, **HAND_FLAT})
    run_steps(env, press, 45)
    pressed_palm = robot.data.body_state_w[:, palm_body_index(env), :3].clone()
    if args_cli.probe:
        print(f"  [PROBE] pressed palm point: {pressed_palm[0].tolist()}")
    run_steps(env, zero, 45)  # back to hover
    drop = pressed_palm.clone()
    drop[:, 2] = 1.07  # tabletop is ~0.99; lying bulb settles from a short fall
    quat = torch.tensor(BULB_LYING_QUAT, device=env.device).expand(env.num_envs, 4)
    bulb.write_root_pose_to_sim(torch.cat([drop, quat], dim=-1))
    bulb.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
    run_steps(env, zero, settle_steps)


def touching_bodies(env) -> str:
    """Names of robot bodies (scene-wide hand_contact sensor) with contact force."""
    sensor = env.scene.sensors["hand_contact"]
    norms = sensor.data.net_forces_w[0].norm(dim=-1)
    pairs = [
        f"{name}={norms[i].item():.2f}N"
        for i, name in enumerate(sensor.body_names)
        if norms[i].item() > 0.05
    ]
    return ", ".join(pairs) if pairs else "none"


def probe_arm_grid(env):
    """Calibration sweep: command a grid of arm poses and print, for each, the palm
    world position and which palm-local axis points most nearly world-up."""
    from isaaclab.utils.math import matrix_from_quat

    robot = env.scene["robot"]
    palm_idx = palm_body_index(env)
    axes = ("+x", "+y", "+z")
    for pitch in (-0.6, -1.0, -1.4):
        for roll in (0.0, 1.57, -1.57, 3.14):
            targets = dict(ARM_PRESS_HOVER)
            targets["right_shoulder_pitch_joint"] = pitch
            targets["right_wrist_roll_joint"] = roll
            run_steps(env, targets_to_actions(env, targets), 45)
            state = robot.data.body_state_w[0, palm_idx]
            rot = matrix_from_quat(state[3:7].unsqueeze(0))[0]  # local axes as columns
            ups = rot[2, :]  # world-z component of each local axis
            best = int(torch.argmax(ups.abs()))
            sign = "+" if ups[best] > 0 else "-"
            print(
                f"  [PROBE] pitch={pitch:+.1f} roll={roll:+.2f} -> palm at "
                f"({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}), "
                f"most-up axis {sign}{axes[best][1]} (z-comps: "
                f"x={ups[0]:+.2f} y={ups[1]:+.2f} z={ups[2]:+.2f})"
            )


def hand_bulb_force(env) -> float:
    return pairwise_force(env.scene.sensors["hand_bulb_contact"]).sum().item()


def scenario_hand(probe: bool = False):
    print("\n[verify] === scenario: bulb-in-hand ===")
    cfg = build_hand_cfg()
    env = make_env("FIATLUX-Insert-v0", cfg)
    try:
        robot = env.scene["robot"]
        attach_video(env, lookat=(0.44, -0.15, 1.10), eye=(1.05, -0.75, 1.45))
        zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

        # let the arm PD settle into the hover pose before placing the bulb
        run_steps(env, zero, 30)
        if probe:
            palm = robot.data.body_state_w[0, palm_body_index(env), :7]
            print(f"  [PROBE] palm pose (world): {palm.tolist()}")
            print(f"  [PROBE] initial contacts: {touching_bodies(env)}")
            _print_prim_bboxes("/World/envs/env_0/Table", max_prims=3)
            probe_arm_grid(env)
            run_steps(env, zero, 45)  # back to the cfg hover pose

        # --- (a) hover: palm above the bulb on the table, no contact yet -----------
        place_bulb_under_palm(env)
        bulb = env.scene["bulb"]
        mon = Monitor(env)
        run_steps(env, zero, 60, mon)
        hover_f = hand_bulb_force(env)
        bulb_z0 = bulb.data.root_pos_w[0, 2].item()
        bulb_speed = bulb.data.root_lin_vel_w.norm().item()
        ok, detail = mon.bounded()
        record("hand:hover_no_explosion", ok, detail)
        record(
            "hand:hover_bulb_at_rest",
            (not mon.nan) and 0.95 < bulb_z0 < 1.25 and bulb_speed < 0.05 and hover_f < 10.0,
            f"bulb resting at z={bulb_z0:.3f} m, speed {bulb_speed:.3f} m/s, "
            f"hand force {hover_f:.1f} N",
        )

        # --- (b) press: lower the palm onto the bulb, hold gently ------------------
        press = targets_to_actions(env, {**ARM_PRESS_DOWN, **HAND_FLAT})
        mon = Monitor(env)
        peak_f, contact_steps = 0.0, 0

        def ramp_press(i):  # hover -> press over 45 steps, then hold
            a = min(i / 45.0, 1.0)
            return a * press  # zero action == hover (the cfg default pose)

        def track(i):
            nonlocal peak_f, contact_steps
            f = hand_bulb_force(env)
            peak_f = max(peak_f, f)
            contact_steps += int(f > 0.5)
            if probe and i % 30 == 0:
                print(
                    f"  [PROBE] press t={i / STEPS_PER_SECOND:.1f}s hand-bulb force {f:.2f} N, "
                    f"bulb z {bulb.data.root_pos_w[0, 2].item():.3f}, touching: {touching_bodies(env)}"
                )

        run_steps(env, ramp_press, 195, mon, per_step=track)  # 1.5 s ramp + 5 s hold
        moved = abs(bulb.data.root_pos_w[0, 2].item() - bulb_z0)
        ok, detail = mon.bounded()
        record("hand:press_no_explosion", ok, detail)
        record(
            "hand:press_contact_sustained",
            (not mon.nan) and contact_steps > 0.3 * 195,
            f"contact in {contact_steps}/195 steps (palm on bulb on table)",
        )
        record(
            "hand:press_gentle",
            0.0 < peak_f < 50.0,
            f"peak press force {peak_f:.1f} N (fragility 50 N)",
        )
        record(
            "hand:press_bulb_stays",
            (not mon.nan) and moved < 0.08,
            f"bulb vertical move {moved * 100:.1f} cm under press",
        )

        # --- (c) release: raise the palm, bulb must stay put (not launched) --------
        mon = Monitor(env)

        def ramp_release(i):  # press -> hover over 45 steps, then hold
            a = max(1.0 - i / 45.0, 0.0)
            return a * press

        run_steps(env, ramp_release, 105, mon)
        bulb_speed = bulb.data.root_lin_vel_w.norm().item()
        bulb_z = bulb.data.root_pos_w[0, 2].item()
        ok, detail = mon.bounded()
        record("hand:release_no_explosion", ok, detail)
        record(
            "hand:release_no_launch",
            (not mon.nan) and bulb_speed < 0.3 and abs(bulb_z - bulb_z0) < 0.08,
            f"after release: bulb z {bulb_z:.3f} m (rest {bulb_z0:.3f}), speed {bulb_speed:.2f} m/s",
        )
    finally:
        if VIDEO is not None:
            print(f"  [INFO] wrote debug video {VIDEO.write()}")
        env.close()


# --------------------------------------------------------------------------- #
# Scenario: fragility / break + drop detection                                 #
# --------------------------------------------------------------------------- #
def scenario_fragility():
    """Three recorded episodes in one env (gentle / crush / drop), scored offline.

    Episodes are segmented by the recorder on the ``time_out`` truncation, so each
    phase runs one short episode; only the steps after each phase's setup are
    recorded. The scorer must flag exactly: gentle -> neither broken nor dropped,
    crush -> broken, free fall -> dropped.
    """
    print("\n[verify] === scenario: fragility / break + drop detection ===")
    out_dir = args_cli.record_bag or tempfile.mkdtemp(prefix="fiatlux_fragility_")
    info(f"recording bag under {out_dir}")

    cfg = build_hand_cfg()
    cfg.episode_length_s = 6.0  # short episodes; time_out truncates each phase
    env = make_env("FIATLUX-Insert-v0", cfg)
    try:
        recorder = TrajectoryRecorder(
            env, policy_spec="scripted:verify_interactions", seed=args_cli.seed
        )
        zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        gentle_press = targets_to_actions(env, {**ARM_PRESS_DOWN, **HAND_FLAT})
        # wedges the bulb between palm and kinematic table: sustained force spans many
        # sensor reads, unlike a ballistic impact spike that can fall between them
        crush_press = targets_to_actions(env, {**ARM_PRESS_CRUSH, **HAND_FLAT})

        def drop_setup(env):
            bulb = env.scene["bulb"]
            spot = torch.tensor([[0.35, -1.2, 1.0]], device=env.device).expand(env.num_envs, 3)
            bulb.write_root_pose_to_sim(torch.cat([spot, bulb.data.root_quat_w], dim=-1))
            bulb.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))

        def press_ramp(target):
            return lambda i: min(i / 45.0, 1.0) * target  # zero action == hover pose

        phases = [
            ("gentle", place_bulb_under_palm, press_ramp(gentle_press)),
            # crush presses into the bare tabletop; the bulb stays at its spawn spot
            ("crush", lambda env: None, press_ramp(crush_press)),
            ("drop", drop_setup, lambda i: zero),
        ]
        for name, setup, actions in phases:
            setup(env)  # un-recorded placement after the previous episode's reset
            done = False
            for i in range(300):
                act = actions(i)
                obs, reward, terminated, truncated, _ = env.step(act)
                recorder.record_step(obs, act, reward, terminated, truncated)
                if args_cli.probe and i % 15 == 0:
                    net = env.scene.sensors["hand_contact"].data.net_forces_w[0].norm(dim=-1).max().item()
                    bulb_z = env.scene["bulb"].data.root_pos_w[0, 2].item()
                    print(f"  [PROBE] {name} i={i} net force {net:.2f} N, bulb z {bulb_z:.3f}")
                if (terminated | truncated).any():
                    done = True
                    break
            if not done:
                raise RuntimeError(f"fragility phase {name!r} never finished an episode")
        recorder.write(out_dir)
    finally:
        env.close()

    episodes, meta = score.load_bag(out_dir)
    score_cfg = score.ScoreConfig()
    if "drop_min_height" in meta:
        score_cfg.drop_min_height = float(meta["drop_min_height"])
    if len(episodes) != 3:
        record("fragility:three_episodes", False, f"expected 3 recorded episodes, got {len(episodes)}")
        return
    gentle, crush, drop = (score.score_episode(ep, score_cfg) for ep in episodes)

    record(
        "fragility:gentle_not_broken",
        (not gentle["broken"]) and not gentle["dropped"],
        f"peak force {gentle['peak_contact_force']:.1f} N (threshold "
        f"{score_cfg.fragility_threshold:.0f}), broken={gentle['broken']}, dropped={gentle['dropped']}",
    )
    record(
        "fragility:crush_breaks",
        crush["broken"],
        f"peak force {crush['peak_contact_force']:.1f} N (threshold {score_cfg.fragility_threshold:.0f})",
    )
    margin_ok = crush["peak_contact_force"] > 2 * gentle["peak_contact_force"] + 1.0
    record(
        "fragility:threshold_separates_regimes",
        gentle["peak_contact_force"] < score_cfg.fragility_threshold < crush["peak_contact_force"]
        and margin_ok,
        f"gentle {gentle['peak_contact_force']:.1f} N << {score_cfg.fragility_threshold:.0f} N "
        f"<< crush {crush['peak_contact_force']:.1f} N",
    )
    record(
        "fragility:drop_detected",
        drop["dropped"],
        f"dropped={drop['dropped']} (drop_min_height {score_cfg.drop_min_height} m from meta)",
    )


# --------------------------------------------------------------------------- #
# Scenario: robot on ladder                                                    #
# --------------------------------------------------------------------------- #
def scenario_ladder(probe: bool = False):
    print("\n[verify] === scenario: robot-on-ladder ===")
    cfg = parse_env_cfg("FIATLUX-Climb-v0", device=args_cli.device, num_envs=1)
    cfg.seed = args_cli.seed
    # deterministic: no per-reset light sampling
    cfg.events.randomize_sky_intensity = None
    cfg.events.randomize_key_light = None
    # the elevated chandelier is an opt-in dressing asset and irrelevant to
    # rung contact; the scene loads without it
    cfg.scene.socket = None
    # FREE root: the robot leans onto the kinematic A-frame and the force balance
    # self-calibrates (a welded root turns every mm of overlap into a kN wedge)
    # same isolation as build_hand_cfg: the stock hand-mount self-collision wedge
    # shakes the arms and would pollute the limb contact readings
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.scene.robot.init_state.pos = LADDER_STANCE_ROOT_POS
    cfg.scene.robot.init_state.rot = LADDER_STANCE_ROOT_ROT
    cfg.scene.robot.init_state.joint_pos = {
        **cfg.scene.robot.init_state.joint_pos,
        **LADDER_STANCE_JOINTS,
    }
    # whole-robot coverage: the settled lean meets the steps through whichever
    # bodies the force balance picks (chest, forearms, knees, hands)
    cfg.scene.limb_ladder_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ladder"],
        history_length=1,
    )
    if args_cli.video:
        from fiatlux_task.viz import make_video_camera_cfg

        cfg.scene.video_cam = make_video_camera_cfg()
    env = make_env("FIATLUX-Climb-v0", cfg)
    try:
        robot = env.scene["robot"]
        attach_video(env, lookat=(1.2, 0.0, 1.0), eye=(0.4, -1.6, 1.6))
        sensor = env.scene.sensors["limb_ladder_contact"]
        zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

        if probe:
            print(f"  [PROBE] bodies: {robot.body_names}")
            print(f"  [PROBE] sensor bodies: {sensor.body_names}")
            _print_prim_bboxes("/World/envs/env_0/Ladder")

        mon = Monitor(env)
        run_steps(env, zero, 90, mon)  # settle onto the rungs
        if probe:
            f = pairwise_force(sensor)
            for name, val in zip(sensor.body_names, f.tolist()):
                print(f"  [PROBE] settled {name}: {val:.2f} N vs ladder")

        n_bodies = len(sensor.body_names)
        force_sums = torch.zeros(n_bodies, device=env.device)
        n_meas = 240

        def track(_i):
            nonlocal force_sums
            force_sums += pairwise_force(sensor)

        run_steps(env, zero, n_meas, mon, per_step=track)
        mean_forces = (force_sums / n_meas).tolist()
        total = sum(mean_forces)
        touching = ", ".join(
            f"{n}={f:.1f}N" for n, f in zip(sensor.body_names, mean_forces) if f > 1.0
        )
        ok, detail = mon.bounded()
        record("ladder:no_explosion", ok, detail)
        # sustained lean contact at body-weight scale (a kN total means bodies are
        # wedged inside the step colliders, not resting on them)
        record(
            "ladder:robot_rests_on_ladder",
            (not mon.nan) and 5.0 < total < 1000.0,
            f"mean robot-ladder contact {total:.1f} N over 8 s (via {touching or 'nothing'})",
        )
        root_z = robot.data.root_pos_w[0, 2].item()
        root_speed = robot.data.root_lin_vel_w.norm().item()
        record(
            "ladder:lean_stance_stable",
            (not mon.nan) and 0.55 < root_z < 0.9 and root_speed < 0.3,
            f"after 11 s lean: pelvis z {root_z:.2f} m, speed {root_speed:.2f} m/s",
        )
        ladder_moved = env.scene["ladder"].data.root_lin_vel_w.norm().item()
        record(
            "ladder:ladder_static",
            ladder_moved < 1e-3,
            f"ladder speed {ladder_moved:.4f} m/s (kinematic)",
        )
    finally:
        if VIDEO is not None:
            print(f"  [INFO] wrote debug video {VIDEO.write()}")
        env.close()


def _print_prim_bboxes(path: str, max_prims: int = 40):
    """Probe helper: world-frame bbox of each renderable prim under ``path``."""
    import isaacsim.core.utils.prims as prim_utils
    from pxr import Usd, UsdGeom

    root = prim_utils.get_prim_at_path(path)
    if not root.IsValid():
        print(f"  [PROBE] no prim at {path}")
        return
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    count = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if prim.IsA(UsdGeom.Gprim) and count < max_prims:
            count += 1
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lo, hi = box.GetMin(), box.GetMax()
            print(
                f"  [PROBE] {prim.GetPath()} x=[{lo[0]:.3f},{hi[0]:.3f}] "
                f"y=[{lo[1]:.3f},{hi[1]:.3f}] z=[{lo[2]:.3f},{hi[2]:.3f}]"
            )


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
SCENARIOS = {
    "socket": lambda: scenario_socket(),
    "hand": lambda: scenario_hand(probe=args_cli.probe),
    "fragility": lambda: scenario_fragility(),
    "ladder": lambda: scenario_ladder(probe=args_cli.probe),
}


def main() -> int:
    names = list(SCENARIOS) if args_cli.scenario == "all" else [args_cli.scenario]
    for name in names:
        SCENARIOS[name]()

    table = PrettyTable(["#", "Check", "Result", "Detail"])
    table.align["Check"] = "l"
    table.align["Detail"] = "l"
    n_pass = 0
    for i, (name, passed, detail) in enumerate(RESULTS, 1):
        table.add_row([i, name, "PASS" if passed else "FAIL", detail])
        n_pass += int(passed)
    print("\n" + table.get_string())
    total = len(RESULTS)
    overall = n_pass == total
    print(f"\n[verify] {n_pass}/{total} checks passed -- OVERALL: {'PASS' if overall else 'FAIL'}")
    if not overall:
        print("[verify] Failing checks:")
        for name, passed, detail in RESULTS:
            if not passed:
                print(f"   - {name}: {detail}")
    return 0 if overall else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:  # noqa: BLE001 -- print the traceback before the process exits
        import traceback

        traceback.print_exc()
    finally:
        # Same teardown caveat as verify_scene.py: Kit's shutdown terminates the
        # process with code 0, so exit with the real result before it can.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
