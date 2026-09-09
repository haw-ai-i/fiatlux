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
- ``hand``     : the bulb held IN the hand: the fingers close around it, it survives
                 on grip alone, the grip stays under the fragility bound for the feature
                 being held (metal cap or glass -- they differ by ~6x), and opening the
                 hand releases it without throwing it.
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
    "--robot",
    type=str,
    default="dex3",
    choices=["dex3", "inspire"],
    help="G1 hand variant. Defaults to dex3: it is the hand the benchmark scores and the only "
    "one GR00T shipped a checkpoint for, so the grasp checks measure it.",
)
parser.add_argument(
    "--probe",
    action="store_true",
    help="Calibration mode: print body names, palm/ladder poses and live contact readings "
    "instead of grading, so fiatlux_task.poses constants can be tuned.",
)
parser.add_argument(
    "--upright-grip",
    action="store_true",
    help="hand scenario: hold the bulb world-upright on its cap (matches "
    "grasp_poses.BULB_IN_ROOT_STANDING) instead of lying across the fingers.",
)
parser.add_argument(
    "--grip",
    choices=["cradle", "cup"],
    default="cradle",
    help="hand scenario: which hand pose to hold the bulb with. 'cradle' closes the fingers "
    "(poses.HAND_CRADLE); 'cup' is the open palm the carry subtasks actually spawn "
    "(poses.HAND_CUP). Inspire only -- Dex3 has no HAND_CUP equivalent.",
)
parser.add_argument(
    "--curl",
    type=float,
    default=None,
    help="hand scenario: override the finger curl (radians) of the pose --grip selects. The "
    "thumb follows at two thirds of it. Sweeps the open-to-closed axis between HAND_CUP's 0.35 "
    "and HAND_CRADLE's 0.9 to find the most open grip that still retains the bulb.",
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
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--scenario",
            name,
            "--seed",
            str(args_cli.seed),
            "--robot",
            args_cli.robot,
        ]
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
    ARM_CRADLE,
    ARM_PRESS_CRUSH,
    ARM_PRESS_DOWN,
    ARM_PRESS_HOVER,
    BULB_LYING_QUAT,
    HAND_CRADLE_BY_VARIANT,
    HAND_FLAT_BY_VARIANT,
    LADDER_STANCE_JOINTS,
    LADDER_STANCE_LEAN_DEG,
    LADDER_STANCE_PELVIS_Z,
    LADDER_STANCE_STANDOFF,
)
from fiatlux_task.recording import TrajectoryRecorder
from fiatlux_task.robots.g1 import (
    G1_GRASP_DISTAL_BODIES,
    G1_PALM_BODY_BY_VARIANT,
    swap_robot_variant,
)
from fiatlux_task.tasks.manager_based.fiatlux_task import mdp
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import rewards as task_rewards
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp.place_terms import lean_stance_against_ladder
from prettytable import PrettyTable

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from isaaclab_tasks.utils import parse_env_cfg

# scripts/score.py is a sibling script, not a package module.
_SCORE_SPEC = importlib.util.spec_from_file_location(
    "fiatlux_score", os.path.join(os.path.dirname(os.path.abspath(__file__)), "score.py")
)
assert _SCORE_SPEC is not None and _SCORE_SPEC.loader is not None
score = importlib.util.module_from_spec(_SCORE_SPEC)
sys.modules["fiatlux_score"] = score  # dataclass resolution needs the module registered
_SCORE_SPEC.loader.exec_module(score)

# Sanity bounds shared with verify_scene.py: anything past these is an explosion,
# not a plausible interaction outcome.
MAX_SPEED = 25.0  # m/s
MAX_ROOT_Z = 2.6  # m
MIN_ROOT_Z = -0.05  # m

STEPS_PER_SECOND = 50  # sim.dt=1/200 * decimation=4 (the family control rate)
CONTACT_N = 0.05  # above sensor noise, below any force that means something
MAX_SLIP_M = 0.06  # a grasp that lets the bulb travel further than this has lost it
# Bulb geometry in its own frame, offsets along local +z from the root. The root sits OUTSIDE
# the geometry (cap bottom at 0.036, per assets.BULB_STAND_Z_OFFSET), so placements seat a
# feature and back the root out.
# Glass is too wide for the Dex3 thumb's 6.5 cm reach to close over; the cap is the grip
# feature. Glass values kept for the geometry record.
BULB_GLASS_RADIUS_M = 0.040
BULB_GLASS_CENTRE_M = 0.131
BULB_CAP_RADIUS_M = 0.021
BULB_CAP_CENTRE_M = 0.054

# What the bulb tolerates, by the feature held -- one number cannot serve both.
# Glass: thin soda-lime shell, ~50-150 N under a hard fingertip; 50 N also matches
# scripts/score.py's fragility_threshold.
# Cap: metal E26 shell, several hundred N, and it needs to be -- IEC 60968 tests the
# cap/glass joint to 3 N.m, and a realistic 0.3-0.5 N.m install torque at the cap's 13 mm
# radius costs tens of N of grip.
GLASS_CONTACT_LIMIT_N = 50.0
CAP_CONTACT_LIMIT_N = 300.0
# How far out along the fingers the bulb sits, per hand. Canonical home is grasp_poses.py (also
# consumed at runtime by nav_terms.settle_carried_payload_live); imported, not duplicated.
from fiatlux_task.grasp_poses import PALM_GRASP_FORWARD_M_BY_VARIANT  # noqa: E402

PALM_GRASP_FORWARD_M = PALM_GRASP_FORWARD_M_BY_VARIANT[args_cli.robot]

# Palm-link local axes as ``(axis_index, sign)`` -- (outward normal, along fingers, across palm).
# Canonical home is robots/g1.py (also consumed at runtime); imported here under the script's
# existing name.
from fiatlux_task.robots.g1 import G1_PALM_LOCAL_AXES as PALM_LOCAL_AXES  # noqa: E402

# Hand poses for the variant under test: the two hands share no joint names.
HAND_FLAT = HAND_FLAT_BY_VARIANT[args_cli.robot]
HAND_CRADLE = HAND_CRADLE_BY_VARIANT[args_cli.robot]
# --grip cup swaps in the open palm the carry subtasks spawn. Dex3 has no cup pose, so it keeps
# the cradle.
if args_cli.grip == "cup" and args_cli.robot == "inspire":
    from fiatlux_task.poses import HAND_CUP  # noqa: E402

    HAND_CRADLE = HAND_CUP
if args_cli.curl is not None and args_cli.robot == "inspire":
    from fiatlux_task.robots.g1 import G1_FINGER_JOINTS, G1_THUMB_JOINTS  # noqa: E402

    HAND_CRADLE = {
        **dict.fromkeys(G1_FINGER_JOINTS, args_cli.curl),
        **dict.fromkeys(G1_THUMB_JOINTS, args_cli.curl * 2.0 / 3.0),
    }


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
def strip_task_cameras(cfg) -> None:
    """Drop the task's own RTX cameras.

    Every task in the family mounts an ego camera and most mount a wrist camera. These
    scenarios measure physics, and an RTX sensor in the cfg makes ``--enable_cameras``
    mandatory. ``--video`` adds its own ``video_cam`` instead, which enables cameras itself.
    """
    cfg.scene.ego_camera = None
    cfg.scene.wrist_camera = None
    # The observation terms that read them must go too, or the manager fails to resolve
    # the sensor entity. Names differ per task; drop whichever this one declares.
    for term in ("ego_rgb", "wrist_rgb"):
        if getattr(cfg.observations.policy, term, None) is not None:
            setattr(cfg.observations.policy, term, None)


def build_insert_cfg(num_envs: int = 1):
    """Insert-task cfg stripped for deterministic, camera-free scenario runs."""
    cfg = parse_env_cfg("FIATLUX-Insert-v0", device=args_cli.device, num_envs=num_envs)
    cfg.seed = args_cli.seed
    strip_task_cameras(cfg)
    if args_cli.robot != "inspire":
        swap_robot_variant(cfg, args_cli.robot)
    # Deterministic resets: zero every randomization range, keep the reset terms so
    # each reset returns entities exactly to their (scenario-crafted) init_state.
    cfg.events.randomize_light = None
    cfg.events.randomize_key_light = None
    cfg.events.randomize_material_tint = None
    # The per-asset zeroed resets above already restore the scenario-crafted init_state;
    # reset_scene_to_default would additionally write root state to the fixed-base rigs, which
    # shifts the press arc.
    cfg.events.reset_all = None
    # Grip friction is a startup randomization; pinned, so contact measurements are stable.
    cfg.events.randomize_hand_material = mdp.hand_grip_material_event(randomize=False)
    cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_socket.params["pose_range"] = {}
    cfg.events.reset_bulb.params["pose_range"] = {}
    # only time_out may end an episode: scenarios hold poses that trip the task
    # terminations every step (the scorer detects drops from bulb height anyway)
    cfg.terminations.success = None
    cfg.terminations.bulb_dropped = None
    cfg.terminations.fell_below = None
    cfg.terminations.fell_over = None
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
def targets_to_actions(env, targets: dict[str, float]) -> torch.Tensor:
    """Invert each action term's own ``target = offset + scale * action`` for named joints;
    every other slot gets zero action (hold default)."""
    act = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    base = 0
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        joint_names = getattr(term, "_joint_names", None)
        if joint_names is not None:
            scale, offset = term._scale, term._offset  # noqa: SLF001
            for i, jn in enumerate(joint_names):
                if jn not in targets:
                    continue
                s = scale if isinstance(scale, float) else scale[:, i]
                o = offset if isinstance(offset, float) else offset[:, i]
                act[:, base + i] = (targets[jn] - o) / s
        base += term.action_dim
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
        self.max_bulb_speed = 0.0
        self.min_z = float("inf")
        self.max_z = float("-inf")

    def step(self):
        robot = self.env.scene["robot"]
        bulb = self.env.scene["fresh_bulb"]
        if torch.isnan(robot.data.root_pos_w).any() or torch.isnan(bulb.data.root_pos_w).any():
            self.nan = True
            return
        bulb_speed = bulb.data.root_lin_vel_w.norm(dim=-1).max().item()
        self.max_bulb_speed = max(self.max_bulb_speed, bulb_speed)
        self.max_speed = max(self.max_speed, robot.data.root_lin_vel_w.norm(dim=-1).max().item(), bulb_speed)
        z = robot.data.root_pos_w[:, 2]
        self.min_z = min(self.min_z, z.min().item())
        self.max_z = max(self.max_z, z.max().item())

    def bounded(self) -> tuple[bool, str]:
        ok = not self.nan and self.max_speed < MAX_SPEED and self.max_z < MAX_ROOT_Z and self.min_z > MIN_ROOT_Z
        detail = (
            "NaN in states"
            if self.nan
            else f"peak speed {self.max_speed:.1f} m/s, root z in [{self.min_z:.2f}, {self.max_z:.2f}] m"
        )
        return ok, detail


def run_steps(env, actions, n, monitor=None, per_step=None, pre_step=None):
    """Step ``n`` env steps under fixed or callable actions.

    ``pre_step`` runs BEFORE each physics step, ``per_step`` after it. Asset state the step
    must honour (pinning a body) belongs in ``pre_step``.
    """
    for i in range(n):
        if pre_step is not None:
            pre_step(i)
        act = actions(i) if callable(actions) else actions
        env_step(env, act)
        if monitor is not None:
            monitor.step()
            if monitor.nan:
                break
        if per_step is not None:
            per_step(i)


def teleport(asset, pos, quat=None) -> None:
    """Move an asset to ``pos`` (and optionally ``quat``) and kill its velocity."""
    n, device = pos.shape[0] if pos.dim() > 1 else 1, pos.device
    pos = pos.reshape(-1, 3).expand(n, 3)
    quat = asset.data.root_quat_w if quat is None else torch.as_tensor(quat, device=device).reshape(-1, 4).expand(n, 4)
    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
    asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=device))


def ramp(target: torch.Tensor, steps: int, out: bool = False):
    """Per-frame action that eases into (or out of) ``target`` over ``steps``, then holds."""
    if out:
        return lambda i: max(1.0 - i / steps, 0.0) * target
    return lambda i: min(i / steps, 1.0) * target


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
    # robot:no_self_collision_noise needs the robot touching nothing but itself. Its root is
    # fixed (a free base has no policy here and cannot balance), and the bench is dropped --
    # the arms hang into its under-shelf crates, and nothing in this scenario rests on it: the
    # socket is kinematic and the bulb is teleported to its seated poses.
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    cfg.scene.table = None
    cfg.scene.fresh_bulb.init_state.pos = TABLETOP_SEATED_BULB_POSITION
    cfg.scene.fresh_bulb.spawn.activate_contact_sensors = True
    # The Omniverse bulb/socket carry their rigid body on the spawned prim itself; the
    # <entity>/base_link nesting was the BEHAVIOR-1K pair's layout.
    cfg.scene.bulb_socket_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Bulb",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Socket"],
        history_length=1,
    )
    env = make_env("FIATLUX-Insert-v0", cfg)
    try:
        bulb = env.scene["fresh_bulb"]
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

        # Stock-robot health: touching nothing, no body may carry contact force. A reading is
        # the robot's own geometry (collider overlap + enabled_self_collisions), and it lands
        # in the unfiltered net-force channel feeding the contact observation and the recorded
        # fragility force -- so it scores policies as "broken". Every body the sensor reports
        # is inspected; its prim_path already scopes it to the right hand and wrist.
        sensor = env.scene.sensors["hand_contact"]
        norms = sensor.data.net_forces_w[0].norm(dim=-1)
        hand_noise = [
            f"{name}={norms[i].item():.0f}N" for i, name in enumerate(sensor.body_names) if norms[i].item() > 5.0
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
        teleport(bulb, target)
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
            f"after 4 s at the success pose: pos_error {err * 100:.1f} cm, bulb_seated={seated}, speed {speed:.3f} m/s",
        )
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Scenario: bulb in hand                                                       #
# --------------------------------------------------------------------------- #
def build_hand_cfg(arm=ARM_PRESS_HOVER, hand=HAND_FLAT):
    """Fixed-root G1 over the bench table, holding ``arm``/``hand`` as its DEFAULT pose.

    Zero action means exactly this pose; the scenarios ramp their actions from it.
    """
    cfg = build_insert_cfg()
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
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
    cfg.scene.robot.init_state.joint_pos = {**cfg.scene.robot.init_state.joint_pos, **arm, **hand}
    cfg.scene.fresh_bulb.spawn.activate_contact_sensors = True
    cfg.scene.hand_bulb_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(right_hand_.*|right_wrist_yaw_link|R_.*)",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Bulb"],
        history_length=1,
    )
    return cfg


def palm_body_index(env) -> int:
    """Index of the right palm body for the variant under test.

    Must be the palm itself: callers position the bulb under the pressing palm, and the wrist
    sits ~4 cm away, which turns a flat press into an off-centre edge contact. Raises rather
    than substituting a neighbouring body.
    """
    robot = env.scene["robot"]
    name = G1_PALM_BODY_BY_VARIANT[args_cli.robot]
    ids, _ = robot.find_bodies(name)
    if not ids:
        raise RuntimeError(f"palm body {name!r} not found among {robot.body_names}")
    return ids[0]


def place_bulb_under_palm(env, arm=None, settle_steps: int = 30):
    """Lay the bulb on the tabletop exactly where the PRESSED palm will arrive.

    ``arm`` is the press pose the bulb will be met with, so each phase places it under its own
    palm arc; defaults to :data:`ARM_PRESS_DOWN`.

    The press pose is dipped once to measure the pressed palm point (the shoulder
    pitch moves the palm along an arc, not straight down), the arm returns to
    hover, and the bulb is released lying just above its rest height so it
    settles on the (kinematic) table without depenetration kicks.
    """
    robot = env.scene["robot"]
    bulb = env.scene["fresh_bulb"]
    zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    press = targets_to_actions(env, {**(arm or ARM_PRESS_DOWN), **HAND_FLAT})
    run_steps(env, press, 45)
    pressed_palm = robot.data.body_state_w[:, palm_body_index(env), :3].clone()
    if args_cli.probe:
        print(f"  [PROBE] pressed palm point: {pressed_palm[0].tolist()}")
    run_steps(env, zero, 45)  # back to hover
    from fiatlux_task.assets import BULB_LIE_Z_OFFSET
    from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import TABLETOP_SURFACE_Z

    drop = pressed_palm.clone()
    # DERIVED from the bulb's own lying rest height: 1 cm of fall, so it settles without a
    # depenetration kick and without being released inside the hovering palm.
    drop[:, 2] = TABLETOP_SURFACE_Z + BULB_LIE_Z_OFFSET + 0.01
    teleport(bulb, drop, BULB_LYING_QUAT)
    run_steps(env, zero, settle_steps)


def touching_bodies(env) -> str:
    """Names of robot bodies (scene-wide hand_contact sensor) with contact force."""
    sensor = env.scene.sensors["hand_contact"]
    norms = sensor.data.net_forces_w[0].norm(dim=-1)
    pairs = [f"{name}={norms[i].item():.2f}N" for i, name in enumerate(sensor.body_names) if norms[i].item() > 0.05]
    return ", ".join(pairs) if pairs else "none"


def probe_arm_grid(env, base=ARM_PRESS_HOVER):
    """Calibration sweep around ``base``: print each pose's palm world position and which
    palm-local axis points most nearly world-up."""
    robot = env.scene["robot"]
    palm_idx = palm_body_index(env)
    axes = ("+x", "+y", "+z")
    for pitch in (-0.6, -1.0, -1.4):
        for roll in (0.0, 1.57, -1.57, 3.14):
            targets = dict(base)
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


def palm_frame(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(palm_origin, outward_normal, along_fingers, across_palm)`` in world coordinates.

    The axes come from :data:`PALM_LOCAL_AXES`, measured per variant rather than assumed --
    the two hands carry their palm face on different local axes.
    """
    robot = env.scene["robot"]
    idx = robot.find_bodies(G1_PALM_BODY_BY_VARIANT[args_cli.robot])[0][0]
    state = robot.data.body_state_w[0, idx]
    rot = matrix_from_quat(state[3:7].unsqueeze(0))[0]
    normal, fingers, across = (rot[:, axis] * sign for axis, sign in PALM_LOCAL_AXES[args_cli.robot])
    return state[:3], normal, fingers, across


def palm_grasp_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Bulb lying ACROSS the fingers, gripped at the CAP, long axis spanning the palm.

    Every offset is bounded by geometry ``--probe`` prints: the digits in this palm frame, and
    the bulb's collider bounds against its root.

    The cap, not the glass: the glass is 8 cm across against 6.5 cm of Dex3 thumb reach off the
    palm plane, so the thumb cannot close over it. The 4.2 cm cap fits the thumb's span.
    ``PALM_GRASP_FORWARD_M`` keeps the grip within the thumb's 2.3 cm reach rather than out at
    the fingertips (12.4 cm). The root is then backed out along the axis, since it lies outside
    the bulb's own geometry.
    """
    origin, normal, fingers, across = palm_frame(env)
    seat = origin + normal * BULB_CAP_RADIUS_M + fingers * PALM_GRASP_FORWARD_M
    pos = seat - across * BULB_CAP_CENTRE_M
    if args_cli.upright_grip:
        # World-upright on the cap (BULB_UPRIGHT_QUAT), the orientation
        # grasp_poses.BULB_IN_ROOT_STANDING/BULB_IN_ROOT_ON_LADDER actually need -- not the
        # lying-across-the-fingers pinch this function otherwise calibrates.
        return pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=pos.device)
    # columns [normal, across x normal, across]: right-handed, since col0 x col1 == col2.
    basis = torch.stack([normal, torch.linalg.cross(across, normal), across], dim=1)
    return pos, quat_from_matrix(basis.unsqueeze(0))[0]


def grasp_point(env) -> torch.Tensor:
    """Where a held bulb's centre should sit (position half of :func:`palm_grasp_pose`)."""
    return palm_grasp_pose(env)[0]


def close_hand_on_bulb(env, closed: torch.Tensor, monitor=None, ramp_steps: int = 60) -> None:
    """Pin the bulb in the hand while the fingers wrap it, then release the pin.

    The hand closes around a bulb already in it; an open hand cannot catch one. Nothing here
    is graded: a pinned bulb is infinitely stiff, so the closing reaction is an artifact of
    the pin. Grade the hold, after this returns.
    """
    bulb = env.scene["fresh_bulb"]
    pos, quat = palm_grasp_pose(env)
    pos, quat = pos.clone(), quat.clone()
    run_steps(
        env,
        ramp(closed, ramp_steps),
        ramp_steps + 15,
        monitor,
        pre_step=lambda _i: teleport(bulb, pos, quat),
    )


def hand_bulb_force(env) -> float:
    return pairwise_force(env.scene.sensors["hand_bulb_contact"]).sum().item()


def scenario_hand(probe: bool = False):
    """Grasp the bulb IN the hand: close, hold on grip alone, release.

    Holding it in the hand keeps the bench out of the test -- no surface height, no drop
    height, no dependence on which way the bulb settles -- and makes the graded quantity the
    one the benchmark scores: grip force against retention.
    """
    print("\n[verify] === scenario: bulb-in-hand ===")
    cfg = build_hand_cfg(arm=ARM_CRADLE)
    env = make_env("FIATLUX-Insert-v0", cfg)
    try:
        bulb = env.scene["fresh_bulb"]
        attach_video(env, lookat=(0.44, 0.18, 1.12), eye=(1.05, -0.35, 1.45))
        zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

        run_steps(env, zero, 60)  # let the arm PD settle into the open palm-up pose
        if probe:
            robot = env.scene["robot"]
            palm = robot.data.body_state_w[0, palm_body_index(env), :7]
            print(f"  [PROBE] palm pose (world): {palm.tolist()}")
            # The palm frame is what PALM_LOCAL_AXES declares; print the axes and the digit
            # positions together so a wrong axis is visible as a number, not only in a render.
            origin, normal, fingers, across = palm_frame(env)
            print(f"  [PROBE] palm origin {[round(v, 4) for v in origin.tolist()]}")
            for label, vec in (("normal", normal), ("along_fingers", fingers), ("across_palm", across)):
                print(f"  [PROBE]   {label:14s} {[round(v, 3) for v in vec.tolist()]}")
            for body in G1_GRASP_DISTAL_BODIES[args_cli.robot]:
                pos = robot.data.body_pos_w[0, robot.find_bodies(body)[0][0]]
                rel = pos - origin
                print(
                    f"  [PROBE]   {body:26s} {[round(v, 4) for v in pos.tolist()]}  "
                    f"palm-frame (n,f,a)=({torch.dot(rel, normal):+.4f}, "
                    f"{torch.dot(rel, fingers):+.4f}, {torch.dot(rel, across):+.4f})"
                )
            print(f"  [PROBE] grasp point: {[round(v, 4) for v in grasp_point(env).tolist()]}")
            # The bulb's ROOT is not inside its geometry (assets.BULB_STAND_Z_OFFSET), so the
            # body lands offset from wherever the root is written. Print the offset.
            bulb_root = env.scene["fresh_bulb"].data.root_pos_w[0]
            print(f"  [PROBE] bulb root: {[round(v, 4) for v in bulb_root.tolist()]}")
            _print_prim_bboxes("/World/envs/env_0/Bulb")
            # How far the palm's own collider stands off its link origin: the seat has to clear
            # this or the bulb is placed inside the palm and the grip force is depenetration.
            _print_prim_bboxes(f"/World/envs/env_0/Robot/{G1_PALM_BODY_BY_VARIANT[args_cli.robot]}")
            print(f"  [PROBE] initial contacts: {touching_bodies(env)}")
            probe_arm_grid(env, ARM_CRADLE)  # palm-pose sweep for recalibrating ARM_CRADLE
            run_steps(env, zero, 60)  # back to the open palm-up cradle

        # --- (a) close the fingers around a bulb placed in the cup ------------------
        hold = targets_to_actions(env, {**ARM_CRADLE, **HAND_CRADLE})
        mon = Monitor(env)
        close_hand_on_bulb(env, hold, mon)
        ok, detail = mon.bounded()
        record("hand:grasp_no_explosion", ok, detail)
        if probe:
            robot0 = env.scene["robot"]
            for jname, target in {**ARM_CRADLE, **HAND_CRADLE}.items():
                jidx = robot0.find_joints(jname)[0][0]
                achieved = robot0.data.joint_pos[0, jidx].item()
                print(f"  [PROBE] joint {jname:28s} target={target:+.3f} achieved={achieved:+.3f}")
            # How far each fingertip actually lands, CLOSED, from the pinned grasp point --
            # in the same palm frame the open-pose printout above used, so the two are
            # directly comparable. A large gap here means the grip is missing the bulb
            # geometrically; no amount of curl retuning fixes that.
            robot = env.scene["robot"]
            origin, normal, fingers, across = palm_frame(env)
            gp = grasp_point(env)
            gp_rel = gp - origin
            print(
                f"  [PROBE] CLOSED grasp point palm-frame (n,f,a)=({torch.dot(gp_rel, normal):+.4f}, "
                f"{torch.dot(gp_rel, fingers):+.4f}, {torch.dot(gp_rel, across):+.4f})"
            )
            for body in G1_GRASP_DISTAL_BODIES[args_cli.robot]:
                pos = robot.data.body_pos_w[0, robot.find_bodies(body)[0][0]]
                rel = pos - origin
                to_gp = (pos - gp).norm().item()
                print(
                    f"  [PROBE] CLOSED {body:26s} {[round(v, 4) for v in pos.tolist()]}  "
                    f"palm-frame (n,f,a)=({torch.dot(rel, normal):+.4f}, {torch.dot(rel, fingers):+.4f}, "
                    f"{torch.dot(rel, across):+.4f})  dist-to-grasp-point={to_gp * 100:.1f}cm"
                )

        # --- (b) hold on grip alone: the pin is gone, only the fingers hold it ------
        mon = Monitor(env)
        peak_f, contact_steps, n_hold = 0.0, 0, 150

        def track(i):
            nonlocal peak_f, contact_steps
            f = hand_bulb_force(env)
            peak_f = max(peak_f, f)
            contact_steps += int(f > CONTACT_N)
            if probe and i % 30 == 0:
                slip = (bulb.data.root_pos_w[0] - grasp_point(env)).norm().item()
                print(f"  [PROBE] hold t={i / STEPS_PER_SECOND:.1f}s force {f:.2f} N, slip {slip * 100:.1f} cm")

        run_steps(env, hold, n_hold, mon, per_step=track)
        if probe:
            # fiatlux_task.grasp_poses' calibration workflow: the bulb's pose in the robot's
            # OWN root frame, after the hold has settled -- this is what BULB_IN_ROOT_STANDING/
            # BULB_IN_ROOT_ON_LADDER should hold, root-frame offsets are pose-invariant so this
            # transfers directly onto a free (randomized) root.
            from isaaclab.utils.math import quat_apply_inverse
            from isaaclab.utils.math import quat_mul as _quat_mul_isl

            root = env.scene["robot"]
            root_pos = root.data.root_pos_w[0]
            root_quat = root.data.root_quat_w[0]
            bulb_pos_in_root = quat_apply_inverse(
                root_quat.unsqueeze(0), (bulb.data.root_pos_w[0] - root_pos).unsqueeze(0)
            )[0]
            bulb_quat_in_root = _quat_mul_isl(
                torch.stack([root_quat[0], -root_quat[1], -root_quat[2], -root_quat[3]]).unsqueeze(0),
                bulb.data.root_quat_w[0].unsqueeze(0),
            )[0]
            print(
                f"  [PROBE] SETTLED bulb pos-in-root = ({bulb_pos_in_root[0]:.4f}, "
                f"{bulb_pos_in_root[1]:.4f}, {bulb_pos_in_root[2]:.4f})  "
                f"quat-in-root(w,x,y,z) = ({bulb_quat_in_root[0]:.6f}, {bulb_quat_in_root[1]:.6f}, "
                f"{bulb_quat_in_root[2]:.6f}, {bulb_quat_in_root[3]:.6f})"
            )
            print(f"  [PROBE] DEBUG root_pos_w={root_pos.tolist()} root_quat_w={root_quat.tolist()}")
            print(f"  [PROBE] DEBUG bulb_pos_w={bulb.data.root_pos_w[0].tolist()}")
            from isaaclab.utils.math import quat_apply as _quat_apply_isl

            roundtrip = root_pos + _quat_apply_isl(root_quat.unsqueeze(0), bulb_pos_in_root.unsqueeze(0))[0]
            print(f"  [PROBE] DEBUG roundtrip world pos = {roundtrip.tolist()}  (should equal bulb_pos_w above)")
        slip = (bulb.data.root_pos_w[0] - grasp_point(env)).norm().item()
        held_f = hand_bulb_force(env)
        ok, detail = mon.bounded()
        record("hand:hold_no_explosion", ok, detail)
        record(
            "hand:hold_retains_bulb",
            (not mon.nan) and held_f > CONTACT_N and slip < MAX_SLIP_M,
            f"after {n_hold / STEPS_PER_SECOND:.1f} s on grip alone: {slip * 100:.1f} cm of slip, "
            f"contact {held_f:.2f} N",
        )
        record(
            "hand:hold_contact_sustained",
            contact_steps > 0.8 * n_hold,
            f"contact in {contact_steps}/{n_hold} steps of the hold",
        )
        # The grip must not itself break the bulb. Which bound applies depends on WHERE the
        # hand holds it, and palm_grasp_pose grips the cap.
        record(
            "hand:hold_gentle",
            0.0 < peak_f < CAP_CONTACT_LIMIT_N,
            f"peak grip force {peak_f:.1f} N on the cap (limit {CAP_CONTACT_LIMIT_N:.0f} N; "
            f"the glass would be {GLASS_CONTACT_LIMIT_N:.0f} N)",
        )

        # --- (c) open the hand: the bulb leaves without being thrown ---------------
        mon = Monitor(env)
        run_steps(env, ramp(hold, 45, out=True), n_hold, mon)
        end_speed = bulb.data.root_lin_vel_w.norm().item()
        ok, detail = mon.bounded()
        record("hand:release_no_explosion", ok, detail)
        # ~1.5 m/s is free fall to the bench; the fingers must not fling it.
        record(
            "hand:release_no_launch",
            (not mon.nan) and mon.max_bulb_speed < 2.5 and end_speed < 0.2,
            f"after release: peak speed {mon.max_bulb_speed:.2f} m/s, at rest {end_speed:.3f} m/s",
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
            env, policy_spec="scripted:verify_interactions", seed=args_cli.seed, out_dir=out_dir
        )
        zero = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        gentle_press = targets_to_actions(env, {**ARM_PRESS_DOWN, **HAND_FLAT})
        # wedges the bulb between palm and kinematic table: sustained force spans many
        # sensor reads, unlike a ballistic impact spike that can fall between them
        crush_press = targets_to_actions(env, {**ARM_PRESS_CRUSH, **HAND_FLAT})

        def drop_setup(env):
            bulb = env.scene["fresh_bulb"]
            spot = torch.tensor([[0.35, -1.2, 1.0]], device=env.device).expand(env.num_envs, 3)
            teleport(bulb, spot)

        def press_ramp(target):
            return ramp(target, 45)  # zero action == hover pose

        phases = [
            ("gentle", place_bulb_under_palm, press_ramp(gentle_press)),
            # The bulb goes under the CRUSH arc, not the gentle one: the contact channel counts
            # only force on the bulb, so a press into the bare tabletop scores nothing.
            ("crush", lambda env: place_bulb_under_palm(env, arm=ARM_PRESS_CRUSH), press_ramp(crush_press)),
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
                    bulb_z = env.scene["fresh_bulb"].data.root_pos_w[0, 2].item()
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
        gentle["peak_contact_force"] < score_cfg.fragility_threshold < crush["peak_contact_force"] and margin_ok,
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
    strip_task_cameras(cfg)
    # deterministic: no per-reset light sampling, no start-pose randomization (the
    # RL env's reset events would perturb the calibrated stance below)
    cfg.events.randomize_sky_intensity = None
    cfg.events.randomize_key_light = None
    cfg.events.randomize_material_tint = None
    cfg.events.reset_robot_joints = None
    cfg.events.reset_robot_root = None
    # the elevated chandelier is an opt-in dressing asset and irrelevant to
    # rung contact; the scene loads without it
    cfg.scene.socket = None
    # FREE root: a welded root turns every mm of overlap into a kN wedge. Resolved against the
    # ladder's own pose so the lean meets the steps rather than the brace side behind them.
    cfg.scene.robot.init_state.pos, cfg.scene.robot.init_state.rot = lean_stance_against_ladder(
        cfg.scene.ladder.init_state.pos,
        cfg.scene.ladder.init_state.rot,
        LADDER_STANCE_STANDOFF,
        LADDER_STANCE_PELVIS_Z,
        LADDER_STANCE_LEAN_DEG,
    )
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
        n_tail = 60  # the final quarter of the measurement window (~2 s)
        root_track: list[list[float]] = []

        def track(_i):
            nonlocal force_sums
            force_sums += pairwise_force(sensor)
            root_track.append(robot.data.root_pos_w[0].tolist())

        run_steps(env, zero, n_meas, mon, per_step=track)
        mean_forces = (force_sums / n_meas).tolist()
        total = sum(mean_forces)
        touching = ", ".join(f"{n}={f:.1f}N" for n, f in zip(sensor.body_names, mean_forces) if f > 1.0)
        ok, detail = mon.bounded()
        record("ladder:no_explosion", ok, detail)
        # sustained lean contact at body-weight scale (a kN total means bodies are
        # wedged inside the step colliders, not resting on them)
        record(
            "ladder:robot_rests_on_ladder",
            (not mon.nan) and 5.0 < total < 1000.0,
            f"mean robot-ladder contact {total:.1f} N over 8 s (via {touching or 'nothing'})",
        )
        # CONVERGENCE, not a snapshot: a passive robot slumped on a ladder settles on its own
        # schedule, so instantaneous speed at a fixed mark reads as noise. Whether the pile is
        # held UP belongs to ladder:robot_rests_on_ladder, which asserts it via contact force.
        tail = root_track[-n_tail:] if len(root_track) >= n_tail else root_track
        p2p = [max(p[ax] for p in tail) - min(p[ax] for p in tail) for ax in range(3)] if tail else [9.9] * 3
        record(
            "ladder:lean_stance_settles",
            (not mon.nan) and bool(tail) and max(p2p) < 0.02,
            f"pelvis peak-to-peak over the final {n_tail} steps: "
            f"x {p2p[0] * 1e3:.1f} mm, y {p2p[1] * 1e3:.1f} mm, z {p2p[2] * 1e3:.1f} mm",
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
