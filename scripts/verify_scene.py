# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load a Fiatlux ladder-family env, step it under a zero/default-hold policy, and assert the scene is solid.

This is a *verification* tool, not training. It checks, with an explicit PASS/FAIL per item:
assets present, robot sanity, gravity/settling, collision coverage, and contact/penetration.

It targets the **ladder task family** (``FIATLUX-Base-v0`` and friends): it expects the scene
entities ``robot`` / ``ladder`` / ``lamp`` / ``bulb`` and the shared ground/lights, and drives the
non-RL ``ManagerBasedEnv`` directly. It is not applicable to ``FIATLUX-Insert-v0``.

Examples
--------
    # headless verification (default base env)
    uv run python scripts/verify_scene.py --headless

    # record an orbiting MP4 of the scene to logs/verify/ (the reliable way to see it headless)
    uv run python scripts/verify_scene.py --record --hold_base --headless --num_envs 1

    # verify a specific task env
    uv run python scripts/verify_scene.py --headless --task FIATLUX-Climb-v0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the Fiatlux ladder scene loads and physics is stable.")
parser.add_argument("--task", type=str, default="FIATLUX-Base-v0", help="Gym id of the env/task to verify.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--steps", type=int, default=200, help="Number of (decimated) env steps to simulate.")
parser.add_argument(
    "--hold_base",
    action="store_true",
    help="Pin the G1 root link (fix_root_link) so it stands and holds its default pose for a clean "
    "visual sanity check. Off by default: the task envs need a FREE base, and a free humanoid cannot "
    "balance passively (it sags/tips without a policy).",
)
parser.add_argument(
    "--keep_alive",
    action="store_true",
    help="After the checks pass, keep stepping/rendering so you can inspect the scene live via your "
    "--livestream client (or a local GUI). Press Ctrl-C to exit.",
)
parser.add_argument(
    "--record",
    action="store_true",
    help="Headlessly render an orbiting MP4 of the scene to logs/verify/ (no streaming needed). "
    "Auto-enables camera rendering; pair with --hold_base --headless for a clean standing G1.",
)
parser.add_argument(
    "--record_steps", type=int, default=240, help="Frames to record (240 ~= one full 360 orbit)."
)
parser.add_argument("--record_fps", type=int, default=30, help="Frames-per-second of the output MP4.")
# AppLauncher contributes --headless, --livestream, --device, --enable_cameras, ...
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --record's orbit camera is an RTX sensor; enable camera rendering at app-launch time so the user
# does not have to pass --enable_cameras separately.
if args_cli.record:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib
import math

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import numpy as np
import torch
from prettytable import PrettyTable

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg

from isaaclab_tasks.utils import parse_env_cfg

# global prims (shared across envs) and the per-env tracked entities we expect
GLOBAL_PRIMS = {"ground": "/World/ground", "dome_light": "/World/DomeLight", "key_light": "/World/KeyLight"}
TRACKED = ["robot", "ladder", "lamp", "bulb"]

# where --record writes MP4s (repo-root logs/ dir, next to the RL runs; gitignored)
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "verify"))


def env0(path: str) -> str:
    """Resolve a regex env prim path (``.../env_.*/X``) to environment 0 (``.../env_0/X``)."""
    return path.replace(".*", "0")


def iter_prims(root_path: str):
    """Iterate a prim subtree, descending into instance proxies (imported robot USDs are
    instanceable, so plain GetChildren() would not see their meshes/colliders)."""
    root = prim_utils.get_prim_at_path(root_path)
    if not root.IsValid():
        return
    yield from Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))


def collider_audit(root_path: str) -> tuple[int, int]:
    """Return (number of geometry prims, number of those carrying a CollisionAPI) under a subtree."""
    n_geom = n_coll = 0
    for prim in iter_prims(root_path):
        if prim.IsA(UsdGeom.Gprim):  # a renderable shape (Cube/Cylinder/Sphere/Mesh/...)
            n_geom += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                n_coll += 1
    return n_geom, n_coll


def maybe_enable_collider_drawing() -> None:
    """Best-effort: draw collision shapes in the viewer so alignment can be eyeballed."""
    if args_cli.headless:
        return
    try:
        import carb

        s = carb.settings.get_settings()
        s.set_bool("/persistent/physics/visualizationDisplayColliders", True)
        s.set_bool("/persistent/physics/visualizationDisplayCollidersAsAabb", False)
        s.set_int("/persistent/physics/visualizationColliderModeAll", 1)
        print("[verify] Collider visualization enabled in the viewer.")
    except Exception as exc:  # noqa: BLE001  (purely cosmetic)
        print(f"[verify] Could not enable collider drawing: {exc}")


def make_record_camera(width: int = 1280, height: int = 720) -> CameraCfg:
    """A pinhole RGB camera added to the scene only when --record; aimed each frame in record_video."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/record_cam",
        update_period=0.0,  # refresh every render
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 1.0e4)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, -4.0, 2.2), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )


def record_video(base, cam, actions, n_steps: int, fps: int, out_path: str) -> str:
    """Step the env while orbiting the camera 360 deg around the scene; write the frames to an MP4.

    A turntable orbit makes the (policy-less, near-static) scene watchable and shows it in 3D -- the
    robot just holds its pose (use --hold_base), so there is nothing else to "perform" yet.
    """
    import imageio.v2 as imageio

    device = base.device
    n_cam = base.num_envs  # one camera per env; we orbit them all together and capture env 0
    center = torch.tensor([0.1, 0.0, 0.9], device=device)  # between lamp(-0.8), robot(0), ladder(+1)
    radius, cam_h = 4.0, 2.2
    frames: list[np.ndarray] = []
    for i in range(n_steps):
        theta = 2.0 * math.pi * i / max(n_steps, 1)
        eye = torch.tensor([0.1 + radius * math.cos(theta), radius * math.sin(theta), cam_h], device=device)
        cam.set_world_poses_from_view(eye.expand(n_cam, 3), center.expand(n_cam, 3))
        base.step(actions)
        rgb = cam.data.output["rgb"][0, ..., :3]  # (H, W, 3) uint8 on device
        frames.append(rgb.detach().cpu().numpy().astype(np.uint8))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=8)
    # also drop a poster PNG: PNGs preview inline in most editors, MP4s do not
    poster = out_path.rsplit(".", 1)[0] + "_poster.png"
    imageio.imwrite(poster, frames[len(frames) // 2])
    print(f"[verify] poster frame : {poster}")
    return out_path


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        results.append((name, passed, detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}{' -- ' + detail if detail else ''}")

    # ----- build & reset -----
    print(f"\n[verify] Loading task '{args_cli.task}' with {args_cli.num_envs} env(s)...")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.hold_base:
        env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True
        print("[verify] --hold_base: G1 root fixed -> it stands and holds the default pose (balancing not tested).")
    if args_cli.record:
        env_cfg.scene.record_cam = make_record_camera()
        print("[verify] --record: orbit RTX camera added to the scene.")
    # Resolve the env class from the Gym registry (proves the id is registered and loadable), then
    # instantiate directly: ManagerBasedEnv is non-RL and its step() returns (obs, extras), so we
    # drive it raw rather than through gym.make's 5-tuple wrappers.
    spec = gym.spec(args_cli.task)
    module_name, class_name = spec.entry_point.split(":")
    env_class = getattr(importlib.import_module(module_name), class_name)
    env = env_class(cfg=env_cfg)
    base = env
    device = base.device
    maybe_enable_collider_drawing()

    base.reset()

    robot = base.scene["robot"]
    props = {name: base.scene[name] for name in ("ladder", "lamp", "bulb")}

    # =========================== 1. ASSETS PRESENT ===========================
    print("\n[verify] (1) Assets present")
    for name, path in GLOBAL_PRIMS.items():
        record(f"asset:{name}", prim_utils.is_prim_path_valid(path), path)
    for name in TRACKED:
        p = env0(base.scene[name].cfg.prim_path)
        record(f"asset:{name}", prim_utils.is_prim_path_valid(p), p)

    # =========================== 2. ROBOT SANITY ===========================
    print("\n[verify] (2) Robot sanity")
    action_dim = base.action_manager.total_action_dim
    record("robot:bodies>0", robot.num_bodies > 0, f"{robot.num_bodies} bodies")
    record("robot:joints>0", robot.num_joints > 0, f"{robot.num_joints} joints/DOFs")
    record("robot:action_dim", action_dim > 0, f"action_dim={action_dim}")
    # default standing pose applied by the reset
    dpose_err = (robot.data.joint_pos - robot.data.default_joint_pos).abs().max().item()
    record("robot:default_pose_applied", dpose_err < 1e-3, f"max|q-q_default|={dpose_err:.2e} rad")

    # ----- record initial state, then roll out under zero (hold-default) actions -----
    actions = torch.zeros((base.num_envs, action_dim), device=device)
    init_root_z = robot.data.root_pos_w[:, 2].clone()
    init_bulb_p = props["bulb"].data.root_pos_w.clone()
    init_kin = {n: props[n].data.root_pos_w.clone() for n in ("ladder", "lamp")}
    step1_root = None
    nan_seen = False
    min_z_seen, max_z_seen, max_speed_seen = float("inf"), float("-inf"), 0.0
    render_viewer = not args_cli.headless

    print(f"\n[verify] Stepping {args_cli.steps} steps under zero/default-hold actions...")
    for i in range(args_cli.steps):
        base.step(actions)
        if render_viewer:
            base.sim.render()
        if torch.isnan(robot.data.root_pos_w).any() or torch.isnan(props["bulb"].data.root_pos_w).any():
            nan_seen = True
            break
        rz = robot.data.root_pos_w[:, 2]
        min_z_seen = min(min_z_seen, rz.min().item())
        max_z_seen = max(max_z_seen, rz.max().item())
        max_speed_seen = max(max_speed_seen, robot.data.root_lin_vel_w.norm(dim=-1).max().item())
        if i == 0:
            step1_root = robot.data.root_pos_w.clone()

    # =========================== 3. GRAVITY / SETTLING (numerical soundness) ===========================
    # NOTE: a free-base humanoid holding a fixed joint pose is an inverted pendulum -- without an
    # active balancing policy it WILL sag/tip. That is expected here; we verify the simulation is
    # *sound* (finite, bounded, stays on the floor, no explosion), not that the robot balances.
    print("\n[verify] (3) Gravity / settling (numerical soundness)")
    record("sim:no_nans", not nan_seen, "NaNs in root states" if nan_seen else "finite throughout")
    record(
        "robot:bounded(no explosion/sink)",
        (not nan_seen) and (max_z_seen < 2.0) and (min_z_seen > -0.05) and (max_speed_seen < 25.0),
        f"root z in [{min_z_seen:.2f}, {max_z_seen:.2f}] m, peak speed {max_speed_seen:.1f} m/s",
    )
    # diagnostic only (not graded): posture of the uncontrolled robot
    final_lin = robot.data.root_lin_vel_w.norm(dim=-1).max().item()
    q = robot.data.root_quat_w  # (w, x, y, z)
    tilt = math.degrees(torch.acos(torch.clamp(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1.0, 1.0)).max().item())
    print(
        f"  [INFO] uncontrolled-robot posture: final root z={robot.data.root_pos_w[:, 2].min().item():.2f} m, "
        f"max tilt={tilt:.0f} deg, final |v|={final_lin:.2f} m/s (sagging/tipping is expected without a policy)"
    )

    # props rest stably: kinematic props must not move; the dynamic bulb must settle near where it
    # started (its init pose is an estimate above the floor, so allow a small drop -- what this
    # catches is falling through the floor, being launched, or exploding).
    for n in ("ladder", "lamp"):
        moved = (props[n].data.root_pos_w - init_kin[n]).norm(dim=-1).max().item()
        record(f"{n}:static", moved < 1e-2, f"max move={moved * 1000:.2f} mm (kinematic)")
    bulb_move = (props["bulb"].data.root_pos_w[:, 2] - init_bulb_p[:, 2]).abs().max().item()
    record("bulb:settles", (bulb_move < 0.15) and not nan_seen, f"vertical move from init = {bulb_move * 1000:.1f} mm")

    # =========================== 4. COLLISION COVERAGE ===========================
    # Imported USD assets (robot and BEHAVIOR-1K props alike) may split visual meshes from
    # dedicated collision meshes, so we require colliders to EXIST under each entity rather
    # than a 1:1 visual-geom:collider match.
    print("\n[verify] (4) Collision coverage (every tracked entity must have colliders)")
    for name in TRACKED:
        root = env0(base.scene[name].cfg.prim_path)
        n_geom, n_coll = collider_audit(root)
        record(f"{name}:colliders", n_coll > 0, f"{n_coll}/{n_geom} geoms have CollisionAPI")

    # =========================== 5. CONTACT / PENETRATION ===========================
    print("\n[verify] (5) Contact / penetration")
    # robot never sank through the floor at any point in the run (negative pelvis z == fell through)
    record("robot:above_floor", min_z_seen > -0.05 and not nan_seen, f"min root z over run={min_z_seen:.3f} m")
    # bulb did not sink through the ground plane
    bulb_z = props["bulb"].data.root_pos_w[:, 2]
    record(
        "bulb:above_floor",
        bool((bulb_z > -0.02).all()) and not nan_seen,
        f"min bulb z={bulb_z.min().item():.3f} m",
    )
    # no large depenetration kick on the very first step (sign of initial interpenetration)
    first_kick = (step1_root[:, 2] - init_root_z).abs().max().item() if step1_root is not None else 0.0
    record("scene:no_initial_interpenetration", first_kick < 0.1, f"first-step root jump={first_kick * 1000:.1f} mm")

    # ----- summary -----
    table = PrettyTable(["#", "Check", "Result", "Detail"])
    table.align["Check"] = "l"
    table.align["Detail"] = "l"
    n_pass = 0
    for i, (name, passed, detail) in enumerate(results, 1):
        table.add_row([i, name, "PASS" if passed else "FAIL", detail])
        n_pass += int(passed)
    print("\n" + table.get_string())
    total = len(results)
    overall = n_pass == total
    print(f"\n[verify] {n_pass}/{total} checks passed -- OVERALL: {'PASS' if overall else 'FAIL'}")
    if not overall:
        print("[verify] Failing checks:")
        for name, passed, detail in results:
            if not passed:
                print(f"   - {name}: {detail}")

    # optional: render an orbiting MP4 of the scene -- the reliable way to "see it" on a headless box
    if args_cli.record:
        import time

        out_path = os.path.join(OUT_DIR, f"{args_cli.task}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        print(f"\n[verify] --record: orbiting camera for {args_cli.record_steps} frames -> {out_path}")
        record_video(base, base.scene["record_cam"], actions, args_cli.record_steps, args_cli.record_fps, out_path)
        print(f"[verify] wrote video: {out_path}")

    # optional: hold the scene open so it can be inspected live (livestream client or GUI).
    # NOTE: AppLauncher forces headless=True whenever livestreaming, so gate on the --keep_alive
    # flag alone -- gating on `not args_cli.headless` would skip this exactly under --livestream.
    if args_cli.keep_alive:
        print(
            "\n[verify] --keep_alive: holding the scene open and rendering. Connect your livestream "
            "client now; press Ctrl-C here to exit."
        )
        try:
            while simulation_app.is_running():
                base.step(actions)
                base.sim.render()
        except KeyboardInterrupt:
            print("\n[verify] keep_alive interrupted -- closing.")

    env.close()
    return 0 if overall else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:  # noqa: BLE001 -- print the traceback before close() hard-exits the process
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        simulation_app.close()
    sys.exit(code)
