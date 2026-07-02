# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load any Fiatlux env, step it under a zero/default-hold policy, and assert the scene is solid.

This is a *verification* tool, not training. It checks, with an explicit PASS/FAIL per item:
assets present, preset initial state, robot sanity, gravity/settling, collision coverage, and
contact/penetration.

It covers the whole task family: the entity list is derived from the task's scene cfg, so
presets that drop entities (tabletop has no ladder, workshop has no table) verify with the
same tool. RL members work too -- their step returns are ignored and mid-run auto-resets do
not disturb the checks. ``FIATLUX-Insert-v0`` carries a wrist-camera sensor, so verifying it
needs ``--enable_cameras``.

Examples
--------
    # headless verification (default base env)
    uv run python scripts/verify_scene.py --headless

    # record an orbiting MP4 of the scene to logs/verify/ (the reliable way to see it headless)
    uv run python scripts/verify_scene.py --record --hold_base --headless --num_envs 1

    # verify a specific task env
    uv run python scripts/verify_scene.py --headless --task FIATLUX-Climb-v0

    # the Insert task needs camera rendering for its wrist-camera sensor
    uv run python scripts/verify_scene.py --headless --enable_cameras --task FIATLUX-Insert-v0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the Fiatlux ladder scene loads and physics is stable.")
parser.add_argument(
    "--task",
    type=str,
    default="FIATLUX-Base-v0",
    help="Gym id of the env/task to verify (any FIATLUX id; Insert also needs --enable_cameras).",
)
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
import torch
from fiatlux_task.viz import make_video_camera_cfg, record_orbit
from prettytable import PrettyTable

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.assets import ArticulationCfg, RigidObjectCfg

from isaaclab_tasks.utils import parse_env_cfg

# global prims (shared across envs) and the per-env tracked entities we expect
# Candidate scene entities; each is checked only when it exists (and is not None) on the
# task's scene cfg, so this one verifier covers every family preset: the tabletop preset
# has no ladder, the workshop presets have no table, dressing cfgs may drop the fixture.
GLOBAL_CANDIDATES = ["ground", "dome_light", "key_light", "room"]
TRACKED_CANDIDATES = ["robot", "ladder", "lamp", "socket", "bulb", "table", "bin"]

# Presence expectations per scene preset (env cfg attr `scene_preset`): entities that MUST
# be present / MUST be absent. Layout *positions* are covered generically by the
# init-state drift check, which compares spawned poses to the cfg's own init_state.
PRESET_PRESENCE = {
    "tabletop": ({"table"}, {"ladder"}),
    "workshop": ({"ladder"}, {"table"}),
    "carry": ({"ladder"}, {"table"}),
    "climb": ({"ladder"}, {"table"}),
    "descend": ({"ladder"}, {"table"}),
    "remove": ({"ladder", "bin"}, {"table"}),
    "install": ({"ladder", "bin"}, {"table"}),
}

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
    """Return (renderable Gprims, prims of ANY type carrying a CollisionAPI) under a subtree.

    Collision is counted on every prim type, not just Gprims: robot USDs (e.g. the Unitree
    G1) author it on per-link ``collisions`` Xforms (PhysicsCollisionAPI +
    PhysicsMeshCollisionAPI on the Xform, visual meshes carry nothing).
    """
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


# Fallback orbit framing when a task cfg carries no orbit hints. Task cfgs override via
# `orbit_center` / `orbit_radius` / `orbit_height` attrs to frame their own preset layout.
# Radius must keep the camera inside the Simple Room (walls ~4.5 m out). Capture machinery
# lives in fiatlux_task.viz.
ORBIT_CENTER = (0.1, 0.0, 1.3)
ORBIT_RADIUS, ORBIT_HEIGHT = 4.0, 2.8


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
        env_cfg.scene.video_cam = make_video_camera_cfg()
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

    # Resolve what this task's scene actually contains (family presets drop entities).
    # Prim paths come from the *cfg* -- AssetBaseCfg entities (table, lights) are bare
    # XFormPrims at runtime with no `.cfg` attribute.
    global_prims = {
        n: getattr(env_cfg.scene, n).prim_path
        for n in GLOBAL_CANDIDATES
        if getattr(env_cfg.scene, n, None) is not None
    }
    tracked = [n for n in TRACKED_CANDIDATES if getattr(env_cfg.scene, n, None) is not None]
    # kinematic set-dressing props (RigidObjects only; the dynamic bulb and the robot are
    # checked separately, AssetBase entities have no physics state to assert on)
    kinematic_props = [
        n for n in tracked
        if n not in ("robot", "bulb") and isinstance(getattr(env_cfg.scene, n), RigidObjectCfg)
    ]
    robot = base.scene["robot"]

    # =========================== 1. ASSETS PRESENT ===========================
    print("\n[verify] (1) Assets present")
    for name, path in global_prims.items():
        record(f"asset:{name}", prim_utils.is_prim_path_valid(path), path)
    for name in tracked:
        p = env0(getattr(env_cfg.scene, name).prim_path)
        record(f"asset:{name}", prim_utils.is_prim_path_valid(p), p)

    # =========================== 1b. INITIAL STATE (preset layout) ===========================
    # The family scene is preset-parametrized. Two layers of checks:
    # (a) presence: the entities the task's preset must spawn / must drop;
    # (b) drift: each rigid entity's spawned root position (env 0, in env-local frame)
    #     matches the cfg's own init_state, so cfg-vs-stage drift is caught for ANY layout.
    print("\n[verify] (1b) Initial state (preset layout)")
    preset = getattr(env_cfg, "scene_preset", None)
    if preset in PRESET_PRESENCE:
        need, forbid = PRESET_PRESENCE[preset]
        for n in sorted(need):
            record(f"preset:{preset}:has_{n}", n in tracked, f"{n} must spawn in this preset")
        for n in sorted(forbid):
            record(f"preset:{preset}:no_{n}", n not in tracked, f"{n} must not spawn in this preset")
    else:
        print(f"  [INFO] no presence expectations for preset {preset!r}")
    env0_origin = base.scene.env_origins[0]
    for name in tracked:
        entity_cfg = getattr(env_cfg.scene, name)
        if not isinstance(entity_cfg, (RigidObjectCfg, ArticulationCfg)):
            continue  # AssetBase entities (table, bin) have no runtime physics state
        spawned = base.scene[name].data.root_pos_w[0] - env0_origin
        expect = torch.tensor(entity_cfg.init_state.pos, device=device)
        drift = (spawned - expect).norm().item()
        # 0.10 m tolerance covers deliberate reset randomization (Insert: +/-0.05) and settling
        record(
            f"init:{name}_at_cfg_pose",
            drift < 0.10,
            f"|spawned - cfg init_state| = {drift * 100:.1f} cm",
        )

    # =========================== 2. ROBOT SANITY ===========================
    print("\n[verify] (2) Robot sanity")
    action_dim = base.action_manager.total_action_dim
    record("robot:bodies>0", robot.num_bodies > 0, f"{robot.num_bodies} bodies")
    record("robot:joints>0", robot.num_joints > 0, f"{robot.num_joints} joints/DOFs")
    record("robot:action_dim", action_dim > 0, f"action_dim={action_dim}")
    # default standing pose applied by the reset (0.1 rad tolerance: task cfgs may
    # deliberately randomize joint offsets on reset, e.g. Insert's +/-0.05 rad)
    dpose_err = (robot.data.joint_pos - robot.data.default_joint_pos).abs().max().item()
    record("robot:default_pose_applied", dpose_err < 0.1, f"max|q-q_default|={dpose_err:.2e} rad")

    # ----- record initial state, then roll out under zero (hold-default) actions -----
    actions = torch.zeros((base.num_envs, action_dim), device=device)
    init_root_z = robot.data.root_pos_w[:, 2].clone()
    init_bulb_p = base.scene["bulb"].data.root_pos_w.clone()
    step1_root = None
    nan_seen = False
    min_z_seen, max_z_seen, max_speed_seen = float("inf"), float("-inf"), 0.0
    render_viewer = not args_cli.headless

    # Kinematic props must not move *within* an episode. RL family members auto-reset
    # finished episodes inside step() and their reset events may deliberately re-pose
    # kinematic props (e.g. Insert re-samples the socket +/- a few cm), so track per-step
    # movement and skip comparisons across a reset (detected via episode_length_buf --
    # an RL-env buffer; the non-RL ManagerBasedEnv never auto-resets mid-run).
    prev_kin = {n: base.scene[n].data.root_pos_w.clone() for n in kinematic_props}
    ep_len_buf = getattr(base, "episode_length_buf", None)
    prev_ep_len = ep_len_buf.clone() if ep_len_buf is not None else None
    max_kin_move = dict.fromkeys(kinematic_props, 0.0)

    print(f"\n[verify] Stepping {args_cli.steps} steps under zero/default-hold actions...")
    for i in range(args_cli.steps):
        base.step(actions)
        if render_viewer:
            base.sim.render()
        if torch.isnan(robot.data.root_pos_w).any() or torch.isnan(base.scene["bulb"].data.root_pos_w).any():
            nan_seen = True
            break
        rz = robot.data.root_pos_w[:, 2]
        min_z_seen = min(min_z_seen, rz.min().item())
        max_z_seen = max(max_z_seen, rz.max().item())
        max_speed_seen = max(max_speed_seen, robot.data.root_lin_vel_w.norm(dim=-1).max().item())
        if i == 0:
            step1_root = robot.data.root_pos_w.clone()
        if ep_len_buf is not None:
            progressed = ep_len_buf > prev_ep_len  # envs that did NOT reset during this step
            prev_ep_len = ep_len_buf.clone()
        else:
            progressed = torch.ones(base.num_envs, dtype=torch.bool, device=device)
        for n in kinematic_props:
            cur = base.scene[n].data.root_pos_w
            if bool(progressed.any()):
                moved = (cur[progressed] - prev_kin[n][progressed]).norm(dim=-1).max().item()
                max_kin_move[n] = max(max_kin_move[n], moved)
            prev_kin[n] = cur.clone()

    # =========================== 3. GRAVITY / SETTLING (numerical soundness) ===========================
    # NOTE: a free-base humanoid holding a fixed joint pose is an inverted pendulum -- without an
    # active balancing policy it WILL sag/tip. That is expected here; we verify the simulation is
    # *sound* (finite, bounded, stays on the floor, no explosion), not that the robot balances.
    print("\n[verify] (3) Gravity / settling (numerical soundness)")
    record("sim:no_nans", not nan_seen, "NaNs in root states" if nan_seen else "finite throughout")
    # z ceiling 2.6: the at-height presets legitimately START the robot at ~1.8 m; the
    # bound guards launches/explosions, not high starting poses.
    record(
        "robot:bounded(no explosion/sink)",
        (not nan_seen) and (max_z_seen < 2.6) and (min_z_seen > -0.05) and (max_speed_seen < 25.0),
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
    for n in kinematic_props:
        record(
            f"{n}:static",
            max_kin_move[n] < 1e-2,
            f"max within-episode move={max_kin_move[n] * 1000:.2f} mm (kinematic)",
        )
    bulb_move = (base.scene["bulb"].data.root_pos_w[:, 2] - init_bulb_p[:, 2]).abs().max().item()
    record("bulb:settles", (bulb_move < 0.15) and not nan_seen, f"vertical move from init = {bulb_move * 1000:.1f} mm")

    # =========================== 4. COLLISION COVERAGE ===========================
    # Imported USD assets (robot and BEHAVIOR-1K props alike) may split visual meshes from
    # dedicated collision meshes, so we require colliders to EXIST under each entity rather
    # than a 1:1 visual-geom:collider match.
    print("\n[verify] (4) Collision coverage (every tracked entity must have colliders)")
    for name in tracked:
        root = env0(getattr(env_cfg.scene, name).prim_path)
        n_geom, n_coll = collider_audit(root)
        record(f"{name}:colliders", n_coll > 0, f"{n_coll} collision prims ({n_geom} visual geoms)")

    # =========================== 5. CONTACT / PENETRATION ===========================
    print("\n[verify] (5) Contact / penetration")
    # robot never sank through the floor at any point in the run (negative pelvis z == fell through)
    record("robot:above_floor", min_z_seen > -0.05 and not nan_seen, f"min root z over run={min_z_seen:.3f} m")
    # bulb did not sink through the ground plane
    bulb_z = base.scene["bulb"].data.root_pos_w[:, 2]
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
        record_orbit(
            base,
            base.scene["video_cam"],
            actions,
            n_steps=args_cli.record_steps,
            fps=args_cli.record_fps,
            out_path=out_path,
            center=tuple(getattr(env_cfg, "orbit_center", ORBIT_CENTER)),
            radius=float(getattr(env_cfg, "orbit_radius", ORBIT_RADIUS)),
            height=float(getattr(env_cfg, "orbit_height", ORBIT_HEIGHT)),
        )
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
    except Exception:  # noqa: BLE001 -- print the traceback before the process exits
        import traceback

        traceback.print_exc()
    finally:
        # SimulationApp.close() ends in a native framework shutdown that terminates the
        # process with exit code 0, so nothing placed after it (sys.exit included) ever
        # runs. main() already closed the env; flush and exit with the real verification
        # result ourselves. os._exit skips Kit's graceful shutdown on purpose -- process
        # teardown releases the GPU, and CI must see a non-zero code on FAIL.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
