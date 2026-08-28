# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the bayonet bulb/socket state machine (issue #54) end-to-end.

The scripted run manipulates bulb poses while robot actions remain zero. It verifies the
reviewed architecture directly:

1. The old bulb starts fully inserted and rotationally locked.
2. Axial motion is rejected until the bulb rotates to the release angle; reversing
   mid-unlock re-locks it.
3. Rotation is rejected while the released bulb travels along the insertion axis.
4. The bulb becomes free only after it passes the configured insertion depth.
5. A fresh bulb never engages an occupied socket or a misaligned entry; it enters
   axially, may start rotation only at full depth, and becomes attached only after the
   bulb itself reaches the configured rotation angle.
6. The completed replacement still satisfies the task success predicate.
7. `(low, high)` parameter ranges sample per env and re-sample on reset.
8. The lock state (`_phase`, `_theta`) reaches the privileged observation group, at a
   known offset, carrying the manager's own state (issue #77).
9. `rotation_sign` releases the way a real bayonet cap does -- counter-clockwise to an
   operator facing the fixture (issue #77).
10. The seated bulb spawns at the fixture's own rotation, which is what `reset()` assumes
    when it zeroes `_prev_twist` (issue #77).
11. Both bulbs filter their collision pair with the socket, so the projection and the contact
    solver cannot fight over the same body again (issue #77).

The fixture stays where the preset mounted it. An earlier rig teleported the socket to the
robot's palm every step for a compact test rig, which cost nothing in the checks -- they all
derive poses from the live socket -- but made a recording show a lamp floating in mid air,
still carrying its randomized wall-mount rotation, so the bulb came out sideways. Scenery
stays put unless a robot moves it.

Both bulbs have gravity and collisions disabled only in this harness, isolating the tensorized
state machine from contact artifacts while retaining a physics-driven free-body check.

Examples
--------
    uv run python scripts/verify_attach.py --headless
    uv run python scripts/verify_attach.py --headless --video /tmp/attach/attach.mp4
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive the bayonet bulb/socket FSM end-to-end.")
parser.add_argument("--seed", type=int, default=0, help="Env seed (deterministic).")
parser.add_argument(
    "--video",
    type=str,
    default=None,
    help="Write an MP4 of the run to this path (implies camera rendering).",
)
parser.add_argument(
    "--check-ranges",
    action="store_true",
    help="Run only the (low, high) parameter-range sampling check (spec item 7). A second "
    "env build in one Isaac Sim process hangs, so this check needs its own invocation.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless
if args_cli.video:
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
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import mate_terms as task_mate
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import rewards as task_rewards
from fiatlux_task.tasks.manager_based.fiatlux_task.replace_env_cfg import (
    BAYONET_INSERTION_DEPTH,
    BAYONET_ROTATION_ANGLE,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed
from fiatlux_task.viz import orbit_pose
from prettytable import PrettyTable

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_tasks.utils import parse_env_cfg

RESULTS: list[tuple[str, bool, str]] = []
VIDEO = None
# Per-captured-frame lock state, for the plot written alongside a video. The mechanic's whole
# state is four numbers and none of them has a visual signature, so a curve shows the unscrew
# far more clearly than any camera angle can.
LOCK_TRACE: list[tuple[int, float, int, float]] = []

# Video framing comes from `viz.fixture_orbit`, the repo's own fixture view: it reads the mount
# from the cfg, keeps the orbit radius inside the room, and gives a wall mount a 180 degree arc
# centred on the direction the socket opening faces. Three hand-rolled aims failed before this
# one -- a fixed world offset framed the robot's torso, a world-up rise hid a ceiling bulb
# behind its shade, and backing away from the robot walked the camera through a wall.
#
# Framing the bulb and the humanoid sharply in ONE shot is not possible anyway: the bayonet
# travel is 34 mm, about 3 percent of frame width at the range needed to see the robot. The
# overlay and the theta plot carry the mechanic's detail, so the camera is free to show scene.
ORBIT: dict | None = None
ORBIT_FRAMES = 227  # nominal run length, so the arc completes over a full recording


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}{' -- ' + detail if detail else ''}", flush=True)


def info(message: str) -> None:
    print(f"  [INFO] {message}", flush=True)


def build_replace_cfg(num_envs: int = 1):
    """Return a deterministic Replace config stripped to the bayonet test rig."""
    # Replace draws its fixture mount at cfg-build time, inside parse_env_cfg and before
    # cfg.seed exists, so the seed has to be declared here or the layout comes from OS
    # entropy. Without this the suite silently tests a different wall or ceiling mount on
    # every run, which is how `bayonet:release_reads_counter_clockwise` was seen reporting
    # two different seat axes for the same commit.
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=num_envs)
    cfg.seed = args_cli.seed
    for event in ("randomize_sky_intensity", "randomize_key_light", "randomize_material_tint"):
        if getattr(cfg.events, event, None) is not None:
            setattr(cfg.events, event, None)
    if getattr(cfg.events, "reset_robot_joints", None) is not None:
        cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
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
    for bulb_cfg in (cfg.scene.fresh_bulb, cfg.scene.old_bulb):
        if getattr(bulb_cfg.spawn, "rigid_props", None) is not None:
            bulb_cfg.spawn.rigid_props.disable_gravity = True
        bulb_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    if args_cli.video:
        from fiatlux_task.viz import make_video_camera_cfg

        cfg.scene.video_cam = make_video_camera_cfg()
    return cfg


def make_env(cfg):
    spec = gym.spec("FIATLUX-Replace-v0")
    module_name, class_name = spec.entry_point.split(":")
    env_class = getattr(importlib.import_module(module_name), class_name)
    env = env_class(cfg=cfg)
    env.reset(seed=args_cli.seed)
    return env


def _spin_about(quat: torch.Tensor, world_axis: torch.Tensor, angle: float) -> torch.Tensor:
    half = 0.5 * angle
    spin = torch.cat(
        [
            torch.cos(torch.tensor([half], device=quat.device)),
            world_axis * torch.sin(torch.tensor(half, device=quat.device)),
        ]
    )
    return quat_mul(spin.unsqueeze(0), quat.unsqueeze(0))[0]


def _build_rig(env, robot, socket, old_bulb, fresh_bulb, zero_action, zeros6):
    seat_offset = torch.tensor(SOCKET_SEAT_OFFSET, device=env.device)
    plug_offset = torch.tensor(BULB_PLUG_OFFSET, device=env.device)
    seat_axis = torch.tensor(SOCKET_SEAT_AXIS, device=env.device)
    _manager = task_attach.attachment_manager(env)
    rotation_sign = 1.0 if _manager is None else _manager.rotation_sign

    def seat_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        socket_quat = socket.data.root_quat_w[0]
        seat_point = socket.data.root_pos_w[0] + quat_apply(socket_quat.unsqueeze(0), seat_offset.unsqueeze(0))[0]
        world_axis = quat_apply(socket_quat.unsqueeze(0), seat_axis.unsqueeze(0))[0]
        return seat_point, socket_quat, world_axis

    def lateral_axis(world_axis: torch.Tensor) -> torch.Tensor:
        reference = torch.tensor([1.0, 0.0, 0.0], device=env.device)
        if abs(torch.dot(world_axis, reference).item()) > 0.9:
            reference = torch.tensor([0.0, 1.0, 0.0], device=env.device)
        lateral = torch.linalg.cross(world_axis, reference)
        return lateral / torch.norm(lateral)

    def bulb_pose(axial_distance: float, rotation: float, lateral_distance: float = 0.0) -> torch.Tensor:
        seat_point, socket_quat, world_axis = seat_geometry()
        # `rotation` is LOCK-POSITIVE (0 = released, +rotation_angle = fully locked), the
        # same convention as the manager's theta. `rotation_sign` maps it onto the physical
        # twist about the seat axis. Without this factor the suite silently assumes
        # rotation_sign == +1 and every rotation check inverts when the sign is flipped.
        bulb_quat = _spin_about(socket_quat, world_axis, rotation_sign * rotation)
        plug_point = seat_point + axial_distance * world_axis + lateral_distance * lateral_axis(world_axis)
        bulb_pos = plug_point - quat_apply(bulb_quat.unsqueeze(0), plug_offset.unsqueeze(0))[0]
        return torch.cat([bulb_pos, bulb_quat])

    def place_bulb(bulb, axial_distance: float, rotation: float, lateral_distance: float = 0.0) -> None:
        bulb.write_root_pose_to_sim(bulb_pose(axial_distance, rotation, lateral_distance).unsqueeze(0))
        bulb.write_root_velocity_to_sim(zeros6)

    def axial_distance(bulb) -> float:
        seat_point, _, world_axis = seat_geometry()
        plug_point = (
            bulb.data.root_pos_w[0] + quat_apply(bulb.data.root_quat_w[0].unsqueeze(0), plug_offset.unsqueeze(0))[0]
        )
        return torch.dot(plug_point - seat_point, world_axis).item()

    def bulb_twist(bulb) -> float:
        _, socket_quat, _ = seat_geometry()
        return task_attach._signed_twist(
            socket_quat.unsqueeze(0),
            bulb.data.root_quat_w[0].unsqueeze(0),
            seat_axis.unsqueeze(0),
        )[0].item()

    def bulb_lateral_distance(bulb) -> float:
        seat_point, _, world_axis = seat_geometry()
        plug_point = (
            bulb.data.root_pos_w[0] + quat_apply(bulb.data.root_quat_w[0].unsqueeze(0), plug_offset.unsqueeze(0))[0]
        )
        displacement = plug_point - seat_point
        axial = torch.dot(displacement, world_axis)
        return torch.norm(displacement - axial * world_axis).item()

    def lock_state_sample() -> tuple[int, float, int, float]:
        """One row of the lock trace: (old phase, old theta, fresh phase, fresh theta)."""
        manager = task_attach.attachment_manager(env)
        if manager is None:
            return (0, 0.0, 0, 0.0)
        return (
            int(manager._phase[task_attach._OLD, 0].item()),
            float(manager._theta[task_attach._OLD, 0].item()),
            int(manager._phase[task_attach._FRESH, 0].item()),
            float(manager._theta[task_attach._FRESH, 0].item()),
        )

    def lock_state_caption() -> str:
        """The state machine's own numbers, for burning into a video frame.

        The unscrew has no visual signature: the bulb is a surface of revolution, so turning it
        about its own axis changes almost nothing on screen. `theta` is the only way to see the
        release happen, and task 1 of #77 is what made it readable from outside the manager.
        """
        manager = task_attach.attachment_manager(env)
        if manager is None:
            return ""
        names = {task_attach._FREE: "FREE", task_attach._AXIAL: "AXIAL", task_attach._ROTATING: "ROTATING"}
        rows = []
        for label, row in (("old", task_attach._OLD), ("fresh", task_attach._FRESH)):
            phase = int(manager._phase[row, 0].item())
            theta = float(manager._theta[row, 0].item())
            rows.append(f"{label:<5} {names.get(phase, phase):<8} theta={theta:5.3f} rad")
        return "\n".join(rows)

    def step() -> None:
        env.step(zero_action)
        if VIDEO is not None:
            aim_camera()
            VIDEO.capture(overlay=lock_state_caption())
            LOCK_TRACE.append(lock_state_sample())

    def drive_pose(bulb, axial_start: float, axial_end: float, rotation_start: float, rotation_end: float, steps: int):
        for index in range(steps):
            fraction = (index + 1) / steps
            axial = axial_start + fraction * (axial_end - axial_start)
            rotation = rotation_start + fraction * (rotation_end - rotation_start)
            place_bulb(bulb, axial, rotation)
            step()

    def animate_free_bulb(bulb, start: torch.Tensor, end: torch.Tensor, quat: torch.Tensor, steps: int) -> None:
        for index in range(steps):
            fraction = (index + 1) / steps
            pos = start * (1.0 - fraction) + end * fraction
            bulb.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))
            bulb.write_root_velocity_to_sim(zeros6)
            step()

    def aim_camera() -> None:
        """Orbit the fixture using the repo's own fixture view (``viz.fixture_orbit``).

        Hand-rolled aims kept failing on this scene, three times: a fixed world offset put the
        robot's torso in the way, a world-up rise hid a ceiling bulb behind its shade, and
        stepping back from the robot walked the camera through the wall a wall fixture is
        mounted on. `fixture_orbit` already solves all of that -- it reads the mount from the
        cfg, keeps the radius inside the room, and gives a wall mount a 180 degree arc centred
        on the direction the socket opening actually faces.

        Orbiting also answers what a fixed camera cannot: the mechanic has no visual signature,
        so a moving viewpoint at least shows the scene it sits in.
        """
        if VIDEO is None or ORBIT is None:
            return
        eye, lookat = orbit_pose(len(VIDEO), ORBIT_FRAMES, **ORBIT)
        VIDEO.set_pose(eye, lookat)

    return (
        seat_geometry,
        place_bulb,
        axial_distance,
        bulb_twist,
        bulb_lateral_distance,
        step,
        drive_pose,
        animate_free_bulb,
        aim_camera,
    )


def _check_parameter_ranges() -> None:
    """Spec §9.7: ``(low, high)`` ranges sample per env and re-sample on reset."""
    cfg = build_replace_cfg(num_envs=2)
    cfg.events.bulb_attachment.params["insertion_depth"] = (0.020, 0.045)
    cfg.events.bulb_attachment.params["rotation_angle"] = (0.6, 2.4)
    env = make_env(cfg)
    try:
        manager = getattr(env, task_attach._ENV_ATTR)
        first_depth, first_angle = manager._depth.clone(), manager._angle.clone()
        env.reset(seed=args_cli.seed + 1)
        depth, angle = manager._depth.clone(), manager._angle.clone()
        per_env = bool((depth[0] != depth[1]).item() or (angle[0] != angle[1]).item())
        re_sampled = bool((depth != first_depth).any().item() or (angle != first_angle).any().item())
        in_range = bool(
            ((depth >= 0.020) & (depth <= 0.045)).all().item() and ((angle >= 0.6) & (angle <= 2.4)).all().item()
        )
        record(
            "bayonet:parameter_ranges_randomize",
            per_env and re_sampled and in_range,
            f"depths={[round(v, 4) for v in depth.tolist()]} m, angles={[round(v, 3) for v in angle.tolist()]} rad",
        )
    finally:
        env.close()


def _check_lock_state_observable(env, manager) -> None:
    """Assert the bayonet lock state reaches the privileged observation group (issue #77).

    Placement matters, not only presence: the group concatenates, so a consumer reads the
    lock columns by offset. This pins them to the tail and checks they carry the manager's
    own reset state -- the old bulb locked at the rotation angle, the fresh bulb free.

    Without this term ``_phase`` and ``_theta`` reach no observation, telemetry or recording
    path, and an operator cannot tell a twist that does not register from one the lock
    clamps away.
    """
    terms = env.observation_manager.active_terms.get("privileged", [])
    if "bulb_lock_state" not in terms:
        record("bayonet:lock_state_observable", False, "no bulb_lock_state term in the privileged group")
        return
    priv = env.observation_manager.compute()["privileged"]
    lock = task_attach.bulb_lock_state(env)
    tail_ok = bool(torch.allclose(priv[:, -4:], lock))
    old_locked = bool(torch.all(lock[:, 0] == 2.0)) and bool(torch.allclose(lock[:, 1], manager._angle))
    fresh_free = bool(torch.all(lock[:, 2] == 0.0)) and bool(torch.all(lock[:, 3] == 0.0))
    record(
        "bayonet:lock_state_observable",
        tail_ok and old_locked and fresh_free,
        f"privileged[{priv.shape[-1]}] tail 4 = {[round(v, 4) for v in lock[0].tolist()]} "
        f"(old ROTATING at theta=angle, fresh FREE)",
    )


def _check_spawn_twist(env, socket, old_bulb) -> None:
    """The seated bulb must spawn at the fixture's own rotation (issue #77 task 4).

    ``reset()`` sets ``_prev_twist = 0`` on the stated premise that "the old bulb's init rot IS
    the fixture rot" (``attach.py`` L159). A scene edit that moves the socket's rotation without
    matching the bulb's breaks it: the first step then reads the mismatch as a real twist and
    integrates a false delta into theta, which can spuriously advance the unlock.

    Measured before any step, so it reads the spawn poses and not the projected ones.

    Both the twist and the FULL orientation have to match. ``_signed_twist`` projects onto the
    seat axis, so a pure swing mismatch -- a local-X tilt, say -- returns zero twist and would
    pass a twist-only test while the bulb is genuinely misaligned. The first interval event then
    snaps it into place and hides the regression.
    """
    axis = torch.tensor(SOCKET_SEAT_AXIS, device=env.device).expand(env.num_envs, 3)
    socket_quat, bulb_quat = socket.data.root_quat_w, old_bulb.data.root_quat_w
    twist = task_attach._signed_twist(socket_quat, bulb_quat, axis)
    swing = task_attach._orientation_error(socket_quat, bulb_quat)
    worst = float(twist.abs().max().item())
    worst_full = float(swing.abs().max().item())
    record(
        "bayonet:seated_bulb_spawns_untwisted",
        worst < 1e-3 and worst_full < 1e-3,
        f"worst |spawn twist| = {worst:.2e} rad, worst full orientation error = {worst_full:.2e} rad "
        f"across {env.num_envs} env(s)",
    )


def _write_lock_plot(video_path: str, rotation_angle: float) -> str | None:
    """Plot the lock trace next to the video, as ``<video>_theta.png``.

    The bulb is a surface of revolution, so its rotation is invisible on camera. The curve is
    the honest view of the mechanic: theta walking down from the lock angle to zero, with the
    phase bands behind it showing when the state machine actually released.
    """
    if not LOCK_TRACE:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        info("matplotlib missing, skipping the theta plot")
        return None

    old_phase = [row[0] for row in LOCK_TRACE]
    old_theta = [row[1] for row in LOCK_TRACE]
    fresh_theta = [row[3] for row in LOCK_TRACE]
    frames = range(len(LOCK_TRACE))

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    # Phase bands behind the curves: the release is a state change, not just a number falling.
    colors = {task_attach._FREE: "#e8f5e9", task_attach._AXIAL: "#fff4e5", task_attach._ROTATING: "#e8eefc"}
    start = 0
    for i in range(1, len(old_phase) + 1):
        if i == len(old_phase) or old_phase[i] != old_phase[start]:
            # Bin EDGES, not frame indices. Spanning `start` to `i - 1` draws nothing at all
            # when a phase lasts a single frame, and clips every other band by one frame. The
            # scripted `unlock_step_pins_axial` produces exactly a one-frame AXIAL phase, so
            # the naive version omits the transition this plot exists to show.
            ax.axvspan(start - 0.5, i - 0.5, color=colors.get(old_phase[start], "#ffffff"), zorder=0)
            start = i
    ax.plot(frames, old_theta, label="old bulb theta", color="#1f4fd8", linewidth=2.0)
    ax.plot(frames, fresh_theta, label="fresh bulb theta", color="#d81f4f", linewidth=1.4, linestyle="--")
    ax.axhline(rotation_angle, color="#666666", linewidth=0.9, linestyle=":", label="lock angle")
    ax.axhline(0.0, color="#666666", linewidth=0.9, linestyle=":")
    ax.set_xlabel("captured frame")
    ax.set_ylabel("theta (rad)")
    ax.set_title("Bayonet lock state (bands = old bulb phase: blue ROTATING, orange AXIAL, green FREE)")
    ax.legend(loc="upper right")
    ax.margins(x=0)
    fig.tight_layout()

    out = video_path.rsplit(".", 1)[0] + "_theta.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _check_socket_pair_filtered(env) -> None:
    """Both bulbs must filter their collision pair with the socket (issue #77 task 6).

    The projection owns a constrained bulb's pose and writes it every step. Leaving the socket
    contact live costs a median 1067 N, about 3100x the bulb's weight, against roughly 5 N of
    tangential force from an 0.1 N.m twist. `theta` never settles and the release never fires.
    ``scripts/step0_contact.py`` measures it, and ``_spawn_bulb_socket_filtered`` fixes it.

    The test is STRUCTURAL, and it has to be, because this suite's rig disables the bulb
    colliders -- which is precisely why the suite could not see the bug in the first place. The
    behavioural half lives in ``scripts/verify_pair_filter.py``, with the colliders on and
    several envs, and in a collider-on run of ``scripts/diagnose_contact_twist.py``.
    """
    from pxr import UsdPhysics

    stage = env.sim.stage
    want = "/World/envs/env_0/Socket"
    problems = []
    checked = 0
    for name in ("Bulb", "OldBulb"):
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
        if not prim or not prim.IsValid():
            continue  # a preset without this bulb is not a fault here
        checked += 1
        if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            problems.append(f"{name} has no FilteredPairsAPI")
            continue
        rel = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
        targets = [str(t) for t in (rel.GetTargets() or [])]
        if want not in targets:
            problems.append(f"{name} filters {targets}, wanted {want}")
    record(
        "bayonet:bulb_socket_pair_filtered",
        checked > 0 and not problems,
        f"{checked} bulb(s) filter their socket pair" if not problems else "; ".join(problems),
    )


def _check_release_direction(manager, seat_geometry) -> None:
    """The release twist must read counter-clockwise to the operator (issue #77 task 2).

    A BA22d bayonet cap releases counter-clockwise and seats clockwise, as seen by whoever
    faces the fixture. ``rotation_sign`` maps the manager's lock-positive theta onto the
    physical twist, so it alone decides which way a person has to turn. It shipped unchosen
    at +1.0, which inverted both halves and left the real release direction completely
    inert -- no rotation, no state change, nothing observable.

    The test does not look at world axes, because Replace mounts the fixture on a random
    wall and the seat axis can point any direction. It uses a mount-independent fact
    instead: the seat axis points from the seat OUTWARD along the insertion axis (positive
    axial travel leaves the socket, which is what ``eject`` tests), so it always points at
    whoever holds the bulb. A positive rotation about an axis aimed at the viewer reads
    counter-clockwise to that viewer, whatever the fixture's orientation.

    Making the rest of the suite sign-agnostic (see ``bulb_pose``) is what makes this check
    necessary: with the rotations expressed in lock-positive units, nothing else here would
    notice the sign flipping back.
    """
    _, _, axis_w = seat_geometry()
    # The physical twist the manager holds while locked. Release drives it toward zero, so
    # the release rotation is positive about the seat axis exactly when this is negative.
    locked_twist = manager.rotation_sign * float(manager._angle[0].item())
    counter_clockwise = locked_twist < 0.0
    record(
        "bayonet:release_reads_counter_clockwise",
        counter_clockwise,
        f"rotation_sign={manager.rotation_sign:+.0f}, seat axis {[round(v, 2) for v in axis_w.tolist()]} "
        f"-> operator turns {'counter-clockwise' if counter_clockwise else 'CLOCKWISE (a real cap releases CCW)'}",
    )


def main() -> int:
    global VIDEO, ORBIT
    torch.manual_seed(args_cli.seed)
    if args_cli.check_ranges:
        _check_parameter_ranges()
        return _summary()
    env = make_env(build_replace_cfg())
    try:
        robot = env.scene["robot"]
        socket = env.scene["socket"]
        old_bulb = env.scene["old_bulb"]
        fresh_bulb = env.scene["fresh_bulb"]
        manager = getattr(env, task_attach._ENV_ATTR, None)
        if manager is None:
            record("bayonet:manager_present", False, "no bulb_attachment term wired on FIATLUX-Replace-v0")
            return _summary()
        record("bayonet:manager_present", True, "mdp.bulb_attachment is enforced every step")
        _check_lock_state_observable(env, manager)
        _check_spawn_twist(env, socket, old_bulb)
        _check_socket_pair_filtered(env)

        zero_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        zeros6 = torch.zeros((env.num_envs, 6), device=env.device)
        if args_cli.video:
            from fiatlux_task.viz import VideoRecorder, fixture_orbit

            VIDEO = VideoRecorder(env, env.scene["video_cam"], args_cli.video, fps=20)
            ORBIT = fixture_orbit(env.cfg)
            info(f"fixture orbit {ORBIT}")

        (
            seat_geometry,
            place_bulb,
            axial_distance,
            bulb_twist,
            bulb_lateral_distance,
            step,
            drive_pose,
            animate_free_bulb,
            aim_camera,
        ) = _build_rig(
            env,
            robot,
            socket,
            old_bulb,
            fresh_bulb,
            zero_action,
            zeros6,
        )
        _check_release_direction(manager, seat_geometry)

        def old_phase() -> int:
            return int(manager._phase[task_attach._OLD, 0].item())

        def fresh_phase() -> int:
            return int(manager._phase[task_attach._FRESH, 0].item())

        def fresh_theta() -> float:
            return manager._theta[task_attach._FRESH, 0].item()

        def old_theta() -> float:
            return manager._theta[task_attach._OLD, 0].item()

        # Warm the renderer BEFORE the first capture. The RTX pipeline returns an empty frame
        # on its first passes, so capturing immediately puts a blank frame at the head of every
        # recording. Step without capturing until it produces pixels.
        if VIDEO is not None:
            aim_camera()
            for _ in range(4):
                env.step(zero_action)

        for _ in range(12):
            step()
        for _ in range(8):
            step()
        aim_camera()

        depth = manager._depth[0].item()
        angle = manager._angle[0].item()
        params_ok = abs(depth - BAYONET_INSERTION_DEPTH) < 1e-5 and abs(angle - BAYONET_ROTATION_ANGLE) < 1e-5
        record(
            "bayonet:parametric_geometry",
            params_ok,
            f"depth={depth:.3f} m, rotation={angle:.3f} rad",
        )

        old_attached = bool(task_attach.old_bulb_attached(env)[0].item())
        old_seat_error = abs(axial_distance(old_bulb))
        record(
            "bayonet:old_starts_locked",
            old_attached and old_seat_error < 0.003,
            f"attached={old_attached}, axial error={old_seat_error * 1000:.1f} mm",
        )

        # A second bulb must never engage an occupied socket, however well aligned.
        fresh_park = fresh_bulb.data.root_pos_w[0].clone()
        fresh_park_quat = fresh_bulb.data.root_quat_w[0].clone()
        place_bulb(fresh_bulb, 0.5 * depth, 0.0)
        step()
        record(
            "bayonet:occupied_socket_rejects_fresh",
            fresh_phase() == task_attach._FREE,
            f"aligned fresh bulb in occupied channel stays FREE (phase={fresh_phase()})",
        )
        fresh_bulb.write_root_pose_to_sim(torch.cat([fresh_park, fresh_park_quat]).unsqueeze(0))
        fresh_bulb.write_root_velocity_to_sim(zeros6)
        step()

        place_bulb(old_bulb, 0.75 * depth, angle)
        step()
        locked_axial = abs(axial_distance(old_bulb))
        record(
            "bayonet:rotation_stage_blocks_translation",
            locked_axial < 0.003 and bool(task_attach.old_bulb_attached(env)[0].item()),
            f"attempted {0.75 * depth * 1000:.1f} mm, retained {locked_axial * 1000:.1f} mm",
        )

        # Reversing a partial unlock must re-lock (theta clamps at the lock angle).
        drive_pose(old_bulb, 0.0, 0.0, angle, 0.4 * angle, 12)
        drive_pose(old_bulb, 0.0, 0.0, 0.4 * angle, angle, 12)
        relocked = bool(task_attach.old_bulb_attached(env)[0].item())
        record(
            "bayonet:reversed_unlock_relocks",
            relocked and abs(old_theta() - angle) < 0.02,
            f"theta={old_theta():.3f} rad after reversal (lock angle {angle:.3f})",
        )

        # A shove landing on the very step the bulb unlocks (twist -> 0 and a large axial
        # offset in one step) must stay pinned at the seat: the unlock transition step
        # begins AXIAL travel only on the *next* step.
        drive_pose(old_bulb, 0.0, 0.0, angle, 0.1, 20)
        place_bulb(old_bulb, 0.9 * depth, 0.0)
        step()
        unlock_shove_axial = abs(axial_distance(old_bulb))
        record(
            "bayonet:unlock_step_pins_axial",
            unlock_shove_axial < 0.003 and old_phase() == task_attach._AXIAL,
            f"unlocked with 0.9*depth shove, retained {unlock_shove_axial * 1000:.1f} mm axial",
        )
        # Restore the locked state so the removal sequence below starts as it expects.
        place_bulb(old_bulb, 0.0, 0.05)
        step()
        drive_pose(old_bulb, 0.0, 0.0, 0.05, angle, 12)

        drive_pose(old_bulb, 0.0, 0.0, angle, 0.0, 24)
        old_released = not bool(task_attach.old_bulb_attached(env)[0].item())
        record(
            "bayonet:bulb_rotation_unlocks_old",
            old_released and old_phase() == task_attach._AXIAL,
            f"bulb rotation={old_theta():.3f} rad, released={old_released}",
        )

        place_bulb(old_bulb, 0.5 * depth, 0.5 * angle, lateral_distance=0.025)
        step()
        axial_twist = abs(bulb_twist(old_bulb))
        axial_position = axial_distance(old_bulb)
        lateral_error = bulb_lateral_distance(old_bulb)
        record(
            "bayonet:axial_stage_allows_only_axis_travel",
            axial_twist < 0.02 and lateral_error < 0.003 and abs(axial_position - 0.5 * depth) < 0.003,
            f"twist={axial_twist:.3f} rad, lateral error={lateral_error * 1000:.1f} mm",
        )

        drive_pose(old_bulb, 0.5 * depth, 1.15 * depth, 0.0, 0.0, 18)
        old_free = old_phase() == task_attach._FREE
        record(
            "bayonet:old_ejects_after_full_depth",
            old_free,
            f"phase={old_phase()}, travel={axial_distance(old_bulb) * 1000:.1f} mm",
        )

        free_start = old_bulb.data.root_pos_w[0].clone()
        _, _, world_axis = seat_geometry()
        free_velocity = torch.cat([0.5 * world_axis, torch.zeros(3, device=env.device)])
        old_bulb.write_root_velocity_to_sim(free_velocity.unsqueeze(0))
        for _ in range(10):
            step()
        free_motion = torch.norm(old_bulb.data.root_pos_w[0] - free_start).item()
        record(
            "bayonet:ejected_bulb_moves_freely",
            old_free and free_motion > 0.02,
            f"physics-driven displacement={free_motion * 100:.1f} cm",
        )

        # Carry the freed bulb to the crate, which is where the task wants it. The old code
        # nudged it by a hardcoded WORLD offset, [0.20, -0.12, 0.06], and the harness disables
        # gravity and collisions, so the bulb simply hung wherever that landed. On seed 0 that
        # direction points into the east wall: the bulb parked mid-air, half buried in it, and
        # stayed in shot for the whole fresh-bulb sequence. Same world-frame mistake the camera
        # aim made twice. The crate is a real scene entity, so this works on any mount.
        old_quat = old_bulb.data.root_quat_w[0].clone()
        old_start = old_bulb.data.root_pos_w[0].clone()
        old_aside = env.scene["bin"].data.root_pos_w[0].clone()
        old_aside[2] += 0.05
        animate_free_bulb(old_bulb, old_start, old_aside, old_quat, 18)

        # A misaligned bulb must never engage the (now empty) channel.
        place_bulb(fresh_bulb, 0.5 * depth, 0.0, lateral_distance=0.03)
        step()
        record(
            "bayonet:misaligned_entry_rejected",
            fresh_phase() == task_attach._FREE,
            f"3 cm lateral offset stays FREE (phase={fresh_phase()})",
        )

        # --- issue #90: the clock angle a bayonet enters at is free -------------------------
        # A cap goes into the bore at whatever angle the operator's wrist happens to be at and
        # turns from there. The gate used to read FULL-frame orientation error, so any entry
        # twist past the tolerance was rejected -- 326 of 358 blocked steps in the 2026-08-21
        # bags had their tilt within tolerance and were refused on twist alone.
        entry_clock = 0.40  # rad, twice the tolerance the full-frame gate allowed
        place_bulb(fresh_bulb, 0.5 * depth, entry_clock)
        step()
        record(
            "bayonet:enters_at_any_clock_angle",
            fresh_phase() == task_attach._AXIAL,
            f"entered at {entry_clock:.3f} rad of twist (phase={fresh_phase()})",
        )

        # ...and the mechanic keeps that angle instead of teleporting the bulb onto the socket's
        # own. The projection used to write `socket_quat` outright, which moved a gripped bulb
        # 0.155 rad in one 20 ms step.
        retained_clock = bulb_twist(fresh_bulb)
        expected_clock = task_attach.attachment_manager(env).rotation_sign * entry_clock
        record(
            "bayonet:entry_clock_angle_preserved",
            # The phase conjunct is load-bearing. Without it this passes whenever the bulb never
            # engaged at all -- nothing constrains a FREE bulb, so its twist is trivially retained.
            fresh_phase() == task_attach._AXIAL and abs(retained_clock - expected_clock) < 0.02,
            f"entered at {expected_clock:+.3f} rad, retained {retained_clock:+.3f} rad "
            f"(phase={fresh_phase()})",
        )

        # The dense alignment reward must not fall as the bulb turns toward the lock. This pins the
        # PROPERTY the two reward helpers have, which is why Install must score the axis-only one.
        #
        # It has to run on a FREE bulb. A constrained one is pose-written every step, so both
        # readings land on the same projected pose, the full-frame term does not move either, and
        # the check reports a difference that is really leftover FSM state.
        drive_pose(fresh_bulb, 0.5 * depth, 1.6 * depth, 0.0, 0.0, 12)  # travel out -> FREE
        place_bulb(fresh_bulb, 2.0 * depth, 0.0, lateral_distance=0.05)
        step()
        free_for_reward = fresh_phase() == task_attach._FREE
        axis_untwisted = task_mate.bulb_axis_alignment_tanh(env, std=0.3)[0].item()
        full_untwisted = task_rewards.object_socket_orientation_tanh(env, std=0.3)[0].item()
        place_bulb(fresh_bulb, 0.5 * depth, angle, lateral_distance=0.05)
        step()
        axis_twisted = task_mate.bulb_axis_alignment_tanh(env, std=0.3)[0].item()
        full_twisted = task_rewards.object_socket_orientation_tanh(env, std=0.3)[0].item()
        record(
            "bayonet:axis_alignment_is_twist_invariant",
            free_for_reward
            and abs(axis_twisted - axis_untwisted) < 0.01
            and (full_untwisted - full_twisted) > 0.1,
            f"axis-only {axis_untwisted:.3f} -> {axis_twisted:.3f} over a {angle:.3f} rad turn; "
            f"full-frame falls {full_untwisted:.3f} -> {full_twisted:.3f} (free={free_for_reward})",
        )

        # And the wiring: Install must score the axis-only term. Read off the cfg class, because
        # this script builds Replace and a second env build in one Isaac process hangs.
        from fiatlux_task.tasks.manager_based.fiatlux_task.install_env_cfg import RewardsCfg as _InstallRewards

        record(
            "bayonet:install_scores_axis_alignment",
            _InstallRewards.align_orientation.func is task_mate.bulb_axis_alignment_tanh,
            f"Install align_orientation -> {_InstallRewards.align_orientation.func.__name__}",
        )

        place_bulb(fresh_bulb, 1.15 * depth, 0.0)
        step()

        drive_pose(fresh_bulb, 1.15 * depth, 0.75 * depth, 0.0, 0.0, 18)
        fresh_axial = fresh_phase() == task_attach._AXIAL
        record(
            "bayonet:fresh_enters_axial_channel",
            fresh_axial,
            f"phase={fresh_phase()}, depth={axial_distance(fresh_bulb) * 1000:.1f} mm",
        )

        place_bulb(fresh_bulb, 0.5 * depth, 0.5 * angle)
        step()
        fresh_axial_twist = abs(bulb_twist(fresh_bulb))
        record(
            "bayonet:fresh_cannot_rotate_during_insertion",
            fresh_axial_twist < 0.02 and fresh_phase() == task_attach._AXIAL,
            f"attempted {0.5 * angle:.3f} rad, retained {fresh_axial_twist:.3f} rad",
        )

        drive_pose(fresh_bulb, 0.5 * depth, -0.002, 0.0, 0.0, 18)
        fresh_at_turning_point = fresh_phase() == task_attach._AXIAL
        record(
            "bayonet:full_insertion_reaches_turning_point",
            fresh_at_turning_point
            and abs(axial_distance(fresh_bulb)) < 0.003
            and not bool(task_attach.fresh_bulb_attached(env)[0].item()),
            f"phase={fresh_phase()}, rotation={fresh_theta():.3f} rad",
        )

        place_bulb(fresh_bulb, 0.0, 0.15 * angle)
        step()
        rotation_started = fresh_phase() == task_attach._ROTATING
        record(
            "bayonet:bulb_twist_selects_rotation_stage",
            rotation_started and fresh_theta() > 0.0,
            f"phase={fresh_phase()}, rotation={fresh_theta():.3f} rad",
        )

        place_bulb(fresh_bulb, 0.75 * depth, fresh_theta())
        step()
        fresh_locked_axial = abs(axial_distance(fresh_bulb))
        record(
            "bayonet:fresh_cannot_translate_while_rotating",
            fresh_locked_axial < 0.003 and fresh_phase() == task_attach._ROTATING,
            f"attempted {0.75 * depth * 1000:.1f} mm, retained {fresh_locked_axial * 1000:.1f} mm",
        )

        rotation_start = fresh_theta()
        drive_pose(fresh_bulb, 0.0, 0.0, rotation_start, angle, 24)
        fresh_attached = bool(task_attach.fresh_bulb_attached(env)[0].item())
        record(
            "bayonet:bulb_rotation_attaches_fresh",
            fresh_attached,
            f"bulb rotation={fresh_theta():.3f} rad, attached={fresh_attached}",
        )

        seat_point, _, _ = seat_geometry()
        shove = seat_point + torch.tensor([0.12, 0.10, 0.05], device=env.device)
        fresh_bulb.write_root_pose_to_sim(torch.cat([shove, fresh_bulb.data.root_quat_w[0]]).unsqueeze(0))
        fresh_bulb.write_root_velocity_to_sim(zeros6)
        step()
        shove_error = abs(axial_distance(fresh_bulb))
        record(
            "bayonet:locked_fresh_rejects_shove",
            shove_error < 0.003 and bool(task_attach.fresh_bulb_attached(env)[0].item()),
            f"re-seated to {shove_error * 1000:.1f} mm axial error",
        )

        crate = env.scene["bin"]
        crate_pos = crate.data.root_pos_w[0].clone()
        crate_pos[2] += 0.05
        old_bulb.write_root_pose_to_sim(torch.cat([crate_pos, old_bulb.data.root_quat_w[0]]).unsqueeze(0))
        old_bulb.write_root_velocity_to_sim(zeros6)
        for _ in range(10):
            step()
        success = bool(task_attach.attached_replacement_success(env)[0].item())
        record("bayonet:replacement_success", success, f"attached_replacement_success={success}")

        finite = not bool(
            torch.isnan(robot.data.root_pos_w).any()
            or torch.isnan(old_bulb.data.root_pos_w).any()
            or torch.isnan(fresh_bulb.data.root_pos_w).any()
        )
        record("bayonet:no_nan", finite, "states finite throughout" if finite else "NaN in states")

        if VIDEO is not None:
            info(f"wrote video {VIDEO.write()} ({len(VIDEO)} frames)")
            plot_path = _write_lock_plot(args_cli.video, angle)
            if plot_path:
                info(f"wrote lock-state plot {plot_path}")
    finally:
        env.close()
    info("run again with --check-ranges for the parameter-range sampling check")
    return _summary()


def _summary() -> int:
    table = PrettyTable()
    table.field_names = ["#", "check", "result", "detail"]
    table.align["check"] = table.align["detail"] = "l"
    passed = 0
    for index, (name, ok, detail) in enumerate(RESULTS, 1):
        table.add_row([index, name, "PASS" if ok else "FAIL", detail])
        passed += int(ok)
    print("\n" + table.get_string(), flush=True)
    failures = [name for name, ok, _ in RESULTS if not ok]
    verdict = "PASS" if not failures else "FAIL"
    print(f"\n[verify] {passed}/{len(RESULTS)} checks passed -- OVERALL: {verdict}", flush=True)
    if failures:
        print("[verify] Failing checks:\n   - " + "\n   - ".join(failures), flush=True)
    return 0 if not failures else 1


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
