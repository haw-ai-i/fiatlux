# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Does a wall-seated bulb hold its ORIENTATION against gravity, no operator, no contact? (issue #171)

``verify_ceiling_hold.py`` answers the ceiling question: on a ceiling mount the seat axis points
straight down, so the bulb's whole weight acts along the release direction, and the axial magnet
(``a321061``) has to -- and does -- beat it. A wall mount rotates the seat axis to roughly
horizontal instead, so gravity acts almost entirely LATERAL to the seat axis: this is the
lateral/tilt centering terms' load to carry, not the axial magnet's, and unlike the axial term
those gains were shipped as "rough starting points... needing the same real-teleop retuning as
every other gain" (module docstring) -- never measured against a wall mount specifically.

Real teleop evidence (S11 insert, wall mount, seed 2, 4 independent episodes, 6 SEATED windows
from 0.2s to 5.7s) found every single one settles with TILT parked at 0.15-0.26 rad against a
0.20 rad ``tilt_tolerance`` -- i.e. right at the boundary the seat admission gate itself uses,
with no margin -- then gets knocked past ``release_threshold`` by an ordinary hand-release
contact transient and can never re-seat, because re-seating needs ``tilt < tilt_tolerance`` too
and that budget is already spent. This script isolates whether that near-tolerance tilt is a
PASSIVE property of the mechanism (gravity alone, no hand, no operator) or an artifact of noisy
teleop contact -- the same isolation ``verify_ceiling_hold.py`` did for the axial defect.

Forces a wall mount (the ``west`` wall's own orientation from ``_sample_fixture_mount``: local
+Z, the seat axis, rotated 90 deg about Y so it points along the wall's inward normal instead of
straight down), spawns the old bulb already seated at that fixture with zero velocity, and steps
forward under constant zero action for several real seconds -- no hand, no disturbance, nothing
overwriting pose or velocity -- reading axial/lateral/tilt straight off the real, contact-resolved
simulation every step, exactly as ``mdp.attach`` computes them.

Supports overriding the lateral/tilt gains (``--tilt_k``/``--tilt_d``/``--max_torque``/
``--lateral_k``/``--lateral_d``/``--max_lateral_force``), for the same job ``verify_no_twist_spin
.py``'s ``--hold_force``/``--bore_depth`` did: sweeping a candidate config's margin before it
ships, not just confirming the shipped one.

Run via ./pyrun (repo root), not a bare .venv/bin/python -- see verify_ceiling_hold.py's docstring
for why.

Example
-------
    ./pyrun scripts/verify_wall_hold.py --headless
    ./pyrun scripts/verify_wall_hold.py --headless --tilt_k 0.15 --max_torque 0.15
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify a wall-seated bulb holds its orientation under gravity alone.")
parser.add_argument("--seed", type=int, default=0, help="Env seed (deterministic).")
parser.add_argument("--seconds", type=float, default=8.0, help="Real simulated seconds to hold and watch.")
parser.add_argument("--tilt_k", type=float, default=None, help="Override the tilt-alignment gain (N*m/rad-ish).")
parser.add_argument("--tilt_d", type=float, default=None, help="Override the tilt damping gain.")
parser.add_argument("--max_torque", type=float, default=None, help="Override the tilt torque saturation (N*m).")
parser.add_argument("--lateral_k", type=float, default=None, help="Override the lateral spring gain (N/m).")
parser.add_argument("--lateral_d", type=float, default=None, help="Override the lateral damping gain.")
parser.add_argument("--max_lateral_force", type=float, default=None, help="Override the lateral force saturation (N).")
parser.add_argument(
    "--release_threshold",
    type=float,
    default=None,
    help="Override the axial excursion (m) that releases a seated bulb. Pair with --axial_kick "
    "to show the before/after directly: the same brush that the shipped 15 mm threshold rides "
    "out releases the bulb at the 8 mm one it replaced.",
)
parser.add_argument(
    "--axial_kick",
    type=float,
    default=0.0,
    help="Spawn the plug this many METRES out along the seat axis -- a finger-brush that has "
    "dragged the bulb part-way out. Checks that the magnet is still engaged there and pulls it "
    "home, rather than having switched off mid-excursion. Pair with --axial_kick_vel.",
)
parser.add_argument(
    "--axial_kick_vel",
    type=float,
    default=0.0,
    help="Outward axial velocity (m/s) to give the plug at spawn. A real brush does not leave "
    "the bulb at rest -- it drags it out and lets go while it is still moving, which is why a "
    "positional kick alone understates it. Measured off the real bags at the moment each "
    "knock-out crossed the threshold: median 0.10 m/s, worst 0.25 m/s.",
)
parser.add_argument(
    "--tilt_perturb",
    type=float,
    default=0.0,
    help="Rotate the bulb by this many radians about an axis perpendicular to the seat axis "
    "before releasing it -- a one-time bump (e.g. a hand knocking it) rather than a perfect "
    "spawn, to test RECOVERY rather than static equilibrium. A passive hold starting from a "
    "perfect spawn can be stable while still never recovering from a real disturbance.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything else follows."""

import importlib
import inspect
import math
import sys

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import torch
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import (
    _WALLS,
    WALL_MOUNT_Z,
    _quat_mul,
    _quat_y_deg,
    _quat_z_deg,
    set_layout_seed,
)

from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_tasks.utils import parse_env_cfg

_TILT_TOLERANCE = inspect.signature(task_attach.bulb_attachment.__call__).parameters["tilt_tolerance"].default


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

    # Force a WALL mount, the "west" wall's own orientation (scene_cfg.py's _sample_fixture_mount
    # wall branch): local +Z (the seat axis) rotated 90 deg about Y so it points along the wall's
    # inward normal (roughly horizontal) rather than straight down, then yawed so +X faces into
    # the room -- +90, matching _sample_fixture_mount exactly, not a stand-in approximation.
    _, wall_x, _, yaw, _ = _WALLS["west"]
    wall_quat = _quat_mul(_quat_z_deg(yaw), _quat_y_deg(90.0))
    _, y, _ = cfg.scene.socket.init_state.pos
    cfg.scene.socket.init_state.pos = (wall_x, y, WALL_MOUNT_Z)
    cfg.scene.socket.init_state.rot = wall_quat
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

    term_params = env.event_manager.get_term_cfg("bulb_attachment").params
    overrides = {
        k: v
        for k, v in (
            ("tilt_k", args_cli.tilt_k),
            ("tilt_d", args_cli.tilt_d),
            ("max_torque", args_cli.max_torque),
            ("lateral_k", args_cli.lateral_k),
            ("lateral_d", args_cli.lateral_d),
            ("max_lateral_force", args_cli.max_lateral_force),
            ("release_threshold", args_cli.release_threshold),
        )
        if v is not None
    }
    term_params.update(overrides)
    defaults = inspect.signature(task_attach.bulb_attachment.__call__).parameters
    live = {
        k: term_params.get(k, defaults[k].default)
        for k in ("tilt_k", "tilt_d", "max_torque", "lateral_k", "lateral_d", "max_lateral_force", "release_threshold")
    }
    print(
        f"\n=== TILT/LATERAL GAINS {'(OVERRIDDEN: ' + str(overrides) + ')' if overrides else '(defaults)'} ===",
        flush=True,
    )
    print(f"  {live}", flush=True)
    # Read by attach.py's own signature rather than duplicated, same reason verify_ceiling_hold.py
    # does: a stale copy here would silently grade a fallen-out or mis-tilted bulb as still held.
    release_threshold = live["release_threshold"]

    device = env.device
    zero_action = torch.zeros((1, env.action_manager.total_action_dim), device=device)
    seat_axis = torch.tensor(SOCKET_SEAT_AXIS, device=device)
    seat_offset = torch.tensor(SOCKET_SEAT_OFFSET, device=device)
    plug_offset = torch.tensor(BULB_PLUG_OFFSET, device=device)

    def seated_pose() -> tuple[torch.Tensor, torch.Tensor]:
        socket_quat = socket.data.root_quat_w[0]
        seat = socket.data.root_pos_w[0] + quat_apply(socket_quat.unsqueeze(0), seat_offset.unsqueeze(0))[0]
        pos = seat - quat_apply(socket_quat.unsqueeze(0), plug_offset.unsqueeze(0))[0]
        return pos, socket_quat

    def state() -> tuple[float, float, float, int]:
        socket_quat = socket.data.root_quat_w
        bulb_quat = old_bulb.data.root_quat_w
        axis_w = quat_apply(socket_quat, seat_axis.unsqueeze(0))
        seat = socket.data.root_pos_w + quat_apply(socket_quat, seat_offset.unsqueeze(0))
        plug = old_bulb.data.root_pos_w + quat_apply(bulb_quat, plug_offset.unsqueeze(0))
        d = plug - seat
        axial = (d * axis_w).sum(dim=1)
        lateral = torch.norm(d - axial.unsqueeze(1) * axis_w, dim=1)
        tilt = task_attach._tilt_error(socket_quat, bulb_quat, seat_axis)
        phase = int(manager._phase[task_attach._OLD, 0].item())
        return float(axial.item()), float(lateral.item()), float(tilt.item()), phase

    # Spawn the old bulb EXACTLY at the seat, aligned with the (now wall) socket, at rest --
    # matching how a real wall-mounted Replace scene starts (old_bulb pre-seated). Written once,
    # before the loop; nothing overwrites it again after this. Any tilt/lateral growth seen below
    # is therefore produced by the mechanism (gravity + real contact + this term's own wrench),
    # not carried over from a sloppy spawn.
    pos, quat = seated_pose()
    if args_cli.axial_kick:
        # Park the plug this far out along the seat axis, at rest -- what a teleop finger-brush
        # leaves behind (issue #171, sixth finding: 12 of 14 real knock-outs were brushes that
        # parked between 8.9 and 13.8 mm, with the hand already off the bulb). The question this
        # answers is whether the magnet, still engaged at 15 mm, actually pulls it home under real
        # contact -- not just whether the force arithmetic says it should.
        axis_w0 = quat_apply(socket.data.root_quat_w, seat_axis.unsqueeze(0))[0]
        pos = pos + args_cli.axial_kick * axis_w0
    if args_cli.tilt_perturb:
        # A one-time bump about world Y -- the gravity-relevant tipping direction for a roughly
        # world-X seat axis -- applied AFTER computing the aligned seated pose, so `pos` still
        # reflects the unperturbed plug offset (a real hand bump rotates the bulb in place, it
        # does not relocate the seat). Composed on the LEFT: this is a world-frame rotation
        # applied to the already-oriented bulb, not a rotation of its local frame.
        bump = torch.tensor(_quat_y_deg(math.degrees(args_cli.tilt_perturb)), device=device)
        quat = quat_mul(bump.unsqueeze(0), quat.unsqueeze(0))[0]
    old_bulb.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))
    vel = torch.zeros((1, 6), device=device)
    if args_cli.axial_kick_vel:
        vel[0, :3] = args_cli.axial_kick_vel * quat_apply(socket.data.root_quat_w, seat_axis.unsqueeze(0))[0]
    old_bulb.write_root_velocity_to_sim(vel)
    env.step(zero_action)  # resolves spawn phase -> SEATED, since it's within tolerance of the seat
    axial0, lateral0, tilt0, phase0 = state()
    axis_z = float(quat_apply(socket.data.root_quat_w, seat_axis.unsqueeze(0))[0, 2].item())
    print(
        f"SETUP mount=wall seat_axis.z={axis_z:.3f} (expect ~0, horizontal) axial0={axial0:.4f} "
        f"lateral0={lateral0:.4f} tilt0={tilt0:.4f} (tilt_perturb={args_cli.tilt_perturb:.4f} rad requested) "
        f"phase0={phase0} (expect phase0=1/SEATED)",
        flush=True,
    )
    if abs(axis_z) > 0.3:
        print("FATAL seat axis is not roughly horizontal -- the wall setup itself is wrong", flush=True)
        return 1
    if phase0 != task_attach._SEATED:
        print("FATAL did not seat at spawn -- the setup itself is wrong, not testing retention", flush=True)
        return 1

    steps = int(round(args_cli.seconds / env.step_dt))
    trace = []
    max_abs_axial = 0.0
    max_lateral = 0.0
    max_tilt = 0.0
    dropped_at = None
    for i in range(steps):
        env.step(zero_action)  # no hand, no disturbance -- gravity + real contact + retention only
        a, lat, tilt, p = state()
        max_abs_axial = max(max_abs_axial, abs(a))
        max_lateral = max(max_lateral, lat)
        max_tilt = max(max_tilt, tilt)
        if p != task_attach._SEATED and dropped_at is None:
            dropped_at = i * env.step_dt
        if i % max(1, steps // 20) == 0 or i == steps - 1:
            trace.append((round(i * env.step_dt, 3), round(a, 4), round(lat, 4), round(tilt, 4), p))

    # "held" mirrors verify_ceiling_hold.py's axial check; "tilt_margin" is this script's own
    # question -- not just whether the phase stayed SEATED (release only checks axial), but
    # whether tilt stayed with real margin below tilt_tolerance rather than parked at its edge.
    # Half of tilt_tolerance is the same kind of margin bar the axial fix set for itself (it
    # demanded multiples of the load, not a bare pass) -- a bulb sitting at 90%+ of the entry
    # gate's own tolerance has no room left for any real disturbance, hand or otherwise.
    held = dropped_at is None and max_abs_axial <= release_threshold
    tilt_margin_ok = max_tilt <= 0.5 * _TILT_TOLERANCE
    print(
        f"\nRESULT held={held} tilt_margin_ok={tilt_margin_ok} "
        f"max_abs_axial={max_abs_axial:.4f} release_threshold={release_threshold:.4f} "
        f"max_lateral={max_lateral:.4f} max_tilt={max_tilt:.4f} tilt_tolerance={_TILT_TOLERANCE:.4f} "
        f"dropped_at={dropped_at} final_phase={state()[3]}",
        flush=True,
    )
    print(f"trace (t, axial, lateral, tilt, phase): {trace}", flush=True)
    env.close()
    return 0 if (held and tilt_margin_ok) else 1


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
