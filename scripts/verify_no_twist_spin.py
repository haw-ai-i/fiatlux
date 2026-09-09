# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Is the ceiling "spin" gone now that the twist torque is? (issue #171)

Reproduces the STARTING STATE OF THE ORIGINAL EVIDENCE rather than an approximation of it. The
bug was reported from real teleop bags at
``gs://fiatlux/teleop-trajectory/issue-evidence-2026-09-08-171-magnet-ceiling-spin/dex3/``
(take1-174550 and take2-174716, byte-identical over the seated phase), so this builds the same
env those were recorded from -- ``FIATLUX-S03-RemoveOldBulb-Teleop-v0``, dex3 hand, seed 3 --
and lets the preset's own randomized layout place the ceiling fixture. The earlier
``verify_ceiling_hold.py`` / ``verify_twist_damping.py`` scripts instead FORCED a ceiling mount
onto ``FIATLUX-Replace-v0``, which is a fair stand-in for the geometry but not the scene the
bug came from; validating against a stand-in is how the twist fix came to be believed at
``--seed 0`` while still failing at the reported seed. So the first thing this does is assert
the reconstructed start matches the bag's recorded first frame, and refuse to report a result
if it does not.

Then it steps forward under ZERO ACTION for the bag's own episode length -- no operator, no
hand contact (the bags show 0.00 N on both hands for the entire seated phase) -- and measures
the same quantities off the live sim that were measured off the bag.

What "no persistent spin" means here is specific, because the original signature was NOT a
physical spin. In the bag, twist rate reverses sign on 140 of 143 consecutive control steps at
+-16 rad/s (mean |twist| 15.23, peak 22.91) before the bulb ejects at step 144 (t=2.88s). That
is a per-step alternation locked to the 50 Hz control rate -- a force flipping direction every
step, not a rotating body. So this checks BOTH magnitude and the alternation rate: a fix that
merely reduced the amplitude while still chattering every step would not have addressed the
cause.

**Expect a FAIL at default gains, for a DIFFERENT reason -- read the metrics, not the verdict.**
As of dropping the twist torque, this script's three criteria split:

- ``no_chatter`` and the spin magnitudes PASS. Sign-flip fraction 0.16 vs the bag's 0.98,
  peak 11.6 rad/s vs 22.9. Held artificially seated (``--hold_force 2.0 --hold_range 0.04``)
  so there is a long window to judge, it is emphatic: 750/750 steps seated over 15s, peak
  |twist| 0.71 rad/s, mean 0.09, and 0.02 over the final second -- i.e. it decays to nothing
  and stays there, where the bag sustained 15.23 indefinitely. The reported spin is gone.
- ``held`` FAILS: the bulb leaves this ceiling seat at t=0.64s (the bag ejected at 2.88s).
  That is a SEPARATE, pre-existing defect in the axial term, not a consequence of removing the
  twist one -- the magnet-shaped law's magnitude DECREASES with distance, so its balance point
  against constant gravity is UNSTABLE: at ``hold_force=0.5``/``hold_range=0.01`` the crossing
  sits at 4.6 mm (peak only 1.46x the bulb's 0.343 N weight), and past it gravity wins and the
  bulb accelerates out. The module docstring's "worst-case sag ~4.6mm against a 20mm threshold"
  reads that number as a resting point; it is a cliff edge. Widening the basin (peak/weight
  5.8, crossing at 193 mm, same 50 N/m local stiffness) holds it indefinitely, which is what
  the ``--hold_force``/``--hold_range`` overrides above demonstrate -- they are a diagnosis of
  that defect, not a proposed fix for it.

Run via ./pyrun (repo root), not a bare .venv/bin/python -- see verify_twist_damping.py's
docstring for why.

Example
-------
    ./pyrun scripts/verify_no_twist_spin.py --headless
    ./pyrun scripts/verify_no_twist_spin.py --headless --seconds 15 --hold_force 2.0 --hold_range 0.04
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the ceiling twist spin is gone, from the bag's own start state.")
parser.add_argument("--seed", type=int, default=3, help="Env seed. 3 is the seed both evidence bags were recorded at.")
parser.add_argument(
    "--seconds",
    type=float,
    default=9.8,
    help="Real simulated seconds to watch. Default matches the evidence bags' 490-step episode.",
)
parser.add_argument("--hand", type=str, default="dex3", choices=("dex3", "inspire"), help="Hand variant (bags: dex3).")
parser.add_argument(
    "--task",
    type=str,
    default="FIATLUX-S03-RemoveOldBulb-Teleop-v0",
    help="Task id. Default is the one the evidence bags' meta.json names.",
)
parser.add_argument(
    "--skip_start_check",
    action="store_true",
    help="Report even if the reconstructed start does not match the bag. For diagnosing the setup only.",
)
parser.add_argument(
    "--hold_force",
    type=float,
    default=None,
    help="Override the axial hold_force (N). Needed to answer the SPIN question separately from "
    "the axial-hold one: at the shipped 0.5 N the bulb leaves this ceiling seat in ~0.6s, which "
    "is too short a window to call a spin persistent or not. Pair with --hold_range.",
)
parser.add_argument(
    "--hold_range",
    type=float,
    default=None,
    help="Override the axial hold_range (m). Raise it alongside --hold_force to keep the local "
    "stiffness (hold_force/hold_range) under the semi-implicit bound mass/step_dt^2 (~88 N/m).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless

# Read by apply_subtask_teleop when the cfg is constructed (inside parse_env_cfg below), so it
# has to be set before that call, not after.
os.environ["FIATLUX_TELEOP_HAND"] = args_cli.hand

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything else follows."""

import importlib
import sys

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX benchmark environments
import fiatlux_teleop  # noqa: F401  -- registers the -Teleop-v0 twins
import gymnasium as gym
import torch
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg

# ---------------------------------------------------------------------------------------------
# The evidence bags' own numbers, measured off run.h5 (both takes agree exactly over the seated
# phase). Recomputed from the recorded socket/bulb poses in the seat frame the same way this
# script computes them live, so they are directly comparable rather than merely indicative.
#
#   task FIATLUX-S03-RemoveOldBulb-Teleop-v0, seed 3, dex3, sim_dt 0.005, decimation 4,
#   step_dt 0.02, 490 steps (9.8s), score 0.0
BAG = {
    "socket_pos": (0.33791128, -0.28950667, 2.20000005),
    "socket_quat": (0.0, 0.0, 1.0, 0.0),  # 180 deg about +Y: the fixture is inverted (ceiling)
    "axial": 0.00325,
    "lateral": 0.00234,
    "tilt": 0.0427,
    "twist": -5.751,  # ALREADY spinning at the first recorded frame
    "seated_steps": 144,  # leaves the seat at step 144 = t 2.88s
    "max_abs_twist": 22.91,
    "mean_abs_twist": 15.23,
    "sign_flips": 140,  # out of 143 consecutive step-to-step transitions while seated
    "sign_flip_steps": 143,
    "hand_contact_N": 0.0,  # both hands, entire seated phase
}

# Pass thresholds. The bag sustained mean |twist| 15.23 rad/s indefinitely and flipped sign on
# 98% of steps, so these are not close calls -- they are set where a genuine residual contact
# jitter can pass but a chattering force law cannot.
MAX_MEAN_TWIST_LAST_SECOND = 1.0  # rad/s; bag: 15.23 sustained, never decaying
MAX_SIGN_FLIP_FRACTION = 0.5  # bag: 0.98. A per-step alternation is the bug's signature.


def build_cfg():
    """The bag's env, as the bag built it: preset layout at its seed, no forced fixture pose."""
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    assert hasattr(cfg.scene, "old_bulb"), (
        f"cfg.scene ({type(cfg.scene)} from {sys.modules[type(cfg.scene).__module__].__file__}) has no "
        "old_bulb -- fiatlux_task likely resolved to the wrong checkout again; check sys.path/pyrun"
    )
    cfg.seed = args_cli.seed
    # Cameras only cost time here (zero action, nothing to see that the numbers do not say) and
    # draw no randoms, so dropping them does not perturb the layout the seed produced.
    for camera in ("ego_camera", "torso_camera", "wrist_camera"):
        if getattr(cfg.scene, camera, None) is not None:
            setattr(cfg.scene, camera, None)
    for group_name in ("policy", "privileged"):
        group = getattr(cfg.observations, group_name, None)
        for term in ("ego_rgb", "torso_rgb", "wrist_rgb"):
            if group is not None and getattr(group, term, None) is not None:
                setattr(group, term, None)
    # Watch the whole window even if a termination would have cut it: the point is what happens
    # to the seated bulb, and a truncated episode would hide a late ejection.
    cfg.episode_length_s = max(args_cli.seconds * 2.0, cfg.episode_length_s)
    for term in [t for t in vars(cfg.terminations) if not t.startswith("_")]:
        if term != "time_out" and getattr(cfg.terminations, term, None) is not None:
            setattr(cfg.terminations, term, None)
    return cfg


class SeatProbe:
    """Seat-frame quantities read off the contact-resolved sim, identically to the bag maths."""

    def __init__(self, env):
        self.socket = env.scene["socket"]
        self.bulb = env.scene["old_bulb"]
        self.manager = task_attach.attachment_manager(env)
        dev = env.device
        self._axis = torch.tensor(SOCKET_SEAT_AXIS, device=dev).unsqueeze(0)
        self._seat_off = torch.tensor(SOCKET_SEAT_OFFSET, device=dev).unsqueeze(0)
        self._plug_off = torch.tensor(BULB_PLUG_OFFSET, device=dev).unsqueeze(0)

    def axis_w(self) -> torch.Tensor:
        return quat_apply(self.socket.data.root_quat_w, self._axis)

    def _displacement(self) -> torch.Tensor:
        seat = self.socket.data.root_pos_w + quat_apply(self.socket.data.root_quat_w, self._seat_off)
        plug = self.bulb.data.root_pos_w + quat_apply(self.bulb.data.root_quat_w, self._plug_off)
        return plug - seat

    def read(self) -> dict:
        d = self._displacement()
        axis = self.axis_w()
        axial = (d * axis).sum(dim=1)
        lateral = torch.norm(d - axial.unsqueeze(1) * axis, dim=1)
        twist = (self.bulb.data.root_ang_vel_w * axis).sum(dim=1)
        tilt = task_attach._tilt_error(self.socket.data.root_quat_w, self.bulb.data.root_quat_w, self._axis)
        return {
            "axial": float(axial.item()),
            "lateral": float(lateral.item()),
            "tilt": float(tilt.item()),
            "twist": float(twist.item()),
            "phase": int(self.manager._phase[task_attach._OLD, 0].item()),
        }


def check_start_matches_bag(env, probe: SeatProbe) -> list[str]:
    """Refuse to grade a run whose starting state is not the one the bug was reported from."""
    socket = env.scene["socket"]
    pos = [round(float(v), 5) for v in socket.data.root_pos_w[0]]
    quat = [round(float(v), 4) for v in socket.data.root_quat_w[0]]
    state = probe.read()
    print("\n=== STARTING STATE vs. THE EVIDENCE BAG ===", flush=True)
    print(f"  task={args_cli.task} hand={args_cli.hand} seed={args_cli.seed} step_dt={env.step_dt}", flush=True)
    print(f"  {'quantity':14s} {'bag':>12s} {'here':>12s}", flush=True)
    mismatches = []
    for label, bag_value, here, tol in (
        ("socket_pos.x", BAG["socket_pos"][0], pos[0], 2e-3),
        ("socket_pos.y", BAG["socket_pos"][1], pos[1], 2e-3),
        ("socket_pos.z", BAG["socket_pos"][2], pos[2], 2e-3),
        ("axial", BAG["axial"], state["axial"], 2e-3),
        ("lateral", BAG["lateral"], state["lateral"], 2e-3),
        ("tilt", BAG["tilt"], state["tilt"], 2e-2),
    ):
        ok = abs(bag_value - here) <= tol
        print(f"  {label:14s} {bag_value:12.5f} {here:12.5f}  {'ok' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            mismatches.append(f"{label}: bag {bag_value:.5f} vs here {here:.5f} (tol {tol})")
    # The seat axis, not the raw quaternion: (0,0,1,0) and (0,0,-1,0) are the same rotation, and
    # the inverted mount is the physically meaningful part (gravity along the release direction).
    axis_z = float(probe.axis_w()[0, 2].item())
    inverted = axis_z < -0.99
    print(f"  {'seat axis z':14s} {-1.0:12.5f} {axis_z:12.5f}  {'ok' if inverted else 'MISMATCH'}", flush=True)
    if not inverted:
        mismatches.append(f"seat axis z {axis_z:.5f} is not the bag's inverted (ceiling) mount")
    print(f"  bag socket_quat={BAG['socket_quat']} here={quat} (equivalent up to sign)", flush=True)
    print(
        f"  NOTE the bag's bulb is already twisting at {BAG['twist']:.2f} rad/s in its FIRST recorded "
        f"frame; here it starts at {state['twist']:.2f} rad/s -- that initial rate is itself an "
        "output of the mechanism under test, so it is reported, not required to match.",
        flush=True,
    )
    return mismatches


def main() -> int:
    cfg = build_cfg()
    spec = gym.spec(args_cli.task)
    module_name, class_name = spec.entry_point.split(":")
    env = getattr(importlib.import_module(module_name), class_name)(cfg=cfg)
    env.reset(seed=args_cli.seed)

    if task_attach.attachment_manager(env) is None:
        print(f"FATAL no bulb_attachment term on {args_cli.task}", flush=True)
        return 1
    probe = SeatProbe(env)

    overrides = {k: v for k, v in (("hold_force", args_cli.hold_force), ("hold_range", args_cli.hold_range)) if v}
    if overrides:
        env.event_manager.get_term_cfg("bulb_attachment").params.update(overrides)
        weight = 0.035 * 9.81
        hf, hr = args_cli.hold_force, args_cli.hold_range
        print(f"\n=== AXIAL HOLD OVERRIDDEN: {overrides} ===", flush=True)
        print(
            "  Isolating the spin question from the axial-hold one. NOT a proposed fix -- it only\n"
            "  keeps the bulb seated long enough for 'persistent spin' to mean something.",
            flush=True,
        )
        if hf and hr:
            # This law's magnitude DECREASES with distance, so its equilibrium against constant
            # gravity is unstable: inside it the bulb recovers, outside it accelerates away. The
            # useful number is therefore where that crossing sits relative to release_threshold.
            print(
                f"  peak/weight={hf / weight:.2f}  gravity crossing at "
                f"{1000 * hr * (hf / weight - 1):.1f} mm (release_threshold 20 mm)  "
                f"local stiffness={hf / hr:.0f} N/m (bound mass/step_dt^2 ~= "
                f"{0.035 / env.step_dt**2:.0f} N/m)",
                flush=True,
            )

    # The preset spawns the old bulb seated; the manager reads that on its first step. Nothing
    # here writes pose or velocity -- unlike the earlier verify_* scripts, the start state is
    # the task's own, which is the whole point of using the bag's task.
    zero_action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    env.step(zero_action)
    if probe.read()["phase"] != task_attach._SEATED:
        print("FATAL the old bulb did not start SEATED -- the setup is wrong, nothing to measure", flush=True)
        return 1

    mismatches = check_start_matches_bag(env, probe)
    if mismatches and not args_cli.skip_start_check:
        print("\nFATAL reconstructed start does not match the evidence bag:", flush=True)
        for m in mismatches:
            print(f"  - {m}", flush=True)
        print("  Not reporting a spin result against a scenario that isn't the reported one.", flush=True)
        env.close()
        return 1

    steps = int(round(args_cli.seconds / env.step_dt))
    release_threshold = 0.02  # attach.py's default; not exposed on the manager
    twists, trace = [], []
    max_abs_axial = max_abs_twist = max_tilt = 0.0
    left_seat_at = None
    for i in range(steps):
        env.step(zero_action)  # zero action: gravity + real contact + the retention wrench only
        s = probe.read()
        seated = s["phase"] == task_attach._SEATED and abs(s["axial"]) <= release_threshold
        if seated:
            twists.append(s["twist"])
            max_abs_twist = max(max_abs_twist, abs(s["twist"]))
        elif left_seat_at is None:
            left_seat_at = i
        max_abs_axial = max(max_abs_axial, abs(s["axial"]))
        max_tilt = max(max_tilt, s["tilt"])
        if i % max(1, steps // 20) == 0 or i == steps - 1:
            trace.append((round(i * env.step_dt, 2), round(s["axial"], 4), round(s["twist"], 2), round(s["tilt"], 3)))

    tail = twists[max(0, len(twists) - int(round(1.0 / env.step_dt))) :]
    mean_tail = sum(abs(w) for w in tail) / max(1, len(tail))
    flips = sum(1 for a, b in zip(twists, twists[1:]) if (a > 0) != (b > 0))
    flip_fraction = flips / max(1, len(twists) - 1)

    held = left_seat_at is None
    quiet = mean_tail < MAX_MEAN_TWIST_LAST_SECOND
    no_chatter = flip_fraction < MAX_SIGN_FLIP_FRACTION
    print(f"\n=== {args_cli.seconds:.1f}s UNDER ZERO ACTION (seated samples: {len(twists)}/{steps}) ===", flush=True)
    print(f"  {'metric':28s} {'bag (broken)':>16s} {'here':>16s}", flush=True)
    for label, bag_value, here in (
        ("steps seated", f"{BAG['seated_steps']}/490", f"{len(twists)}/{steps}"),
        ("max |twist| rad/s", f"{BAG['max_abs_twist']:.2f}", f"{max_abs_twist:.2f}"),
        (
            "mean |twist| rad/s",
            f"{BAG['mean_abs_twist']:.2f}",
            f"{sum(abs(w) for w in twists) / max(1, len(twists)):.2f}",
        ),
        ("mean |twist| last 1s", "15.23 (no decay)", f"{mean_tail:.2f}"),
        (
            "sign flips / steps",
            f"{BAG['sign_flips']}/{BAG['sign_flip_steps']} = 0.98",
            f"{flips}/{max(1, len(twists) - 1)} = {flip_fraction:.2f}",
        ),
        ("max |axial| m", ">2.0 (fell to floor)", f"{max_abs_axial:.4f}"),
        ("max tilt rad", "~1.9 (ejected)", f"{max_tilt:.3f}"),
        (
            "ejected at",
            "step 144 / t=2.88s",
            "never" if held else f"step {left_seat_at} / t={left_seat_at * env.step_dt:.2f}s",
        ),
    ):
        print(f"  {label:28s} {bag_value:>16s} {here:>16s}", flush=True)
    print(f"  trace (t, axial, twist, tilt): {trace}", flush=True)

    verdict = "PASS" if (held and quiet and no_chatter) else "FAIL"
    print(
        f"\nRESULT {verdict} held={held} quiet={quiet} (mean |twist| last 1s {mean_tail:.2f} < "
        f"{MAX_MEAN_TWIST_LAST_SECOND}) no_chatter={no_chatter} (sign-flip fraction "
        f"{flip_fraction:.2f} < {MAX_SIGN_FLIP_FRACTION})",
        flush=True,
    )
    env.close()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
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
