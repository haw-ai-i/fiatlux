# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What did the bayonet lock do in a recorded session? (issue #92, PR #128)

Reads the lock telemetry a run.h5 bag carries (``old_bulb_phase`` / ``old_bulb_theta`` /
``lock_rotation_angle`` / ``lock_insertion_depth``, recorded since ``bulb_lock_telemetry``
landed) and reconstructs what the state machine did: every phase transition, a step-by-step
trace around each pop-out (a locked or inserted bulb going FREE), whether a FREE bulb ever
sat inside the fixture's envelope while off the channel -- travel the collision filter of
``_spawn_bulb_socket_filtered`` no longer blocks -- and the contact-force and speed peaks.

This is how the 2026-09-01 evidence bags
(``gs://fiatlux/teleop-trajectory/issue-evidence-2026-09-01-92-release-sessions/``) were
diagnosed: theta unwound its full quarter-turn lock in 8-14 control steps while a hand
squeezed the seated bulb (solver spin 30-50 rad/s -- no visible rotation, the bulb is a
surface of revolution), the freed bulb left at up to 5.6 m/s, and the "1.6 s palm jam" of
bench ep06 was a bulb lying on the bench under the open hand from 0.2 s after the fingers
opened. ``tests/test_attach.py`` replays the ep06 unwind against the state machine's
contact bounds.

Needs only h5py + numpy -- no Isaac Sim. Point it at one or more bags:

    python scripts/analyze_lock_telemetry.py "path/to/*/run.h5" --window 6.0 9.0
"""

import argparse
import glob
import json

import h5py
import numpy as np

# Keep in sync with fiatlux_task.assets; duplicated so the script needs no repo install.
SEAT_OFFSET = np.array([0.0, 0.0, 0.036259])
PLUG_OFFSET = np.array([0.0, 0.0, 0.036259])
FREE, AXIAL, ROTATING = 0, 1, 2


def quat_apply(q, v):
    """Rotate vectors ``v`` (N,3) by wxyz quaternions ``q`` (N,4)."""
    qv, w = q[:, 1:], q[:, 0:1]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_mul(a, b):
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def quat_inv(q):
    out = q.copy()
    out[:, 1:] *= -1.0
    return out


def signed_twist(socket_q, bulb_q):
    """Signed twist about socket-local z; mirrors ``attach._signed_twist``."""
    rel = quat_mul(quat_inv(socket_q), bulb_q)
    rel = rel * np.where(rel[:, :1] < 0.0, -1.0, 1.0)
    return 2.0 * np.arctan2(rel[:, 3], rel[:, 0])


def _bulb_channels(g: dict) -> tuple[str, str, str, str | None] | None:
    """Which channels carry the OLD bulb's kinematics, across two recorders.

    The bench recorder logs every scene entity by name (``old_bulb_pos`` / ``_quat`` /
    ``_lin_vel`` / ``_ang_vel``); the canonical ``recording.py`` logs only the single tracked
    bulb as ``bulb_pos`` / ``_quat`` / ``_lin_vel`` and no angular-velocity channel at all. The
    lock telemetry is about the old bulb, so prefer the explicit ``old_bulb_*`` set; fall back to
    ``bulb_*`` (correct whenever the tracked bulb IS the old bulb, e.g. the Remove tasks) and
    derive spin from the quaternion trace when no ``*_ang_vel`` exists.
    """
    for pos, quat, lin, ang in (
        ("old_bulb_pos", "old_bulb_quat", "old_bulb_lin_vel", "old_bulb_ang_vel"),
        ("bulb_pos", "bulb_quat", "bulb_lin_vel", "bulb_ang_vel"),
    ):
        if pos in g and quat in g and lin in g:
            return pos, quat, lin, (ang if ang in g else None)
    return None


def _axis_spin_from_quat(bq: np.ndarray, axis_w: np.ndarray, dt: float) -> np.ndarray:
    """Signed spin about ``axis_w`` (rad/s) from consecutive orientations, for bags with no
    recorded angular velocity. Small-angle finite difference of the relative rotation."""
    rel = quat_mul(bq[1:], quat_inv(bq[:-1]))
    rel = rel * np.where(rel[:, :1] < 0.0, -1.0, 1.0)
    omega = 2.0 * rel[:, 1:] / dt  # angle-axis rate, small-angle
    spin = (omega * axis_w[1:]).sum(axis=1)
    return np.concatenate([spin[:1], spin])


def analyze(path: str, windows: list[tuple[float, float]]) -> None:
    with h5py.File(path, "r") as f:
        d = f["data/demo_0"]
        meta = json.loads(f.attrs["meta"])
        g = {k: d[k][:] for k in d.keys()}
    if "old_bulb_phase" not in g:
        print(f"\n=== {path}: no lock telemetry in this bag (task wires no mdp.bulb_attachment) ===")
        return
    channels = _bulb_channels(g)
    if channels is None:
        print(f"\n=== {path}: lock telemetry present but no bulb kinematics channels found ===")
        return
    pos_k, quat_k, lin_k, ang_k = channels
    dt = meta["step_dt"]
    T = len(g["old_bulb_phase"])
    t = np.arange(T) * dt

    phase = g["old_bulb_phase"].astype(int)
    theta = g["old_bulb_theta"]
    depth = g["lock_insertion_depth"]

    sq, sp = g["socket_quat"], g["socket_pos"]
    bq, bp = g[quat_k], g[pos_k]
    axis_w = quat_apply(sq, np.tile([0.0, 0.0, 1.0], (T, 1)))
    seat = sp + quat_apply(sq, np.tile(SEAT_OFFSET, (T, 1)))
    plug = bp + quat_apply(bq, np.tile(PLUG_OFFSET, (T, 1)))
    disp = plug - seat
    axial = (disp * axis_w).sum(axis=1)
    lateral = np.linalg.norm(disp - axial[:, None] * axis_w, axis=1)
    twist = signed_twist(sq, bq)
    speed = np.linalg.norm(g[lin_k], axis=1)
    spin = (g[ang_k] * axis_w).sum(axis=1) if ang_k else _axis_spin_from_quat(bq, axis_w, dt)
    if channels[0] == "bulb_pos":
        print(f"  (note: {path} has no old_bulb_* channels; reading the tracked bulb_* entity)")

    force_r = np.linalg.norm(g["contact_force"], axis=2).sum(axis=1)
    force_l = np.linalg.norm(g["contact_force_left"], axis=2).sum(axis=1)
    force = force_r + force_l

    print(
        f"\n=== {path}  T={T} ({T * dt:.1f}s)  lock_angle={g['lock_rotation_angle'][0]:.3f} rad"
        f"  depth={depth[0]:.4f} m ==="
    )
    trans = np.nonzero(np.diff(phase) != 0)[0] + 1
    if len(trans) == 0:
        print("  no old-bulb phase transitions")
    for i in trans:
        print(
            f"  t={t[i]:7.2f}s  phase {phase[i - 1]}->{phase[i]}  theta {theta[i - 1]:+.3f}->{theta[i]:+.3f}"
            f"  axial={axial[i]:+.4f}  lat={lateral[i]:.4f}  |v|={speed[i]:5.2f} m/s"
            f"  spin={spin[i]:+7.1f} rad/s  Fhand={force[i]:7.1f} N"
        )
    pops = [i for i in trans if phase[i] == FREE and phase[i - 1] != FREE]
    for i in pops:
        lo = max(0, i - 12)
        print(f"  -- pop-out at t={t[i]:.2f}s")
        print("     t      phase theta   d_theta twist   axial    lat     |v|    spin    Fhand")
        prev = theta[lo - 1] if lo > 0 else theta[lo]
        for j in range(lo, min(T, i + 9)):
            print(
                f"     {t[j]:6.2f} {phase[j]:3d} {theta[j]:+7.3f} {theta[j] - prev:+7.3f} {twist[j]:+7.3f}"
                f" {axial[j]:+7.4f} {lateral[j]:7.4f} {speed[j]:6.2f} {spin[j]:+8.1f} {force[j]:8.1f}"
            )
            prev = theta[j]
    # "Inside the fixture envelope but off the channel": FREE, axially within the socket
    # depth, and laterally between the channel radius and the socket's outer collar (AABB
    # half-extents ~0.074-0.081 m, scene_cfg SOCKET note). The upper bound matters -- without
    # it a bulb resting on the bench a metre away, merely at seat height, counts as "inside".
    ghost = (phase == FREE) & (axial > -0.01) & (axial < depth + 0.005) & (lateral > 0.02) & (lateral < 0.085)
    if ghost.any():
        idx = np.nonzero(ghost)[0]
        print(
            f"  ghost-fixture: FREE bulb inside the fixture envelope but off the channel on"
            f" {ghost.sum()} steps (first t={t[idx[0]]:.2f}s, max lat={lateral[idx].max():.3f} m)"
        )
    print(
        f"  peaks: |v| {speed.max():.2f} m/s at t={t[speed.argmax()]:.2f}s;"
        f" hand contact {force.max():.1f} N at t={t[force.argmax()]:.2f}s"
    )

    for lo, hi in windows:
        s0, s1 = int(lo / dt), min(T, int(hi / dt))
        print(f"  -- window [{lo}-{hi}s]:")
        print("     t      phase theta   axial    lat     |v|    spin    Fhand")
        for j in range(s0, s1, max(1, (s1 - s0) // 24)):
            print(
                f"     {t[j]:6.2f} {phase[j]:3d} {theta[j]:+7.3f} {axial[j]:+7.4f} {lateral[j]:7.4f}"
                f" {speed[j]:6.2f} {spin[j]:+8.1f} {force[j]:8.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bags", nargs="+", help="run.h5 paths or globs")
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        action="append",
        default=[],
        metavar=("START_S", "END_S"),
        help="also dump a coarse trace of this time window (repeatable)",
    )
    args = parser.parse_args()
    paths = sorted(p for pattern in args.bags for p in glob.glob(pattern))
    if not paths:
        raise SystemExit("no bags matched")
    for path in paths:
        analyze(path, [tuple(w) for w in args.window])


if __name__ == "__main__":
    main()
