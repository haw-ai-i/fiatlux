"""Convert a LadderMan reference motion into this project's MotionClip format.

LadderMan (Amazon FAR et al., Apache-2.0, github.com/amazon-far/LadderMan) ships a recorded
G1 ladder climb at `data/motion/ladder_data/ladder_data_1/motion.npz`. It is a 29-DoF G1
joint-space motion with full body kinematics, which is exactly what the climbing pipeline
needs as a reference -- so it stands in for human capture and retargeting.

Two conventions were established empirically rather than assumed, and this script re-checks
both every run so a future motion file cannot silently break them:

  * JOINT ORDER is MuJoCo/URDF -- the same order as `gear_sonic_contract.JOINT_ORDER`, so no
    permutation is applied. Verified by the knee signature: knees bend one way only, and they
    sit at columns 3 and 9, where MuJoCo order puts them (IsaacLab order would be 9 and 10).
  * ROOT QUATERNION is wxyz -- frame 0 is exactly [1,0,0,0] and the first component averages
    0.99 across the take, so the scalar is leading. MotionClip expects wxyz too.

Note the SONIC *encoder* wants IsaacLab order; that remap lives in `groot.py`
(`_ISAAC_ORDER_MUJOCO_IDX`) and is applied there, not here.

    python ladderman_to_clip.py motion.npz --out ladderman_climb.npz
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from gear_sonic_contract import JOINT_ORDER

# Fiatlux step-ladder geometry (assets.py): treads ascend to the top platform.
TREAD_Z = (0.236, 0.47, 0.71, 0.94)
PLATFORM_Z = 1.18


def _check_joint_order(jp: np.ndarray) -> str:
    lo, hi = jp.min(0), jp.max(0)
    knees = sorted(np.argsort((hi - lo) * (lo > -0.15))[-2:].tolist())
    expect = sorted([JOINT_ORDER.index("left_knee_joint"), JOINT_ORDER.index("right_knee_joint")])
    return "MuJoCo/URDF" if knees == expect else f"UNKNOWN (knees at {knees}, expected {expect})"


def _check_quat(q: np.ndarray) -> str:
    return "wxyz" if abs(q[:, 0]).mean() > abs(q[:, 3]).mean() else "xyzw"


def _foot_steps(z: np.ndarray, min_rise: float = 0.10) -> list[tuple[int, int, float]]:
    """Upward transitions of one foot: (lift frame, land frame, rise). A step is a rise that
    settles, which is what distinguishes a tread transition from a swing."""
    steps, i, n = [], 1, len(z)
    while i < n:
        if z[i] - z[i - 1] > 0.002:                       # rising
            j = i
            while j + 1 < n and z[j + 1] >= z[j] - 0.002:
                j += 1
            if z[j] - z[i - 1] >= min_rise:
                steps.append((i - 1, j, float(z[j] - z[i - 1])))
            i = j + 1
        else:
            i += 1
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="LadderMan motion.npz")
    ap.add_argument("--out", default="ladderman_climb.npz")
    ap.add_argument("--fps", type=int, default=50, help="LadderMan ships 50 Hz already")
    args = ap.parse_args()

    d = np.load(args.src)                                  # no allow_pickle: numeric arrays only
    need = {"joint_pos", "body_pos_w", "body_quat_w"}
    missing = need - set(d.files)
    if missing:
        raise SystemExit(f"{args.src}: missing {sorted(missing)}")

    jp = np.asarray(d["joint_pos"], dtype=np.float32)
    bp = np.asarray(d["body_pos_w"], dtype=np.float32)
    bq = np.asarray(d["body_quat_w"], dtype=np.float64)
    if jp.shape[1] != len(JOINT_ORDER):
        raise SystemExit(f"expected {len(JOINT_ORDER)} joints, got {jp.shape[1]}")

    order, quat = _check_joint_order(jp), _check_quat(bq[:, 0])
    print(f"  joint order : {order}")
    print(f"  root quat   : {quat}")
    if "UNKNOWN" in order:
        raise SystemExit("joint order could not be confirmed -- refusing to write a clip that "
                         "would silently scramble the legs")
    root_rot = bq[:, 0] if quat == "wxyz" else bq[:, 0][:, [3, 0, 1, 2]]

    np.savez(args.out, dof=jp, root_rot=root_rot,
             root_trans_offset=bp[:, 0].astype(np.float32), fps=args.fps)
    T = jp.shape[0]
    print(f"\n  wrote {args.out}: {T} frames @ {args.fps} Hz = {T / args.fps:.1f} s")

    # --- does this motion contain a step the size of one fiatlux tread? ---
    z0 = bp[0, :, 2]
    feet = np.argsort(z0)[:2]
    print(f"\n  tread rises in this project: "
          f"{', '.join(f'{b - a:.3f}' for a, b in zip((0.0,) + TREAD_Z, TREAD_Z))} m "
          f"(then {PLATFORM_Z - TREAD_Z[-1]:.3f} to the platform)")
    print(f"\n  {'body':>5} {'step':>5} {'lift t':>8} {'land t':>8} {'rise m':>8}  nearest tread rise")
    for f in feet:
        for k, (i, j, rise) in enumerate(_foot_steps(bp[:, f, 2])):
            gaps = [abs(rise - (b - a)) for a, b in zip((0.0,) + TREAD_Z, TREAD_Z)]
            best = int(np.argmin(gaps))
            print(f"  {f:>5} {k:>5} {i / args.fps:>7.2f}s {j / args.fps:>7.2f}s {rise:>8.3f}"
                  f"   tread {best} (off by {gaps[best]:+.3f} m)")
    print("\n  A step whose rise is within a few cm of a tread rise is the one to retarget first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
