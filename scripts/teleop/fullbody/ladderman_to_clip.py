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
  * SAMPLE RATE is 120 Hz, not 50. `joint_pos`/`body_*` carry 1704 frames while `object_contact`
    carries 710; LadderMan's `step_dt: 0.02` makes the contact track 50 Hz, so the take is
    710/50 = 14.20 s and the motion arrays run 1704/14.20 = 120.0 Hz exactly. An earlier version
    of this script defaulted to 50 and would have played every clip at 2.4x slow motion.

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
    ap.add_argument("--meta",
                    help="the clip's meta.json. When given, joint order is verified by NAME "
                         "against JOINT_ORDER, which is exact. The knee-signature heuristic "
                         "below only infers order, and it assumes a natural gait -- a clip we "
                         "synthesised ourselves can trip it while being perfectly ordered. "
                         "Names win when we have them; the heuristic stays for files without.")
    ap.add_argument("--fps", type=int, default=120,
                    help="motion arrays are 120 Hz; only object_contact is 50 (see docstring)")
    ap.add_argument("--anchor-mode", default="xyz", choices=["xyz", "xy"],
                    help="'xyz' shifts all three axes so frame 0 matches the anchor -- correct "
                         "when both clips START THE SAME WAY, e.g. both standing on the floor. "
                         "'xy' leaves z alone, which is what a motion that starts on the PLATFORM "
                         "needs: its heights are already right and only its ground-plane position "
                         "has to move. Using xyz there drags the whole descent underground.")
    ap.add_argument("--anchor",
                    help="one of OUR clips. The raw motion sits in LadderMan's world with the "
                         "ladder near the origin, so replaying it against our scene puts the "
                         "robot metres from our ladder. Frame 0 of both is the robot standing "
                         "with an identity root quaternion, so anchoring on frame-0 pelvis is a "
                         "pure translation. Without it the clip is written unplaced.")
    ap.add_argument("--segment", default="full", choices=["ascent", "descent", "full"],
                    help="the take climbs two rungs and then descends, so the halves are "
                         "separable: 'ascent' for S02, 'descent' for S04/S12.")
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

    if args.meta:
        import json as _json
        names = list(_json.load(open(args.meta))["joint_names"])
        if names != list(JOINT_ORDER):
            raise SystemExit("meta.json joint_names differ from JOINT_ORDER -- refusing")
        order = "MuJoCo/URDF (verified by name against meta.json)"
    else:
        order = _check_joint_order(jp)
    quat = _check_quat(bq[:, 0])
    print(f"  joint order : {order}")
    print(f"  root quat   : {quat}")
    if "UNKNOWN" in order:
        raise SystemExit("joint order could not be confirmed -- refusing to write a clip that "
                         "would silently scramble the legs")
    root_rot = bq[:, 0] if quat == "wxyz" else bq[:, 0][:, [3, 0, 1, 2]]

    root_trans = bp[:, 0].astype(np.float64)
    peak = int(np.argmax(root_trans[:, 2]))
    seg = {"ascent": slice(0, peak + 1), "descent": slice(peak, len(jp)),
           "full": slice(0, len(jp))}[args.segment]
    print(f"  peak        : frame {peak} (t={peak / args.fps:.2f}s, pelvis "
          f"{root_trans[peak, 2]:.3f} m) -> '{args.segment}' = frames {seg.start}..{seg.stop}")

    if args.anchor:
        a = np.load(args.anchor)
        a0 = np.asarray(a["root_trans_offset"], np.float64)[0]
        q0 = np.asarray(a["root_rot"], np.float64)[0]
        if abs(q0[0] - 1.0) > 1e-3 or np.abs(q0[1:]).max() > 1e-3:
            print(f"  WARNING     : anchor frame-0 root_rot {np.round(q0, 4)} is not identity, so "
                  f"a translation alone will not place this clip correctly")
        delta = a0 - root_trans[0]
        if args.anchor_mode == "xy":
            delta[2] = 0.0
        elif abs(delta[2]) > 0.25:
            raise SystemExit(
                f"anchor would move z by {delta[2]*1000:+.0f} mm. Both clips are ground-referenced, "
                f"so a shift that large means they do not start the same way -- a motion that "
                f"begins on the platform needs --anchor-mode xy. Refusing to bury the clip.")
        root_trans = root_trans + delta
        print(f"  anchored on : {args.anchor} -> translation {np.round(delta, 4)} "
              f"(dz {delta[2] * 1000:+.1f} mm; a small dz confirms a shared convention)")
    else:
        print("  anchored on : NOTHING -- clip stays in LadderMan's frame and will not line up "
              "with our ladder. Pass --anchor to place it.")

    np.savez(args.out, dof=jp[seg], root_rot=root_rot[seg],
             root_trans_offset=root_trans[seg].astype(np.float32), fps=args.fps)
    T = jp[seg].shape[0]
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
