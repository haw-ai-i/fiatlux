#!/usr/bin/env python3
"""Author a ladder reference that CLIMBS TO THE TOP AND STANDS ON THE PLATFORM.

Why this exists. LadderMan's shipped reference is a mid-ladder cycle: `contact_idx` is fixed, the
hand contact labels only ever address rungs `contact_idx+1` and `+2`, and the clip ends by
descending. Traced against our ladder the feet finish on rung 1 (0.471 m) with the pelvis near
1.33 m, where S02's success gate wants the pelvis above

    platform 1.152 + TOP_STANCE_PELVIS_OFFSET 0.787 - height_slack 0.15 = 1.789 m

so a teacher trained on that clip cannot score, by about 0.46 m. It is not a tuning problem: the
hands ride three rungs above the feet for the whole motion, and standing ON the platform would
need rungs above the platform, which do not exist.

How this is built, and why it is not the hand-authored motion that already failed us. The legs
are REAL retargeted human stepping, reused: frames 330..781 of the source clip are a gait cycle
whose joint pose repeats to 0.145 rad while the body rises one rung. Repeating that cycle,
translated by OUR rung pitch each time, climbs our ladder with real motion rather than invented
motion. Only the final mount -- feet onto the platform, arms down, stand up -- is synthesized,
and it is synthesized as a TRAINING reference, not a trajectory to replay: the teacher's RL
refines it under torque limits, which is exactly the gap that made open-loop replay fall at
1.33 s. Check it with measure_balance.py before spending a training run on it.
"""
from __future__ import annotations

import argparse
import json
import math

import pathlib as _pathlib

import numpy as np

# Our training ladder, matching lm_train_*.sbatch.
INCL_DEG, SPACING, BASE = 70.3, 0.2492, (-0.58, 0.0, 0.196)
RUNG_H = 0.080
PLATFORM_TOP = 1.152          # the real S02 asset
# Where the benchmark's platform stance sits, measured from RUNG 0 -- which is what `BASE` is,
# so this is the frame the setback must be applied in. Derived from the asset's own tread table
# (scripts/teleop/fullbody/ladder_treads.py) rather than written down: tread 1's centre is rung 0,
# and STEP_LADDER_TOP_OFFSET[1] is where `stand_robot_on_ladder_top` stages the pelvis.
#
# An earlier version used 0.590, which is the stance measured from the bbox FRONT FACE, and then
# applied it from BASE -- 109 mm too far back on both references. The face is 0.4895 m out from
# the root; rung 0 is only 0.3804. Two different landmarks, one frame.
_STANCE_LOCAL_Y = 0.10          # scene_cfg.STEP_LADDER_TOP_OFFSET[1]
try:
    import sys as _sys
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "teleop" / "fullbody"))
    from ladder_treads import TREADS_LOCAL as _TL
    _RUNG0_Y = (_TL[0]["y_front"] + _TL[0]["y_back"]) / 2
    PLATFORM_SETBACK = _STANCE_LOCAL_Y - _RUNG0_Y
except Exception:                                  # tread table unavailable -> the measured value
    PLATFORM_SETBACK = 0.4804
PELVIS_STANCE = 0.787         # scene_cfg.TOP_STANCE_PELVIS_OFFSET
CYCLE = (330, 781)            # measured: pose repeats to 0.145 rad, rises 0.251 m
# The descent is NOT the ascent reversed -- the source contains a real descent, and descending
# uses far less knee flexion (you lower on the hips and arms), so it needs its own cycle and its
# own acceptance test. Frames 910..1294: pose repeats to 0.104 rad, both feet drop ~0.24 m and
# both clear the rung they leave. The outro is the source's own real step-down to the floor.
CYCLE_DOWN = (910, 1294)
OUTRO_DOWN = (1294, 1704)
FLOOR_STANCE = 0.940          # subtask_tiers/balance.py LADDER_FLOOR_STANCE_HEIGHT


def rung_top(n: int) -> float:
    return BASE[2] + RUNG_H / 2 + n * SPACING * math.sin(math.radians(INCL_DEG))


def rung_x(n: int) -> float:
    return BASE[0] + n * SPACING * math.cos(math.radians(INCL_DEG))


def blend(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Cosine cross-fade of the last n frames of `a` into the first n of `b`."""
    n = min(n, len(a), len(b))
    if n == 0:
        return np.concatenate([a, b])
    w = (1 - np.cos(np.linspace(0, math.pi, n))) / 2
    w = w.reshape((n,) + (1,) * (a.ndim - 1))
    mixed = a[-n:] * (1 - w) + b[:n] * w
    return np.concatenate([a[:-n], mixed, b[n:]])



def ground_ending(P: np.ndarray, la: int, ra: int, ankle_to_sole: float,
                  surface: float, ramp: int) -> np.ndarray:
    """Slide the end of the motion vertically so the feet finish ON the surface.

    The source clip is a human retarget and its feet do not sit exactly on its own rungs -- the
    ankle-to-sole offset solves to 0.093/0.106/0.120 m at three different rests, a 27 mm spread.
    Mid-climb that is a constant bias RL absorbs. At the ENDING it is not: both success gates read
    an ABSOLUTE pelvis height (S02 above 1.789, S04/S12 below 0.940), so a foot left floating 60 mm
    high moves the pelvis 60 mm the wrong way and eats the margin. Correct it over a ramp so the
    fix cannot introduce a step.
    """
    sole = float(min(P[-1, la, 2], P[-1, ra, 2])) - ankle_to_sole
    err = sole - surface
    if abs(err) < 1e-4:
        return P
    n = min(ramp, len(P))
    w = (1 - np.cos(np.linspace(0, math.pi, n))) / 2
    P = P.copy()
    P[-n:, :, 2] -= err * w[:, None]
    print(f"[GROUND] ending sole {sole:+.3f} -> {surface:.3f} ({err*1000:+.0f} mm) "
          f"ramped over {n} frames")
    return P


def trim_to_travel(J, P, Q, JV, V, W, src_f, cyc_k, budget, fps, hold_s=0.6):
    """Stop the descent while it is still inside S04/S12's arrival radius.

    `descended_from_ladder` admits a pelvis within LADDER_SUCCESS_XY_RADIUS of the ladder ROOT,
    and the platform stance starts 0.10 m on the far side, so the whole descent has only
    0.700 m of horizontal travel to spend. The dismount costs 0.472 (the pelvis leans out past
    the feet as it steps down) and three rung cycles cost 0.252, which leaves nothing for the
    source's own step-away at the bottom. So cut the clip at the last frame still inside the
    budget, then HOLD that pose -- a trimmed clip that ends mid-stride would fail the gate's
    `robot_standing` and speed conjuncts even while standing in the right place.
    """
    xy = P[:, 0, :2]
    travel = np.linalg.norm(xy - xy[0], axis=1)
    ok = np.where(travel <= budget)[0]
    if len(ok) == 0 or ok[-1] >= len(J) - 1:
        print(f"[TRAVEL] {travel[-1]:.3f} m <= budget {budget:.3f} m; no trim needed")
        return J, P, Q, JV, V, W, src_f, cyc_k
    cut = int(ok[-1]) + 1
    n_hold = int(hold_s * fps)
    print(f"[TRAVEL] {travel[-1]:.3f} m exceeds the {budget:.3f} m arrival budget; "
          f"cut at frame {cut}/{len(J)} ({travel[cut-1]:.3f} m) + {hold_s:.1f}s hold")
    J, P, Q = J[:cut], P[:cut], Q[:cut]
    JV, V, W = JV[:cut], V[:cut], W[:cut]
    src_f, cyc_k = src_f[:cut], cyc_k[:cut]
    J = np.concatenate([J, np.repeat(J[-1][None], n_hold, 0)])
    P = np.concatenate([P, np.repeat(P[-1][None], n_hold, 0)])
    Q = np.concatenate([Q, np.repeat(Q[-1][None], n_hold, 0)])
    JV = np.concatenate([JV, np.zeros((n_hold, *JV.shape[1:]), JV.dtype)])
    V = np.concatenate([V, np.zeros((n_hold, *V.shape[1:]), V.dtype)])
    W = np.concatenate([W, np.zeros((n_hold, *W.shape[1:]), W.dtype)])
    src_f = src_f + [src_f[-1]] * n_hold
    cyc_k = cyc_k + [cyc_k[-1]] * n_hold
    return J, P, Q, JV, V, W, src_f, cyc_k


def build_down(a, d, meta, jp, bp, bq, bn, jv, bv, bw) -> int:
    """Start standing on the platform, descend to the floor.

    Mirror of the climb: a synthesized dismount off the platform, then the source's real descent
    cycle looped one rung lower each time, then the source's own real step-down to the floor. The
    only invented motion is the dismount, exactly as the mount is the only invented part going up.
    """
    i0, i1 = CYCLE_DOWN
    o0, o1 = OUTRO_DOWN
    step_down = np.array([-SPACING * math.cos(math.radians(INCL_DEG)), 0.0,
                          -SPACING * math.sin(math.radians(INCL_DEG))])
    la, ra = bn.index("left_ankle_roll_link"), bn.index("right_ankle_roll_link")
    ankle_to_sole = 0.11

    # --- dismount: standing on the platform, step back down to the cycle's entry pose ---------
    n = int(a.stand_s * a.fps)
    s = (1 - np.cos(np.linspace(0, math.pi, n))) / 2
    entry_j, entry_p, entry_q = jp[i0].copy(), bp[i0].copy(), bq[i0].copy()
    entry_sole = float(min(entry_p[la, 2], entry_p[ra, 2])) - ankle_to_sole
    # place the cycle entry so its feet start on the rung just below the platform
    first_rung = 3
    lift = (rung_top(first_rung) + ankle_to_sole) - float(min(entry_p[la, 2], entry_p[ra, 2]))
    off = np.array([0.0, 0.0, lift])

    stand_q = np.zeros(jp.shape[1])
    for jn, ji in {nm: i for i, nm in enumerate(meta["joint_names"])}.items():
        if jn.endswith("knee_joint"):
            stand_q[ji] = 0.1
        elif jn.endswith("hip_pitch_joint") or jn.endswith("ankle_pitch_joint"):
            stand_q[ji] = -0.05
    start_p = (entry_p + off).copy()
    start_p[:, 2] += (PLATFORM_TOP - rung_top(first_rung))
    start_p[0, 2] = PLATFORM_TOP + PELVIS_STANCE
    # Mirror of the ascent's mount: the platform sits PLATFORM_SETBACK behind the step face, so
    # the dismount starts there and steps FORWARD onto the top rung, instead of starting already
    # on the rung. Without this the clip begins 0.58 m in front of where S04/S12 stage the robot.
    want = PLATFORM_SETBACK if a.setback is None else a.setback
    setback = ((BASE[0] + want) - float(start_p[0, 0])) if want else 0.0
    start_p[:, 0] += setback
    print(f"[START ] standing on the platform: pelvis {start_p[0,2]:.3f} m, "
          f"soles {PLATFORM_TOP:.3f}, x {start_p[0,0]:+.3f} (set back {setback:+.3f} m)")

    J = stand_q[None] + (entry_j - stand_q)[None] * s[:, None]
    P = start_p[None] + ((entry_p + off) - start_p)[None] * s[:, None, None]
    Q = np.repeat(entry_q[None], n, axis=0)
    # The dismount is synthesized and smooth; orientation is held, so angular velocity is zero.
    JV = np.gradient(J, 1.0 / a.fps, axis=0)
    V = np.gradient(P, 1.0 / a.fps, axis=0)
    W = np.zeros_like(V)

    # --- loop the real descent cycle, one rung lower each time --------------------------------
    src_f, cyc_k = [i0] * n, [0] * n
    for k in range(a.cycles + 1):
        cj, cp, cq = jp[i0:i1].copy(), bp[i0:i1].copy() + off, bq[i0:i1].copy()
        J = blend(J, cj, a.splice); P = blend(P, cp, a.splice); Q = blend(Q, cq, a.splice)
        JV = blend(JV, jv[i0:i1].copy(), a.splice)
        V = blend(V, bv[i0:i1].copy(), a.splice)
        W = blend(W, bw[i0:i1].copy(), a.splice)
        nd = min(a.splice, len(src_f), i1 - i0)
        if nd:
            src_f = src_f[:-nd]; cyc_k = cyc_k[:-nd]
        src_f += list(range(i0, i1)); cyc_k += [k] * (i1 - i0)
        off = off + step_down
    print(f"[DESC  ] {a.cycles+1} cycles -> {len(J)} frames ({len(J)/a.fps:.2f}s)")

    # --- outro: the source's own real step-down to the floor ----------------------------------
    oj, op, oq = jp[o0:o1].copy(), bp[o0:o1].copy() + off - step_down, bq[o0:o1].copy()
    J = blend(J, oj, a.splice); P = blend(P, op, a.splice); Q = blend(Q, oq, a.splice)
    JV = blend(JV, jv[o0:o1].copy(), a.splice)
    V = blend(V, bv[o0:o1].copy(), a.splice)
    W = blend(W, bw[o0:o1].copy(), a.splice)
    nd = min(a.splice, len(src_f), o1 - o0)
    if nd:
        src_f = src_f[:-nd]; cyc_k = cyc_k[:-nd]
    src_f += list(range(o0, o1)); cyc_k += [a.cycles] * (o1 - o0)

    if a.max_travel:
        J, P, Q, JV, V, W, src_f, cyc_k = trim_to_travel(
            J, P, Q, JV, V, W, src_f, cyc_k, a.max_travel, a.fps)
    P = ground_ending(P, la, ra, ankle_to_sole, 0.0, int(1.5 * a.fps))

    # --- contact labels: absolute rung indices, descending ------------------------------------
    SRC = 2
    src_oc = d["object_contact"]
    oc120 = np.zeros((len(J), 2), dtype=np.int32)
    for f in range(min(len(J), len(src_f))):
        sv = src_oc[min(int(src_f[f] / a.fps * 50), len(src_oc) - 1)]
        for h in (0, 1):
            oc120[f, h] = 0 if sv[h] == 0 else max(1, int(sv[h]) + SRC + (a.cycles - cyc_k[f]))
    oc120[:n] = 0                      # both hands free while standing on the platform
    n50 = int(round(len(J) / a.fps * 50))
    oc = oc120[np.clip((np.arange(n50) * a.fps / 50).astype(int), 0, len(J) - 1)]

    out = {
        "joint_pos": J.astype(np.float64),
        "joint_vel": JV.astype(np.float64),
        "body_pos_w": P.astype(np.float32),
        "body_quat_w": Q.astype(np.float32),
        "body_lin_vel_w": V.astype(np.float32),
        "body_ang_vel_w": W.astype(np.float32),
        "object_contact": oc,
    }
    out = fit_to_loader(out, a.fps)
    np.savez(a.out, **out)
    lo_sole = float(min(P[-1, la, 2], P[-1, ra, 2])) - ankle_to_sole
    print(f"[CONTACT] max rung {int(oc.max())} -> num_rungs >= {int(oc.max())+1}, contact_idx: 0")
    print(f"[WROTE ] {a.out}: {len(J)} frames @ {a.fps:g} fps = {len(J)/a.fps:.2f} s")
    print(f"[WROTE ] pelvis {P[0,0,2]:.3f} -> {P[-1,0,2]:.3f} m; final sole {lo_sole:+.3f}")
    print(f"[CHECK ] final pelvis {'CLEARS' if P[-1,0,2] < FLOOR_STANCE else 'MISSES'} "
          f"the S04/S12 gate (needs BELOW {FLOOR_STANCE:.3f} m)")
    return 0


def fit_to_loader(out: dict, fps: float, target_fps: int = 50) -> dict:
    """Trim the clip to a length LadderMan's resampler can actually load.

    `utils/motion.py::interpolate` builds its time bases with
    ``np.arange(0, T/source_fps, 1/source_fps)``, and for some T that returns T+1 samples --
    pure float accumulation, nothing to do with the frame ratio. `np.interp` is then handed
    xp of length T+1 against fp of length T and raises ``fp and xp are not of the same
    length``, 6 minutes into a training job, with the clip itself perfectly well-formed.
    Their shipped 1704-frame clip lands clean; our 2254-frame one does not.

    So reproduce their arithmetic here and drop whole frames from the END until both the
    source base and the 50 Hz target base come out the length they must be. Costs a few
    milliseconds of the final standing hold and turns a remote crash into a local print.
    """
    T0 = out["joint_pos"].shape[0]
    T = T0
    while T > 1:
        end_t = T / fps
        ts_src = np.arange(0, end_t, 1.0 / fps)
        ts_tgt = np.arange(0, end_t, 1.0 / target_fps)
        if len(ts_tgt) and len(ts_src) and ts_tgt[-1] > ts_src[-1]:
            ts_tgt = ts_tgt[:-1]
        if len(ts_src) == T and len(ts_tgt) == out["object_contact"].shape[0] - (T0 - T) * 0:
            # contact is rebuilt below against the final T, so only the source base must agree
            if len(ts_src) == T:
                break
        T -= 1
    if T != T0:
        print(f"[LOADER] {T0} frames would make np.arange return {T0 + 1}; trimmed to {T} "
              f"({(T0 - T) / fps * 1000:.0f} ms off the end)")
    o = {k: (v[:T] if hasattr(v, "shape") and v.shape and v.shape[0] == T0 else v)
         for k, v in out.items()}
    # the 50 Hz contact track is re-derived from the trimmed length, the way the loader expects
    end_t = T / fps
    ts_src = np.arange(0, end_t, 1.0 / fps)
    ts_tgt = np.arange(0, end_t, 1.0 / target_fps)
    if len(ts_tgt) and ts_tgt[-1] > ts_src[-1]:
        ts_tgt = ts_tgt[:-1]
    oc = out["object_contact"]
    idx = np.clip((ts_tgt * target_fps).astype(int), 0, len(oc) - 1)
    o["object_contact"] = oc[idx]
    assert len(ts_src) == T, f"source base {len(ts_src)} != {T}"
    assert o["object_contact"].shape[0] == len(ts_tgt), "contact track does not match the 50 Hz base"
    print(f"[LOADER] T={T} -> source base {len(ts_src)}, 50 Hz base {len(ts_tgt)}, "
          f"contact {o['object_contact'].shape[0]}  OK")
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycles", type=int, default=3, help="extra gait cycles (one rung each)")
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--stand-s", type=float, default=2.0, help="seconds of mount + stand")
    ap.add_argument("--direction", default="up", choices=["up", "down"],
                    help="'up' climbs the ladder and stands on the platform (S02). 'down' starts "
                         "standing on the platform and descends to the floor (S04/S12), using "
                         "the source's own descent cycle rather than the ascent time-reversed.")
    ap.add_argument("--splice", type=int, default=24, help="cross-fade frames at each splice")
    ap.add_argument("--max-travel", dest="max_travel", type=float, default=None,
                    help="DOWN only: stop the clip once the pelvis has moved this far "
                         "horizontally, then hold. S04/S12 allow 0.70 m from the platform "
                         "stance -- see measure_climb_geometry.py.")
    ap.add_argument("--setback", type=float, default=None,
                    help="metres behind the step face to place the platform stance "
                         f"(default {PLATFORM_SETBACK}); 0 keeps the reference's own top-of-ladder "
                         "pose. Going DOWN this is additive travel and can bust the arrival "
                         "radius -- see measure_climb_geometry.py.")
    a = ap.parse_args()

    d = np.load(a.motion, allow_pickle=True)
    meta = json.load(open(a.meta))
    bn = list(meta["body_names"])
    jp, bp, bq = d["joint_pos"].astype(np.float64), d["body_pos_w"].astype(np.float64), d["body_quat_w"].astype(np.float64)
    # Carry the source's OWN velocity tracks rather than differentiating its positions. The
    # retarget's wrist links jitter by 20-35 mm between frames (and jump 220 mm at frame 11),
    # so np.gradient on body_pos_w reports 9.5 m/s where the stored track says 3.3 -- and
    # body_ang_vel_w, which has no positional counterpart at all, came out identically zero
    # against the shipped clip's 20.7 rad/s. A reference whose angular velocity is zero
    # everywhere teaches the tracker to hold still while climbing.
    jv = d["joint_vel"].astype(np.float64)
    bv = d["body_lin_vel_w"].astype(np.float64)
    bw = d["body_ang_vel_w"].astype(np.float64)

    if a.direction == "down":
        return build_down(a, d, meta, jp, bp, bq, bn, jv, bv, bw)

    i0, i1 = CYCLE
    # One rung along OUR ladder axis.
    step = np.array([SPACING * math.cos(math.radians(INCL_DEG)), 0.0,
                     SPACING * math.sin(math.radians(INCL_DEG))])
    print(f"[CYCLE ] frames {i0}..{i1} ({(i1-i0)/a.fps:.2f}s); translating each repeat by "
          f"{np.round(step,4)} (our rung pitch)")

    # --- intro: everything up to the cycle start -------------------------------------------
    J, P, Q = jp[:i0].copy(), bp[:i0].copy(), bq[:i0].copy()
    # Velocity is invariant under the pure translation each cycle repeat applies, so the source
    # tracks splice exactly like the poses do.
    JV, V, W = jv[:i0].copy(), bv[:i0].copy(), bw[:i0].copy()
    off = np.zeros(3)
    # provenance per OUTPUT frame: which source frame it came from, and which cycle (rung offset).
    # The contact labels are rebuilt from this so a hand target rises with the climb instead of
    # staying pinned to the two rungs a fixed `contact_idx` can address.
    src_f = list(range(i0))
    cyc_k = [0] * i0

    # --- repeat the cycle, each repeat one rung higher ---------------------------------------
    for k in range(a.cycles + 1):
        cj, cp, cq = jp[i0:i1].copy(), bp[i0:i1].copy(), bq[i0:i1].copy()
        cp = cp + off
        J = blend(J, cj, a.splice)
        P = blend(P, cp, a.splice)
        Q = blend(Q, cq, a.splice)
        JV = blend(JV, jv[i0:i1].copy(), a.splice)
        V = blend(V, bv[i0:i1].copy(), a.splice)
        W = blend(W, bw[i0:i1].copy(), a.splice)
        n_drop = min(a.splice, len(src_f), i1 - i0)
        if n_drop:
            src_f = src_f[:-n_drop]; cyc_k = cyc_k[:-n_drop]
        src_f += list(range(i0, i1))
        cyc_k += [k] * (i1 - i0)
        off = off + step
    print(f"[CLIMB ] {a.cycles+1} cycles -> {len(J)} frames ({len(J)/a.fps:.2f}s), "
          f"total lift {off[2]:.3f} m")

    # --- mount + stand: feet to the platform, arms down, pelvis to standing height ----------
    la, ra = bn.index("left_ankle_roll_link"), bn.index("right_ankle_roll_link")
    ankle_to_sole = 0.11          # solved from the clip's own rest heights (0.093/0.106/0.120)
    sole_now = float(min(P[-1, la, 2], P[-1, ra, 2])) - ankle_to_sole
    lift = (PLATFORM_TOP + ankle_to_sole) - float(min(P[-1, la, 2], P[-1, ra, 2]))
    pelvis_now = float(P[-1, 0, 2])
    pelvis_target = PLATFORM_TOP + PELVIS_STANCE
    print(f"[MOUNT ] feet sole at {sole_now:.3f} m -> platform {PLATFORM_TOP:.3f} (lift {lift:+.3f})")
    print(f"[MOUNT ] pelvis {pelvis_now:.3f} -> {pelvis_target:.3f} m "
          f"(gate needs > {PLATFORM_TOP + PELVIS_STANCE - 0.15:.3f})")

    n = int(a.stand_s * a.fps)
    s = (1 - np.cos(np.linspace(0, math.pi, n))) / 2        # smooth 0 -> 1

    # Legs and arms relax toward an upright stance; this is the only invented motion.
    stand_q = np.zeros(jp.shape[1])
    name_idx = {nm: i for i, nm in enumerate(meta["joint_names"])}
    for nm, val in ((".*_knee_joint", 0.1), (".*_hip_pitch_joint", -0.05), (".*_ankle_pitch_joint", -0.05)):
        key = nm.replace(".*_", "")
        for jn, ji in name_idx.items():
            if jn.endswith(key):
                stand_q[ji] = val

    # The mount also has to carry the body BACK onto the platform: lifting in place tops out
    # near the step face, which on the real A-frame is its front edge. This is a rigid
    # translation of an already-synthesized segment -- the feet slide rather than step -- which
    # is the same standard the rest of the mount is held to, since the teacher's RL refines it.
    want = PLATFORM_SETBACK if a.setback is None else a.setback
    setback = ((BASE[0] + want) - float(P[-1, 0, 0])) if want else 0.0
    print(f"[MOUNT ] pelvis x {P[-1,0,0]:+.3f} -> {BASE[0] + PLATFORM_SETBACK:+.3f} "
          f"(set back {setback:+.3f} m onto the platform)")

    mj = J[-1] + (stand_q - J[-1]) * s[:, None]
    mp = np.repeat(P[-1][None], n, axis=0)
    mp[:, :, 2] += (s[:, None] * lift)
    mp[:, :, 0] += (s[:, None] * setback)
    mp[:, 0, 2] = pelvis_now + (pelvis_target - pelvis_now) * s
    mq = np.repeat(Q[-1][None], n, axis=0)

    # The mount is synthesized and smooth, so its own velocities ARE the derivative of its
    # poses -- and its orientation is held constant, so its angular velocity is genuinely zero.
    mv = np.gradient(mp, 1.0 / a.fps, axis=0)
    mjv = np.gradient(mj, 1.0 / a.fps, axis=0)
    mw = np.zeros_like(mv)

    J = blend(J, mj, a.splice)
    P = blend(P, mp, a.splice)
    Q = blend(Q, mq, a.splice)
    JV = blend(JV, mjv, a.splice)
    V = blend(V, mv, a.splice)
    W = blend(W, mw, a.splice)
    P = ground_ending(P, la, ra, ankle_to_sole, PLATFORM_TOP, int(1.5 * a.fps))

    # --- contact labels: ABSOLUTE rung indices, so they climb with the robot -----------------
    # LadderMan computes `obj_idx = object_contact + ladder_contact_idx` and `ladder_contact_idx`
    # is one fixed number, so the shipped labels {0,1,2} can only ever address two rungs. Writing
    # absolute indices here and running with `contact_idx: 0` makes the hand target advance one
    # rung per gait cycle -- no change to their code.
    #   0  -> hand free. rewards.py: `rew*in_range + 1 - in_range` pays full marks and
    #         terminations.py never counts a lost contact, so 0 really is "unconstrained".
    #   v>0 -> rung v.  Rung 0 is therefore unaddressable, which costs nothing: the hands lead
    #         the feet by three rungs and never need the bottom one.
    SRC_CONTACT_IDX = 2                      # the base the shipped labels were written against
    src_oc = d["object_contact"]
    n_motion = len(J)
    oc120 = np.zeros((n_motion, 2), dtype=np.int32)
    climb_frames = n_motion - n
    for f in range(min(climb_frames, len(src_f))):
        sv = src_oc[min(int(src_f[f] / a.fps * 50), len(src_oc) - 1)]
        for h in (0, 1):
            oc120[f, h] = 0 if sv[h] == 0 else int(sv[h]) + SRC_CONTACT_IDX + cyc_k[f]
    oc120[climb_frames:] = 0                 # hands release for the mount and stand
    n50 = int(round(n_motion / a.fps * 50))
    oc = oc120[np.clip((np.arange(n50) * a.fps / 50).astype(int), 0, n_motion - 1)]
    hi = int(oc.max())
    print(f"[CONTACT] absolute rung indices, max {hi} -> needs num_rungs >= {hi+1} "
          f"and contact_idx: 0")

    out = {
        "joint_pos": J.astype(np.float64),
        "joint_vel": JV.astype(np.float64),
        "body_pos_w": P.astype(np.float32),
        "body_quat_w": Q.astype(np.float32),
        "body_lin_vel_w": V.astype(np.float32),
        "body_ang_vel_w": W.astype(np.float32),
        "object_contact": oc,
    }
    out = fit_to_loader(out, a.fps)
    np.savez(a.out, **out)
    print(f"[WROTE ] {a.out}")
    print(f"[WROTE ] {len(J)} frames @ {a.fps:g} fps = {len(J)/a.fps:.2f} s; "
          f"object_contact {oc.shape} @50Hz")
    print(f"[WROTE ] pelvis z {P[:,0,2].min():.3f} .. {P[:,0,2].max():.3f} m "
          f"(final {P[-1,0,2]:.3f})")
    print(f"[CHECK ] final pelvis {'CLEARS' if P[-1,0,2] > PLATFORM_TOP + PELVIS_STANCE - 0.15 else 'MISSES'}"
          f" the S02 gate at {PLATFORM_TOP + PELVIS_STANCE - 0.15:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
