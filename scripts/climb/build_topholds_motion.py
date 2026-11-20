"""Re-author a climb clip so both hands only ever hold a LOAD-BEARING TOP FACE.

build_handlegrab_motion.py stays runnable. This replaces its hold model outright, for reasons
that were measured, not guessed:

1. NO STRINGER HOLDS. The termination that polices contact (`hdmi/terminations.py`) counts a
   hand as holding only when the contact force points up, within `max_force_angle_deg` of +z.
   A stringer is a vertical face, so its normal is lateral and it can NEVER satisfy that --
   and the Dex3 fingers are never commanded, so the hand cannot wrap it either. Only a surface
   the hand presses DOWN on counts. The previous build leaned on stringers through the whole
   mid-climb, which is also what put a hand through the rail in the render.

2. The hold set is therefore exactly the six upward faces of the real asset, clustered off the
   mesh (`approx=sdf`, so the render mesh IS the collision surface):
       step1 0.237  step2 0.472  step3 0.707  step4 0.942   platform 1.149   handrail 1.842
   The platform is 0.143 m^2 -- five times a step -- and the handrail is the red top cap.

3. Reach is the FK-measured shoulder-to-PALM figure, `G1_PALM_REACH = 0.419` m, not the
   0.3776 m shoulder-to-wrist-origin used before. That error is why an earlier pass called the
   handrail unreachable on step 3. It is not: the feet reach step 3 at 50 Hz step 450 and the
   handrail comes into both hands' reach at 442.

4. Segments, not per-frame picks. The old schedule re-chose a hold every frame and flipped in
   as little as 4 steps, which is the arm glitching/"fighting" seen in the render. Here a hold
   must last `--min_hold` seconds or it is merged away, and the wrist target eases between
   holds on a raised cosine with an upward arc so the hand lifts over edges instead of
   dragging through them.

5. Labels are valid. LadderMan maps a label to ladder body `label + contact_idx`, and its
   ladder is `rung_0..rung_{N-1}` + 2 rails. With the ported geometry the object list is
   [dummy, step1..step4, platform, handrail] and `contact_idx: 0`, so 0 still means "free"
   and 1..6 name the six faces. The old clips emitted 3..7 and 13..18, which index past the end.

    python scripts/climb/build_topholds_motion.py --base <clip> --out <dir> --mesh <npz>
"""
import argparse
import json
import pathlib
import shutil

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--base", required=True, help="clip dir to re-author (read only)")
ap.add_argument("--out", required=True)
ap.add_argument("--mesh", required=True, help="ladder_mesh.npz, triangles in the ladder frame")
ap.add_argument("--robot", default="dex3", choices=["dex3", "inspire"],
                help="G1 hand variant. The scene default is INSPIRE everywhere -- every task "
                     "in the repo is authored against it and swapped explicitly -- so without "
                     "this the climb work silently ran on the wrong hand. Dex3 is what this "
                     "project uses, so it is the default here.")
ap.add_argument("--envs", type=int, default=64, help="frames solved in parallel")
ap.add_argument("--iters", type=int, default=12, help="IK iterations per batch")
ap.add_argument("--reuse_q", action="store_true",
                help="load q_out_cache.npy from --out and skip the IK pass")
ap.add_argument("--plan_only", action="store_true",
                help="print the hold schedule and exit, before the ~1 h IK pass")
ap.add_argument("--reach", type=float, default=0.4161,
                help="max shoulder-to-CONTACT distance: FK-measured shoulder-to-PALM with the\n                     DEX3 hand and the arm straight. assets.py's G1_PALM_REACH = 0.419 is the\n                     INSPIRE figure.")
ap.add_argument("--palm_off", type=float, default=0.0416,
                help="wrist-to-palm offset, FK-measured on DEX3 (wrist_yaw_link to\n                     hand_palm_link; max wrist reach 0.3746). The "
                     "wrist target sits this far back from the contact point, toward the "
                     "shoulder.")
ap.add_argument("--grip_up", type=float, default=0.025,
                help="floor on how far the wrist stays above the face its palm rests on")
ap.add_argument("--min_hold", type=float, default=0.40, help="seconds; shorter segments merge")
ap.add_argument("--transit", type=float, default=0.40,
                help="seconds the hand spends moving between two holds. Without this the "
                     "wrist target jumps straight from one face to the next -- 0.69 m in a "
                     "single frame for platform->handrail -- which is the arm glitching.")
ap.add_argument("--null_gain", type=float, default=0.35,
                help="how hard the redundant arm swivel is pulled toward the base clip's own "
                     "arm pose. Four arm joints solve a 3-D wrist target, so one degree of "
                     "freedom is free; nothing constrained it, and neighbouring frames settled "
                     "into different elbow configurations and snapped between them. This is "
                     "applied in the NULL SPACE, so it steers the swivel without pulling the "
                     "hand off its hold.")
ap.add_argument("--tie", type=float, default=0.0,
                help="a HIGHER hold wins only when it is no farther than this beyond the "
                     "nearest one. The hand otherwise takes the NEAREST reachable hold, so it "
                     "never reaches PAST a near support to a far one: off the floor that keeps "
                     "both hands on step 4 at 0.33 m instead of stretching over it to the "
                     "platform at 0.35 m, and it hands over to the platform at step 112 and to "
                     "the handrail at 442, where each in turn becomes the nearest.")
ap.add_argument("--clear", type=float, default=0.055,
                help="a moving hand stays at least this far above the ladder silhouette")
ap.add_argument("--ik_tol", type=float, default=0.025,
                help="wrist error above which a frame counts as an IK convergence failure and "
                     "is re-made by interpolating its arm angles from good neighbours. The "
                     "solver starts each frame from the base clip's pose and runs a fixed "
                     "number of Gauss-Newton steps, so isolated frames stall in a bad basin "
                     "while their neighbours land on 0 mm -- and a frame whose arm angles jump "
                     "away from its neighbours IS the visible glitch.")
ap.add_argument("--max_patch", type=int, default=16,
                help="longest run of failed frames that may be interpolated; longer runs are "
                     "left alone, because they are a real reach limit and not a stall")
ap.add_argument("--hold_tol", type=float, default=0.040,
                help="if the hand still ends this far from its target, the clip stops claiming "
                     "a hold there and the label goes to 0")
ap.add_argument("--hold_margin", type=float, default=0.010,
                help="headroom a hold needs before the clip claims it. --reach is the arm's "
                     "ABSOLUTE straight-arm maximum, so without this a hold is taken the "
                     "instant it is theoretically touchable -- measured, the handrail hold "
                     "began with 1.6 mm of headroom and several frames were 3 mm BEYOND the "
                     "arm. The IK cannot make those, so the hand sat short, against the bar.")
ap.add_argument("--park", type=float, default=0.26,
                help="while NEITHER hold is in reach, the hand waits this far out from the "
                     "shoulder, pointed away from the ladder, instead of reaching for a hold "
                     "it cannot get to. Aiming at an unreachable hold leaves the hand short "
                     "ALONG THE RAY to it, and that ray grazes the handrail's front face -- "
                     "which is why every transit target could be 55 mm clear and the hand "
                     "still came within 6 mm.")
ap.add_argument("--rise_frac", type=float, default=0.25,
                help="fraction of a transit spent rising straight up off the old hold")
ap.add_argument("--descend_frac", type=float, default=0.25,
                help="fraction spent descending straight down onto the new one")
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import fiatlux_task.tasks  # noqa: E402,F401
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ARM_JOINTS = ["{s}_shoulder_pitch_joint", "{s}_shoulder_roll_joint",
              "{s}_shoulder_yaw_joint", "{s}_elbow_joint"]

base = pathlib.Path(a.base)
meta = json.loads((base / "meta.json").read_text())
d = dict(np.load(base / "motion.npz"))
fps = float(meta["fps"])
T = d["joint_pos"].shape[0]
file_joints = list(meta["joint_names"])
clip_bodies = list(meta["body_names"])
print(f"[base] {T} frames @ {fps} fps, {len(file_joints)} joints, {len(clip_bodies)} bodies")

# ---- the six top faces, in the clip's own frame -------------------------------------------
# The runtime's mapping (ladderman_runtime._set_frame, rung0_local=(0,-0.3804,0.196)):
#     ladder_x = -clip_y      ladder_y = clip_x + 0.1996      ladder_z = clip_z
# so a ladder-frame point (x,y,z) is clip (y-0.1996, -x, z). Verified against the runtime.
def lad_to_clip(x, y, z):
    return np.array([y - 0.1996, -x, z])


def clip_to_lad(p):
    return np.array([-p[1], p[0] + 0.1996, p[2]])


STEPS = [(0.237, -0.387), (0.472, -0.301), (0.707, -0.216), (0.942, -0.130)]  # (z, ladder y_ctr)
HOLDS = {}
for _k, (_z, _yc) in enumerate(STEPS):
    _a = lad_to_clip(-0.247, _yc - 0.038, _z)
    _b = lad_to_clip(+0.247, _yc + 0.038, _z)
    HOLDS[f"step{_k+1}"] = dict(x=(min(_a[0], _b[0]), max(_a[0], _b[0])),
                                y=(min(_a[1], _b[1]), max(_a[1], _b[1])), z=_z)
HOLDS["platform"] = dict(x=(-0.302, 0.075), y=(-0.208, 0.206), z=1.1494)
HOLDS["handrail"] = dict(x=(-0.030, 0.066), y=(-0.241, 0.241), z=1.8424)
ORDER = ["step1", "step2", "step3", "step4", "platform", "handrail"]   # ascending in z

# ---- how high the ladder stands under any point, so a moving hand can pass OVER it ---------
# A transit used to be a straight line between two holds plus a fixed arc, with no reference
# to the mesh at all -- which is how a moving hand went through the structure. Here every
# transit waypoint is lifted above the real silhouette beneath it.
_TRI = np.load(a.mesh)["T"]                                   # ladder-frame triangles
_TV = _TRI.reshape(-1, 3)
_TC = np.stack([_TV[:, 1] - 0.1996, -_TV[:, 0], _TV[:, 2]], axis=1)   # ... in CLIP frame
_CELL = 0.02
_GLO = _TC[:, :2].min(0) - 0.40
_GHI = _TC[:, :2].max(0) + 0.40
_GN = np.ceil((_GHI - _GLO) / _CELL).astype(int) + 1
_HMAP = np.full(tuple(_GN), -1e9)
_gi = np.clip(((_TC[:, :2] - _GLO) / _CELL).astype(int), 0, _GN - 1)
np.maximum.at(_HMAP, (_gi[:, 0], _gi[:, 1]), _TC[:, 2])
# dilate so a hand passing within `clear` HORIZONTALLY of structure is lifted over it too
_R = int(np.ceil(0.10 / _CELL))
_DIL = _HMAP.copy()
for _dx in range(-_R, _R + 1):
    for _dy in range(-_R, _R + 1):
        if _dx * _dx + _dy * _dy > _R * _R:
            continue
        _DIL = np.maximum(_DIL, np.roll(np.roll(_HMAP, _dx, 0), _dy, 1))
_HMAP = _DIL
print(f"[mesh] silhouette map {_GN[0]}x{_GN[1]} cells at {_CELL*1000:.0f} mm, "
      f"peak {_HMAP.max():.3f} m")

# Take each face's height from the mesh itself, as the TRUE top of its footprint. The nominal
# values above came from clustering upward-facing triangles and taking an area-weighted mean,
# which is biased LOW on every face here: each one has a chamfered edge whose upward-tilted
# triangles pull the mean below the flat top. Measured against the mesh that bias put the palm
# 5 mm (platform) to 18 mm (handrail) INSIDE the solid, which is a hand clipping the ladder.
for _n in ORDER:
    _h = HOLDS[_n]
    _m = ((_TC[:, 0] >= _h["x"][0] - 0.005) & (_TC[:, 0] <= _h["x"][1] + 0.005)
          & (_TC[:, 1] >= _h["y"][0] - 0.005) & (_TC[:, 1] <= _h["y"][1] + 0.005)
          & (np.abs(_TC[:, 2] - _h["z"]) < 0.06))
    if not _m.any():
        raise SystemExit(f"[mesh] no ladder surface under the {_n} footprint -- check the frame")
    _top = float(_TC[_m, 2].max())
    print(f"[mesh] {_n:9s} top {_top:.4f} m (nominal {_h['z']:.4f}, {(_top-_h['z'])*1000:+.1f} mm)")
    _h["z"] = _top


_TLO = np.stack([_TC.reshape(-1, 3, 3)[:, :, i].min(1) for i in range(3)], axis=1)
_THI = np.stack([_TC.reshape(-1, 3, 3)[:, :, i].max(1) for i in range(3)], axis=1)
_TCL = _TC.reshape(-1, 3, 3)                                   # clip-frame triangles


def _exact_on(p, ids):
    t = _TCL[ids]
    A, B, C = t[:, 0], t[:, 1], t[:, 2]
    ab, ac = B - A, C - A
    nn = np.cross(ab, ac)
    den = np.maximum((nn * nn).sum(1), 1e-20)
    ap = p - A
    u = (np.cross(ap, ac) * nn).sum(1) / den
    v = (np.cross(ab, ap) * nn).sum(1) / den
    ins = (u >= 0) & (v >= 0) & (u + v <= 1)
    q = A + u[:, None] * ab + v[:, None] * ac
    best = np.where(ins, ((q - p) ** 2).sum(1), np.inf)
    bq = np.where(ins[:, None], q, 0.0)
    for p0_, ed in ((A, ab), (A, ac), (B, C - B)):
        w_ = np.clip(((p - p0_) * ed).sum(1) / np.maximum((ed * ed).sum(1), 1e-20), 0, 1)
        cd = p0_ + w_[:, None] * ed
        dd = ((cd - p) ** 2).sum(1)
        m_ = dd < best
        best = np.where(m_, dd, best)
        bq = np.where(m_[:, None], cd, bq)
    i_ = int(best.argmin())
    return float(np.sqrt(best[i_])), bq[i_]


def nearest_on_ladder(p, k=3072):
    """Exact nearest ladder point to a CLIP-frame point, as (distance, point).

    A height map can route a hand OVER an obstacle but not AROUND one, which is why the
    transit still slid up the FRONT FACE of the handrail. The prune here is by AABB lower
    bound -- a true lower bound -- and the pruned answer is rechecked against the smallest
    bound left out, so the result is exact. A nearest-CENTROID prune is not, and was measured
    wrong by up to 87 mm on this mesh.
    """
    dx = np.maximum(np.maximum(_TLO[:, 0] - p[0], p[0] - _THI[:, 0]), 0.0)
    dy = np.maximum(np.maximum(_TLO[:, 1] - p[1], p[1] - _THI[:, 1]), 0.0)
    dz = np.maximum(np.maximum(_TLO[:, 2] - p[2], p[2] - _THI[:, 2]), 0.0)
    lb = np.sqrt(dx * dx + dy * dy + dz * dz)
    if lb.size <= k:
        return _exact_on(p, np.arange(lb.size))
    part = np.argpartition(lb, k)
    dist, q = _exact_on(p, part[:k])
    if dist > lb[part[k]]:
        return _exact_on(p, np.arange(lb.size))
    return dist, q


def ladder_top_under(p):
    """Highest ladder surface at this clip-frame x,y (-inf where the ladder is not)."""
    g = np.clip(((p[:2] - _GLO) / _CELL).astype(int), 0, _GN - 1)
    return float(_HMAP[g[0], g[1]])
LABEL = {n: i + 1 for i, n in enumerate(ORDER)}                        # 0 stays "hand free"
for _n in ORDER:
    _h = HOLDS[_n]
    print(f"[hold] {_n:9s} z {_h['z']:.4f}  clip x [{_h['x'][0]:+.3f},{_h['x'][1]:+.3f}]"
          f"  y [{_h['y'][0]:+.3f},{_h['y'][1]:+.3f}]  -> label {LABEL[_n]}")

lw = clip_bodies.index("left_wrist_yaw_link")
rw = clip_bodies.index("right_wrist_yaw_link")
pel = clip_bodies.index("pelvis")
sh = [clip_bodies.index("left_shoulder_pitch_link"), clip_bodies.index("right_shoulder_pitch_link")]
y_left = float(np.median(d["body_pos_w"][:, lw, 1]))
y_right = float(np.median(d["body_pos_w"][:, rw, 1]))
LEFT_SIGN = 1.0 if y_left > y_right else -1.0
print(f"[side] robot-left is {'+y' if LEFT_SIGN > 0 else '-y'} in the clip frame")


def contact_on(name, shoulder, is_left):
    """The point on this face the PALM rests on, nearest to this shoulder.

    The handrail is one bar across the full width, so each hand is kept on its own half;
    the other faces are wide enough that clamping the shoulder's own x,y is sufficient.
    """
    h = HOLDS[name]
    ylo, yhi = h["y"]
    if name == "handrail":
        out = LEFT_SIGN if is_left else -LEFT_SIGN
        ylo, yhi = (0.03, yhi) if out > 0 else (ylo, -0.03)
    return np.array([float(np.clip(shoulder[0], *h["x"])),
                     float(np.clip(shoulder[1], ylo, yhi)),
                     h["z"]])


def target_on(name, shoulder, is_left):
    """Where the WRIST goes so the palm lands on that contact point.

    The palm sticks out past the wrist, so the wrist sits one palm-length back along the line
    to the shoulder -- NOT straight up off the face. Lifting straight up was wrong twice over:
    it asked the IK for points up to 0.44 m from the shoulder when the wrist reaches only
    0.3776 m (the arm then solves straight, with the hand short of the hold -- holding air),
    and on a far, low reach it drove the wrist down through the structure instead of resting
    the hand on top of it.

    Placed this way, |wrist - shoulder| = |contact - shoulder| - PALM_OFF, so gating the
    CONTACT at G1_PALM_REACH keeps every wrist target inside the wrist's own reach.
    """
    c = contact_on(name, shoulder, is_left)
    v = shoulder - c
    nv = float(np.linalg.norm(v))
    w = c + (v / nv) * a.palm_off if nv > 1e-6 else c + np.array([0.0, 0.0, a.palm_off])
    w[2] = max(w[2], HOLDS[name]["z"] + a.grip_up)      # never below the face it rests on
    # Never ask the solver for a point outside the arm. Measured, the handrail hold opened
    # with targets up to 3 mm BEYOND the wrist's maximum: the arm then straightens toward the
    # bar and the hand ends up short, resting against it. Clamped, the target sits just in
    # front of the face instead, which is where the hand can actually be.
    vw = w - shoulder
    nw = float(np.linalg.norm(vw))
    lim = a.reach - a.palm_off
    if nw > lim:
        w = shoulder + vw / nw * lim
    return w


def reach_to(name, shoulder, is_left):
    return float(np.linalg.norm(contact_on(name, shoulder, is_left) - shoulder))


# ---- per-hand schedule: the HIGHEST reachable face, then merged into real segments ---------
raw = [[None, None] for _ in range(T)]
for i in range(T):
    for c in (0, 1):
        s = d["body_pos_w"][i, sh[c]]
        ok = [(n, reach_to(n, s, c == 0)) for n in ORDER]
        ok = [(n, r) for n, r in ok if r <= a.reach - a.hold_margin]
        if not ok:
            raw[i][c] = None
        else:
            near = min(r for _, r in ok)
            raw[i][c] = [n for n, r in ok if r <= near + a.tie][-1]

MINF = max(1, int(a.min_hold * fps))


def segments(col):
    out, cur, st = [], raw[0][col], 0
    for i in range(1, T):
        if raw[i][col] != cur:
            out.append([st, i, cur]); cur = raw[i][col]; st = i
    out.append([st, T, cur])
    # drop runs shorter than --min_hold by extending the previous segment over them
    merged = []
    for s in out:
        if merged and (s[1] - s[0]) < MINF:
            merged[-1][1] = s[1]
        elif merged and merged[-1][2] == s[2]:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    # the merge above extends the PREVIOUS segment, so a too-short FIRST one survives it;
    # hand it to its successor instead, or the clip opens with a 0.12 s grab and a swing away
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < MINF:
        merged[0][2] = None
    # a merge can leave two neighbours with the same hold; coalesce again
    coal = []
    for s in merged:
        if coal and coal[-1][2] == s[2]:
            coal[-1][1] = s[1]
        else:
            coal.append(s)
    return coal


def add_transits(sl_):
    """Carve a transit window out of every boundary between two DIFFERENT holds.

    A hold change is a real arm movement, not an instant one: platform -> handrail moves the
    wrist 0.69 m. Without a window here the target steps across that in one frame and the arm
    snaps, which is what the previous build's render showed.
    """
    h = max(1, int(a.transit * fps / 2))
    out_ = []
    for k, (s, e, v) in enumerate(sl_):
        s2, e2 = s, e
        if k > 0 and sl_[k - 1][2] is not None and v is not None and sl_[k - 1][2] != v:
            s2 = s + h
        if k + 1 < len(sl_) and sl_[k + 1][2] is not None and v is not None and sl_[k + 1][2] != v:
            e2 = e - h
        if s2 >= e2:                                   # segment too short to carve; leave it
            s2, e2 = s, e
        if out_ and out_[-1][1] < s2:
            out_.append([out_[-1][1], s2, None])       # the transit itself
        out_.append([s2, e2, v])
    return out_


segs = [add_transits(segments(0)), add_transits(segments(1))]
for c, lab in ((0, "L"), (1, "R")):
    print(f"[plan] {lab}: " + "  ".join(
        f"{s/fps*50:.0f}-{e/fps*50:.0f}:{v or 'transit'}" for s, e, v in segs[c]))
both_free = sum(1 for i in range(T)
                if all(next(v for s, e, v in segs[c] if s <= i < e) is None for c in (0, 1)))
print(f"[plan] frames with BOTH hands in transit: {both_free}/{T} = {both_free/T*100:.1f}%")

# ---- wrist target per frame, easing across transits ----------------------------------------
tgt = np.zeros((T, 2, 3))
lab_f = np.zeros((T, 2), dtype=int)
transit_overreach = [0, 0]
transit_minclear = [1e9]
for c in range(2):
    for k, (s, e, v) in enumerate(segs[c]):
        if v is not None:
            for i in range(s, e):
                tgt[i, c] = target_on(v, d["body_pos_w"][i, sh[c]], c == 0)
                lab_f[i, c] = LABEL[v]
        else:
            # ease from where the last hold left the hand to where the next one takes it,
            # arcing up so the palm clears the edge it is leaving instead of dragging through
            prev = segs[c][k - 1][2] if k > 0 else None
            nxt = segs[c][k + 1][2] if k + 1 < len(segs[c]) else None
            # RISE, TRAVERSE, DESCEND -- the ordinary approach path, and the only shape that
            # cannot bring a hand at a face from below or from the side. The previous version
            # blended straight from hold to hold, lifted over the silhouette with a taper that
            # went to ZERO at both ends, and then clamped to arm reach. Two faults: the clamp
            # rescales the whole vector from the shoulder, so it pulled the point back DOWN and
            # INTO the ladder, undoing the lift; and the taper removed the lift exactly where
            # the hand was approaching the handrail, so it sank 56 mm below the bar and came up
            # into it from underneath. Here the horizontal move happens at a safe height, and
            # the hand only ever descends onto a face from directly above it.
            def ends(i):
                sp_ = d["body_pos_w"][i, sh[c]]
                q0 = target_on(prev, sp_, c == 0) if prev else d["body_pos_w"][i, lw if c == 0 else rw]
                q1 = target_on(nxt, sp_, c == 0) if nxt else q0
                return sp_, q0, q1

            def park_at(i):
                """A clear, reachable place to wait: straight out from the shoulder, away
                from the ladder."""
                sp_ = d["body_pos_w"][i, sh[c]]
                dd_, qq_ = nearest_on_ladder(sp_)
                v_ = sp_ - qq_
                nv_ = float(np.linalg.norm(v_))
                v_ = v_ / nv_ if nv_ > 1e-6 else np.array([0.0, 0.0, 1.0])
                return sp_ + v_ * a.park

            # Start the approach only once the destination is actually reachable; wait parked
            # until then. The gap between two holds can be far longer than one arm movement --
            # the platform goes out of reach at step 362 and the handrail only comes in at 437 --
            # and reaching across that whole gap is what dragged the hand along the bar.
            # The approach occupies the LAST --transit seconds of the gap and the hand waits
            # before that. It cannot be deferred until the destination is reachable: the hold
            # segment begins exactly when it becomes reachable, so keying off that collapsed
            # the approach to a single frame and the hand teleported onto the bar.
            ap_s = s if nxt is None else max(s, e - int(a.transit * fps))
            parked = ap_s > s + 2

            _, e0, e1 = ends(s)
            _, f0, f1 = ends(e - 1)
            if parked:
                e0 = park_at(ap_s)                   # the move now starts from the park point
            # one safe height for the whole transit, from the tallest thing under the traverse
            zs = max(e0[2], e1[2], f0[2], f1[2])
            for t_ in np.linspace(0.0, 1.0, 25):
                for (g0, g1) in ((e0, e1), (f0, f1)):
                    zs = max(zs, ladder_top_under((1 - t_) * g0 + t_ * g1) + a.clear)
            uA, uB = a.rise_frac, 1.0 - a.descend_frac
            R = a.reach - a.palm_off
            path = np.zeros((e - s, 3))
            holds_xy = np.zeros(e - s, dtype=bool)              # True where xy must not be moved
            for i in range(s, e):
                sp, p0, p1 = ends(i)
                if parked and i < ap_s:
                    # ease off the old hold, then simply wait, clear of the ladder
                    pk = park_at(i)
                    w = min(1.0, (i - s) / max(1.0, 0.30 * fps))
                    w = 0.5 - 0.5 * np.cos(np.pi * w)
                    path[i - s] = (1 - w) * p0 + w * pk
                    holds_xy[i - s] = False
                    continue
                if parked:
                    p0 = park_at(ap_s)
                lo_, hi_ = (ap_s, e) if parked else (s, e)
                u = 0.0 if hi_ - lo_ <= 1 else (i - lo_) / (hi_ - lo_ - 1)
                if u < uA:                                     # rise, straight up off the hold
                    w = 0.5 - 0.5 * np.cos(np.pi * (u / uA))
                    p = np.array([p0[0], p0[1], p0[2] + (zs - p0[2]) * w])
                    holds_xy[i - s] = True
                elif u < uB:                                   # traverse, at the safe height
                    w = 0.5 - 0.5 * np.cos(np.pi * ((u - uA) / (uB - uA)))
                    p = np.array([(1 - w) * p0[0] + w * p1[0],
                                  (1 - w) * p0[1] + w * p1[1], zs])
                else:                                          # descend, straight onto the hold
                    w = 0.5 - 0.5 * np.cos(np.pi * ((u - uB) / (1.0 - uB)))
                    p = np.array([p1[0], p1[1], zs + (p1[2] - zs) * w])
                    holds_xy[i - s] = True
                path[i - s] = p
            if len(path) > 4:                                  # round the two corners
                ker = np.ones(5) / 5.0
                for ax in range(3):
                    pad = np.r_[[path[0, ax]] * 2, path[:, ax], [path[-1, ax]] * 2]
                    path[:, ax] = np.convolve(pad, ker, mode="valid")
            # Reach is enforced HORIZONTALLY, leaving z alone, so it can no longer undo the
            # clearance. The floor is re-applied after, and only across the traverse: a hand
            # descending onto a face must be allowed below the safe height, and directly above
            # a top face is free space anyway.
            for it_ in range(3):
                for i in range(s, e):
                    sp = d["body_pos_w"][i, sh[c]]
                    k = i - s
                    lo_, hi_ = (ap_s, e) if parked else (s, e)
                    u = 1.0 if i < lo_ else (0.0 if hi_ - lo_ <= 1 else (i - lo_) / (hi_ - lo_ - 1))
                    if (not (parked and i < ap_s)) and uA <= u < uB:
                        need = ladder_top_under(path[k]) + a.clear
                        if need > path[k, 2]:
                            path[k, 2] = need
                    dz = path[k, 2] - sp[2]
                    room = float(np.sqrt(max(R * R - dz * dz, 0.0)))
                    vh = path[k, :2] - sp[:2]
                    nh = float(np.linalg.norm(vh))
                    if nh > room and not holds_xy[k]:
                        path[k, :2] = sp[:2] + vh / max(nh, 1e-9) * room
                    # Last and highest priority: stand the target off the REAL surface, on the
                    # robot's side of it. A target out of reach only leaves the hand short; a
                    # target against the ladder drags it along the metal, which is how the hand
                    # came up the front of the handrail even with a correct over-the-top route.
                    dmin, qn = nearest_on_ladder(path[k])
                    if dmin < a.clear:
                        away = sp - qn
                        na = float(np.linalg.norm(away))
                        if na > 1e-6:
                            path[k] = qn + away / na * a.clear
            for i in range(s, e):
                tgt[i, c] = path[i - s]
                lab_f[i, c] = 0
            for i in range(s, e):
                dmin, _ = nearest_on_ladder(path[i - s])
                transit_minclear[0] = min(transit_minclear[0], dmin)
            n_bad = sum(1 for i in range(s, e)
                        if float(np.linalg.norm(path[i - s] - d["body_pos_w"][i, sh[c]])) > R + 1e-6)
            if n_bad:
                transit_overreach[c] += n_bad

print(f"[plan] closest any TRANSIT target comes to the ladder: "
      f"{transit_minclear[0]*1000:.1f} mm (asked for {a.clear*1000:.0f})")
if any(transit_overreach):
    print(f"[plan] transit frames whose safe height is past arm reach: "
          f"L {transit_overreach[0]}, R {transit_overreach[1]} -- the hand is left short of "
          f"the target there, which keeps it ABOVE the ladder rather than inside it")

if a.plan_only:
    import sys as _sys
    print("[plan] --plan_only: stopping before IK")
    _sys.exit(0)

# ---- Isaac: numerical IK against real FK ---------------------------------------------------
set_layout_seed(0)
cfg = parse_env_cfg(a.task, num_envs=a.envs)
if a.robot != "inspire":
    from fiatlux_task.robots.g1 import swap_robot_variant
    swap_robot_variant(cfg, a.robot)
print(f"[robot] {a.robot}: {cfg.scene.robot.spawn.usd_path.split('/')[-1]}")
env = gym.make(a.task, cfg=cfg).unwrapped
env.reset()
rob = env.scene["robot"]
origins = env.scene.env_origins
bn = list(rob.body_names)
jn = list(rob.joint_names)
dev = env.device
N = a.envs

j_src = [file_joints.index(n) for n in jn if n in file_joints]
j_dst = torch.tensor([jn.index(n) for n in jn if n in file_joints], device=dev)
print(f"[map ] {len(j_dst)} of {len(jn)} robot joints driven by the clip")

arm_ids, wrist_ids = {}, {}
for s in ("left", "right"):
    arm_ids[s] = torch.tensor([jn.index(t.format(s=s)) for t in ARM_JOINTS], device=dev)
    wrist_ids[s] = bn.index(f"{s}_wrist_yaw_link")

q_out = d["joint_pos"].copy()


def fk_refresh():
    env.scene.write_data_to_sim()
    env.sim.render()


print(f"[ik  ] solving {T} frames in batches of {N}, {a.iters} iterations each", flush=True)
err_hist = []
ik_err = np.zeros((T, 2))
_cache = pathlib.Path(a.out) / "q_out_cache.npy"
_reuse = a.reuse_q and _cache.exists()
if _reuse:
    q_out = np.load(_cache)
    _ec = _cache.with_name("ik_err_cache.npy")
    if _ec.exists():
        ik_err = np.load(_ec)
    print(f"[ik  ] reusing solved angles from {_cache}", flush=True)
for b0 in ([] if _reuse else range(0, T, N)):
    fr = np.arange(b0, min(b0 + N, T))
    n = len(fr)
    qf = rob.data.default_joint_pos.clone()
    qf[:n][:, j_dst] = torch.as_tensor(d["joint_pos"][fr][:, j_src], dtype=torch.float32, device=dev)
    root = torch.zeros(N, 7, device=dev); root[:, 3] = 1.0
    root[:n, :3] = torch.as_tensor(d["body_pos_w"][fr, pel], dtype=torch.float32, device=dev) + origins[:n]
    root[:n, 3:] = torch.as_tensor(d["body_quat_w"][fr, pel], dtype=torch.float32, device=dev)
    rob.write_root_pose_to_sim(root)
    rob.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))

    for s, col in (("left", 0), ("right", 1)):
        tt = torch.zeros(N, 3, device=dev)
        tt[:n] = torch.as_tensor(tgt[fr, col], dtype=torch.float32, device=dev)
        act = torch.zeros(N, dtype=torch.bool, device=dev); act[:n] = True
        ids = arm_ids[s]
        ref_cols = [file_joints.index(t.format(s=s)) for t in ARM_JOINTS]
        qref = torch.zeros(N, len(ids), device=dev)
        qref[:n] = torch.as_tensor(d["joint_pos"][fr][:, ref_cols], dtype=torch.float32, device=dev)
        I3 = torch.eye(3, device=dev)
        Ij = torch.eye(len(ids), device=dev)
        for _ in range(a.iters):
            rob.write_joint_state_to_sim(qf, torch.zeros_like(qf)); fk_refresh()
            w = rob.data.body_pos_w[:, wrist_ids[s]] - origins
            e = tt - w
            J = torch.zeros(N, 3, len(ids), device=dev)
            for c in range(len(ids)):
                qp = qf.clone(); qp[:, ids[c]] += 1e-3
                rob.write_joint_state_to_sim(qp, torch.zeros_like(qp)); fk_refresh()
                J[:, :, c] = ((rob.data.body_pos_w[:, wrist_ids[s]] - origins) - w) / 1e-3
            JT = J.transpose(1, 2)
            Jp = JT @ torch.linalg.inv(J @ JT + 1e-4 * I3)       # damped pseudo-inverse
            prim = (Jp @ e.unsqueeze(-1)).squeeze(-1)            # reach the target
            null = Ij - Jp @ J                                   # ... without disturbing it
            sec = (null @ (qref - qf[:, ids]).unsqueeze(-1)).squeeze(-1)
            dq = prim + a.null_gain * sec
            dq = dq.clamp(-0.25, 0.25) * act.float().unsqueeze(-1)
            qf[:, ids] += dq
            qf[:, ids] = qf[:, ids].clamp(-2.6, 2.6)
        rob.write_joint_state_to_sim(qf, torch.zeros_like(qf)); fk_refresh()
        fin = (tt - (rob.data.body_pos_w[:, wrist_ids[s]] - origins)).norm(dim=-1)
        err_hist.append(float(fin[:n].mean()))
        ik_err[fr, col] = fin[:n].cpu().numpy()

    for s in ("left", "right"):
        for t in ARM_JOINTS:
            nm = t.format(s=s)
            q_out[fr, file_joints.index(nm)] = qf[:n, jn.index(nm)].cpu().numpy()
    if b0 % (N * 4) == 0:
        print(f"[ik  ] frames {b0:4d}-{fr[-1]:4d}  mean wrist error "
              f"{np.mean(err_hist[-2:])*1000:6.1f} mm", flush=True)

if not _reuse:
    print(f"[ik  ] done. overall mean wrist error {np.mean(err_hist)*1000:.1f} mm", flush=True)
    _cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(_cache, q_out)
    np.save(_cache.with_name("ik_err_cache.npy"), ik_err)
    print(f"[ik  ] cached -> {_cache}", flush=True)

# ---- repair frames the solver stalled on ----------------------------------------------------
arm_cols = {s_: [file_joints.index(t.format(s=s_)) for t in ARM_JOINTS]
            for s_ in ("left", "right")}
patched = dropped_runs = 0
for col, s_ in ((0, "left"), (1, "right")):
    badf = ik_err[:, col] > a.ik_tol
    i = 0
    while i < T:
        if not badf[i]:
            i += 1; continue
        j = i
        while j < T and badf[j]:
            j += 1
        if i > 0 and j < T and (j - i) <= a.max_patch:
            for cjt in arm_cols[s_]:
                q_out[i:j, cjt] = np.interp(np.arange(i, j), [i - 1, j],
                                            [q_out[i - 1, cjt], q_out[j, cjt]])
            patched += j - i
        else:
            dropped_runs += 1
        i = j
print(f"[fix ] frames over {a.ik_tol*1000:.0f} mm: "
      f"{int((ik_err > a.ik_tol).sum())} of {2*T}; interpolated {patched}, "
      f"left {dropped_runs} run(s) alone as a real reach limit")

# ---- rebuild body tracks from the solved pose ----------------------------------------------
bp = d["body_pos_w"].copy()
bq = d["body_quat_w"].copy()
clip_in_rob = [bn.index(b) for b in clip_bodies if b in bn]
clip_rows = [i for i, b in enumerate(clip_bodies) if b in bn]
for b0 in range(0, T, N):
    fr = np.arange(b0, min(b0 + N, T)); n = len(fr)
    qf = rob.data.default_joint_pos.clone()
    qf[:n][:, j_dst] = torch.as_tensor(q_out[fr][:, j_src], dtype=torch.float32, device=dev)
    root = torch.zeros(N, 7, device=dev); root[:, 3] = 1.0
    root[:n, :3] = torch.as_tensor(d["body_pos_w"][fr, pel], dtype=torch.float32, device=dev) + origins[:n]
    root[:n, 3:] = torch.as_tensor(d["body_quat_w"][fr, pel], dtype=torch.float32, device=dev)
    rob.write_root_pose_to_sim(root)
    rob.write_joint_state_to_sim(qf, torch.zeros_like(qf)); fk_refresh()
    P = (rob.data.body_pos_w - origins[:, None, :]).cpu().numpy()
    Q = rob.data.body_quat_w.cpu().numpy()
    bp[np.ix_(fr, clip_rows)] = P[:n][:, clip_in_rob]
    bq[np.ix_(fr, clip_rows)] = Q[:n][:, clip_in_rob]

bv = np.gradient(bp, 1.0 / fps, axis=0)
jv = np.gradient(q_out, 1.0 / fps, axis=0)

# ---- contact labels at 50 Hz, downsampled from the 120 Hz schedule --------------------------
oc50 = d["object_contact"]
n50 = oc50.shape[0]
ach = np.linalg.norm(bp[:, [lw, rw], :] - tgt, axis=-1)        # [T,2] what the hand reached
miss = (lab_f > 0) & (ach > a.hold_tol)
lab_f = np.where(miss, 0, lab_f)
print(f"[fix ] hand-frames still over {a.hold_tol*1000:.0f} mm from their target: "
      f"{int(miss.sum())} -- label dropped to 0 there, so the clip never claims a hold the "
      f"hand did not make")
oc_new = np.zeros_like(oc50)
for i in range(n50):
    f = min(int(i / 50 * fps), T - 1)
    oc_new[i] = lab_f[f]

out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
np.savez(out / "motion.npz", joint_pos=q_out, joint_vel=jv, body_pos_w=bp, body_quat_w=bq,
         body_lin_vel_w=bv, body_ang_vel_w=d["body_ang_vel_w"], object_contact=oc_new)
shutil.copy(base / "meta.json", out / "meta.json")
(out / "holds.json").write_text(json.dumps(
    dict(holds=HOLDS, label=LABEL, contact_idx=0,
         objects=["dummy"] + ORDER, reach=a.reach, palm_off=a.palm_off,
         grip_up=a.grip_up), indent=1))

# ---- diagnostics ----------------------------------------------------------------------------
print(f"[out ] {out/'motion.npz'}")
print(f"[out ] pelvis {bp[0, pel, 2]:.3f} -> {bp[-1, pel, 2]:.3f} m")
uniq = sorted(set(int(v) for v in np.unique(oc_new)))
inv = {v: k for k, v in LABEL.items()}
print(f"[out ] contact labels used: {uniq} -> "
      + ", ".join(f"{v}:{inv.get(v, 'free')}" for v in uniq))
print(f"[out ] max label {max(uniq)}; valid range for the ported ladder is 0..{len(ORDER)}")
for v in uniq:
    if v == 0:
        continue
    c = int((oc_new == v).any(axis=1).sum())
    print(f"[out ] steps with a hand on {inv[v]:9s}: {c:4d}/{n50}")
c0 = int((oc_new == 0).all(axis=1).sum())
print(f"[out ] steps with BOTH hands in transit: {c0:4d}/{n50}")
# a hand must never end up below the face it is holding -- that is the "through the rail" bug
bad = 0
for i in range(n50):
    f = min(int(i / 50 * fps), T - 1)
    for c, wi in ((0, lw), (1, rw)):
        v = int(oc_new[i, c])
        if v and bp[f, wi, 2] < HOLDS[inv[v]]["z"] - 0.01:
            bad += 1
print(f"[out ] hand-frames BELOW the face they hold: {bad} (must be 0)")
env.close(); app.close()
