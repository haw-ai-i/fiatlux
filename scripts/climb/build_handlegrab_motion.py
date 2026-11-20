"""Re-author a climb clip so the hands take REAL holds, including the top handle.

build_railgrip_motion.py stays runnable. This fixes three things in it, all found by measuring
the placed clip IN THE SIM (measure_clip_clearance.py) rather than through a fitted offline
transform:

1. Its rail model came from the incline formula `BASE + dir*s` at 70.3 deg. The mesh says the
   stringer runs nearer 67 deg with a different offset, so the modelled rail was up to 140 mm
   from the real one near the top -- matching the 159 mm the right hand actually sits off the
   ladder at tread 3. Here every rail target is SNAPPED to the real mesh.
2. `--reach` was 0.34 against a true arm of 0.3776 m, so reachable holds were dropped and the
   hand released to whatever the base clip happened to be doing.
3. Rail holds were labelled `num_rungs + 2*k + side`, reaching 18. LadderMan's command manager
   indexes rungs with that and the training env has 8. Here a hold takes the NEAREST RUNG BY
   HEIGHT, clamped to the rung count; 0 still means "hand free", as the clip already uses.

Measured on the base clip, in sim, with the ladder's own pose: both hands are over 50 mm off the
ladder in 96.2% of frames (median 75 mm left, 108 mm right). The top cap -- the red handle -- is
a 0.494 m bar at z 1.861 and is the only load-bearing feature up there. It comes within true arm
reach at 50 Hz step ~452 (pelvis 1.42 m), late in the tread-3 phase.

    python scripts/climb/build_handlegrab_motion.py --base <clip> --out <dir> --holds <json>
"""
import argparse
import json
import math
import pathlib
import shutil

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--base", required=True, help="clip dir to re-author (read only)")
ap.add_argument("--out", required=True)
ap.add_argument("--envs", type=int, default=64, help="frames solved in parallel")
ap.add_argument("--reuse_q", action="store_true",
                help="load q_out_cache.npy from --out and skip the IK pass")
ap.add_argument("--iters", type=int, default=12, help="IK iterations per batch")
ap.add_argument("--num_rungs", type=int, default=5)
ap.add_argument("--rail_segments", type=int, default=6)
ap.add_argument("--rail_offset", type=float, default=0.225)
ap.add_argument("--rail_z_lo", type=float, default=0.25)
ap.add_argument("--rail_z_hi", type=float, default=1.35)
ap.add_argument("--hand_back", type=float, default=0.060, help="wrist this far back from the rail")
ap.add_argument("--hand_up", type=float, default=0.020)
ap.add_argument("--plan_only", action="store_true",
                help="print the hold schedule and exit, before the ~1 h IK pass")
ap.add_argument("--holds", required=True,
                help="holds_clip.json: the cap bar and rail points taken off the real mesh")
ap.add_argument("--mesh", required=True, help="ladder_mesh.npz, triangles in the ladder frame")
ap.add_argument("--grip_up", type=float, default=0.035,
                help="wrist this far above a bar's top surface when the hand wraps it")
ap.add_argument("--reach", type=float, default=0.365,
                help="max shoulder-to-wrist-target distance. MEASURED max is 0.3776 m "
                     "(upper 0.1934 + fore 0.1842); the old default of 0.34 dropped holds that "
                     "were in fact reachable, which is how the previous build ended up with a "
                     "hand 226 mm from the ladder.")
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

INCL, SPACING, BASE, RUNG_H = 70.3, 0.2492, (-0.58, 0.0, 0.196), 0.080
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

# ---- rail segment geometry, in the clip's own frame (rung 0 at BASE) -----------------------
ang = math.radians(INCL)
dirv = np.array([math.cos(ang), 0.0, math.sin(ang)])
s_lo = (a.rail_z_lo - BASE[2]) / math.sin(ang)
s_hi = (a.rail_z_hi - BASE[2]) / math.sin(ang)
seg_len = (s_hi - s_lo) / a.rail_segments
seg_c = np.array([np.asarray(BASE) + dirv * (s_lo + (k + 0.5) * seg_len)
                  for k in range(a.rail_segments)])          # [K,3] centreline
seg_z = seg_c[:, 2]
print("[rail] segment top-z: " + ", ".join(f"{z:.3f}" for z in seg_z))

# ---- which side is which, read off the clip --------------------------------------------
lw = clip_bodies.index("left_wrist_yaw_link")
rw = clip_bodies.index("right_wrist_yaw_link")
pel = clip_bodies.index("pelvis")
y_left = float(np.median(d["body_pos_w"][:, lw, 1]))
y_right = float(np.median(d["body_pos_w"][:, rw, 1]))
LEFT_SIGN = 1.0 if y_left > y_right else -1.0
print(f"[side] left wrist median y {y_left:+.3f}, right {y_right:+.3f} -> robot-left is "
      f"{'+y' if LEFT_SIGN > 0 else '-y'}")
# robot stands on the low-x side of rung 0 and climbs toward +x
x_pel0 = float(d["body_pos_w"][0, pel, 0])
BACK = -1.0 if x_pel0 < BASE[0] else 1.0
print(f"[side] pelvis x at frame 0 {x_pel0:+.3f} vs rung0 x {BASE[0]:+.3f} -> 'back' is "
      f"{'-x' if BACK < 0 else '+x'}")

# ---- per-frame hand plan, at the clip's own 120 Hz --------------------------------------
oc50 = d["object_contact"]                      # [939,2] at 50 Hz
n50 = oc50.shape[0]
idx50 = np.clip((np.arange(T) / fps * 50).astype(int), 0, n50 - 1)
oc120 = oc50[idx50]                             # old labels lifted to 120 Hz

wz_l = d["body_pos_w"][:, lw, 2]
wz_r = d["body_pos_w"][:, rw, 2]
pelz = d["body_pos_w"][:, pel, 2]

NR = a.num_rungs
REACH = a.reach
TREAD0_Z, TREAD_RISE = 0.236, 0.2347          # measured off the asset
PLATFORM_IDX = NR - 1                         # the top tread / platform

# ---- real holds, taken off the ladder mesh ------------------------------------------------
# The clip frame and the ladder's own frame are related by the mapping the RUNTIME uses
# (ladderman_runtime._set_frame, rung0_local=(0,-0.3804,0.196), climb_dir_local=(0,1,0)):
#     ladder_x = -clip_y      ladder_y = clip_x + 0.1996      ladder_z = clip_z
# Verified: clip rung 0 (-0.58, 0, 0.196) -> ladder (0, -0.3804, 0.196), the tread's true
# lateral CENTRE. An earlier offline fit put it at ladder x +0.2442, the tread's outer edge on
# the stringer, and every clearance number computed through it was wrong.
HOLDS = json.loads(pathlib.Path(a.holds).read_text())
_TRI = np.load(a.mesh)["T"]                                   # ladder-local triangles
_CENT = _TRI.mean(1)
_NRM = np.cross(_TRI[:, 1] - _TRI[:, 0], _TRI[:, 2] - _TRI[:, 0])
_NRM /= np.maximum(np.linalg.norm(_NRM, axis=1, keepdims=True), 1e-12)


def clip_to_lad(p):
    p = np.asarray(p, dtype=float)
    return np.array([-p[1], p[0] + 0.1996, p[2]])


def lad_to_clip(p):
    p = np.asarray(p, dtype=float)
    return np.array([p[1] - 0.1996, -p[0], p[2]])


def rot_lad_to_clip(v):
    """The ROTATION part of lad_to_clip, for directions such as surface normals."""
    v = np.asarray(v, dtype=float)
    return np.array([v[1], -v[0], v[2]])


def _closest_on_tris(p, ids):
    t = _TRI[ids]
    A, B, C = t[:, 0], t[:, 1], t[:, 2]
    AB, AC = B - A, C - A
    nn = np.cross(AB, AC)
    den = np.maximum((nn * nn).sum(1), 1e-20)
    ap = p - A
    u = (np.cross(ap, AC) * nn).sum(1) / den
    v = (np.cross(AB, ap) * nn).sum(1) / den
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    q = A + u[:, None] * AB + v[:, None] * AC
    best = np.where(inside, ((q - p) ** 2).sum(1), np.inf)
    bq = np.where(inside[:, None], q, np.full_like(q, np.nan))
    for P0, E in ((A, AB), (A, AC), (B, C - B)):
        s = np.clip(((p - P0) * E).sum(1) / np.maximum((E * E).sum(1), 1e-20), 0, 1)
        cand = P0 + s[:, None] * E
        dd = ((cand - p) ** 2).sum(1)
        m = dd < best
        best = np.where(m, dd, best)
        bq = np.where(m[:, None], cand, bq)
    k = int(np.argmin(best))
    return float(np.sqrt(best[k])), bq[k], _NRM[ids[k]]


# Centroid pruning is NOT safe on this mesh -- the stringers are long thin triangles whose
# centroids sit far from a surface that is close, and a k-nearest-centroid query was measured
# wrong by up to 87 mm. Snapping happens a few hundred times, so just use every triangle.
_ALL = np.arange(_TRI.shape[0])


def snap(p_clip, out_sign):
    """Nearest real ladder surface to a clip-frame point, with the wrist standing off it.

    `out_sign` is the clip-y direction that points AWAY from the ladder's centreline for this
    hand, so the wrist ends up outside a stringer and above the cap rather than inside the metal.
    """
    d, q_lad, n_lad = _closest_on_tris(clip_to_lad(p_clip), _ALL)
    q = lad_to_clip(q_lad)
    n = rot_lad_to_clip(n_lad)                            # a direction: rotate, do NOT translate
    n = n / max(np.linalg.norm(n), 1e-9)
    if abs(n[2]) >= 0.5:
        n = n if n[2] > 0 else -n                         # a top face: stand the wrist above it
    else:
        if np.sign(n[1]) != np.sign(out_sign):            # a side face: stand it outside
            n = -n
    return d, q + n * a.grip_up


# shoulder_pitch_link's ORIGIN is the joint itself, parented to the torso, so it does not move
# when the arm angles are re-solved -- a valid reach origin even though the arms are exactly what
# this script rewrites.
sh_l = clip_bodies.index("left_shoulder_pitch_link")
sh_r = clip_bodies.index("right_shoulder_pitch_link")

CAP = HOLDS["cap"]


def cap_target(shoulder, out_sign):
    """Closest grip on the real top bar to this shoulder, kept on this hand's half."""
    xlo, xhi = CAP["clip_x"]
    ylo, yhi = (0.03, CAP["clip_y"][1]) if out_sign > 0 else (CAP["clip_y"][0], -0.03)
    x = float(np.clip(shoulder[0], xlo, xhi))
    y = float(np.clip(shoulder[1], ylo, yhi))
    return np.array([x, y, CAP["z"] + a.grip_up])


def rail_target(shoulder, out_sign):
    """A grip on the stringer at the shoulder's own height, snapped to the real mesh."""
    z = float(np.clip(shoulder[2], TREAD0_Z, CAP["z"] - 0.05))
    seed = np.array([shoulder[0], out_sign * 0.226, z])
    d, t = snap(seed, out_sign)
    return (t if d < 0.25 else None)


def rung_label(z):
    """Nearest real tread by height, 1-based so that 0 keeps meaning 'hand free'."""
    k = int(round((z - TREAD0_Z) / TREAD_RISE)) + 1
    return int(np.clip(k, 1, NR - 1))


def plan(old_label, shoulder, is_left):
    """-> (contact label, wrist xyz) or (0, None) when no real hold is in reach.

    The handle wins whenever it is reachable: it is the only load-bearing feature in the upper
    climb (horizontal, |normal_z| ~ 1), where every stringer is a vertical face that an open
    hand cannot hold. Below that, the stringer at the shoulder's own height.
    """
    if old_label == 0:
        return 0, None
    out_sign = LEFT_SIGN if is_left else -LEFT_SIGN
    cap = cap_target(shoulder, out_sign)
    if np.linalg.norm(cap - shoulder) <= REACH:
        return rung_label(cap[2]), cap
    rail = rail_target(shoulder, out_sign)
    if rail is not None and np.linalg.norm(rail - shoulder) <= REACH:
        return rung_label(rail[2]), rail
    return 0, None


plans = [[plan(int(oc120[i, 0]), d["body_pos_w"][i, sh_l], True),
          plan(int(oc120[i, 1]), d["body_pos_w"][i, sh_r], False)] for i in range(T)]
_held = sum(1 for f in plans for h in f if h[0] != 0)
_cap = sum(1 for f in plans for h in f if h[0] != 0 and h[1][2] > CAP["z"] - 0.01)
_rel = sum(1 for i, f in enumerate(plans) for c, h in zip(oc120[i], f)
           if int(c) != 0 and h[0] == 0)
print(f"[plan] {_held} of {2*T} hand-frames take a REAL hold ({_cap} of them the TOP HANDLE); "
      f"{_rel} released as out of reach (reach {REACH:.3f} m)")
_first = next((i for i, f in enumerate(plans)
               if all(h[0] != 0 and h[1][2] > CAP["z"] - 0.01 for h in f)), None)
if _first is not None:
    print(f"[plan] both hands on the handle from frame {_first} = 50 Hz step "
          f"{int(_first / fps * 50)}")

if a.plan_only:
    import sys as _sys
    print("[plan] --plan_only: stopping before IK")
    _sys.exit(0)

# ---- Isaac: numerical IK against real FK -------------------------------------------------
set_layout_seed(0)
cfg = parse_env_cfg(a.task, num_envs=a.envs)
env = gym.make(a.task, cfg=cfg).unwrapped
env.reset()
rob = env.scene["robot"]
origins = env.scene.env_origins
bn = list(rob.body_names)
jn = list(rob.joint_names)
dev = env.device
N = a.envs

missing = [b for b in clip_bodies if b not in bn]
if missing:
    print(f"[warn] clip bodies absent from this robot: {missing}")
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
_cache = pathlib.Path(a.out) / 'q_out_cache.npy'
_reuse = a.reuse_q and _cache.exists()
if _reuse:
    q_out = np.load(_cache)
    print(f'[ik  ] reusing solved angles from {_cache}', flush=True)
for b0 in ([] if _reuse else range(0, T, N)):
    fr = np.arange(b0, min(b0 + N, T))
    n = len(fr)
    qf = rob.data.default_joint_pos.clone()
    qf[:n][:, j_dst] = torch.as_tensor(d["joint_pos"][fr][:, j_src], dtype=torch.float32, device=dev)
    rp = torch.as_tensor(d["body_pos_w"][fr, pel], dtype=torch.float32, device=dev)
    rq = torch.as_tensor(d["body_quat_w"][fr, pel], dtype=torch.float32, device=dev)
    root = torch.zeros(N, 7, device=dev); root[:, 3] = 1.0
    root[:n, :3] = rp + origins[:n]
    root[:n, 3:] = rq
    rob.write_root_pose_to_sim(root)
    rob.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))

    for s, col in (("left", 0), ("right", 1)):
        tgt = torch.zeros(N, 3, device=dev)
        act = torch.zeros(N, dtype=torch.bool, device=dev)
        for i, f in enumerate(fr):
            cidx, p = plans[f][col]
            if p is not None:
                tgt[i] = torch.as_tensor(p, dtype=torch.float32, device=dev); act[i] = True
        if not bool(act.any()):
            continue
        ids = arm_ids[s]
        for _ in range(a.iters):
            rob.write_joint_state_to_sim(qf, torch.zeros_like(qf)); fk_refresh()
            w = rob.data.body_pos_w[:, wrist_ids[s]] - origins
            e = tgt - w
            J = torch.zeros(N, 3, len(ids), device=dev)
            for c in range(len(ids)):
                qp = qf.clone(); qp[:, ids[c]] += 1e-3
                rob.write_joint_state_to_sim(qp, torch.zeros_like(qp)); fk_refresh()
                J[:, :, c] = ((rob.data.body_pos_w[:, wrist_ids[s]] - origins) - w) / 1e-3
            JT = J.transpose(1, 2)
            A = JT @ J + 1e-4 * torch.eye(len(ids), device=dev)
            dq = torch.linalg.solve(A, JT @ e.unsqueeze(-1)).squeeze(-1)
            dq = dq.clamp(-0.25, 0.25) * act.float().unsqueeze(-1)
            qf[:, ids] += dq
            qf[:, ids] = qf[:, ids].clamp(-2.6, 2.6)
        rob.write_joint_state_to_sim(qf, torch.zeros_like(qf)); fk_refresh()
        fin = (tgt - (rob.data.body_pos_w[:, wrist_ids[s]] - origins)).norm(dim=-1)
        err_hist.append(float(fin[act].mean()) if bool(act.any()) else 0.0)

    # write the solved arm angles back into the clip's joint order
    for s in ("left", "right"):
        for t in ARM_JOINTS:
            nm = t.format(s=s)
            q_out[fr, file_joints.index(nm)] = qf[:n, jn.index(nm)].cpu().numpy()
    if b0 % (N * 4) == 0:
        print(f"[ik  ] frames {b0:4d}-{fr[-1]:4d}  mean wrist error "
              f"{np.mean(err_hist[-2:])*1000:6.1f} mm", flush=True)

if not _reuse:
    print(f"[ik  ] done. overall mean wrist error {np.mean(err_hist)*1000:.1f} mm", flush=True)
    # checkpoint: everything below this line is seconds of work, but a crash there
    # used to cost the entire IK pass (~55 min). Never again.
    _cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(_cache, q_out)
    print(f"[ik  ] cached -> {_cache}", flush=True)

# ---- rebuild body tracks from the solved pose, so joints and bodies agree -----------------
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

# velocities are finite differences of the new tracks
bv = np.gradient(bp, 1.0 / fps, axis=0)
jv = np.gradient(q_out, 1.0 / fps, axis=0)

# ---- contact labels at 50 Hz --------------------------------------------------------------
oc_new = np.zeros_like(oc50)
for i in range(n50):
    f = min(int(i / 50 * fps), T - 1)
    oc_new[i, 0] = plans[f][0][0]
    oc_new[i, 1] = plans[f][1][0]

out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
np.savez(out / "motion.npz", joint_pos=q_out, joint_vel=jv, body_pos_w=bp, body_quat_w=bq,
         body_lin_vel_w=bv, body_ang_vel_w=d["body_ang_vel_w"], object_contact=oc_new)
shutil.copy(base / "meta.json", out / "meta.json")

uniq = sorted(set(int(v) for v in np.unique(oc_new)))
print(f"[out ] {out/'motion.npz'}")
print(f"[out ] contact labels used: {uniq}  (max -> num_rungs+2*segments >= {max(uniq)+1})")
print(f"[out ] pelvis {bp[0, pel, 2]:.3f} -> {bp[-1, pel, 2]:.3f} m")
for lbl, v in (("free", 0), ("platform", PLATFORM_IDX)):
    c = int((oc_new == v).any(axis=1).sum())
    print(f"[out ] steps with a hand on {lbl:9s}: {c:4d}/{n50}")
rail = int((oc_new >= NR).any(axis=1).sum())
print(f"[out ] steps with a hand on a RAIL     : {rail:4d}/{n50}")
env.close(); app.close()
