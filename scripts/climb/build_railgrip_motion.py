"""Re-author the top-out reference so the HANDS HOLD THE RAILS instead of phantom rungs.

A sibling of build_topout_motion.py, which stays runnable and is still the source of the legs.

Why. Measured off the benchmark ladder's own mesh (plane intersection, not vertex sampling):
the climbing-side rail is 40.0 x 74.5 mm, centres +/-0.225 m, continuous from the floor to
z ~= 1.35 m, and there is NOTHING on the climbing face above that. The previous reference
commanded a hand onto a rung at 1.409 m or 1.644 m (empty air) or the rear top cap at 1.878 m
for 478 of its 939 steps -- 51%. It also never addressed a rail, because LadderMan builds each
rail as ONE long cuboid whose object_pos_w is a fixed midpoint (see ladder_rails.py, which
replaces them with per-height segments).

What changes, and what deliberately does NOT:
  * legs, pelvis, torso: UNTOUCHED. Rungs 0-3 land on our treads to within 0.8 mm; that part of
    the old clip is real retargeted human stepping and is the best thing in it.
  * arms: re-solved so each wrist sits on the rail on its own side, by numerical IK against the
    robot's real forward kinematics. An analytic 2-link solve would be biased -- shoulder-to-
    wrist at the zero pose is 0.285 m against link origins summing to 0.378 m, so the origins
    are not collinear.
  * contact labels: rail segments while climbing, the PLATFORM for the final push-up, free once
    standing. No rung above index 4 is ever addressed again.

Left/right and the approach direction are derived FROM THE CLIP, not assumed: LadderMan names
its -y rail "left" while a robot facing +x has its left on +y, and guessing that wrong would
cross the arms silently.

    python scripts/climb/build_railgrip_motion.py --headless \
        --base /home/yujin-chen/psi0_ft/climb/topout-rungs-2026-11-18 \
        --out  /home/yujin-chen/psi0_ft/climb/topout-rails
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
ap.add_argument("--reach", type=float, default=0.34,
                help="max shoulder-to-wrist-target distance. MEASURED max is 0.378 m "
                     "(upper 0.1934 + fore 0.1842); this leaves margin.")
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
PLATFORM_IDX = NR - 1                           # rung 4 = the platform
CAP_L, CAP_R = NR + 2 * a.rail_segments, NR + 2 * a.rail_segments + 1
TOP_CAP_Z = 1.845
REACH = a.reach                                 # wrist target must sit inside this of the shoulder

# Shoulder positions come straight from the clip. shoulder_pitch_link's ORIGIN is the joint
# itself, parented to the torso, so it does not move when the arm angles are re-solved -- which
# makes it a valid reach origin even though the arms are exactly what this script rewrites.
sh_l = clip_bodies.index("left_shoulder_pitch_link")
sh_r = clip_bodies.index("right_shoulder_pitch_link")

s_cap = (TOP_CAP_Z - BASE[2]) / math.sin(ang)
cap_c = np.asarray(BASE) + dirv * s_cap


def candidates(is_left):
    """Every REAL hold on this ladder, on this hand's side, highest last."""
    sgn = LEFT_SIGN if is_left else -LEFT_SIGN
    out = []
    for k in range(a.rail_segments):
        c = seg_c[k].copy(); c[1] = sgn * a.rail_offset
        out.append((NR + 2 * k + (0 if sgn < 0 else 1),
                    c + np.array([BACK * a.hand_back, 0.0, a.hand_up])))
    pf = np.asarray(BASE) + dirv * (PLATFORM_IDX * SPACING)
    out.append((PLATFORM_IDX, pf + np.array([BACK * a.hand_back,
                                             sgn * 0.16, RUNG_H / 2 + a.hand_up])))
    cc = cap_c.copy(); cc[1] = sgn * 0.220
    out.append(((CAP_L if sgn < 0 else CAP_R), cc + np.array([BACK * a.hand_back, 0.0, 0.0])))
    return sorted(out, key=lambda t: t[1][2])


CAND = {True: candidates(True), False: candidates(False)}


def plan(old_label, shoulder, is_left):
    """-> (contact index, desired wrist xyz), or (0, None) when the hand must be free.

    The previous version clamped an out-of-range hold to the topmost rail and let the IK chase
    it; mean wrist error grew to 827 mm by the end of the clip, because once the pelvis passes
    ~1.3 m the rails are a metre below a 0.378 m arm. A hold that cannot be reached is not a
    hold: the hand is released instead, which is also what a person does on a step ladder when
    the rail drops below their waist.
    """
    if old_label == 0:
        return 0, None
    best = None
    for cidx, p in CAND[is_left]:
        if np.linalg.norm(p - shoulder) <= REACH:
            best = (cidx, p)                     # candidates are sorted low->high: keep the top
    return best if best is not None else (0, None)


plans = [[plan(int(oc120[i, 0]), d["body_pos_w"][i, sh_l], True),
          plan(int(oc120[i, 1]), d["body_pos_w"][i, sh_r], False)] for i in range(T)]
_held = sum(1 for f in plans for h in f if h[0] != 0)
_rel = sum(1 for i, f in enumerate(plans) for c, h in zip(oc120[i], f)
           if int(c) != 0 and h[0] == 0)
print(f"[plan] {_held} of {2*T} hand-frames hold a REAL hold; {_rel} released as out of reach "
      f"(reach {REACH:.2f} m)")

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
