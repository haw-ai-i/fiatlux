"""Phase 1: replay a reference climb clip onto THIS benchmark's ladder, kinematically.

Write-through replay, modelled on fiatlux-s06eval/scripts/replay_render_ego.py: every frame
writes the root pose and joint state straight to the sim and renders. ``sim.step()`` is never
called, so physics never resolves a contact. This answers GEOMETRY ONLY -- whether the foot
would land on the tread and clear its front edge -- and says nothing about balance or whether
the foot would hold.

The clip (LadderMan, Apache-2.0, converted by ladderman_to_clip.py) climbs a ladder whose rungs
sit at a different height and depth from ours, so the clip is ALIGNED rather than played raw:

  yaw   -- the clip is turned so the robot faces the ladder's step face (LADDER_STEP_FACE_LOCAL)
  place -- the swing foot's LANDING position is solved with OUR robot's own forward kinematics
           (write the landing frame, read where the foot actually went, translate by the error),
           so the placement does not inherit LadderMan's link lengths or foot geometry.

Nothing about the ladder is hardcoded: S02 re-randomises its layout every launch, so the ladder
pose and every tread's world AABB are read live off the stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--clip", default="/home/yujin-chen/climb_motion/ladderman_climb.npz")
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--robot", default="dex3", choices=["inspire", "dex3"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--start", type=int, default=185, help="first clip frame")
ap.add_argument("--end", type=int, default=301, help="last clip frame (inclusive)")
ap.add_argument("--tread", type=int, default=1, help="target tread, 1 = lowest")
ap.add_argument("--swing", default="right", choices=["left", "right"])
ap.add_argument("--out", required=True)
ap.add_argument("--measure-only", action="store_true", help="dump ladder geometry and stop")
ap.add_argument("--probe-views", action="store_true", help="render candidate camera views and stop")
ap.add_argument(
    "--no-align",
    action="store_true",
    help="clip root poses are ALREADY world-frame for this ladder; do not yaw/shift them",
)
ap.add_argument("--cam-dist", type=float, default=1.25)
ap.add_argument("--cam-lift", type=float, default=0.55)
ap.add_argument("--cam-aim", type=float, default=None, help="world z to aim at (default: the tread)")
ap.add_argument("--fps", type=int, default=50)
ap.add_argument("--img-width", type=int, default=1280)
ap.add_argument("--img-height", type=int, default=720, dest="img_h")
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fiatlux_task.tasks  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed
from fiatlux_task.viz import make_video_camera_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
FOOT = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}
LADDER_STEP_FACE_LOCAL = (0.0, -1.0, 0.0)  # mdp/place_terms.py


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def yaw_quat(t):
    return np.array([math.cos(t / 2), 0.0, 0.0, math.sin(t / 2)])


def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_rot(q, v):
    w, u = q[0], np.asarray(q[1:])
    v = np.asarray(v, float)
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def mesh_inventory(stage, ladder_prim_path, scale):
    """World AABB of every mesh under the ladder prim. The step treads are NOT named prims --
    they come out of the convex decomposition -- so they are found by shape, not by name."""
    from pxr import Gf, Usd, UsdGeom

    out = []
    root = stage.GetPrimAtPath(ladder_prim_path)
    if not root or not root.IsValid():
        print(f"[WARN  ] no prim at {ladder_prim_path}", flush=True)
        return out
    for prim in Usd.PrimRange(root):
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            continue
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        P = np.array([[*m.Transform(Gf.Vec3d(q[0], q[1], q[2]))] for q in pts], float) * scale
        lo, hi = P.min(0), P.max(0)
        out.append(
            {
                "name": prim.GetName(),
                "path": str(prim.GetPath()),
                "lo": lo.tolist(),
                "hi": hi.tolist(),
                "npts": len(pts),
            }
        )
    out.sort(key=lambda t: t["hi"][2])
    return out


from ladder_treads import (
    TREAD_WIDTH,
    TREAD_X_CENTRE,  # noqa: E402
    TREADS_LOCAL,
    tread_world,
    verify_treads,
)


def main() -> int:
    os.makedirs(a.out, exist_ok=True)
    set_layout_seed(a.seed)
    cfg = parse_env_cfg(a.task, device=a.device, num_envs=1, use_fabric=True)
    cfg.seed = a.seed
    if hasattr(cfg, "couple_ladder_to_fixture"):
        cfg.couple_ladder_to_fixture = True  # belt and braces; physics never runs anyway
    if a.robot != "inspire":
        from fiatlux_task.robots.g1 import swap_robot_variant

        swap_robot_variant(cfg, a.robot)
    cfg.scene.video_cam = make_video_camera_cfg(a.img_width, a.img_h)

    env = gym.make(a.task, cfg=cfg)
    base = env.unwrapped
    dev = base.device
    env.reset(seed=a.seed)
    robot = base.scene["robot"]
    vcam = base.scene["video_cam"]

    # ---- live ladder pose + tread geometry -------------------------------------------------
    ladder = base.scene["ladder"]
    lp = ladder.data.root_pos_w[0].detach().cpu().numpy().astype(float)
    lq = ladder.data.root_quat_w[0].detach().cpu().numpy().astype(float)
    print(
        f"[LADDER] pos {np.round(lp, 4)}  quat(wxyz) {np.round(lq, 4)}  yaw {math.degrees(quat_yaw(lq)):+.2f} deg",
        flush=True,
    )

    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    lpath = ladder.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
    inv = mesh_inventory(stage, lpath, scale)
    print(f"[LADDER] prim {lpath}: {len(inv)} meshes", flush=True)
    if a.measure_only:
        for m in inv:
            lo, hi = np.array(m["lo"]), np.array(m["hi"])
            print(
                f"   {m['name'][:34]:34s} z[{lo[2]:+.3f},{hi[2]:+.3f}] "
                f"x[{lo[0]:+.3f},{hi[0]:+.3f}] y[{lo[1]:+.3f},{hi[1]:+.3f}] pts {m['npts']}",
                flush=True,
            )
    if True:
        # The ladder is ONE mesh, so treads are found in the VERTEX CLOUD, not from prim bounds.
        # Dumped once here so the surface-finding can be iterated offline without relaunching Kit.
        from pxr import Gf, Usd, UsdGeom

        rootp = stage.GetPrimAtPath(lpath)
        clouds = {}
        for prim in Usd.PrimRange(rootp):
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get() if mesh else None
            if not pts:
                continue
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            clouds[prim.GetName()] = (
                np.array([[*m.Transform(Gf.Vec3d(q[0], q[1], q[2]))] for q in pts], np.float32) * scale
            )
        clouds_all = clouds
        np.savez_compressed(os.path.join(a.out, "ladder_vertices.npz"), **clouds)
        print(
            f"[OK    ] wrote {a.out}/ladder_vertices.npz  " + ", ".join(f"{k}:{v.shape[0]}" for k, v in clouds.items()),
            flush=True,
        )
    body = max(clouds_all.items(), key=lambda kv: len(kv[1]))[1] if clouds_all else np.zeros((0, 3))
    if not verify_treads(body, lp, quat_yaw(lq)):
        print(
            "[FATAL] the ladder asset no longer matches TREADS_LOCAL -- re-measure before trusting a landing",
            flush=True,
        )
        env.close()
        return 2
    treads = [
        dict(t, width=TREAD_WIDTH, x_centre=TREAD_X_CENTRE, depth=t["y_back"] - t["y_front"], n=0) for t in TREADS_LOCAL
    ]
    print(f"[LADDER] {len(treads)} tread colliders (stage metersPerUnit {scale})", flush=True)
    for i, t in enumerate(treads, 1):
        print(
            f"   tread {i}  top z {t['top']:.4f}  W {t['width']:.4f}  D {t['depth']:.4f}  "
            f"y_front {t['y_front']:+.4f}  y_back {t['y_back']:+.4f}  ({t['n']} verts)",
            flush=True,
        )
    if len(treads) > 1:
        rises = np.diff([t["top"] for t in treads])
        runs = np.diff([(t["y_front"] + t["y_back"]) / 2 for t in treads])
        print(f"   rises {np.round(rises, 4)}   runs {np.round(runs, 4)}", flush=True)

    # sole offset: ankle_roll link height while the robot stands on the floor at reset
    fid = robot.find_bodies([FOOT[a.swing]], preserve_order=True)[0][0]
    oid = robot.find_bodies([FOOT["left" if a.swing == "right" else "right"]], preserve_order=True)[0][0]
    sole_off = float(robot.data.body_pos_w[0, fid, 2].item())
    print(f"[ROBOT ] {FOOT[a.swing]} at reset z = {sole_off:.4f} m  (sole offset)", flush=True)

    meta = {
        "task": a.task,
        "seed": a.seed,
        "ladder_pos": lp.tolist(),
        "ladder_quat": lq.tolist(),
        "ladder_yaw_deg": math.degrees(quat_yaw(lq)),
        "treads": treads,
        "sole_offset_m": sole_off,
        "stage_meters_per_unit": scale,
    }
    json.dump(meta, open(os.path.join(a.out, "ladder_geometry.json"), "w"), indent=2)
    print(f"[OK    ] wrote {a.out}/ladder_geometry.json", flush=True)
    if a.measure_only:
        env.close()
        return 0

    # ---- clip ------------------------------------------------------------------------------
    clip = np.load(a.clip)
    dof = clip["dof"].astype(np.float64)
    rrot = clip["root_rot"].astype(np.float64)  # wxyz
    rtra = clip["root_trans_offset"].astype(np.float64)
    idx = [list(robot.joint_names).index(n) for n in JOINT_ORDER]
    if a.end < 0:
        a.end = len(dof) - 1
    frames = list(range(a.start, a.end + 1))
    print(f"[CLIP  ] {a.clip}  frames {a.start}-{a.end} ({len(frames)}) of {len(dof)}", flush=True)

    # ---- alignment -------------------------------------------------------------------------
    if not treads:
        print("[FATAL] no treads found", flush=True)
        env.close()
        return 2
    tread = treads[a.tread - 1]
    tread_ctr_w, tread_front_w = tread_world(tread, lp, quat_yaw(lq))
    target = tread_ctr_w + np.array([0.0, 0.0, sole_off])
    thi = np.array([tread_ctr_w[0], tread_ctr_w[1], tread["top"]])
    face_w = quat_rot(lq, LADDER_STEP_FACE_LOCAL)  # outward normal of the step side
    desired_yaw = math.atan2(-face_w[1], -face_w[0])  # robot +x points INTO the ladder
    dyaw = 0.0 if a.no_align else desired_yaw - quat_yaw(rrot[a.end])
    qy = yaw_quat(dyaw)
    print(f"[ALIGN ] tread {a.tread} top z {thi[2]:.4f}  centre ({target[0]:+.3f},{target[1]:+.3f})", flush=True)
    print(
        f"[ALIGN ] step face {np.round(face_w, 3)}  desired yaw {math.degrees(desired_yaw):+.2f} deg  "
        f"clip yaw {math.degrees(quat_yaw(rrot[a.end])):+.2f}  delta {math.degrees(dyaw):+.2f}",
        flush=True,
    )

    def t_(x):
        return torch.tensor(np.asarray(x, np.float32), device=dev).unsqueeze(0)

    nq = robot.num_joints

    def pose_of(i, shift):
        q = quat_mul(qy, rrot[i])
        p = quat_rot(qy, rtra[i]) + shift
        jp = robot.data.default_joint_pos[0].detach().cpu().numpy().copy()
        jp[idx] = dof[i]
        return p, q, jp

    def write(i, shift):
        p, q, jp = pose_of(i, shift)
        robot.write_root_pose_to_sim(torch.cat([t_(p), t_(q)], dim=-1))
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
        robot.write_joint_state_to_sim(t_(jp), torch.zeros((1, nq), device=dev))
        base.scene.write_data_to_sim()
        base.sim.render()

    # solve the shift with OUR FK: write the landing frame at zero shift, read the foot, correct.
    # --no-align skips this: the clip is already world-frame for THIS ladder (climb_sequence.py).
    write(a.end, np.zeros(3))
    foot0 = robot.data.body_pos_w[0, fid].detach().cpu().numpy().astype(float)
    shift = np.zeros(3) if a.no_align else target - foot0
    write(a.end, shift)
    foot1 = robot.data.body_pos_w[0, fid].detach().cpu().numpy().astype(float)
    print(
        f"[ALIGN ] landing-frame foot {np.round(foot0, 4)} -> shift {np.round(shift, 4)} -> "
        f"{np.round(foot1, 4)}   residual {np.linalg.norm(foot1 - target) * 1000:.2f} mm",
        flush=True,
    )

    # ---- camera: side-on at the target tread, so clearance is readable ----------------------
    # Side-on view along the tread's WIDTH axis, close enough that an 8 cm tread is readable.
    # Placed relative to the landing target (the robot is there), not to room coordinates.
    along = quat_rot(lq, (1.0, 0.0, 0.0))  # ladder local +x = tread width direction
    # 45 deg from the open climbing side: the -y side of this room is a wall (probe_views A/F
    # render flat 53,53,53 from inside it) and a pure +y side view looks through the ladder's
    # own rails, so neither pure axis works.
    aim_z = a.cam_aim if a.cam_aim is not None else target[2]
    aim = np.array([target[0], target[1], aim_z])
    eye = aim + face_w * a.cam_dist - along * a.cam_dist + np.array([0.0, 0.0, a.cam_lift])
    look = aim + np.array([0.0, 0.0, -0.02])
    print(f"[CAM   ] eye {np.round(eye, 3)} -> look {np.round(look, 3)}", flush=True)
    vcam.set_world_poses_from_view(
        torch.tensor(eye, dtype=torch.float32, device=dev).expand(1, 3),
        torch.tensor(look, dtype=torch.float32, device=dev).expand(1, 3),
    )

    if a.probe_views:
        import imageio.v2 as _iio

        write(a.end, shift)
        cands = {
            "A_side_near": target + along * 1.6 + face_w * 0.3 + np.array([0, 0, 0.5]),
            "B_side_far": target + along * 3.0 + face_w * 0.6 + np.array([0, 0, 1.2]),
            "C_front": target + face_w * 2.2 + np.array([0, 0, 0.9]),
            "D_front_near": target + face_w * 1.3 + np.array([0, 0, 0.4]),
            "E_opp_side": target - along * 1.6 + face_w * 0.3 + np.array([0, 0, 0.5]),
            "F_high": target + face_w * 2.0 + along * 2.0 + np.array([0, 0, 1.8]),
        }
        for nm, e in cands.items():
            vcam.set_world_poses_from_view(
                torch.tensor(e, dtype=torch.float32, device=dev).expand(1, 3),
                torch.tensor(target, dtype=torch.float32, device=dev).expand(1, 3),
            )
            for _ in range(3):
                base.sim.render()
            vcam.update(0.0)
            img = vcam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            _iio.imwrite(os.path.join(a.out, f"view_{nm}.png"), img)
            print(f"[VIEW  ] {nm:12s} eye {np.round(e, 2)}  mean rgb {img.reshape(-1, 3).mean(0).round(1)}", flush=True)
        env.close()
        return 0

    # ---- replay ----------------------------------------------------------------------------
    vid, rec = [], []
    for k, i in enumerate(frames):
        write(i, shift)
        base.sim.render()
        vcam.update(0.0)
        vid.append(vcam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
        bp = robot.data.body_pos_w[0].detach().cpu().numpy().astype(float)
        bq = robot.data.body_quat_w[0].detach().cpu().numpy().astype(float)
        rec.append(
            {
                "frame": int(i),
                "t": i / a.fps,
                "swing_foot": bp[fid].tolist(),
                "swing_quat": bq[fid].tolist(),
                "stance_foot": bp[oid].tolist(),
                "root": bp[0].tolist(),
                # every link, so a knee or shin through the ladder cannot pass unseen:
                # the foot-vs-tread gate alone does not check the rest of the robot
                "bodies": bp.tolist(),
            }
        )
        if k % 20 == 0:
            print(f"[REPLAY] {k}/{len(frames)}  swing foot z {bp[fid][2]:.4f}", flush=True)

    out_mp4 = os.path.join(a.out, "replay.mp4")
    imageio.mimwrite(out_mp4, vid, fps=max(1, a.fps // 2), codec="libx264", quality=8)
    json.dump(
        {
            "meta": meta,
            "body_names": list(robot.body_names),
            "target": target.tolist(),
            "shift": shift.tolist(),
            "tread_index": a.tread,
            "tread": tread,
            "swing": a.swing,
            "frames": rec,
        },
        open(os.path.join(a.out, "replay.json"), "w"),
        indent=2,
    )
    print(f"[OK    ] wrote {out_mp4} ({len(vid)} frames) and replay.json", flush=True)
    env.close()
    return 0


code = main()
app.close()
raise SystemExit(code)
