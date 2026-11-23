"""render_reference.py with the BULB drawn in the right hand (S10). A sibling, not an edit.

Each frame also writes the fresh bulb's pose: the clip's right wrist_yaw_link pose composed with the
seat measured in the benchmark (measure_bulb_in_hand.py). The fingers keep S10's own cupped pose.
Still kinematic: no physics, nothing is held, nothing can be crushed or dropped.

--- original docstring ---
Render the reference clip the teacher trains on, kinematically, on the real S02 ladder.

THIS IS NOT PHYSICS. Every frame writes the reference's root pose and joint positions straight
onto the robot and renders; `sim.step()` is never called, so no contact is ever resolved, no
torque limit is ever applied and nothing can fall. It shows what the clip ASKS FOR, which is a
different question from whether a G1 can do it -- `waist_pitch` is already known to saturate at
its 50 N.m limit on this motion.

The clip is placed through `LadderManObs`, the same adapter that feeds the policy, so the pose
drawn here is the pose the teacher is rewarded for tracking -- not a re-derivation that could
drift from it.

    python scripts/climb/render_reference.py --headless --motion <dir> --out <dir>
"""

import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description="Kinematic render of the training reference.")
ap.add_argument("--task", default="FIATLUX-S10-ClimbWithBulb-v0")
ap.add_argument("--motion", required=True)
ap.add_argument("--out", required=True)
ap.add_argument(
    "--robot",
    default="dex3",
    choices=["dex3", "inspire"],
    help="G1 hand variant. The scene default is INSPIRE everywhere -- every task "
    "in the repo is authored against it and swapped explicitly -- so without "
    "this the climb work silently ran on the wrong hand. Dex3 is what this "
    "project uses, so it is the default here.",
)
ap.add_argument("--layout_seed", type=int, default=0)
ap.add_argument("--stride", type=int, default=1, help="render every Nth of the 50 Hz steps")
ap.add_argument("--fps", type=int, default=50)
ap.add_argument("--img_width", type=int, default=1280)
ap.add_argument("--img_height", type=int, default=720)
ap.add_argument(
    "--probe_views",
    action="store_true",
    help="render candidate camera placements at a mid-climb frame and stop. The -y "
    "side of this room is a WALL: a camera put there renders a flat texture and "
    "every frame comes out identical, which looks exactly like a successful "
    "render until you difference two frames. Probe, then choose.",
)
ap.add_argument(
    "--cam_eye", type=float, nargs=3, default=None, help="explicit world eye position, overriding the derived placement"
)
ap.add_argument("--cam_out", type=float, default=2.9, help="metres out from the step face")
ap.add_argument("--cam_side", type=float, default=1.7)
ap.add_argument("--cam_lift", type=float, default=1.25)
ap.add_argument(
    "--cam_look_z",
    type=float,
    default=0.95,
    help="height above the ladder root to aim at; raise it so a pelvis that ends at 1.94 m still has headroom in frame",
)
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import fiatlux_task.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402
from fiatlux_task.viz import make_video_camera_cfg  # noqa: E402
from ladderman_runtime import LadderManObs  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

set_layout_seed(a.layout_seed)
cfg = parse_env_cfg(a.task, num_envs=1)
if a.robot != "inspire":
    from fiatlux_task.robots.g1 import swap_robot_variant

    swap_robot_variant(cfg, a.robot)
print(f"[robot] {a.robot}: {cfg.scene.robot.spawn.usd_path.split('/')[-1]}")
cfg.scene.video_cam = make_video_camera_cfg(a.img_width, a.img_height)
env = gym.make(a.task, cfg=cfg)
base = env.unwrapped
base.reset()

ob = LadderManObs(base, a.motion)
robot = base.scene["robot"]
vcam = base.scene["video_cam"]
dev = base.device
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)

# Frame the whole ladder from the open climbing side. Derived from the scene, not hardcoded:
# the clip's own frame-0 root marks where the robot stands, so the vector from the ladder root
# to it is the step-facing direction whatever the layout draw did.
lad = base.scene["ladder"].data.root_pos_w[0].cpu().numpy().astype(float)
start = ob.ref_body_pos[0][ob.root_mo].cpu().numpy().astype(float)
outv = start[:2] - lad[:2]
outv = outv / (np.linalg.norm(outv) + 1e-9)
side = np.array([-outv[1], outv[0]])
eye = np.array(
    [
        lad[0] + outv[0] * a.cam_out + side[0] * a.cam_side,
        lad[1] + outv[1] * a.cam_out + side[1] * a.cam_side,
        lad[2] + a.cam_lift,
    ]
)
look = np.array([lad[0], lad[1], lad[2] + a.cam_look_z])
if a.cam_eye is not None:
    eye = np.array(a.cam_eye, dtype=float)


def aim(e, l):
    vcam.set_world_poses_from_view(
        torch.tensor(e, dtype=torch.float32, device=dev).expand(1, 3),
        torch.tensor(l, dtype=torch.float32, device=dev).expand(1, 3),
    )


def stage(i):
    t = torch.full((1,), i, dtype=torch.long, device=dev)
    robot.write_root_pose_to_sim(
        torch.cat([ob.ref_body_pos[t][:, ob.root_mo], ob.ref_body_quat[t][:, ob.root_mo]], dim=-1)
    )
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
    q = ob.ref_joint_pos[t]
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ob.j_fx)
    base.scene.write_data_to_sim()


def shoot():
    base.sim.render()
    vcam.update(0.0)
    return vcam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)


if a.probe_views:
    import imageio.v2 as _iio

    stage(ob.motion_len // 2)
    mid = np.array([lad[0], lad[1], lad[2] + 0.95])
    cands = {}
    for name, (ov, sv, lift) in {
        "out_near": (2.2, 0.0, 1.1),
        "out_far": (3.6, 0.0, 1.4),
        "out_left": (2.6, 1.8, 1.2),
        "out_right": (2.6, -1.8, 1.2),
        "side_left": (0.6, 3.0, 1.2),
        "side_right": (0.6, -3.0, 1.2),
        "high_out": (2.4, 1.0, 2.6),
        "back_out": (-2.6, 1.4, 1.3),
    }.items():
        e = np.array([lad[0] + outv[0] * ov + side[0] * sv, lad[1] + outv[1] * ov + side[1] * sv, lad[2] + lift])
        aim(e, mid)
        img = shoot()
        f = img.reshape(-1, 3).astype(float)
        cands[name] = (e, f.mean(0), f.std(0).mean())
        _iio.imwrite(str(out / f"view_{name}.png"), img)
        print(
            f"[VIEW  ] {name:11s} eye {np.round(e, 2)}  mean {f.mean(0).round(1)}  detail(std) {f.std(0).mean():6.2f}",
            flush=True,
        )
    print("[VIEW  ] a buried/flat view has LOW std; pick a high-std one and pass --cam_eye", flush=True)
    env.close()
    app.close()
    sys.exit(0)

aim(eye, look)
print(f"[CAM   ] eye {np.round(eye, 3)} -> look {np.round(look, 3)}", flush=True)
print(f"[MOTION] {ob.motion_len} steps at 50 Hz = {ob.motion_len / 50:.2f} s", flush=True)

from isaaclab.utils.math import quat_apply, quat_mul  # noqa: E402

bulb = base.scene["fresh_bulb"]
_wr = ob.lm_body_names.index("right_wrist_yaw_link")
_BP = torch.tensor([[0.1192, 0.0540, -0.1310]], device=dev)
_BQ = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]], device=dev)
frames = list(range(0, ob.motion_len, max(1, a.stride)))
vid = []
zs = []
with torch.inference_mode():
    for k, i in enumerate(frames):
        t = torch.full((1,), i, dtype=torch.long, device=dev)
        root_p = ob.ref_body_pos[t][:, ob.root_mo]
        root_q = ob.ref_body_quat[t][:, ob.root_mo]
        robot.write_root_pose_to_sim(torch.cat([root_p, root_q], dim=-1))
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        q = ob.ref_joint_pos[t]
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ob.j_fx)
        # WITHOUT this the renderer reads a stale pose: write_*_to_sim updates the PhysX buffers
        # and the data cache, but the render path reads fabric, which only refreshes when the
        # scene is flushed. robot.data then reports a climb that the video does not show.
        wp = ob.ref_body_pos[t][:, _wr]
        wq = ob.ref_body_quat[t][:, _wr]
        bulb.write_root_pose_to_sim(torch.cat([wp + quat_apply(wq, _BP), quat_mul(wq, _BQ)], dim=-1))
        bulb.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        base.scene.write_data_to_sim()
        base.sim.render()
        vcam.update(0.0)
        vid.append(vcam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
        zs.append(float(robot.data.root_pos_w[0, 2]))
        if k % 100 == 0:
            print(f"[FRAME ] {i:4d}/{ob.motion_len}  pelvis {zs[-1]:.3f} m", flush=True)

# Guard against exactly the bug above: if the pose never reached the renderer, every frame is
# identical and the mean absolute difference between the first and middle frame is ~0.
_d = float(np.abs(vid[0].astype(np.int16) - vid[len(vid) // 2].astype(np.int16)).mean())
print(f"[CHECK ] mean |frame0 - frame{len(vid) // 2}| = {_d:.2f} / 255", flush=True)
if _d < 1.0:
    print("[CHECK ] FAILED: the frames are identical -- the render did not see the motion.", flush=True)

mp4 = out / "reference.mp4"
imageio.mimwrite(mp4, vid, fps=max(1, a.fps // max(1, a.stride)), codec="libx264", quality=8)
print(f"[RESULT] {len(vid)} frames -> {mp4}", flush=True)
print(
    f"[RESULT] pelvis {zs[0]:.3f} -> {zs[-1]:.3f} m  (+{(zs[-1] - zs[0]) * 1000:.0f} mm), max {max(zs):.3f}", flush=True
)
env.close()
app.close()
