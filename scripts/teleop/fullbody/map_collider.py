"""Map what the ladder's COLLIDER actually occupies -- not what it renders as.

PhysX reported the robot's knees inside the ladder for 202 of 295 frames while a KD-tree over
the ladder's RENDER mesh said -4 mm. The collider is a convex decomposition, which bulges out
over every concavity, so the solid PhysX sees is materially fatter than the drawn ladder. Every
clearance number planned against the render mesh is therefore suspect.

This sweeps PhysX overlap_sphere over the ladder's bounding volume in the ladder's own frame and
records where the collider really is, at two radii:

  point probe  (1 mm)   -- the collider's own solid volume
  knee probe   (55 mm)  -- where a G1 knee could actually be

The knee probe answers the question that matters: is there room between these treads for a leg
at all, or is the decomposition filling the space a climber has to move through?
"""
from __future__ import annotations
import argparse, json, math, os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--step", type=float, default=0.025, help="grid spacing, m")
ap.add_argument("--radii", default="0.001,0.055", help="probe radii, m")
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import carb, gymnasium as gym, numpy as np
import fiatlux_task.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed
from omni.physx import get_physx_scene_query_interface


def main() -> int:
    os.makedirs(a.out, exist_ok=True)
    set_layout_seed(a.seed)
    cfg = parse_env_cfg(a.task, device=a.device, num_envs=1, use_fabric=False)  # scene queries miss collider transforms kept in Fabric
    cfg.seed = a.seed
    from fiatlux_task.robots.g1 import swap_robot_variant
    swap_robot_variant(cfg, "dex3")
    env = gym.make(a.task, cfg=cfg); base = env.unwrapped
    env.reset(seed=a.seed)
    # move the robot far away so it cannot be what the probe hits
    robot = base.scene["robot"]
    import torch
    far = torch.tensor([[50.0, 50.0, 0.5, 1.0, 0.0, 0.0, 0.0]], device=base.device)
    robot.write_root_pose_to_sim(far); base.scene.write_data_to_sim(); base.sim.step(render=False)

    ladder = base.scene["ladder"]
    lp = ladder.data.root_pos_w[0].detach().cpu().numpy().astype(float)
    lq = ladder.data.root_quat_w[0].detach().cpu().numpy().astype(float)
    w, x, y, z = lq
    lyaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    c, s = math.cos(lyaw), math.sin(lyaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    print(f"[LADDER] pos {np.round(lp,4)} yaw {math.degrees(lyaw):+.2f} deg", flush=True)

    # DEBUG: a point known to be inside tread 1, to learn what a hit object actually exposes
    seen = []
    def probe_report(hit):
        seen.append({k: str(getattr(hit, k, None)) for k in
                     ("rigid_body", "collision", "protoIndex", "shape", "body")})
        return True
    for nm, local in (("tread1 centre", (0.0, -0.3804, 0.2300)),
                      ("tread3 centre", (0.0, -0.2142, 0.7000)),
                      ("open air",      (0.0, -0.3804, 0.3500))):
        seen.clear()
        wp = R @ np.array(local) + lp
        n = get_physx_scene_query_interface().overlap_sphere(0.001, carb.Float3(*[float(v) for v in wp]),
                                                             probe_report, False)
        print(f"[DEBUG ] {nm:14s} local {local} -> {n} hits", flush=True)
        for h in seen[:4]:
            print(f"           {h}", flush=True)
    hit_flag = {"v": False}
    def report(hit):
        p = str(getattr(hit, "rigid_body", "") or "") + "|" + str(getattr(hit, "collision", "") or "")
        if "Ladder" in p:
            hit_flag["v"] = True
            return False          # stop at the first ladder hit
        return True               # keep looking past the floor/table

    def occupied(world_pt, radius):
        hit_flag["v"] = False
        get_physx_scene_query_interface().overlap_sphere(
            float(radius), carb.Float3(*[float(v) for v in world_pt]), report, False)
        return hit_flag["v"]

    xs = np.arange(-0.325, 0.326, a.step)
    ys = np.arange(-0.600, 0.401, a.step)
    zs = np.arange(0.000, 1.301, a.step)
    print(f"[GRID  ] {len(xs)}x{len(ys)}x{len(zs)} = {len(xs)*len(ys)*len(zs)} points "
          f"at {a.step*1000:.0f} mm, ladder-local", flush=True)
    out = {}
    for radius in [float(r) for r in a.radii.split(",")]:
        occ = np.zeros((len(xs), len(ys), len(zs)), bool)
        for i, xv in enumerate(xs):
            for j, yv in enumerate(ys):
                for k, zv in enumerate(zs):
                    occ[i, j, k] = occupied(R @ np.array([xv, yv, zv]) + lp, radius)
            if i % 5 == 0:
                print(f"[PROBE ] r={radius*1000:.0f}mm  x slice {i}/{len(xs)}  "
                      f"occupied so far {occ.sum()}", flush=True)
        out[f"occ_{int(radius*1000)}mm"] = occ
        print(f"[PROBE ] r={radius*1000:4.0f} mm -> {occ.sum()} of {occ.size} cells solid "
              f"({100*occ.sum()/occ.size:.1f}%)", flush=True)

    np.savez_compressed(os.path.join(a.out, "collider_map.npz"),
                        xs=xs, ys=ys, zs=zs, ladder_pos=lp, ladder_yaw=lyaw, **out)
    json.dump({"step": a.step, "radii": a.radii, "ladder_pos": lp.tolist(),
               "ladder_yaw_deg": math.degrees(lyaw),
               "shape": [len(xs), len(ys), len(zs)]},
              open(os.path.join(a.out, "collider_map.json"), "w"), indent=2)
    print(f"[OK    ] wrote {a.out}/collider_map.npz", flush=True)
    env.close(); return 0


code = main()
app.close()
raise SystemExit(code)
