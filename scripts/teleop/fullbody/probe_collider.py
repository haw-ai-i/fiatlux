"""Map the ladder's COLLIDER by sweeping a probe body through it and asking PhysX.

`map_collider.py` tried PhysX `overlap_sphere` and got 0 hits everywhere, including a 1 mm probe
inside a tread. This instead reuses the path that demonstrably works (`verify_contacts.py`): put
a real sphere in the scene, move it to each grid point, step physics with gravity off, and read a
contact sensor filtered to the ladder. A reported force means the sphere's collision shape
overlaps the ladder's -- the same currency as the 113 kN knee numbers.

The ladder has to be held still while this runs. S02 ships it as a DYNAMIC rigid body, and with
gravity zeroed nothing holds it: the first cell that overlaps a rail shoves the whole ladder, and
with no weight on it, it keeps going (measured: 215 mm in 300 frames, still at 0.08 m/s). The first
run of this map read 15 solid cells out of 50,949 -- all of them in the first two x-slabs, none
after -- which was the ladder leaving, not free space. Flipping the same asset to kinematic keeps
the identical collider and makes it immovable; the probe then has to be the dynamic one, because
PhysX generates no contacts between two kinematic actors. Drift is asserted at the end of the run.

Run it at the radius you care about:
    5 mm   the collider's own solid volume
   55 mm   where a G1 knee could actually fit
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--radius", type=float, required=True)
ap.add_argument("--step", type=float, default=0.025)
ap.add_argument("--xlim", default="-0.325,0.325")
ap.add_argument("--ylim", default="-0.600,0.400")
ap.add_argument("--zlim", default="0.000,1.300")
ap.add_argument("--force-eps", type=float, default=1e-4)
ap.add_argument(
    "--ladder",
    default="kinematic",
    choices=["kinematic", "dynamic"],
    help="kinematic: same collider, immovable (correct). dynamic: S02's shipped "
    "ladder, which the probe shoves out of the grid.",
)
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import fiatlux_task.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> int:
    os.makedirs(a.out, exist_ok=True)
    set_layout_seed(a.seed)
    cfg = parse_env_cfg(a.task, device=a.device, num_envs=1, use_fabric=True)
    cfg.seed = a.seed
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    # See the module docstring: hold the ladder, and make the probe the dynamic body of the pair.
    if a.ladder == "kinematic":
        cfg.scene.ladder.spawn.rigid_props.kinematic_enabled = True
    cfg.scene.probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Probe",
        spawn=sim_utils.SphereCfg(
            radius=a.radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=(a.ladder == "dynamic"), disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )
    cfg.scene.probe_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Probe",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ladder"],
        history_length=1,
        track_air_time=False,
    )
    env = gym.make(a.task, cfg=cfg)
    base = env.unwrapped
    dev = base.device
    env.reset(seed=a.seed)
    robot = base.scene["robot"]
    robot.write_root_pose_to_sim(torch.tensor([[60.0, 60.0, 0.5, 1, 0, 0, 0]], device=dev))

    probe = base.scene["probe"]
    sensor = base.scene["probe_contact"]
    ladder = base.scene["ladder"]
    lp = ladder.data.root_pos_w[0].detach().cpu().numpy().astype(float)
    q = ladder.data.root_quat_w[0].detach().cpu().numpy().astype(float)
    lyaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    c, s = math.cos(lyaw), math.sin(lyaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    lad0 = ladder.data.root_pos_w[0].clone()
    print(f"[LADDER] pos {np.round(lp, 4)} yaw {math.degrees(lyaw):+.2f}  mode={a.ladder}", flush=True)

    def lim(t):
        lo, hi = (float(v) for v in t.split(","))
        return np.arange(lo, hi + 1e-9, a.step)

    xs, ys, zs = lim(a.xlim), lim(a.ylim), lim(a.zlim)
    tot = len(xs) * len(ys) * len(zs)
    print(f"[GRID  ] r={a.radius * 1000:.0f}mm  {len(xs)}x{len(ys)}x{len(zs)} = {tot} points", flush=True)

    occ = np.zeros((len(xs), len(ys), len(zs)), bool)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
    done = 0
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            for k, zv in enumerate(zs):
                w = R @ np.array([xv, yv, zv]) + lp
                probe.write_root_pose_to_sim(
                    torch.cat([torch.tensor([w], dtype=torch.float32, device=dev), quat], dim=-1)
                )
                probe.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
                base.scene.write_data_to_sim()
                base.sim.step(render=False)
                base.scene.update(base.sim.get_physics_dt())
                fm = sensor.data.force_matrix_w
                occ[i, j, k] = bool(torch.linalg.norm(fm[0, 0, 0, :]).item() > a.force_eps) if fm is not None else False
                done += 1
        drift = float(torch.linalg.norm(ladder.data.root_pos_w[0] - lad0))
        print(
            f"[PROBE ] x {i + 1}/{len(xs)}  solid so far {occ.sum()}/{done}  ladder drift {drift * 1000:.2f} mm",
            flush=True,
        )
        if drift > 1e-3:
            print(
                "[FATAL] the ladder moved -- every cell after the first contact is measured "
                "against a ladder that is no longer where the grid says. Aborting.",
                flush=True,
            )
            return 3

    np.savez_compressed(
        os.path.join(a.out, f"collider_r{int(a.radius * 1000)}mm.npz"),
        xs=xs,
        ys=ys,
        zs=zs,
        occ=occ,
        ladder_pos=lp,
        ladder_yaw=lyaw,
        radius=a.radius,
    )
    print(
        f"[RESULT] r={a.radius * 1000:.0f}mm: {occ.sum()} of {occ.size} cells solid "
        f"({100 * occ.sum() / occ.size:.2f}%)",
        flush=True,
    )
    json.dump(
        {
            "radius": a.radius,
            "step": a.step,
            "solid": int(occ.sum()),
            "total": int(occ.size),
            "ladder_mode": a.ladder,
            "ladder_drift_mm": float(torch.linalg.norm(ladder.data.root_pos_w[0] - lad0)) * 1000.0,
            "ladder_pos": lp.tolist(),
            "ladder_yaw_deg": math.degrees(lyaw),
        },
        open(os.path.join(a.out, f"collider_r{int(a.radius * 1000)}mm.json"), "w"),
        indent=2,
    )
    env.close()
    return 0


code = main()
app.close()
raise SystemExit(code)
