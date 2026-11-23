"""Where is the centre of mass, and can the ankles hold it there?

`verify_contacts.py` answers "does anything overlap the ladder", and a search driven by that
alone has a degenerate optimum: move the body away until nothing touches. It did exactly that --
lean -0.42 scores zero contact by sitting the robot in mid-air behind the ladder. Clearance is
only half a climb; this is the other half.

Gravity stays OFF (the pose is written, not simulated) -- this measures the STATIC demand of the
written pose, not a fall. For each frame: the whole-body CoM from the real per-body masses, its
horizontal offset from the supporting foot along the ladder's fore-aft axis, and the ankle pitch
torque that offset demands, against what the actuator can actually deliver.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--robot", default="dex3", choices=["inspire", "dex3"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--stride", type=int, default=4)
ap.add_argument(
    "--hands",
    action="store_true",
    help="credit the hands on the rails. Without this the support set is the feet "
    "alone, which is what the chair-sit climb was scored against.",
)
ap.add_argument(
    "--heel",
    type=float,
    default=0.080,
    help="m of sole behind the ankle; credited as support (conservative if smaller)",
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

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    set_layout_seed(a.seed)
    cfg = parse_env_cfg(a.task, device=a.device, num_envs=1, use_fabric=True)
    cfg.seed = a.seed
    if a.robot != "inspire":
        from fiatlux_task.robots.g1 import swap_robot_variant

        swap_robot_variant(cfg, a.robot)
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.scene.ladder.spawn.rigid_props.kinematic_enabled = True

    env = gym.make(a.task, cfg=cfg)
    base = env.unwrapped
    dev = base.device
    env.reset(seed=a.seed)
    robot = base.scene["robot"]
    ladder = base.scene["ladder"]

    from retarget_clearance import ARM_CHAIN, leg_jacobian, torso_frame

    # what each arm joint can actually deliver (fiatlux_task/robots/g1.py: _ARM_EFFORT)
    TAU = np.array([25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0])

    def quat_to_mat_np(q):
        w, x, y, z = (float(v) for v in q)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def arm_force_capacity(q7, u_torso, side):
        """Largest palm force along u the arm can hold, from tau = J^T F and the joint limits.

        This is the honest version of a reach-times-torque estimate: it uses the Jacobian at the
        configuration the clip is actually in, so a braced near-straight arm scores high and a
        folded one scores low, which is the whole point of bracing.
        """
        q = np.concatenate([q7, [0.0]])  # trailing pinned palm joint
        J, _, p_palm = leg_jacobian(q, ARM_CHAIN[side])
        tau_unit = np.abs(J[:3, :7].T @ u_torso)  # torque per 1 N of palm force
        with np.errstate(divide="ignore"):
            f = float(np.min(np.where(tau_unit > 1e-9, TAU / np.maximum(tau_unit, 1e-9), np.inf)))
        return f, p_palm

    masses = robot.root_physx_view.get_masses()[0].detach().cpu().numpy().astype(float)
    bodies = list(robot.data.body_names)
    M = float(masses.sum())
    ia = [bodies.index(n) for n in bodies if n.endswith("ankle_roll_link")]
    foot_names = [bodies[i] for i in ia]

    # ankle pitch capability, straight from the articulation
    try:
        eff = robot.root_physx_view.get_dof_max_forces()[0].detach().cpu().numpy().astype(float)
        jn = list(robot.joint_names)
        ank = [eff[jn.index(n)] for n in jn if "ankle_pitch" in n]
        ankle_cap = float(min(ank)) if ank else float("nan")
    except Exception:
        ankle_cap = float("nan")

    # ladder fore-aft: the step face normal, in world
    q = ladder.data.root_quat_w[0].detach().cpu().numpy().astype(float)
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    # ladder-local -y is the climbing side; +out = away from the treads
    out = np.array([-math.cos(yaw) * 0.0 - math.sin(yaw) * -1.0, math.sin(yaw) * 0.0 + math.cos(yaw) * -1.0, 0.0])
    out /= np.linalg.norm(out)
    print(f"[SETUP ] total mass {M:.2f} kg over {len(bodies)} bodies; feet {foot_names}")
    print(f"[SETUP ] ankle pitch effort limit {ankle_cap:.1f} N m; 'out' axis {np.round(out, 3)}")
    try:
        jn = list(robot.joint_names)
        wi = jn.index("waist_pitch_joint")
        lim = robot.data.joint_limits[0, wi].detach().cpu().numpy()
        print(
            f"[SETUP ] waist_pitch limits {lim[0]:+.3f}..{lim[1]:+.3f} rad "
            f"({math.degrees(lim[0]):+.0f}..{math.degrees(lim[1]):+.0f} deg)"
        )
    except Exception as e:
        print(f"[SETUP ] waist_pitch limits unavailable ({e})")

    clip = np.load(a.clip)
    dof = clip["dof"].astype(np.float32)
    rrot = clip["root_rot"].astype(np.float32)
    rtra = clip["root_trans_offset"].astype(np.float32)
    from retarget_clearance import JOINT_ORDER as JO

    idx = [list(robot.joint_names).index(n) for n in JO]
    n = len(dof)
    nq = robot.num_joints
    zeros6 = torch.zeros((1, 6), device=dev)

    def t_(x):
        return torch.tensor(np.asarray(x, np.float32), device=dev).unsqueeze(0)

    rows = []
    for i in range(0, n, a.stride):
        jp = robot.data.default_joint_pos[0].detach().cpu().numpy().copy()
        jp[idx] = dof[i]
        robot.write_root_pose_to_sim(torch.cat([t_(rtra[i]), t_(rrot[i])], dim=-1))
        robot.write_root_velocity_to_sim(zeros6)
        robot.write_joint_state_to_sim(t_(jp), torch.zeros((1, nq), device=dev))
        base.scene.write_data_to_sim()
        base.sim.step(render=False)
        base.scene.update(base.sim.get_physics_dt())
        try:
            bp = robot.data.body_com_pos_w[0].detach().cpu().numpy().astype(float)
        except Exception:
            bp = robot.data.body_pos_w[0].detach().cpu().numpy().astype(float)
        com = (masses[:, None] * bp).sum(0) / M
        feet = bp[ia]
        low = int(np.argmin(feet[:, 2]))  # the foot carrying the weight
        stance = feet[low]
        # how far behind the support edge the CoM sits, along the ladder's fore-aft axis
        d = float(np.dot(com[:2] - stance[:2], out[:2])) - a.heel
        cap, lever = 0.0, 0.0
        if a.hands:
            # the hands pull the robot back toward the ladder: -out, expressed in the torso frame
            R_t_p, p_t_p = torso_frame(dof[i][12:15])
            Rp = quat_to_mat_np(rrot[i])
            Rtw = Rp @ R_t_p
            ptw = np.asarray(rtra[i], float) + Rp @ p_t_p
            hs = []
            for k, side in enumerate(("left", "right")):
                f, p_palm = arm_force_capacity(dof[i][15 + 7 * k : 22 + 7 * k], Rtw.T @ (-out), side)
                cap += f
                hs.append((ptw + Rtw @ p_palm)[2])
            lever = float(np.mean(hs) - stance[2])  # grip height above the planted foot
        rows.append((i, com.tolist(), stance.tolist(), d, foot_names[low], cap, lever))

    d_all = np.array([r[3] for r in rows])
    worst = int(np.argmax(d_all))
    frac_out = float((d_all > 0).mean())
    tau = d_all * M * 9.81
    print(f"\n[RESULT] {len(rows)} frames")
    print(f"[RESULT] CoM is BEHIND the support edge in {100 * frac_out:.0f}% of frames")
    print(
        f"[RESULT] worst overhang {d_all.max() * 1000:+.0f} mm at frame {rows[worst][0]} "
        f"(median {np.median(d_all) * 1000:+.0f} mm)"
    )
    print(
        f"[RESULT] ankle pitch torque to hold the worst frame: {tau.max():.0f} N m "
        f"vs {ankle_cap:.0f} N m available  -> {tau.max() / ankle_cap:.1f}x over"
        if ankle_cap == ankle_cap
        else ""
    )
    if a.hands:
        caps = np.array([r[5] for r in rows])
        # moment the ankle cannot cover, and the hand force that would have to cover it
        resid = np.maximum(tau - ankle_cap, 0.0)
        lev = np.array([max(r[6], 0.05) for r in rows])  # measured grip height above the foot
        need = resid / lev
        over = need > caps
        print(f"[HANDS ] arm force capacity along the pull direction: {caps.min():.0f}..{caps.max():.0f} N (both arms)")
        print(f"[HANDS ] grip height above the planted foot: {lev.min():.2f}..{lev.max():.2f} m")
        print(
            f"[HANDS ] hand force needed to cover what the ankle cannot: "
            f"{need.max():.0f} N peak; short in {100 * over.mean():.0f}% of frames"
        )
        print(f"[SCORE2] hand_need_N={need.max():.1f} hand_cap_N={caps.min():.1f} frac_hand_short={over.mean():.3f}")
    print(
        f"[SCORE ] frac_com_outside={frac_out:.3f} worst_overhang_mm={d_all.max() * 1000:.1f} "
        f"median_overhang_mm={float(np.median(d_all)) * 1000:.1f} peak_ankle_Nm={tau.max():.1f} "
        f"ankle_cap_Nm={ankle_cap:.1f} mass_kg={M:.2f}"
    )
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(
            {
                "clip": a.clip,
                "mass_kg": M,
                "ankle_cap_Nm": ankle_cap,
                "frac_com_outside": frac_out,
                "worst_overhang_m": float(d_all.max()),
                "median_overhang_m": float(np.median(d_all)),
                "peak_ankle_Nm": float(tau.max()),
                "heel_m": a.heel,
                "rows": [
                    {"frame": r[0], "com": r[1], "stance": r[2], "overhang_m": r[3], "stance_body": r[4]} for r in rows
                ],
            },
            open(a.out, "w"),
            indent=2,
        )
        print(f"[OK    ] wrote {a.out}")
    env.close()
    return 0


code = main()
app.close()
raise SystemExit(code)
