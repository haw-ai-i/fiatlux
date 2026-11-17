"""Does the climb stand up with gravity ON and only the actuators holding it?

Everything measured so far is static and kinematic. `verify_contacts.py` writes the pose every
frame and zeroes gravity, so it answers "does this geometry overlap"; `measure_balance.py` asks
"if the robot were frozen here, could the ankle hold it". Neither simulates a climb.

This does. The root is FREE -- never written after the first frame -- and the clip's joint values
go in as PD position TARGETS, exactly as a controller would issue them. Gravity is on. Whatever
happens next is the robot's own doing.

What it reports, in order of what you would want to know:
  * how long it stays up, and what "up" stopped meaning (tilt, or the pelvis dropping)
  * which joint loses its target first, and by how much -- that separates "the posture is wrong"
    from "the posture is fine but it needs a balance controller"
  * whether the ladder itself moves, since S02 ships it as a dynamic body

A prior result sets expectations: a no-policy G1 held in a scripted pose topples in about 1.5 s
(ankle stiffness 150 against roughly 250 N m/rad of gravitational load). So a short survival time
here is the EXPECTED result and is not by itself evidence against the posture. The useful signal
is where it diverges and in what order.
"""
from __future__ import annotations
import argparse, json, math, os, sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--robot", default="dex3", choices=["inspire", "dex3"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ladder", default="dynamic", choices=["kinematic", "dynamic"],
                help="dynamic is what S02 ships and is the honest test; kinematic separates "
                     "'the robot fell' from 'the ladder moved under it'.")
ap.add_argument("--gain-scale", type=float, default=1.0,
                help="multiply joint stiffness/damping. NOT a claim about the real robot -- a "
                     "diagnostic. If the posture holds at high gain WITHIN the effort limits, the "
                     "posture is feasible and what is missing is feedforward, i.e. a balance "
                     "controller. If it falls even then, the posture is wrong.")
ap.add_argument("--settle", type=int, default=60, help="physics steps to settle before playback")
ap.add_argument("--speed", type=float, default=1.0, help="<1 plays the clip slower")
ap.add_argument("--tilt-limit", type=float, default=40.0, help="deg of base tilt counted as a fall")
ap.add_argument("--passive", default="",
                help="comma-separated joint-name substrings to HOLD AT DEFAULT instead of\n                      commanding from the clip. LadderMan drives 23 of 29 joints and leaves\n                      the six wrists passive, so replaying its clip with --passive wrist is\n                      the apples-to-apples comparison; empty (default) keeps every joint\n                      commanded, which is the original behaviour.")
ap.add_argument("--drop-limit", type=float, default=0.25, help="m of pelvis drop counted as a fall")
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import gymnasium as gym, numpy as np, torch
import fiatlux_task.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    os.makedirs(a.out, exist_ok=True)
    set_layout_seed(a.seed)
    cfg = parse_env_cfg(a.task, device=a.device, num_envs=1, use_fabric=True)
    cfg.seed = a.seed
    if a.robot != "inspire":
        from fiatlux_task.robots.g1 import swap_robot_variant
        swap_robot_variant(cfg, a.robot)
    if a.ladder == "kinematic":
        cfg.scene.ladder.spawn.rigid_props.kinematic_enabled = True
    # gravity is LEFT ALONE -- that is the whole point

    env = gym.make(a.task, cfg=cfg); base = env.unwrapped; dev = base.device
    env.reset(seed=a.seed)
    robot = base.scene["robot"]; ladder = base.scene["ladder"]
    dt = base.sim.get_physics_dt()
    if a.gain_scale != 1.0:
        robot.write_joint_stiffness_to_sim(robot.data.joint_stiffness * a.gain_scale)
        robot.write_joint_damping_to_sim(robot.data.joint_damping * math.sqrt(a.gain_scale))
        print(f"[GAINS ] stiffness x{a.gain_scale:g}, damping x{math.sqrt(a.gain_scale):.2f} "
              f"(effort limits unchanged)", flush=True)
    print(f"[SIM   ] gravity {tuple(cfg.sim.gravity)}  dt {dt}  ladder={a.ladder}", flush=True)

    clip = np.load(a.clip)
    dof = clip["dof"].astype(np.float32)
    rrot = clip["root_rot"].astype(np.float32)
    rtra = clip["root_trans_offset"].astype(np.float32)
    fps = float(clip["fps"]) if "fps" in clip else 50.0
    from retarget_clearance import JOINT_ORDER as JO
    jn = list(robot.joint_names)
    idx = [jn.index(n) for n in JO]
    subs = [s.strip() for s in a.passive.split(",") if s.strip()]
    pas = [c for c, n in enumerate(JO) if any(s in n for s in subs)]
    act = [c for c in range(len(JO)) if c not in pas]
    idx_act = [idx[c] for c in act]
    if pas:
        print(f"[PASSIVE] holding {len(pas)} joints at default, commanding {len(act)}: "
              f"{', '.join(JO[c] for c in pas)}", flush=True)
    nq = robot.num_joints
    sub = max(int(round((1.0 / fps) / dt / max(a.speed, 1e-6))), 1)
    print(f"[CLIP  ] {len(dof)} frames @ {fps:g} fps -> {sub} physics steps per frame "
          f"({len(dof)/fps:.1f} s of motion)", flush=True)

    def t_(x):
        return torch.tensor(np.asarray(x, np.float32), device=dev).unsqueeze(0)

    # --- start from the clip's first pose, then let go -----------------------------------------
    jp0 = robot.data.default_joint_pos[0].detach().cpu().numpy().copy()
    jp0[idx_act] = dof[0][act]
    robot.write_root_pose_to_sim(torch.cat([t_(rtra[0]), t_(rrot[0])], dim=-1))
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
    robot.write_joint_state_to_sim(t_(jp0), torch.zeros((1, nq), device=dev))
    lad0 = ladder.data.root_pos_w[0].clone()
    tgt = t_(jp0)
    for _ in range(a.settle):
        robot.set_joint_position_target(tgt)
        base.scene.write_data_to_sim(); base.sim.step(render=False)
        base.scene.update(dt)
    pelvis0 = float(robot.data.root_pos_w[0, 2])
    print(f"[SETTLE] {a.settle} steps; pelvis at {pelvis0:.4f} m, "
          f"commanded {rtra[0][2]:.4f} m (drift {(pelvis0-rtra[0][2])*1000:+.0f} mm)", flush=True)

    # --- play the clip as targets, root free ---------------------------------------------------
    rows = []
    fall_t, fall_why = None, ""
    worst = np.zeros(len(JO))
    sat = np.zeros(len(JO))
    eff_lim = robot.root_physx_view.get_dof_max_forces()[0].detach().cpu().numpy()[idx]
    for i in range(len(dof)):
        jp = robot.data.default_joint_pos[0].detach().cpu().numpy().copy()
        jp[idx_act] = dof[i][act]
        tgt = t_(jp)
        for _ in range(sub):
            robot.set_joint_position_target(tgt)
            base.scene.write_data_to_sim(); base.sim.step(render=False)
            base.scene.update(dt)
        q = robot.data.root_quat_w[0].detach().cpu().numpy().astype(float)
        up_z = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)          # body +z . world +z
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
        pz = float(robot.data.root_pos_w[0, 2])
        err = np.abs(robot.data.joint_pos[0].detach().cpu().numpy()[idx] - dof[i])
        # A passive joint is not tracking the clip by choice, so scoring it against the clip
        # reports a deviation we asked for as if it were a failure. Torque still counts: a
        # held-at-default wrist can still saturate under the hand's own weight.
        err[pas] = 0.0
        worst = np.maximum(worst, err)
        tq = np.abs(robot.data.applied_torque[0].detach().cpu().numpy()[idx])
        sat = np.maximum(sat, tq / np.maximum(eff_lim, 1e-6))
        drop = rtra[i][2] - pz
        lad_d = float(torch.linalg.norm(ladder.data.root_pos_w[0] - lad0))
        rows.append({"frame": i, "t": i / fps, "tilt_deg": tilt, "pelvis_z": pz,
                     "pelvis_drop_m": float(drop), "max_joint_err_rad": float(err.max()),
                     "worst_joint": JO[int(np.argmax(err))], "ladder_drift_m": lad_d})
        if fall_t is None:
            if tilt > a.tilt_limit:
                fall_t, fall_why = i / fps, f"base tilted {tilt:.0f} deg"
            elif drop > a.drop_limit:
                fall_t, fall_why = i / fps, f"pelvis {drop*1000:.0f} mm below the commanded height"
        if i % 50 == 0:
            print(f"[PLAY  ] t={i/fps:5.2f}s tilt {tilt:5.1f} deg  drop {drop*1000:+6.0f} mm  "
                  f"worst joint {JO[int(np.argmax(err))]} {math.degrees(err.max()):.1f} deg  "
                  f"ladder {lad_d*1000:.0f} mm", flush=True)

    dur = len(dof) / fps
    print(f"\n[RESULT] played {dur:.1f} s of clip")
    if fall_t is None:
        print(f"[RESULT] STAYED UP for the whole climb "
              f"(worst tilt {max(r['tilt_deg'] for r in rows):.1f} deg, "
              f"worst drop {max(r['pelvis_drop_m'] for r in rows)*1000:.0f} mm)")
    else:
        print(f"[RESULT] FELL at t={fall_t:.2f}s of {dur:.1f}s -- {fall_why}")
    order = np.argsort(-worst)[:6]
    print("[RESULT] joints that lost their target by the most:")
    for k in order:
        print(f"   {JO[k]:28s} {math.degrees(worst[k]):7.1f} deg")
    k_sat = np.argsort(-sat)[:5]
    print("[TORQUE] closest to the effort limit (1.00 = saturated):")
    for k in k_sat:
        print(f"   {JO[k]:28s} {sat[k]:5.2f} of limit {eff_lim[k]:6.1f} N m")
    print(f"[LADDER] moved {max(r['ladder_drift_m'] for r in rows)*1000:.0f} mm during the replay")
    print(f"[SCORE ] survived_s={fall_t if fall_t is not None else dur:.2f} "
          f"of_s={dur:.2f} fell={int(fall_t is not None)} "
          f"worst_tilt_deg={max(r['tilt_deg'] for r in rows):.1f} "
          f"worst_joint_err_deg={math.degrees(worst.max()):.1f} "
          f"worst_joint={JO[int(np.argmax(worst))]} "
          f"ladder_moved_mm={max(r['ladder_drift_m'] for r in rows)*1000:.0f}", flush=True)
    json.dump({"clip": a.clip, "ladder": a.ladder, "fps": fps, "sub_steps": sub,
               "fell": fall_t is not None, "fall_t_s": fall_t, "fall_reason": fall_why,
               "duration_s": dur, "worst_joint_err_rad": worst.tolist(), "joints": JO,
               "rows": rows}, open(os.path.join(a.out, "dynamic.json"), "w"), indent=2)
    print(f"[OK    ] wrote {a.out}/dynamic.json", flush=True)
    env.close(); return 0


code = main()
app.close()
raise SystemExit(code)
