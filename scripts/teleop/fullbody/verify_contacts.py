"""Ask PhysX, not a KD-tree, which parts of the robot actually hit the ladder.

The kinematic replay never calls sim.step(), so nothing ever consults the ladder's colliders and
a limb can sit inside a tread with no complaint. The KD-tree check in climb_sequence.py is only
a stand-in: it tests link origins and segment samples against the RENDER mesh using hand-estimated
link radii.

This runs the same replay but STEPS PHYSICS each frame, with:
  * gravity zeroed, so the only forces are contact/depenetration -- the robot cannot fall or be
    dragged, and a reported force means geometry genuinely overlaps;
  * the pose rewritten before every step, so nothing accumulates and each frame is judged on its
    own written pose;
  * a contact sensor over EVERY robot body (not the feet-and-hands subset the task ships),
    filtered to the ladder, so a knee or a thigh is reported as readily as a foot.

Non-zero force on a body = that body's real collision mesh overlaps the ladder's real collision
mesh at that frame. That is ground truth, and it replaces the radius guesses.
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
ap.add_argument("--stride", type=int, default=2, help="check every Nth frame")
ap.add_argument("--force-eps", type=float, default=1e-3, help="N, above which a contact counts")
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True   # S02 spawns the ego camera; the flag is required to build the scene even though we never render
a.headless = True
app = AppLauncher(a).app

import gymnasium as gym, numpy as np, torch
import fiatlux_task.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.sensors import ContactSensorCfg
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

    # Gravity off: the robot is posed by hand every frame, so the only thing we want PhysX to
    # produce is the reaction to geometry overlapping geometry.
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    # Every body, not the shipped feet+hands subset.
    cfg.scene.ladder_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ladder"],
        history_length=1,
        track_air_time=False,
    )

    env = gym.make(a.task, cfg=cfg); base = env.unwrapped; dev = base.device
    env.reset(seed=a.seed)
    robot = base.scene["robot"]; sensor = base.scene["ladder_contact"]
    names = list(sensor.body_names)
    print(f"[SENSOR] {len(names)} robot bodies filtered against the ladder", flush=True)
    print(f"[SIM   ] gravity {tuple(cfg.sim.gravity)}  dt {base.sim.get_physics_dt()}", flush=True)

    clip = np.load(a.clip)
    dof = clip["dof"].astype(np.float32)
    rrot = clip["root_rot"].astype(np.float32)
    rtra = clip["root_trans_offset"].astype(np.float32)
    from retarget_clearance import JOINT_ORDER as JO
    idx = [list(robot.joint_names).index(n) for n in JO]
    n = len(dof)
    print(f"[CLIP  ] {a.clip}: {n} frames, checking every {a.stride}", flush=True)

    def t_(x):
        return torch.tensor(np.asarray(x, np.float32), device=dev).unsqueeze(0)

    peak = np.zeros(len(names))
    frames_hit = {i: [] for i in range(len(names))}
    zeros6 = torch.zeros((1, 6), device=dev)
    nq = robot.num_joints
    checked = 0
    for i in range(0, n, a.stride):
        jp = robot.data.default_joint_pos[0].detach().cpu().numpy().copy()
        jp[idx] = dof[i]
        robot.write_root_pose_to_sim(torch.cat([t_(rtra[i]), t_(rrot[i])], dim=-1))
        robot.write_root_velocity_to_sim(zeros6)
        robot.write_joint_state_to_sim(t_(jp), torch.zeros((1, nq), device=dev))
        base.scene.write_data_to_sim()
        base.sim.step(render=False)                      # <-- PhysX actually runs here
        base.scene.update(base.sim.get_physics_dt())
        fm = sensor.data.force_matrix_w                  # (1, bodies, filters, 3)
        if fm is None:
            print("[FATAL] sensor returned no force matrix", flush=True); return 2
        f = torch.linalg.norm(fm[0, :, 0, :], dim=-1).detach().cpu().numpy()
        for b in np.flatnonzero(f > a.force_eps):
            frames_hit[int(b)].append(i)
            peak[b] = max(peak[b], float(f[b]))
        checked += 1
        if checked % 50 == 0:
            print(f"[CHECK ] {i}/{n}", flush=True)

    hits = [(names[b], len(v), peak[b], v[0], v[-1]) for b, v in frames_hit.items() if v]
    hits.sort(key=lambda r: -r[2])
    print(f"\n[RESULT] {checked} frames stepped through PhysX with the real collision meshes")
    if not hits:
        print("[RESULT] NO CONTACT: no robot body overlaps the ladder at any checked frame.")
    else:
        print(f"[RESULT] {len(hits)} bodies contact the ladder:")
        print(f"   {'body':34s} {'frames':>7} {'peak N':>10} {'t range':>16}")
        for nm, cnt, pk, f0, f1 in hits:
            print(f"   {nm:34s} {cnt:7d} {pk:10.1f}  {f0/50:6.2f}-{f1/50:6.2f}s")
    FOOT = ("ankle_roll",)
    nonfoot = [(nm, cnt, pk) for nm, cnt, pk, _, _ in hits if not any(f in nm for f in FOOT)]
    nf_frames = sum(c for _, c, _ in nonfoot)
    nf_peak = max([p for _, _, p in nonfoot], default=0.0)
    foot_frames = sum(c for nm, c, pk, _, _ in hits if any(f in nm for f in FOOT))
    print(f"[SCORE ] nonfoot_contact_frames={nf_frames} nonfoot_peak_N={nf_peak:.1f} "
          f"nonfoot_bodies={len(nonfoot)} foot_contact_frames={foot_frames} checked={checked}",
          flush=True)
    json.dump({"clip": a.clip, "stride": a.stride, "checked": checked,
               "nonfoot_contact_frames": nf_frames, "nonfoot_peak_N": nf_peak,
               "nonfoot_bodies": len(nonfoot), "foot_contact_frames": foot_frames,
               "bodies": names, "peak_force_N": peak.tolist(),
               "frames": {names[b]: v for b, v in frames_hit.items() if v}},
              open(os.path.join(a.out, "contacts.json"), "w"), indent=2)
    print(f"[OK    ] wrote {a.out}/contacts.json", flush=True)
    env.close(); return 0


code = main()
app.close()
raise SystemExit(code)
