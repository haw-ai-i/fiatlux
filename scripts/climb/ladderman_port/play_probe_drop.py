"""Why does the reference sag in the trainer? Open-loop reference playback, step by step.

Runs the training reset (random phase), commands the reference joint positions for N steps, and
prints per step: pelvis error against the reference, foot and hand support, leg joint error and
torque saturation. Switches, to separate causes:

  +prefill=true    the action buffer is filled with the reference action at reset. As shipped it
                   is zeros, so for the first 2-6 physics substeps the PD target is the DEFAULT
                   standing pose, whatever pose the robot was reset into.
  +no_noise=true   no reset noise on root pose, root velocity or joints.
  +no_delay=true   action delay 0 and no smoothing.
"""
import os
import sys

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type
from isaaclab.app import AppLauncher

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FILE_PATH, ".."))
sys.path.insert(0, REPO_ROOT)
import active_adaptation  # noqa: E402,F401


@hydra.main(config_path=os.path.join(REPO_ROOT, "cfg"), config_name="play", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    app = AppLauncher(OmegaConf.to_container(cfg.app)).app  # noqa: F841
    from scripts.helpers import make_env_policy
    env, policy, vecnorm = make_env_policy(cfg)
    rollout_policy = policy.get_rollout_policy("eval")
    base = env.base_env
    base.train()
    cmd = base.command_manager
    act = base.action_manager
    robot = base.scene["robot"]
    cs = base.scene["contact_forces"]
    prefill, no_noise, no_delay = (bool(cfg.get(k, False)) for k in ("prefill", "no_noise", "no_delay"))
    steps = int(cfg.get("probe_steps", 25))
    if no_noise:
        cmd.pose_range = torch.zeros_like(cmd.pose_range)
        cmd.velocity_range = torch.zeros_like(cmd.velocity_range)
        cmd.init_joint_pos_noise = cmd.init_joint_pos_noise * 0.0
        cmd.init_joint_vel_noise = cmd.init_joint_vel_noise * 0.0
    if no_delay:
        act.min_delay = act.max_delay = 0
        act.alpha_range = (1.0, 1.0)
    print(f"[drop] prefill={prefill} no_noise={no_noise} no_delay={no_delay}  delay {act.min_delay}-{act.max_delay} substeps, "
          f"alpha {act.alpha_range}, decimation {base.decimation}", flush=True)

    tr = list(cmd.tracking_joint_indices_asset)
    ids = act.joint_ids.tolist() if torch.is_tensor(act.joint_ids) else list(act.joint_ids)
    perm = [tr.index(i) for i in ids]
    names = [robot.joint_names[i] for i in ids]
    leg = [k for k, n in enumerate(names) if any(s in n for s in ("hip", "knee", "ankle"))]
    arm = [k for k, n in enumerate(names) if any(s in n for s in ("shoulder", "elbow"))]
    feet = [k for k, n in enumerate(cs.body_names) if "ankle" in n]
    hands = [k for k, n in enumerate(cs.body_names) if "wrist" in n]
    W = float(robot.root_physx_view.get_masses()[0].sum()) * 9.81
    lim = robot.data.joint_effort_limits[:, ids] if hasattr(robot.data, "joint_effort_limits") else None
    pel = cmd.asset.body_names.index("pelvis")

    def ref_action():
        return ((cmd.ref_joint_pos[:, perm] - act.default_joint_pos[:, ids]) / act.action_scaling).clamp(-10.0, 10.0)

    akp = float(cfg.get("ankle_kp", 0.0))          # DIAGNOSTIC ONLY: stiffer ankle pitch
    if akp > 0:
        aid, anm = robot.find_joints([".*_ankle_pitch_joint"])
        for nm_, a_ in robot.actuators.items():
            jl = a_.joint_indices if not isinstance(a_.joint_indices, slice) else list(range(robot.num_joints))
            jl = jl.tolist() if torch.is_tensor(jl) else list(jl)
            for c, j in enumerate(jl):
                if j in aid:
                    a_.stiffness[:, c] = akp
                    a_.damping[:, c] = a_.damping[:, c] * (akp / 40.0) ** 0.5
            print(f"[drop] actuator group {nm_}: {type(a_).__name__}", flush=True)
        kpw = robot.data.joint_stiffness.clone(); kpw[:, aid] = akp
        try:
            robot.write_joint_stiffness_to_sim(kpw)
        except Exception as ex:
            print("[drop] write stiffness:", ex, flush=True)
    efl = float(cfg.get("ankle_effort", 0.0))
    if bool(cfg.get("dump_colliders", False)):
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
        import numpy as np
        stage = base.sim.stage if hasattr(base.sim, "stage") else __import__("omni.usd").usd.get_context().get_stage()
        root = [p_ for p_ in stage.Traverse() if p_.GetName() == "left_wrist_yaw_link" and "env_0" in str(p_.GetPath())][0]
        print("[coll] link prim", root.GetPath(), flush=True)
        xc = UsdGeom.XformCache()
        Linv = xc.GetLocalToWorldTransform(root).GetInverse()
        try:
            import omni.physx
            from omni.physx import get_physx_property_query_interface
            from omni.physx.bindings._physx import PhysxPropertyQueryMode
            sid = __import__("pxr").UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
            def _cb(*a, **k): pass
            def _shape(info):
                print(f"[coll] physx collider: {getattr(info,'path',None)} aabb min {getattr(info,'aabb_local_min',None)} max {getattr(info,'aabb_local_max',None)} volume {getattr(info,'volume',None)}", flush=True)
            for cp in [root.GetPath().AppendPath("collisions")]:
                get_physx_property_query_interface().query_prim(stage_id=sid, prim_id=__import__("pxr").PhysicsSchemaTools.sdfPathToInt(cp),
                    query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS, collider_fn=_shape, finished_fn=_cb, timeout_ms=4000)
                get_physx_property_query_interface().query_prim(stage_id=sid, prim_id=__import__("pxr").PhysicsSchemaTools.sdfPathToInt(root.GetPath()),
                    query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS, collider_fn=_shape, finished_fn=_cb, timeout_ms=4000)
        except Exception as ex:
            print("[coll] property query failed:", repr(ex), flush=True)
        try:
            nshape = robot.root_physx_view.max_shapes
            print(f"[coll] robot total physx shapes {nshape}; links {robot.num_bodies}", flush=True)
            lv = base.sim.physics_sim_view.create_rigid_body_view("/World/envs/env_*/Robot/left_wrist_yaw_link")
            print(f"[coll] left_wrist_yaw_link physx shapes: {lv.max_shapes}", flush=True)
            lv2 = base.sim.physics_sim_view.create_rigid_body_view("/World/envs/env_*/Robot/torso_link")
            print(f"[coll] torso_link physx shapes: {lv2.max_shapes}", flush=True)
        except Exception as ex:
            print("[coll] shape count failed:", repr(ex), flush=True)
        for p_ in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if p_.HasAPI(UsdPhysics.CollisionAPI) or p_.IsA(UsdGeom.Mesh) or p_.IsA(UsdGeom.Gprim):
                ap = UsdPhysics.MeshCollisionAPI(p_).GetApproximationAttr().Get() if p_.HasAPI(UsdPhysics.MeshCollisionAPI) else None
                info = ""
                if p_.IsA(UsdGeom.Mesh):
                    pts = np.array(UsdGeom.Mesh(p_).GetPointsAttr().Get() or [])
                    if len(pts):
                        M = np.array(xc.GetLocalToWorldTransform(p_) * Linv)
                        q_ = pts @ M[:3, :3] + M[3, :3]
                        info = f"{len(pts)} pts, in link frame x [{q_[:,0].min():+.3f},{q_[:,0].max():+.3f}] y [{q_[:,1].min():+.3f},{q_[:,1].max():+.3f}] z [{q_[:,2].min():+.3f},{q_[:,2].max():+.3f}]"
                co = PhysxSchema.PhysxCollisionAPI(p_) if p_.HasAPI(PhysxSchema.PhysxCollisionAPI) else None
                print(f"[coll]   schemas {list(p_.GetAppliedSchemas())} instanceable {p_.IsInstanceable()} proxy {p_.IsInstanceProxy()}", flush=True)
                print(f"[coll] {str(p_.GetPath()).split('left_wrist_yaw_link')[-1] or '/'} type {p_.GetTypeName()} collision {p_.HasAPI(UsdPhysics.CollisionAPI)} approx {ap} "
                      f"contactOffset {co.GetContactOffsetAttr().Get() if co else None} restOffset {co.GetRestOffsetAttr().Get() if co else None} {info}", flush=True)
    td_ = env.reset()
    cmd.update()          # the reference arrays are stale until the first step; without this the first target is a wrong pose
    if prefill:
        a = ref_action()
        act.action_buf[:] = a.unsqueeze(-1)
        act.applied_action[:] = a
    kp = robot.data.joint_stiffness[0, ids] if hasattr(robot.data, "joint_stiffness") else None
    if kp is not None:
        print("[drop] kp: " + ", ".join(f"{n.replace('_joint','')} {float(kp[k]):.0f}" for k, n in enumerate(names) if n.startswith(("left", "waist"))), flush=True)
    d0 = (robot.data.joint_pos[:, ids] - act.default_joint_pos[:, ids]).abs()
    print(f"[drop] reset pose vs DEFAULT pose, median |dq| rad: legs {float(d0[:, leg].median()):.2f} (max joint median {float(d0[:, leg].median(0).values.max()):.2f}), "
          f"arms {float(d0[:, arm].median()):.2f} (max {float(d0[:, arm].median(0).values.max()):.2f})", flush=True)
    def per_joint(tag):
        e = (robot.data.joint_pos[:, ids] - cmd.ref_joint_pos[:, perm]) * 57.3
        print(f"[drop] {tag} q-ref deg, median (p95 |.|): " + ", ".join(
            f"{names[k].replace('_joint','')} {float(e[:, k].median()):+.1f} ({float(e[:, k].abs().quantile(0.95)):.0f})" for k in leg), flush=True)
        tq = robot.data.applied_torque[:, ids]
        print(f"[drop] {tag} torque N.m median |.|: " + ", ".join(f"{names[k].replace('_joint','')} {float(tq[:, k].abs().median()):.0f}" for k in leg), flush=True)
    # kinematic agreement at the reset frame itself, before any physics: robot bodies against the clip's
    try:
        base.sim.forward() if hasattr(base.sim, "forward") else None
        robot.update(0.0)
        mo = cmd._motion_reset
        mb = list(cmd.dataset.body_names) if hasattr(cmd.dataset, "body_names") else None
        print("[drop] motion body names available:", mb is not None, flush=True)
        o = base.scene.env_origins
        qj = (robot.data.joint_pos - mo.joint_pos[:, cmd.asset_joint_idx_motion]).abs().max() * 57.3
        print(f"[drop] at reset, fresh state: max |q - clip q| over all joints/envs {float(qj):.2f} deg", flush=True)
        for b in ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_knee_link"):
            e = (robot.data.body_link_pos_w[:, robot.body_names.index(b)] - o - mo.body_pos_w[:, mb.index(b)]) * 1000
            print(f"[drop] at reset {b}: robot - clip mm, median x {float(e[:,0].median()):+.1f} y {float(e[:,1].median()):+.1f} z {float(e[:,2].median()):+.1f}; "
                  f"p95 |.| {float(e.norm(dim=-1).quantile(0.95)):.1f}", flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc(); print("[drop] reset check failed:", ex, flush=True)
    per_joint("at reset")
    sl = robot.data.soft_joint_pos_limits[0, ids]
    hl = robot.data.joint_pos_limits[0, ids] if hasattr(robot.data, "joint_pos_limits") else sl
    print("[drop] limits soft/hard deg: " + ", ".join(f"{names[k].replace('_joint','')} [{float(sl[k,0])*57.3:.0f},{float(sl[k,1])*57.3:.0f}]/[{float(hl[k,0])*57.3:.0f},{float(hl[k,1])*57.3:.0f}]" for k in leg if names[k].startswith("left")), flush=True)
    alive = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
    print("[drop] step | pelvis z err mm (median, p10) | pelvis x err mm | feet Fz %W | hands Fz %W | leg |q-ref| max deg | "
          "leg |target-q| max deg | envs w/ a leg joint at torque limit %", flush=True)
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in range(steps):
            td_ = rollout_policy(td_)
            td_["action"] = ref_action()
            td, td_ = env.step_and_maybe_reset(td_)
            done = td.get(("next", "done"), td.get("done")).reshape(-1).bool()
            alive &= ~done
            m = alive
            ez = (robot.data.body_link_pos_w[m, pel, 2] - cmd.ref_root_pos_w[m, 2]) * 1000
            ex = (robot.data.body_link_pos_w[m, pel, 0] - cmd.ref_root_pos_w[m, 0]) * 1000
            F = cs.data.net_forces_w[m]
            qe = (robot.data.joint_pos[m][:, ids] - cmd.ref_joint_pos[m][:, perm]).abs()[:, leg].max(1).values * 57.3
            te = (robot.data.joint_pos_target[m][:, ids] - robot.data.joint_pos[m][:, ids]).abs()[:, leg].max(1).values * 57.3
            tq = robot.data.applied_torque[m][:, ids]
            sat = ((tq.abs() >= 0.98 * lim[m])[:, leg].any(1).float().mean() * 100) if lim is not None else float("nan")
            if i in (0, 2, 4, 9, 14, 24):
                kn = list(cmd.tracking_keypoint_names)
                out = []
                for b in ("left_ankle_roll_link", "right_ankle_roll_link", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"):
                    e = (robot.data.body_link_pos_w[m][:, robot.body_names.index(b)] - cmd.ref_body_pos_w[m][:, kn.index(b)]) * 1000
                    out.append(f"{b.replace('_link','')} x {float(e[:, 0].median()):+.0f} z {float(e[:, 2].median()):+.0f}")
                planted = cmd.ref_body_pos_w[m][:, kn.index("left_ankle_roll_link")]
                print(f"[drop] body pos err mm after step {i+1}: " + "; ".join(out), flush=True)
            if i in (0, 1, 2, 3, 5, 9):
                kn = list(cmd.tracking_keypoint_names)
                hold = torch.stack([cmd.ref_object_contact_left_hand.squeeze(-1), cmd.ref_object_contact_right_hand.squeeze(-1)], 1)
                jn_all = robot.joint_names
                for h, sd in enumerate(("left", "right")):
                    b = f"{sd}_wrist_yaw_link"
                    e = (robot.data.body_link_pos_w[:, robot.body_names.index(b)] - cmd.ref_body_pos_w[:, kn.index(b)]) * 1000
                    Fh = cs.data.net_forces_w[:, cs.body_names.index(b)]
                    aj = [jn_all.index(f"{sd}_{n}_joint") for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
                    qe = (robot.data.joint_pos[:, aj] - robot.data.joint_pos_target[:, aj]) * 57.3
                    for v in (0, 4, 5, 6):
                        mm = alive & (hold[:, h] == v)
                        if mm.sum() < 20:
                            continue
                        print(f"[hands] step {i+1} {sd[0].upper()} hold {v} (n {int(mm.sum())}): wrist err mm x {float(e[mm,0].median()):+.0f} y {float(e[mm,1].median()):+.0f} z {float(e[mm,2].median()):+.0f} | "
                              f"F N x {float(Fh[mm,0].median()):+.0f} z {float(Fh[mm,2].median()):+.0f} |F| med {float(Fh[mm].norm(dim=-1).median()):.0f} p90 {float(Fh[mm].norm(dim=-1).quantile(0.9)):.0f} | "
                              f"q-target deg sp {float(qe[mm,0].median()):+.1f} sr {float(qe[mm,1].median()):+.1f} sy {float(qe[mm,2].median()):+.1f} el {float(qe[mm,3].median()):+.1f} wr {float(qe[mm,4].median()):+.1f} wp {float(qe[mm,5].median()):+.1f} wy {float(qe[mm,6].median()):+.1f}", flush=True)
            if i in (0, 1, 9):
                per_joint(f"after step {i+1}")
            if i < 8 or (i + 1) % 5 == 0:
                print(f"[drop] {i+1:3d} | {float(ez.median()):+6.0f} {float(ez.quantile(0.1)):+6.0f} | {float(ex.median()):+5.0f} | "
                      f"{float(F[:, feet, 2].sum(1).median())/W*100:5.0f} | {float(F[:, hands, 2].sum(1).median())/W*100:5.0f} | "
                      f"{float(qe.median()):5.1f} | {float(te.median()):5.1f} | {float(sat):4.0f}   (n {int(m.sum())})", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
