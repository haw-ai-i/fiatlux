"""The step onto a tread, policy driving, in numbers the benchmark recorder prints too (FX_TRACE_COM=1).

Per step, medians over the robots still up, everything in the clip's frame (x toward the ladder, z up):
the whole-body centre of mass relative to each ankle and its forward speed, each ankle against the
clip, and the contact force on each foot and on the left hand.

    +lo=440 +upto=540 +every=2
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
    env.base_env.eval()
    base = env.base_env
    cmd = base.command_manager
    robot = base.scene["robot"]
    cs = base.scene["contact_forces"]
    lo = int(cfg.get("lo", 440)); hi = int(cfg.get("upto", 540)); every = int(cfg.get("every", 2))
    bn = list(robot.body_names)
    la, ra = bn.index("left_ankle_roll_link"), bn.index("right_ankle_roll_link")
    kn = list(cmd.tracking_keypoint_names)
    kla, kra = kn.index("left_ankle_roll_link"), kn.index("right_ankle_roll_link")
    f_l, _ = cs.find_bodies("left_ankle_roll_link"); f_r, _ = cs.find_bodies("right_ankle_roll_link"); f_h, _ = cs.find_bodies("left_wrist_.*_link")
    mass = robot.root_physx_view.get_masses().to(base.device)                    # [N, B]
    print(f"[MOUNT ] robot mass median {float(mass.sum(1).median()):.2f} kg; frame: x toward the ladder, z up; medians over robots still up", flush=True)
    td_ = env.reset()
    N = base.num_envs
    alive = torch.ones(N, dtype=torch.bool, device=base.device)
    prev = None
    md = lambda x: float(x.median())
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in range(hi):
            td_ = rollout_policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            done = td.get(("next", "done"), td.get("done")).reshape(-1).bool()
            alive &= ~done
            if not alive.any():
                break
            com = (mass[:, :, None] * robot.data.body_com_pos_w).sum(1) / mass.sum(1, keepdim=True)
            if i + 1 >= lo and (i + 1) % every == 0:
                m = alive
                o = base.scene.env_origins
                L = robot.data.body_link_pos_w[:, la]; R = robot.data.body_link_pos_w[:, ra]
                eL = (L - cmd.ref_body_pos_w[:, kla])[m] * 1000; eR = (R - cmd.ref_body_pos_w[:, kra])[m] * 1000
                cL = (com - L)[m] * 1000; cR = (com - R)[m] * 1000
                vx = ((com - prev)[:, 0] / (0.02 * 1))[m] if prev is not None else torch.zeros(1, device=base.device)
                F = cs.data.net_forces_w
                fl = F[:, f_l].sum(1)[m]; fr = F[:, f_r].sum(1)[m]; fh = F[:, f_h].sum(1)[m]
                print(f"[MOUNT ] {i+1:4d} alive {int(m.sum()):3d} | com-Lankle x {md(cL[:,0]):+5.0f} z {md(cL[:,2]):+5.0f}  com-Rankle x {md(cR[:,0]):+5.0f} mm  com vx {md(vx):+5.2f} m/s"
                      f" | L ankle err x {md(eL[:,0]):+4.0f} y {md(eL[:,1]):+4.0f} z {md(eL[:,2]):+4.0f}  R ankle err x {md(eR[:,0]):+4.0f} y {md(eR[:,1]):+4.0f} z {md(eR[:,2]):+4.0f} mm"
                      f" | L foot F x {md(fl[:,0]):+5.0f} z {md(fl[:,2]):+5.0f}  R foot F x {md(fr[:,0]):+5.0f} z {md(fr[:,2]):+5.0f}  L hand F x {md(fh[:,0]):+4.0f} z {md(fh[:,2]):+4.0f} N", flush=True)
            prev = com.clone()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
