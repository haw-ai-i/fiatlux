"""From the ground, policy driving: medians over the robots still up of the root error against the
reference (robot yaw frame), the tilt, and the joints furthest from the reference. The trainer-side
twin of record_climb.py --trace in the fiatlux repo.

    +upto=200 +every=10
"""
import os
import sys

import hydra
import numpy as np
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
    term = base.termination_funcs["cum_ladder_lost_contact_steps"]
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    hi = int(cfg.get("upto", 200)); every = int(cfg.get("every", 10))
    td_ = env.reset()
    N = base.num_envs
    alive = torch.ones(N, dtype=torch.bool, device=base.device)
    jn = list(robot.joint_names)
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in range(hi):
            td_ = rollout_policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            done = td.get(("next", "done"), td.get("done")).reshape(-1).bool()
            alive &= ~done
            if not alive.any():
                break
            if (i + 1) % every == 0:
                m = alive
                e = quat_apply_inverse(yaw_quat(robot.data.root_quat_w[m]), robot.data.root_pos_w[m] - cmd.ref_root_pos_w[m]) * 1000
                g = robot.data.projected_gravity_b[m]
                ref = cmd.future_ref_motion.joint_pos[:, 0][:, cmd.asset_joint_idx_motion][m]
                je = (robot.data.joint_pos[m][:, :ref.shape[1]] - ref).median(0).values
                top = torch.argsort(je.abs(), descending=True)[:5].tolist()
                act = td_["action"] if "action" in td_.keys() else None
                print(f"[TRACE ] {i+1:4d} alive {int(m.sum()):3d} root err fwd {float(e[:,0].median()):+5.0f} left {float(e[:,1].median()):+5.0f} up {float(e[:,2].median()):+5.0f} mm; "
                      f"tilt fwd {float(torch.rad2deg(torch.atan2(g[:,0], -g[:,2])).median()):+5.1f} side {float(torch.rad2deg(torch.atan2(g[:,1], -g[:,2])).median()):+5.1f} deg; joints: "
                      + ", ".join(f"{jn[k].replace('_joint','')} {float(torch.rad2deg(je[k])):+.0f}" for k in top), flush=True)
                lb = getattr(base, "ladder_body", None)
                if lb is not None:
                    dp = (lb.data.root_link_pos_w - base.scene.env_origins)[m] * 1000
                    lg = lb.data.projected_gravity_b[m] if hasattr(lb.data, "projected_gravity_b") else None
                    tilt = torch.rad2deg(torch.acos((-lg[:, 2]).clamp(-1, 1))) if lg is not None else torch.zeros(1, device=base.device)
                    print(f"[LADDER] {i+1:4d} free body moved: median |xy| {float(dp[:, :2].norm(dim=-1).median()):.1f} mm, max {float(dp[:, :2].norm(dim=-1).max()):.0f} mm; "
                          f"z median {float(dp[:, 2].median()):+.1f} mm; tilt median {float(tilt.median()):.2f} max {float(tilt.max()):.1f} deg; "
                          f"object 0 at {[round(float(v), 3) for v in (base.ladder.data.object_pos_w[m][0, 0] - base.scene.env_origins[m][0]).tolist()]}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
