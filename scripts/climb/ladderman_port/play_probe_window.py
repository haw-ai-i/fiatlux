"""From the ground, policy driving: step by step through a window of the clip, what does each hand
satisfy of the hand-hold rule, and when do robots get terminated?

    +win=[600,1050]  clip steps to report (every `+every` steps)
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
    lo, hi = [int(x) for x in cfg.get("win", [600, 1050])]
    every = int(cfg.get("every", 10))
    kn = list(cmd.tracking_keypoint_names)
    td_ = env.reset()
    N = base.num_envs
    alive = torch.ones(N, dtype=torch.bool, device=base.device)
    died = np.full(N, -1)
    cause = [""] * N
    terms = [k for k, f in base.termination_funcs.items() if hasattr(f, "cum_steps")]
    print("[win] step | alive | per hand L,R: label | within 18 cm % | above target % | force ok % | rule counter median | "
          "eef-target median xyz mm | wrist F median xz N | pelvis z err mm | feet Fz %W", flush=True)
    W = float(robot.root_physx_view.get_masses()[0].sum()) * 9.81
    feet = [k for k, n in enumerate(cs.body_names) if "ankle" in n]
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in range(hi + 1):
            td_ = rollout_policy(td_)
            prev = {k: base.termination_funcs[k].cum_steps.clone() for k in terms}
            td, td_ = env.step_and_maybe_reset(td_)
            done = td.get(("next", "done"), td.get("done")).reshape(-1).bool()
            for e in torch.nonzero(alive & done).reshape(-1).tolist():
                died[e] = i + 1
                cause[e] = "+".join(k.replace("cum_", "") for k in terms if int(prev[k][e]) >= base.termination_funcs[k].min_steps - 1)
            alive &= ~done
            if not alive.any():
                break
            if i + 1 >= lo and (i + 1) % every == 0:
                m = alive
                diff = (cmd.contact_eef_pos_w - cmd.contact_target_pos_w)[m]
                F = cs.data.net_forces_w[m][:, term.body_ids]
                fn = F.norm(dim=-1)
                ang = F[..., 2] / (fn + 1e-9) > np.cos(np.deg2rad(term.max_force_angle_deg))
                fok = ang & (F[..., 2] > term.min_upward_force)
                lab = torch.stack([cmd.ref_object_contact_left_hand.squeeze(-1), cmd.ref_object_contact_right_hand.squeeze(-1)], 1)[m]
                ez = (robot.data.body_link_pos_w[m, robot.body_names.index("pelvis"), 2] - cmd.ref_root_pos_w[m, 2]) * 1000
                fz = cs.data.net_forces_w[m][:, feet, 2].sum(1).median() / W * 100
                parts = []
                for h in range(2):
                    parts.append(f"{int(lab[:, h].median())} | {float((diff[:, h].norm(dim=-1) < term.pos_thres).float().mean())*100:3.0f} | "
                                 f"{float((diff[:, h, 2] > 0).float().mean())*100:3.0f} | {float(fok[:, h].float().mean())*100:3.0f} | "
                                 f"[{int(diff[:, h, 0].median()*1000):+4d},{int(diff[:, h, 1].median()*1000):+4d},{int(diff[:, h, 2].median()*1000):+4d}] | "
                                 f"F[{float(F[:, h, 0].median()):+4.0f},{float(F[:, h, 2].median()):+4.0f}]")
                print(f"[win] {i+1:4d} | {int(m.sum()):3d} | L {parts[0]} || R {parts[1]} || ctr {int(term.cum_steps[m].median()):2d} | pz {float(ez.median()):+4.0f} | feet {float(fz):3.0f}", flush=True)
    from collections import Counter
    d = died[died > 0]
    print("[win] terminations by 50-step window: " + " ".join(f"{a}:{int(((d >= a) & (d < a + 50)).sum())}" for a in range(0, hi + 50, 50) if ((d >= a) & (d < a + 50)).sum()), flush=True)
    print("[win] causes in window: " + str(Counter(c for e, c in enumerate(cause) if died[e] >= lo).most_common(5)), flush=True)
    print(f"[win] alive at end of window: {int(alive.sum())}/{N}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
