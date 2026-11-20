"""Which half of the hand-hold termination fails: position or force?

cum_ladder_lost_contact_steps needs, for each hand the clip says is on a hold, (a) the hand point
within pos_thres of the hold target, (b) that point ABOVE the target (strict_position) and (c) an
upward contact force on the wrist_yaw_link. This runs the training reset, steps 20 times (before
the 30-step limit can fire) and reports each condition's pass rate per hold id.
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
    term = base.termination_funcs["cum_ladder_lost_contact_steps"]
    cs = base.scene["contact_forces"]
    print("[hand] sensor bodies:", list(cs.body_names), flush=True)
    print("[hand] term bodies:", [cs.body_names[i] for i in term.body_ids.tolist()],
          "pos_thres", term.pos_thres, "strict", term.strict_position, flush=True)
    act = base.action_manager
    zero = bool(cfg.get("zero_residual", False))
    if zero:
        tr = list(cmd.tracking_joint_indices_asset)
        ids = act.joint_ids.tolist() if torch.is_tensor(act.joint_ids) else list(act.joint_ids)
        perm = [tr.index(i) if i in tr else -1 for i in ids]
    print("[hand] mode:", "open-loop reference" if zero else "policy", flush=True)
    td_ = env.reset()
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in range(int(cfg.get("probe_steps", 20))):
            td_ = rollout_policy(td_)
            if zero:
                tgt = act.default_joint_pos[:, act.joint_ids].clone()
                for a_i, r_i in enumerate(perm):
                    if r_i >= 0:
                        tgt[:, a_i] = cmd.ref_joint_pos[:, r_i]
                td_["action"] = ((tgt - act.default_joint_pos[:, act.joint_ids]) / act.action_scaling).clamp(-10.0, 10.0)
            td, td_ = env.step_and_maybe_reset(td_)
    for k, f in base.termination_funcs.items():
        if hasattr(f, "cum_steps"):
            print(f"[hand] counter {k}: over threshold now in {(f.cum_steps > 0).float().mean()*100:.0f}% of envs", flush=True)
    diff = cmd.contact_eef_pos_w - cmd.contact_target_pos_w          # [N, 2, 3]
    F = cs.data.net_forces_w                                          # [N, bodies, 3]
    Fw = F[:, term.body_ids]
    hold = torch.stack([cmd.ref_object_contact_left_hand.squeeze(-1), cmd.ref_object_contact_right_hand.squeeze(-1)], 1)
    hand_ids = [k for k, n in enumerate(cs.body_names) if any(s in n for s in ("hand", "wrist", "palm", "thumb", "index", "middle"))]
    for h, nm in enumerate(("L", "R")):
        for v in sorted(set(hold[:, h].tolist())):
            m = hold[:, h] == v
            if v == 0 or m.sum() == 0:
                continue
            d = diff[m, h]
            f = Fw[m, h]
            fn = f.norm(dim=-1)
            other = F[m][:, hand_ids].norm(dim=-1)                    # every hand-ish body
            print(f"[hand] {nm} hold id {int(v)}: n {int(m.sum())}  "
                  f"dist<thres {(d.norm(dim=-1) < term.pos_thres).float().mean()*100:.0f}%  "
                  f"above target {(d[:, 2] > 0).float().mean()*100:.0f}%  "
                  f"wrist force up>5N {(f[:, 2] > term.min_upward_force).float().mean()*100:.0f}%  "
                  f"| median diff xyz mm {[int(x*1000) for x in d.median(0).values.tolist()]}  "
                  f"wrist |F| median {fn.median():.1f} N  Fz median {f[:, 2].median():.1f} N", flush=True)
            wz = cmd.asset.data.body_link_pos_w[m][:, cmd.contact_eef_body_indices_asset[h], 2] - base.scene.env_origins[m, 2]
            print(f"[hand]     abs z (m): eef {float((cmd.contact_eef_pos_w[m, h, 2] - base.scene.env_origins[m, 2]).median()):.4f}  "
                  f"target {float((cmd.contact_target_pos_w[m, h, 2] - base.scene.env_origins[m, 2]).median()):.4f}  wrist link {float(wz.median()):.4f}", flush=True)
            print("[hand]     median |F| per body: " + ", ".join(
                f"{cs.body_names[k]} {other[:, j].median():.0f}" for j, k in enumerate(hand_ids) if other[:, j].median() > 0.5), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
