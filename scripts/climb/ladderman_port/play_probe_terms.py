"""From the clip's start, policy driving: which termination ends each robot and when, what each reward
term pays, and what the carry hand goes through (contact force, velocity change per step).

For checking the opt-in carry-hand terms (bulb_hand.py) against a checkpoint before any training:
how often would each rule fire on the teacher as it is, and in which part of the climb.

    +upto=2000 +hand=right_wrist_.*_link
"""

import os
import sys
from collections import Counter, defaultdict

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
    robot = base.scene["robot"]
    cs = base.scene["contact_forces"]
    hand_ids, hand_names = cs.find_bodies(str(cfg.get("hand", "right_wrist_.*_link")))
    link = robot.find_bodies("right_wrist_yaw_link")[0][0]
    hi = int(cfg.get("upto", 2000))
    bins = int(cfg.get("bin", 250))
    td_ = env.reset()
    N = base.num_envs
    alive = torch.ones(N, dtype=torch.bool, device=base.device)
    ended = {}  # env -> (step, [terms])
    rew_sum = defaultdict(float)
    rew_n = 0
    events = 0
    f_bin = defaultdict(list)
    j_bin = defaultdict(list)
    prev_v = robot.data.body_com_lin_vel_w[:, link].clone()
    print(f"[terms ] terminations: {list(base.termination_funcs)}", flush=True)
    print(
        "[terms ] rewards: " + "; ".join(f"{g}: {list(grp.funcs)}" for g, grp in base.reward_groups.items()),
        flush=True,
    )
    print(f"[terms ] carry hand links: {hand_names}", flush=True)
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in range(hi):
            td_ = rollout_policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            nxt = td["next"]
            done = nxt["done"].reshape(-1).bool()
            for e in torch.nonzero(alive & done).reshape(-1).tolist():
                fired = [
                    k
                    for k in base.termination_funcs
                    if ("stats", "termination", k) in nxt.keys(True, True)
                    and bool(nxt["stats", "termination", k][e].any())
                ]
                ended[e] = (i + 1, fired or ["time limit / clip end"])
            m = alive & ~done
            if m.any():
                f = cs.data.net_forces_w_history[:, :, hand_ids].norm(dim=-1).amax(dim=(1, 2))[m]
                v = robot.data.body_com_lin_vel_w[:, link]
                jump = (v - prev_v).norm(dim=-1)[m]
                # who is the hand touching? the sensor reads ALL contact, so say where the hand is when it is loaded
                thr = float(cfg.get("event_force", 0.0))
                if thr > 0 and events < int(cfg.get("events", 30)):
                    fa = cs.data.net_forces_w_history[:, :, hand_ids].norm(dim=-1).amax(dim=(1, 2))
                    for e in torch.nonzero(m & (fa > thr)).reshape(-1).tolist()[:3]:
                        from isaaclab.utils.math import quat_apply_inverse, yaw_quat

                        fw = cs.data.net_forces_w[e, hand_ids]  # [links, 3]
                        k = int(fw.norm(dim=-1).argmax())
                        rq = yaw_quat(robot.data.root_quat_w[e : e + 1])
                        hp = quat_apply_inverse(
                            rq, (robot.data.body_link_pos_w[e, link] - robot.data.root_pos_w[e])[None]
                        )[0]
                        fb = quat_apply_inverse(rq, fw[k][None])[0]
                        lad = getattr(base, "ladder_body", None)
                        dl = (
                            float((robot.data.body_link_pos_w[e, link, :2] - lad.data.root_link_pos_w[e, :2]).norm())
                            if lad is not None
                            else float("nan")
                        )
                        bp = robot.data.body_link_pos_w[e]
                        d = (bp - bp[link]).norm(dim=-1)
                        skip = {
                            n
                            for n in robot.body_names
                            if n.startswith("right_") and any(t in n for t in ("wrist", "elbow", "shoulder"))
                        }
                        near = sorted(((float(d[j]), n) for j, n in enumerate(robot.body_names) if n not in skip))[:3]
                        print(
                            f"[event ] step {i + 1:4d} robot {e:2d}: {float(fa[e]):5.1f} N on {hand_names[k]}, force in the body frame (fwd, left, up) "
                            f"{[round(float(x)) for x in fb]}; hand at (fwd, left, up) {[round(float(x), 3) for x in hp]} from the pelvis, "
                            f"{dl:.2f} m from the ladder's root in the floor plane; nearest other links: "
                            + ", ".join(f"{n} {dd:.2f} m" for dd, n in near),
                            flush=True,
                        )
                        events += 1
                b = (i // bins) * bins
                f_bin[b].append(f.cpu())
                j_bin[b].append(jump.cpu())
                for g, grp in base.reward_groups.items():
                    for k in grp.funcs:
                        if ("stats", g, k) in nxt.keys(True, True):
                            pass
                rew_n += 1
            prev_v = robot.data.body_com_lin_vel_w[:, link].clone()
            alive &= ~done
            if not alive.any():
                break
    # reward terms: the env's own running means (weight applied, per step)
    for k, v in sorted(base.stats_ema.items()):
        if k.startswith("reward."):
            print(f"[reward] {k}: {v:+.4f}", flush=True)
    for b in sorted(f_bin):
        f = torch.cat(f_bin[b])
        j = torch.cat(j_bin[b])
        print(
            f"[hand  ] steps {b:4d}-{b + bins - 1:4d}: robot-steps {len(f):6d}; contact force median {float(f.median()):5.1f} N, 99th pct {float(f.quantile(0.99)):6.1f}, max {float(f.max()):6.1f}; "
            f"over 15 N {100 * float((f > 15).float().mean()):4.1f}%, over 50 N {100 * float((f > 50).float().mean()):4.1f}%; "
            f"velocity change per step: 99th pct {float(j.quantile(0.99)):.2f} m/s, max {float(j.max()):.2f}",
            flush=True,
        )
    c = Counter(t for _, ts in ended.values() for t in ts)
    print(f"[ended ] {len(ended)} of {N} robots ended before step {hi}; {int(alive.sum())} still running", flush=True)
    for t, n in c.most_common():
        st = sorted(s for s, ts in ended.values() if t in ts)
        print(f"[ended ] {t}: {n} robots; steps min {st[0]}, median {st[len(st) // 2]}, max {st[-1]}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
