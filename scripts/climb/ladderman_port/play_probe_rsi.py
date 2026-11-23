"""Where in the clip does a teacher die, and which termination kills it?

A sibling of play_probe.py. That one forces start_t = 0 and reports pelvis gain. This one keeps
the TRAINING reset (random clip phase, reference state init) and, for every finished episode,
records the start phase, the phase it ended at, and every termination term whose counter had
reached its limit. Output: an .npz of episodes plus a printed table per 100-step clip window.

    python scripts/play_probe_rsi.py algo=teacher task=teacher checkpoint_path=<...> \
        headless=true task.num_envs=512 +probe_steps=600 +probe_out=<file.npz> <overrides>
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
    base = env.base_env
    base.train()  # keep the training reset: random phase
    cmd = base.command_manager
    robot = base.scene["robot"]
    steps = int(cfg.get("probe_steps", 600))
    out = str(cfg.get("probe_out", "probe_rsi.npz"))
    terms = [k for k, f in base.termination_funcs.items() if hasattr(f, "cum_steps")]
    td_ = env.reset()
    L = int(cmd.motion_len[0])  # only known after the first reset
    print(f"[rsi] envs={base.num_envs} steps={steps} motion_len={L} terms={terms}", flush=True)
    t0 = cmd.t.clone()
    z0 = (robot.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2]).clone()
    ep = []  # start_t, end_t, pelvis dz, then one flag per term
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in range(steps):
            td_ = rollout_policy(td_)
            t_before = cmd.t.clone()
            z_before = robot.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2]
            prev = {
                k: (base.termination_funcs[k].cum_steps.clone(), base.termination_funcs[k].error.clone()) for k in terms
            }
            td, td_ = env.step_and_maybe_reset(td_)
            done = td.get(("next", "done"), td.get("done")).reshape(-1).bool()
            for e in torch.nonzero(done).reshape(-1).tolist():
                # a counter one short of its limit the step before, and still over threshold, fired
                fl = [int(prev[k][0][e]) >= base.termination_funcs[k].min_steps - 1 for k in terms]
                ep.append([int(t0[e]), int(t_before[e]) + 1, float(z_before[e] - z0[e])] + [float(x) for x in fl])
            if done.any():
                t0[done] = cmd.t[done]
                z0[done] = (robot.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2])[done]
            if (i + 1) % 100 == 0:
                print(f"[rsi] step {i + 1}: {len(ep)} episodes finished", flush=True)

    ep = np.array(ep)
    np.savez(out, episodes=ep, terms=np.array(terms), motion_len=L)
    s, e_, fl = ep[:, 0], ep[:, 1], ep[:, 3:]
    surv = e_ >= L - 1
    print(
        f"[rsi] {len(ep)} episodes; reached clip end {surv.mean() * 100:.1f}%; "
        f"median length {np.median(e_ - s):.0f} steps",
        flush=True,
    )
    print(
        "[rsi] overall cause (share of failed episodes): "
        + "; ".join(f"{k} {fl[~surv][:, j].mean() * 100:.0f}%" for j, k in enumerate(terms)),
        flush=True,
    )
    print("[rsi] by START window: n, reached end %, median steps survived, top causes", flush=True)
    for a in range(0, L, 100):
        m = (s >= a) & (s < a + 100)
        if m.sum() == 0:
            continue
        f = fl[m & ~surv]
        top = sorted(((f[:, j].mean() if len(f) else 0, k) for j, k in enumerate(terms)), reverse=True)[:3]
        print(
            f"[rsi] start {a:4d}-{a + 99:4d}: n {int(m.sum()):4d}  end {surv[m].mean() * 100:5.1f}%  "
            f"median {np.median((e_ - s)[m]):5.0f}  " + ", ".join(f"{k} {v * 100:.0f}%" for v, k in top),
            flush=True,
        )
    print(
        "[rsi] deaths by END window (where in the clip episodes die): "
        + " ".join(f"{a}:{int(((e_ >= a) & (e_ < a + 50) & ~surv).sum())}" for a in range(0, L, 50)),
        flush=True,
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
