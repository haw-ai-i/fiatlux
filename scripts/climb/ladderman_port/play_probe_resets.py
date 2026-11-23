"""Resets as TRAINING makes them: which of them throw the robot, and at which phase of the clip.

Training mode (random-phase starts, pose and joint noise, exploration noise on the actions). Every
episode start is recorded with the clip step it started at; for its first `window` steps the script
keeps the hardest contact force on the ankle and wrist links and the fastest any robot body moves.
A start is a "launch" when a body passes `launch_speed` m/s in that window. Printed: the share of
launched starts by clip phase, the worst ones, and how far any robot ever got from its own
environment's origin (a robot thrown across the grid is what makes PhysX's pair buffers explode).

    +steps=1500 +window=6 +launch_speed=6.0 +phase_bin=100
"""

import os
import sys
from collections import defaultdict

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
    rollout_policy = (
        policy.get_rollout_policy("train")
        if "train" in getattr(policy, "rollout_modes", ["train"])
        else policy.get_rollout_policy("eval")
    )
    base = env.base_env
    base.train()
    cmd = base.command_manager
    robot = base.scene["robot"]
    cs = base.scene["contact_forces"]
    ank, _ = cs.find_bodies(".*_ankle_roll_link")
    wri, _ = cs.find_bodies(".*_wrist_.*_link")
    steps = int(cfg.get("steps", 1500))
    window = int(cfg.get("window", 6))
    launch_speed = float(cfg.get("launch_speed", 6.0))
    pbin = int(cfg.get("phase_bin", 100))
    N = base.num_envs
    dev = base.device
    td_ = env.reset()
    age = torch.zeros(N, dtype=torch.long, device=dev)
    t0 = cmd.t.clone()
    pk_f_ank = torch.zeros(N, device=dev)
    pk_f_wri = torch.zeros(N, device=dev)
    pk_v = torch.zeros(N, device=dev)
    rows = []  # (start step, ankle peak, wrist peak, speed peak)
    far = 0.0
    far_info = None
    vmax_all = 0.0
    print(
        f"[resets] {N} robots, {steps} steps, training mode: {base.training}; clip length {int(cmd.dataset.motion_lengths.max()) if hasattr(cmd, 'dataset') and hasattr(cmd.dataset, 'motion_lengths') else '?'}",
        flush=True,
    )
    with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
        for i in range(steps):
            td_ = rollout_policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            done = td["next", "done"].reshape(-1).bool()
            fa = cs.data.net_forces_w_history[:, :, ank].norm(dim=-1).amax(dim=(1, 2))
            fw = cs.data.net_forces_w_history[:, :, wri].norm(dim=-1).amax(dim=(1, 2))
            v = robot.data.body_lin_vel_w.norm(dim=-1).amax(dim=1)
            live = ~done
            inw = live & (age < window)
            pk_f_ank[inw] = torch.maximum(pk_f_ank[inw], fa[inw])
            pk_f_wri[inw] = torch.maximum(pk_f_wri[inw], fw[inw])
            pk_v[inw] = torch.maximum(pk_v[inw], v[inw])
            d = (robot.data.root_pos_w - base.scene.env_origins)[:, :2].norm(dim=-1)
            if float(d[live].max() if live.any() else 0.0) > far:
                j = int(torch.where(live, d, torch.zeros_like(d)).argmax())
                far = float(d[j])
                far_info = (i, j, int(age[j]), int(t0[j]), float(v[j]))
            vmax_all = max(vmax_all, float(v[live].max()) if live.any() else 0.0)
            age += 1
            # an episode's window is over, or the episode ended: file it
            fin = done | (age == window)
            for j in torch.nonzero(fin & (age <= window)).reshape(-1).tolist():
                rows.append((int(t0[j]), float(pk_f_ank[j]), float(pk_f_wri[j]), float(pk_v[j])))
            if done.any():
                ids = torch.nonzero(done).reshape(-1)
                age[ids] = 0
                t0[ids] = cmd.t[ids]
                pk_f_ank[ids] = 0
                pk_f_wri[ids] = 0
                pk_v[ids] = 0
    R = torch.tensor(rows) if rows else torch.zeros(0, 4)
    print(
        f"[resets] {len(R)} episode starts recorded; launch = a body over {launch_speed:.0f} m/s within {window} steps of the start",
        flush=True,
    )
    if len(R):
        L = R[:, 3] > launch_speed
        print(
            f"[resets] launched starts: {int(L.sum())} of {len(R)} ({100 * float(L.float().mean()):.2f}%); fastest body in any start window {float(R[:, 3].max()):.1f} m/s; "
            f"ankle force 99th pct {float(R[:, 1].quantile(0.99)):.0f} N, max {float(R[:, 1].max()):.0f}; wrist 99th pct {float(R[:, 2].quantile(0.99)):.0f} N, max {float(R[:, 2].max()):.0f}",
            flush=True,
        )
        byb = defaultdict(list)
        for r in R.tolist():
            byb[int(r[0]) // pbin * pbin].append(r)
        for b in sorted(byb):
            x = torch.tensor(byb[b])
            l = x[:, 3] > launch_speed
            print(
                f"[phase ] starts at clip steps {b:4d}-{b + pbin - 1:4d}: {len(x):5d} starts, launched {int(l.sum()):4d} ({100 * float(l.float().mean()):5.1f}%), "
                f"ankle force median {float(x[:, 1].median()):6.0f} N, 99th pct {float(x[:, 1].quantile(0.99)):7.0f}; fastest body {float(x[:, 3].max()):5.1f} m/s",
                flush=True,
            )
        top = R[R[:, 3].argsort(descending=True)[:6]]
        print(
            "[worst ] "
            + "; ".join(
                f"start at clip step {int(a)}: {d_:.1f} m/s, ankle {b_:.0f} N, wrist {c_:.0f} N"
                for a, b_, c_, d_ in top.tolist()
            ),
            flush=True,
        )
    print(
        f"[far   ] fastest body at any time {vmax_all:.1f} m/s; furthest a live robot got from its own origin (floor plane): {far:.2f} m"
        + (
            f" (step {far_info[0]}, {far_info[2]} steps into an episode that started at clip step {far_info[3]}, moving {far_info[4]:.1f} m/s)"
            if far_info
            else ""
        ),
        flush=True,
    )
    try:
        sp = base.scene.cfg.env_spacing
        print(f"[far   ] environment spacing {sp} m", flush=True)
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
