"""Measure what a LadderMan teacher actually DOES in LadderMan's own env.

A sibling of their ``scripts/play.py``, not a modification of it: that script prints the stats
dict and exports ONNX, and neither tells you the one thing we need -- how high the pelvis gets.

Why this exists. Our teacher reports ``train/stats/success = 0.658``, but their success is
``(t >= motion_len - 1)``: the episode survived to the end of the clip from wherever it started,
which is a uniformly random phase under reference state init. It is NOT a height check. So the
scalar cannot distinguish "it climbs" from "it survives a second-long tail of the clip". This
logs pelvis height per step so the question is answered by a number instead of an inference.

Two modes, selected by ``+zero_residual``:

  false (default)  the policy drives, as in play.py. Answers: does the teacher climb AT HOME?
  true             the action is overwritten so the PD target IS the reference joint position,
                   i.e. open-loop replay through the same action path. Answers: does an open-loop
                   replay fall over in THEIR env too? If it does, the identical result in the
                   fiatlux env says nothing about the fiatlux env, and our open-loop test was
                   uninformative. If it stands here and falls there, the gap is our robot or our
                   physics -- not the reference and not the policy.

``env.base_env.eval()`` makes their command force ``start_t = 0``, so both modes run the full
climb from the bottom rather than from a random phase.

    python scripts/play_probe.py algo=teacher task=teacher checkpoint_path=<...> \
        headless=true task.num_envs=4 +probe_steps=1200 +zero_residual=false <ladder overrides>
"""

import itertools
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
    env.base_env.eval()  # forces start_t = 0: the full climb, not a phase

    base = env.base_env
    robot = base.scene["robot"]
    cmd = base.command_manager
    zero_residual = bool(cfg.get("zero_residual", False))
    steps = int(cfg.get("probe_steps", 1200))

    # Map the command's tracked joints onto the action manager's joint order so a reference joint
    # position can be written as an action. Their action term is
    #     pos_target[:, joint_ids] = default[:, joint_ids] + action * action_scaling
    # so the action that commands `q` is (q - default) / action_scaling.
    act = base.action_manager
    perm = None
    if zero_residual:
        tr_asset = list(cmd.tracking_joint_indices_asset)
        act_ids = act.joint_ids.tolist() if torch.is_tensor(act.joint_ids) else list(act.joint_ids)
        missing = [i for i in act_ids if i not in tr_asset]
        if missing:
            names = [robot.joint_names[i] for i in missing]
            print(
                f"[probe] NOTE: {len(missing)} actuated joints are not tracked by the command "
                f"and will be held at their default: {names}",
                flush=True,
            )
        perm = [tr_asset.index(i) if i in tr_asset else -1 for i in act_ids]
        print(
            f"[probe] zero-residual mode: forcing the PD target onto the reference for "
            f"{sum(p >= 0 for p in perm)} of {len(act_ids)} actuated joints",
            flush=True,
        )

    z0 = (robot.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2]).clone()
    print(
        f"[probe] mode={'open-loop reference' if zero_residual else 'policy'}  "
        f"envs={base.num_envs}  steps={steps}  motion_len={int(cmd.motion_len[0]) if hasattr(cmd.motion_len, '__len__') else int(cmd.motion_len)}",
        flush=True,
    )
    print(f"[probe] spawn pelvis {[round(v, 3) for v in z0.tolist()]}", flush=True)

    best = z0.clone()
    alive = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
    died_at = torch.full((base.num_envs,), -1, dtype=torch.long, device=base.device)

    td_ = env.reset()
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in itertools.count():
            if i >= steps:
                break
            td_ = rollout_policy(td_)
            if zero_residual:
                ref = cmd.ref_joint_pos  # [N, n_tracked]
                tgt = act.default_joint_pos[:, act.joint_ids].clone()
                for a_i, r_i in enumerate(perm):
                    if r_i >= 0:
                        tgt[:, a_i] = ref[:, r_i]
                td_["action"] = ((tgt - act.default_joint_pos[:, act.joint_ids]) / act.action_scaling).clamp(
                    -10.0, 10.0
                )
            td, td_ = env.step_and_maybe_reset(td_)

            z = robot.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2]
            best = torch.maximum(best, torch.where(alive, z, torch.full_like(z, -9e9)))
            done = td.get(("next", "done"), td.get("done")).reshape(-1).bool()
            newly = alive & done
            died_at[newly] = i + 1
            alive = alive & ~done
            if (i + 1) % 100 == 0 or not alive.any():
                print(
                    f"[probe] step {i + 1:5d}  alive {int(alive.sum())}/{base.num_envs}  "
                    f"pelvis now {z.max().item():.3f}  best {best.max().item():.3f}",
                    flush=True,
                )
            if not alive.any():
                break

    gain = (best - z0) * 1000.0
    print("[probe] ---- result ----", flush=True)
    for e in range(base.num_envs):
        print(
            f"[probe] env {e}: died at step {int(died_at[e]) if died_at[e] >= 0 else 'survived'}"
            f"   spawn {z0[e]:.3f}  best {best[e]:.3f}  gain {gain[e]:+.0f} mm",
            flush=True,
        )
    print(
        f"[probe] BEST GAIN {gain.max().item():+.0f} mm  (one tread rise is 234 mm; the clip's own end is +1179 mm)",
        flush=True,
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
