"""Trainer twin of fiatlux scripts/climb/ladder_push_test.py: the same wrench on the free ladder body,
before any robot touches it. Run with +task.command.ladder_config.free=asset (or true)."""

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
    lad = base.ladder_body
    px = base.sim.cfg.physx
    print(
        f"[push] sim: dt {base.physics_dt} x {base.decimation}; solver_type {px.solver_type}, min pos/vel iterations {px.min_position_iteration_count}/{px.min_velocity_iteration_count}, "
        f"stabilization {px.enable_stabilization}, bounce threshold {px.bounce_threshold_velocity}",
        flush=True,
    )
    v = lad.root_physx_view
    print(
        f"[push] ladder mass {float(v.get_masses()[0].sum()):.2f} kg, com {[round(float(x), 3) for x in v.get_coms()[0].flatten()[:3]]}, inertia {[round(float(x), 3) for x in v.get_inertias()[0].flatten()[[0, 4, 8]]]}",
        flush=True,
    )
    try:
        mp = v.get_material_properties()[0]
        print(
            f"[push] ladder shape materials (static, dynamic, restitution): {[[round(float(x), 2) for x in r] for r in mp.reshape(-1, 3)[:4].tolist()]}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print("[push] material read failed:", e, flush=True)
    kind = str(cfg.task.command.ladder_config.get("free"))
    if kind == "asset":  # body frame = the ladder's own: +y is the climb direction
        r_com = torch.tensor([0.0, -0.1694 + 0.012, 0.944 - 0.826])
        fwd_axis = 1
    else:  # box body: frame = the clip's, +x is the climb direction, com (-0.2116, 0, 0.826)
        r_com = torch.tensor([-0.369 + 0.2116, 0.0, 0.944 - 0.826])
        fwd_axis = 0
    N = base.num_envs
    p0 = lad.data.root_link_pos_w.clone()
    td_ = env.reset()
    p0 = lad.data.root_link_pos_w.clone()
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for k in range(30):  # the hands reach tread 4 at about step 37: stop before
            imp = [float(x) for x in cfg.get("impulse", [84.0, 46.0, 2])]
            hold = [float(x) for x in cfg.get("hold", [20.0, 10.0])]
            down, fwd = (
                (imp[0], imp[1])
                if 10 <= k < 10 + int(imp[2])
                else (hold[0], hold[1])
                if k >= 10 + int(imp[2])
                else (0.0, 0.0)
            )
            F = torch.zeros(3)
            F[fwd_axis] = fwd
            F[2] = -down
            T = torch.linalg.cross(r_com, F)
            lad.set_external_force_and_torque(
                F.to(base.device)[None, None].expand(N, 1, 3), T.to(base.device)[None, None].expand(N, 1, 3)
            )
            td_ = rollout_policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            if k % 3 == 0 or k in (10, 11, 12):
                g = lad.data.projected_gravity_b
                tl = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))
                d = (lad.data.root_link_pos_w - p0) * 1000
                print(
                    f"[push] step {k:3d}  push down/forward {down:.0f}/{fwd:.0f} N  tilt median {float(tl.median()):.2f} max {float(tl.max()):.2f} deg  origin moved z median {float(d[:, 2].median()):+.1f} mm, |xy| median {float(d[:, :2].norm(dim=-1).median()):.1f} mm",
                    flush=True,
                )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
