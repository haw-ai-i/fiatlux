"""Record video of a LadderMan teacher climbing in LadderMan's OWN env.

A sibling of play_probe.py. That script answered "does it climb" with a number (+1213 mm over
914 steps, 1 of 4 envs). This one shows it.

Their scene is built inside LadderEnv.__init__ from a local InteractiveSceneCfg, so a camera
cannot be injected through the config. It is created here instead, after the sim is already
playing, and aimed per frame at whichever environment is climbing highest and still alive --
startup randomises body materials, so only some environments make it and which one is not known
in advance.

    python scripts/play_video.py algo=teacher task=teacher checkpoint_path=<...> \
        headless=true app.enable_cameras=true task.num_envs=4 +probe_steps=1000 +out=<dir>
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
    cfg.app.enable_cameras = True
    app = AppLauncher(OmegaConf.to_container(cfg.app)).app  # noqa: F841

    import numpy as np
    import imageio.v2 as imageio
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from scripts.helpers import make_env_policy

    env, policy, vecnorm = make_env_policy(cfg)
    rollout_policy = policy.get_rollout_policy("eval")
    env.base_env.eval()
    base = env.base_env
    robot = base.scene["robot"]
    dev = base.device
    out = str(cfg.get("out", "/tmp/lm_video"))
    os.makedirs(out, exist_ok=True)
    steps = int(cfg.get("probe_steps", 1000))

    # The camera is created AFTER env.reset(), and sim.reset() is never called. Calling
    # sim.reset() on an already-running sim to initialise a late-added sensor re-made buffers
    # underneath the ladder's RigidObjectCollection and the next scene.reset() indexed out of
    # bounds -- a CUDA device-side assert in rigid_object_collection.reset (job 15490885).
    td_ = env.reset()
    cam = Camera(CameraCfg(
        prim_path="/World/probe_cam", height=720, width=1280, data_types=["rgb"],
        update_period=0.0,
        spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, clipping_range=(0.05, 1.0e4)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, -4.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0),
                                   convention="world"),
    ))
    # Sensors initialise on the timeline PLAY event (SensorBase._initialize_callback). The
    # sim is ALREADY playing when this camera is constructed, so that event never fires, the
    # sensor allocates no buffers, and cam.update() dies on a missing _timestamp (job
    # 15492093). Run the initialisation the callback would have run -- directly, not via the
    # callback, which swallows errors into builtins.ISAACLAB_CALLBACK_EXCEPTION.
    if not cam.is_initialized:
        cam._initialize_impl()
        cam._is_initialized = True
    base.sim.render()
    cam.update(0.0)
    print(f"[video] camera ready, envs={base.num_envs}", flush=True)

    origins = base.scene.env_origins
    z0 = (robot.data.root_pos_w[:, 2] - origins[:, 2]).clone()
    best = z0.clone()
    alive = torch.ones(base.num_envs, dtype=torch.bool, device=dev)
    frames, picked = [], []

    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in itertools.count():
            if i >= steps or not alive.any():
                break
            td_ = rollout_policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            z = robot.data.root_pos_w[:, 2] - origins[:, 2]
            best = torch.maximum(best, torch.where(alive, z, torch.full_like(z, -9e9)))

            # follow the highest env that is still up; that is the one worth watching
            cand = torch.where(alive, z, torch.full_like(z, -9e9))
            e = int(cand.argmax())
            picked.append(e)
            tgt = robot.data.root_pos_w[e].detach().cpu().numpy().astype(float)
            lad = base.scene["ladder"].data.object_pos_w[e, 0].detach().cpu().numpy().astype(float) \
                if hasattr(base.scene["ladder"].data, "object_pos_w") else tgt
            outv = tgt[:2] - lad[:2]
            n = np.linalg.norm(outv)
            outv = outv / n if n > 1e-6 else np.array([0.0, -1.0])
            side = np.array([-outv[1], outv[0]])
            eye = np.array([tgt[0] + outv[0] * 2.6 + side[0] * 1.9,
                            tgt[1] + outv[1] * 2.6 + side[1] * 1.9, tgt[2] + 0.7])
            look = np.array([tgt[0], tgt[1], tgt[2] - 0.15])
            cam.set_world_poses_from_view(
                torch.tensor(eye, dtype=torch.float32, device=dev).expand(1, 3),
                torch.tensor(look, dtype=torch.float32, device=dev).expand(1, 3))
            base.sim.render()
            cam.update(0.0)
            frames.append(cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))

            done = td.get(("next", "done"), td.get("done")).reshape(-1).bool()
            alive = alive & ~done
            if (i + 1) % 100 == 0:
                print(f"[video] step {i+1:4d}  alive {int(alive.sum())}/{base.num_envs}  "
                      f"following env {e}  pelvis {float(z[e]):.3f}  best {float(best.max()):.3f}",
                      flush=True)

    gain = (best - z0) * 1000.0
    d = float(np.abs(frames[0].astype(np.int16) - frames[len(frames)//2].astype(np.int16)).mean())
    print(f"[video] frame-difference check: {d:.2f}/255 "
          f"({'OK' if d >= 1.0 else 'FAILED - frames identical, camera saw nothing'})", flush=True)
    mp4 = os.path.join(out, "ladderman_climb.mp4")
    imageio.mimwrite(mp4, frames, fps=50, codec="libx264", quality=8)
    print(f"[video] {len(frames)} frames -> {mp4}", flush=True)
    for e in range(base.num_envs):
        print(f"[video] env {e}: best {float(best[e]):.3f}  gain {float(gain[e]):+.0f} mm", flush=True)
    print(f"[video] BEST GAIN {float(gain.max()):+.0f} mm", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Isaac shutdown can hang for tens of minutes: job 15492093 held a node for the full
        # 90 min wall after failing in its first 30 seconds. Fail fast instead.
        os._exit(1)
