"""Drive the trained G1 rough-terrain locomotion policy with a velocity command + follow camera.

The policy (RL, balances itself) walks/climbs; YOU send the walk command:
  keyboard:  W/S = forward/back,  A/D = turn,  Q/E = strafe,  X = stop,  ESC = quit
The policy lifts the legs + keeps balance; you just steer. Run WITHOUT --headless to drive live.
--input scripted --headless renders a close-up walk video (validation).
"""
import argparse
import os
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input", choices=["keyboard", "scripted"], default="scripted")
parser.add_argument("--checkpoint", type=str,
                    default="/home/yujin-chen/SO101_project/IsaacLab/logs/rsl_rl/g1_rough/2026-07-30_20-44-22/model_2999.pt")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-G1-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.input == "scripted":
    args.headless = True
    args.enable_cameras = True   # offscreen render for the video
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402

FR = "/tmp/fiatlux-xr/drive_frames"

# --- env (1 robot), keyboard-friendly command (no auto-resample / heading) ---
env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
env_cfg.commands.base_velocity.heading_command = False
env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
env_cfg.commands.base_velocity.debug_vis = True
# disable auto-resets so it doesn't teleport/"jump" to a new spawn while you're driving it
env_cfg.episode_length_s = 1.0e9
for _t in ("time_out", "base_contact", "illegal_contact", "bad_orientation", "terrain_out_of_bounds"):
    if hasattr(env_cfg.terminations, _t):
        setattr(env_cfg.terminations, _t, None)
# flat ground -> the policy stands much stiller (rough terrain makes it fidget/balance constantly)
env_cfg.scene.terrain.terrain_type = "plane"
env_cfg.scene.terrain.terrain_generator = None
try:
    env_cfg.curriculum.terrain_levels = None
except Exception:
    pass
try:
    env_cfg.curriculum.terrain_levels = None   # keep a fixed (harder) terrain instead of curriculum reset
except Exception:
    pass

env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
env = RslRlVecEnvWrapper(env, clip_actions=None)

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
runner = OnPolicyRunner(env, agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else agent_cfg, log_dir=None, device="cuda:0")
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
print(f"[drive] loaded policy from {args.checkpoint}", flush=True)

uenv = env.unwrapped
cmd_term = uenv.command_manager.get_term("base_velocity")
robot = uenv.scene["robot"]
dev = uenv.device

# --- input ---
keys = set()
if args.input == "keyboard":
    import carb  # noqa
    import omni.appwindow  # noqa
    _iface = carb.input.acquire_input_interface()
    _kbd = omni.appwindow.get_default_app_window().get_keyboard()

    def _on_key(e):
        if e.type == carb.input.KeyboardEventType.KEY_PRESS:
            keys.add(e.input.name)
        elif e.type == carb.input.KeyboardEventType.KEY_RELEASE:
            keys.discard(e.input.name)
        return True
    _iface.subscribe_to_keyboard_events(_kbd, _on_key)
    print("KEYBOARD: W/S forward/back, A/D turn, Q/E strafe, X stop, ESC quit", flush=True)


def read_cmd(step):
    if args.input == "keyboard":
        vx = (0.8 if "W" in keys else 0.0) - (0.8 if "S" in keys else 0.0)
        vy = (0.5 if "Q" in keys else 0.0) - (0.5 if "E" in keys else 0.0)
        wz = (1.0 if "A" in keys else 0.0) - (1.0 if "D" in keys else 0.0)
        if "X" in keys:
            vx = vy = wz = 0.0
        return vx, vy, wz
    return (0.0, 0.0, 0.0) if step < 40 else (0.8, 0.0, 0.0)   # scripted: stand then walk forward


os.makedirs(FR, exist_ok=True)
for f in os.listdir(FR):
    os.remove(os.path.join(FR, f))

obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]
step = 0
MAXF = 500 if args.input == "scripted" else 10 ** 9
while app.is_running() and step < MAXF:
    if args.input == "keyboard" and "ESCAPE" in keys:
        break
    vx, vy, wz = read_cmd(step)
    cmd_term.vel_command_b[:, 0] = vx
    cmd_term.vel_command_b[:, 1] = vy
    cmd_term.vel_command_b[:, 2] = wz
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
    # follow camera on the robot
    p = robot.data.root_pos_w[0].cpu().numpy()
    uenv.sim.set_camera_view(eye=(p[0] - 2.2, p[1] - 2.6, p[2] + 1.4), target=(p[0], p[1], p[2] + 0.3))
    if args.input == "scripted":
        frame = uenv.render()
        if frame is not None:
            Image.fromarray(frame).save(f"{FR}/f{step:05d}.png")
    step += 1

print(f"steps={step}", flush=True)
print("DRIVE_DONE", flush=True)
env.close(); app.close()
