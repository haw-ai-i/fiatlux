"""Does SONIC's Balance policy actually crouch when the height command is varied?

sonic_teleop pins slot 3 of the 86-d observation to HEIGHT_CMD = 0.74 for every step. The
policy was trained with that input, so the question is whether it is live. This sweeps it
from the nominal stand down and records the settled pelvis height, hands-off, on flat ground.
"""

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="FIATLUX-S07-ApproachNewBulb-Teleop-v0")
p.add_argument("--settle", type=int, default=120)
p.add_argument("--hold", type=int, default=150)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(["--headless", "--enable_cameras"])
app = AppLauncher(args).app

import collections
import os  # noqa: E402

import fiatlux_task  # noqa: F401,E402
import fiatlux_teleop  # noqa: F401,E402
import gymnasium as gym  # noqa: E402
import numpy as np
import onnxruntime as ort
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

POL = os.environ["SONIC_POLICY_DIR"]
DEFAULT_29 = np.array(
    [
        -0.1,
        0,
        0,
        0.3,
        -0.2,
        0,
        -0.1,
        0,
        0,
        0.3,
        -0.2,
        0,
        0,
        0,
        0,
        0.2,
        0.2,
        0,
        0.6,
        0,
        0,
        0,
        0.2,
        -0.2,
        0,
        0.6,
        0,
        0,
        0,
    ],
    dtype=np.float32,
)
DEFAULT_15 = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0], dtype=np.float32)
CMD_SCALE = np.array([2.0, 2.0, 0.5], np.float32)
ANG = 0.25
DPOS = 1.0
DVEL = 0.05
ASCALE = 0.25
HIST, OBS = 6, 86

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
cfg.sim.dt = 0.005
cfg.decimation = 4
cfg.terminations = type(cfg.terminations)()
cfg.scene.robot.spawn.articulation_props.fix_root_link = False
env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
sess = ort.InferenceSession(
    os.path.join(POL, "GR00T-WholeBodyControl-Balance.onnx"), providers=["CPUExecutionProvider"]
)
inp = sess.get_inputs()[0].name
names = [
    n
    for n in (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
]
act_idx, _ = robot.find_joints(names, preserve_order=True)
sidx, _ = robot.find_joints(names + [n for n in robot.joint_names if n not in names][:14], preserve_order=True)
env.reset()
last = np.zeros(15, np.float32)
hist = collections.deque([np.zeros(OBS, np.float32)] * HIST, maxlen=HIST)


def obs(hcmd):
    o = np.zeros(OBS, np.float32)
    o[0:3] = 0.0
    o[3] = hcmd
    o[4:7] = 0.0
    o[7:10] = robot.data.root_ang_vel_b[0].cpu().numpy() * ANG
    o[10:13] = robot.data.projected_gravity_b[0].cpu().numpy()
    q = robot.data.joint_pos[0, sidx].cpu().numpy()
    qd = robot.data.joint_vel[0, sidx].cpu().numpy()
    o[13:42] = (q - DEFAULT_29) * DPOS
    o[42:71] = qd * DVEL
    o[71:86] = last
    return o


zero = torch.zeros(1, env.action_space.shape[1], device=env.device)
print(f"{'height_cmd':>11} {'settled pelvis z':>17} {'drop vs 0.74':>13}")
base = None
for h in (0.74, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40):
    env.reset()
    hist = collections.deque([obs(h)] * HIST, maxlen=HIST)
    last = np.zeros(15, np.float32)
    for k in range(args.settle + args.hold):
        hist.append(obs(h))
        flat = np.concatenate(hist).astype(np.float32)[None]
        last = sess.run(None, {inp: flat})[0][0]
        tgt = torch.as_tensor(last * ASCALE + DEFAULT_15, device=env.device).unsqueeze(0)
        robot.set_joint_position_target(tgt, joint_ids=act_idx)
        env.step(zero)
    z = float(robot.data.root_pos_w[0, 2])
    if base is None:
        base = z
    print(f"{h:>11.2f} {z:>17.3f} {z - base:>13.3f}")
env.close()
app.close()
