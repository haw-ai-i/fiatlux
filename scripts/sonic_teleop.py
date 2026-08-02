"""Loco-manipulation in the REAL Insert environment.

Runs the actual ``FIATLUX-Insert-Teleop-v0`` env (its real table/fixture + socket + bulb + the tuned
``controller_rel`` bimanual arm teleop), but FREES the base and drives the legs+waist with the
pre-trained NVIDIA SONIC policy so you can walk up to the table and insert.

  * arms  = the env's own tuned teleop (controller pose -> IK, trigger -> grip)  [unchanged]
  * legs  = SONIC (balance + walk), commanded by the LEFT thumbstick + buttons
  * both ride on the same Pico controllers over CloudXR.

Reuse trick: a walk retargeter is appended to the env's ``controller_rel`` device, so its
``advance()`` returns ``[arm_action..., vx, vy, wz, stop, lean]`` -- we hand the arm part to
``env.step`` and feed the walk part to SONIC. The env is retimed to SONIC's 200 Hz / 50 Hz.
"""
import argparse

from isaaclab.app import AppLauncher

_POLICY_DIR = "/home/yujin-chen/robotica_project/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/policy"

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="FIATLUX-Insert-Teleop-v0")
parser.add_argument("--teleop_device", default="controller_rel")
parser.add_argument("--hand", default="dex3", choices=["dex3", "inspire"])
parser.add_argument("--walk_onnx", default=f"{_POLICY_DIR}/GR00T-WholeBodyControl-Walk.onnx")
parser.add_argument("--balance_onnx", default=f"{_POLICY_DIR}/GR00T-WholeBodyControl-Balance.onnx")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.xr = True          # CloudXR stereo render
args.device = "cuda:0"  # env physics on GPU (xr otherwise defaults to cpu)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import collections  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from isaaclab.devices.device_base import DeviceBase  # noqa: E402
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg  # noqa: E402
from isaaclab.devices.teleop_device_factory import create_teleop_device  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import fiatlux_task  # noqa: F401,E402

# ---------------------------------------------------------------------------
# SONIC contract (identical to sonic_drive.py)
# ---------------------------------------------------------------------------
SONIC_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
DEFAULT_15 = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0], dtype=np.float32)
DEFAULT_29 = np.zeros(29, dtype=np.float32)
DEFAULT_29[:15] = DEFAULT_15
ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.5, 1.0, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
HEIGHT_CMD, LEAN_MAG = 0.74, 0.30
OBS_DIM, HIST_LEN, N_ACT = 86, 6, 15

_ROW = DeviceBase.MotionControllerDataRowIndex.INPUTS.value
_IDX = DeviceBase.MotionControllerInputIndex
_TL = DeviceBase.TrackingTarget.CONTROLLER_LEFT
_TR = DeviceBase.TrackingTarget.CONTROLLER_RIGHT


class WalkRetargeter(RetargeterBase):
    """LEFT stick -> walk; RIGHT stick X -> turn; RIGHT btn -> stop; LEFT X/Y -> lean. Output [vx,vy,wz,stop,lean]."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    @staticmethod
    def _read(data, target):
        cd = data.get(target) if data else None
        if cd is None or len(cd) <= _ROW:
            return 0.0, 0.0, 0.0, 0.0
        inp = cd[_ROW]
        return (float(inp[_IDX.THUMBSTICK_X.value]), float(inp[_IDX.THUMBSTICK_Y.value]),
                float(inp[_IDX.BUTTON_0.value]), float(inp[_IDX.BUTTON_1.value]))

    def retarget(self, data):
        lx, ly, lb0, lb1 = self._read(data, _TL)
        rx, ry, rb0, rb1 = self._read(data, _TR)
        dz = self.cfg.deadzone
        lx = lx if abs(lx) > dz else 0.0        # deadzone: idle thumbstick drift must not walk the robot
        ly = ly if abs(ly) > dz else 0.0
        rx = rx if abs(rx) > dz else 0.0
        ms = self.cfg.movement_scale
        stop = 1.0 if rb0 > 0.5 else 0.0
        lean = (1.0 if lb0 > 0.5 else 0.0) - (1.0 if lb1 > 0.5 else 0.0)
        return torch.tensor([ly * ms, -lx * ms, -rx, stop, lean],
                            device=self.cfg.sim_device, dtype=torch.float32)

    def get_requirements(self):
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]


@dataclass
class WalkRetargeterCfg(RetargeterCfg):
    movement_scale: float = 0.5
    deadzone: float = 0.12
    retargeter_type: type = WalkRetargeter


def main():
    # --- env: real Insert-Teleop scene, retimed for SONIC, base freed ---
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError("expected a ManagerBasedRLEnv task")
    # The Insert-Teleop env defaults to Inspire and needs a patch to switch to Dex3; other teleop
    # envs (e.g. Carry-Teleop) are built Dex3-native in their cfg, so only patch Insert here.
    if args.hand.lower() == "dex3" and "Insert" in args.task:
        from fiatlux_task.tasks.manager_based.fiatlux_task.insert_teleop_env_cfg import apply_dex3_hands
        apply_dex3_hands(env_cfg)

    env_cfg.sim.dt = 0.005                      # 200 Hz (SONIC's rate)
    env_cfg.decimation = 4                      # -> 50 Hz control
    env_cfg.sim.render_interval = 4
    env_cfg.terminations.time_out = None
    # FREE the base so SONIC can balance + walk (the teleop env bolts it down for stationary insert)
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = False
    # Harden the spawn against the intermittent PhysX launch: cap depenetration velocity (a bad
    # contact can't fling the free base metres up) and drop the random joint-offset reset (it
    # perturbs the free-base start pose out of SONIC's balance basin). Keep the bulb reset.
    env_cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity = 1.0
    for _ev in ("reset_robot_joints", "reset_robot", "randomize_robot_root", "push_robot"):
        if getattr(env_cfg.events, _ev, None) is not None:
            setattr(env_cfg.events, _ev, None)
    # operator-paced: no automatic terminations (a "fall" term would auto-reset mid-instability)
    for _t in ("time_out", "success", "object_dropped", "robot_fell", "fall_terminated",
               "bad_orientation", "base_contact", "illegal_contact"):
        if getattr(env_cfg.terminations, _t, None) is not None:
            setattr(env_cfg.terminations, _t, None)
    # Take the robot's x,y from whatever env is loaded (task-specific placement); keep SONIC's
    # standing joint stance so the balance policy starts in-distribution, and raise the spawn a
    # touch so the feet clear the floor. Generalizes across tasks (Insert, Carry, ...).
    _p = env_cfg.scene.robot.init_state.pos
    env_cfg.scene.robot.init_state.pos = (_p[0], _p[1], max(_p[2], 0.80))
    env_cfg.scene.robot.init_state.joint_pos = {
        ".*_hip_pitch_joint": -0.1, ".*_knee_joint": 0.3, ".*_ankle_pitch_joint": -0.2,
    }

    if args.xr:
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    dev = env.device

    # append the walk retargeter + unify ALL retargeter devices to the env device, so advance()'s
    # torch.cat over the retargeter outputs doesn't mix cpu/cuda. Then advance() = [arm_action..., walk(5)].
    dev_cfg = env_cfg.teleop_devices.devices[args.teleop_device]
    for rt in dev_cfg.retargeters:
        rt.sim_device = str(dev)
    dev_cfg.retargeters = list(dev_cfg.retargeters) + [WalkRetargeterCfg(sim_device=str(dev))]

    # wire the AR-menu / keyboard RESET (an empty callbacks dict = the reset button does nothing).
    reset_flag = {"do": False}

    def _reset():
        reset_flag["do"] = True
        print("[sonic-insert] reset requested", flush=True)

    teleop = create_teleop_device(args.teleop_device, env_cfg.teleop_devices.devices,
                                  {"R": _reset, "RESET": _reset})
    print(f"[sonic-insert] teleop device: {args.teleop_device} (+walk)", flush=True)

    # --- SONIC policy + joint maps ---
    jn = robot.joint_names
    sidx = [jn.index(n) for n in SONIC_JOINTS]
    act_idx = torch.tensor(sidx[:15], device=dev)
    walk_sess = ort.InferenceSession(args.walk_onnx, providers=["CPUExecutionProvider"])
    bal_sess = ort.InferenceSession(args.balance_onnx, providers=["CPUExecutionProvider"])
    in_name = walk_sess.get_inputs()[0].name
    print(f"[sonic-insert] loaded Walk + Balance ONNX (input '{in_name}', 516->15)", flush=True)

    loco_cmd = np.zeros(3, dtype=np.float32)
    rpy_cmd = np.zeros(3, dtype=np.float32)
    obs_hist = collections.deque([np.zeros(OBS_DIM, dtype=np.float32)] * HIST_LEN, maxlen=HIST_LEN)
    last_action = np.zeros(N_ACT, dtype=np.float32)

    def build_obs():
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[0:3] = loco_cmd * CMD_SCALE
        o[3] = HEIGHT_CMD
        o[4:7] = rpy_cmd
        o[7:10] = robot.data.root_ang_vel_b[0].cpu().numpy() * ANG_VEL_SCALE
        o[10:13] = robot.data.projected_gravity_b[0].cpu().numpy()
        q = robot.data.joint_pos[0, sidx].cpu().numpy()
        qd = robot.data.joint_vel[0, sidx].cpu().numpy()
        o[13:42] = (q - DEFAULT_29) * DOF_POS_SCALE
        o[42:71] = qd * DOF_VEL_SCALE
        o[71:86] = last_action
        return o

    env.reset()
    # true world spawn (to re-home on reset) + the position-hold target.
    spawn_root = robot.data.root_state_w[:, 0:7].clone()
    home_xy = spawn_root[0, 0:2].cpu().numpy().copy()
    HOLD_KP, HOLD_VMAX, WALK_TH, HOLD_DB, WARMUP = 0.8, 0.25, 0.06, 0.10, 100

    # "hold current pose" arm action (root-frame EE pose + open grip, per arm), so the arm doesn't
    # fling before the controller streams. Order matches the action manager: R_arm(7), R_grip(1),
    # L_arm(7), L_grip(1).
    def rest_arm_action():
        parts = []
        for ee_name in ("right_wrist_yaw_link", "left_wrist_yaw_link"):
            bid = robot.body_names.index(ee_name)
            ee_w = robot.data.body_state_w[:, bid, 0:7]
            p_b, q_b = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, ee_w[:, 0:3], ee_w[:, 3:7])
            parts += [p_b[0], q_b[0], torch.ones(1, device=dev)]
        return torch.cat(parts)

    last_arm = rest_arm_action()

    # SETTLE: PIN the base perfectly upright at the spawn while the legs/feet plant + the arms reach
    # their hold pose, THEN hand a level, zero-velocity robot to SONIC. Without this the free base
    # settles slightly pitched and SONIC lurches forward catching it (a near-faceplant). ~0.35 s.
    pin_pose = robot.data.root_state_w[:, 0:7].clone()
    zero_vel = torch.zeros((env.num_envs, 6), device=dev)
    leg_default = torch.as_tensor(DEFAULT_15, device=dev).unsqueeze(0)
    for _k in range(40):
        robot.set_joint_position_target(leg_default, joint_ids=act_idx)
        robot.write_root_pose_to_sim(pin_pose)                   # hold base upright while feet plant
        robot.write_root_velocity_to_sim(zero_vel)
        env.step(rest_arm_action().repeat(env.num_envs, 1))
    robot.write_root_pose_to_sim(pin_pose)                       # final: level + still, then release to SONIC
    robot.write_root_velocity_to_sim(zero_vel)
    obs_hist = collections.deque([build_obs()] * HIST_LEN, maxlen=HIST_LEN)  # warm history w/ real state

    # SONIC settle-in: let it center its weight (the one-time "back up") NOW, before the operator
    # connects. Short (faster-ready pref) -> a bit of settle-in movement may still show on connect.
    for _s in range(50):
        obs_hist.append(build_obs())
        flat = np.concatenate(obs_hist).astype(np.float32)[None]
        last_action = bal_sess.run(None, {in_name: flat})[0][0]
        leg_target = torch.as_tensor(last_action * ACTION_SCALE + DEFAULT_15, device=dev)
        robot.set_joint_position_target(leg_target.unsqueeze(0), joint_ids=act_idx)
        env.step(rest_arm_action().repeat(env.num_envs, 1))
    spawn_root = robot.data.root_state_w[:, 0:7].clone()         # centered pose = re-home + hold target
    home_xy = spawn_root[0, 0:2].cpu().numpy().copy()
    # FIXED base-frame rest arm pose (captured once). Re-solving the redundant 7-DOF IK every frame
    # lets the null-space drift -> the arm oscillates while "holding still" (worse on a swaying base);
    # holding one fixed target keeps it steady and lets it ride with the body.
    rest_arm = rest_arm_action()

    # Re-anchor the controller_rel arm retargeters to THIS scene's live robot. They default to the
    # Insert *table* world coords, so on any other scene (e.g. the ladder Carry env) the arm reaches
    # for a world point far from the robot and flails. Rebake root + EE-start + workspace to live.
    from scipy.spatial.transform import Rotation as _Rot  # noqa: E402
    _rpos = robot.data.root_pos_w[0].cpu().numpy()
    _rq = robot.data.root_quat_w[0].cpu().numpy()          # w, x, y, z
    _R = _Rot.from_quat([_rq[1], _rq[2], _rq[3], _rq[0]])
    for _rt in getattr(teleop, "_retargeters", None) or []:
        if not hasattr(_rt, "_root_pos"):                 # only the Se3Rel arm retargeters
            continue
        _rt._root_pos = _rpos.astype(np.float32)
        _rt._root_R = _R
        _rt._root_R_T = _R.as_matrix().T.astype(np.float32)
        _right = getattr(_rt, "_target", None) == DeviceBase.TrackingTarget.CONTROLLER_RIGHT
        _eeb = robot.body_names.index("right_wrist_yaw_link" if _right else "left_wrist_yaw_link")
        _ee = robot.data.body_state_w[0, _eeb, 0:3].cpu().numpy().astype(np.float32)
        _rt._init_pos = _ee.copy()
        _rt._pos = _ee.copy()
        _rt._lo = _ee - np.array([0.45, 0.45, 0.45], dtype=np.float32)
        _rt._hi = _ee + np.array([0.45, 0.45, 0.45], dtype=np.float32)
        _rt._prev = None
        _rt._smooth = None

    print(f"[sonic-insert] settled+centered at pelvis=({spawn_root[0,0]:.2f},{spawn_root[0,1]:.2f},{spawn_root[0,2]:.2f})", flush=True)
    print("Teleop ready. In the Isaac Sim UI: AR panel -> Start AR, then connect the Pico. "
          "LEFT stick = walk, RIGHT stick X = turn, RIGHT btn = stop, LEFT X/Y = lean. "
          "Arms: the usual controller_rel teleop (grip-clutch + move, trigger to grasp).", flush=True)

    n_walk = 5
    step_i = 0
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                if reset_flag["do"]:
                    reset_flag["do"] = False
                    env.reset()
                    # env.reset() doesn't re-home a FREE base (it did when the base was bolted) -> force
                    # the root pose + velocity + joints back to the spawn so the robot actually returns.
                    robot.write_root_pose_to_sim(spawn_root)
                    robot.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=dev))
                    robot.write_joint_state_to_sim(
                        robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
                    home_xy = spawn_root[0, 0:2].cpu().numpy().copy()
                    for _rt in getattr(teleop, "_retargeters", None) or []:
                        if hasattr(_rt, "reset"):
                            _rt.reset()                       # re-reference arm targets (no drift after reset)
                    obs_hist = collections.deque([np.zeros(OBS_DIM, dtype=np.float32)] * HIST_LEN, maxlen=HIST_LEN)
                    last_action = np.zeros(N_ACT, dtype=np.float32)
                    loco_cmd[:] = 0.0
                    rpy_cmd[:] = 0.0
                    last_arm = rest_arm
                    print("[sonic-insert] reset done", flush=True)

                # Take arm + walk from the controller ONLY when it's streaming; otherwise hold.
                out = teleop.advance()
                if out is not None:
                    last_arm = out[:-n_walk]
                    walk = out[-n_walk:].detach().cpu().numpy()
                    loco_cmd[:] = walk[:3]
                    if walk[3] > 0.5:
                        loco_cmd[:] = 0.0
                    rpy_cmd[1] = walk[4] * LEAN_MAG
                else:
                    last_arm = rest_arm              # no controller -> FIXED rest pose (no IK re-solve jitter)
                    loco_cmd[:] = 0.0
                    rpy_cmd[:] = 0.0

                # Position hold: SONIC is a velocity policy with no position feedback, so cmd=0 slowly
                # glides the base away. Steer a gentle velocity back to home -- but only AFTER a warmup so
                # it doesn't fight the fragile settle; while walking, home follows the robot.
                base_xy = robot.data.root_pos_w[0, 0:2].cpu().numpy()
                if step_i < WARMUP or np.linalg.norm(loco_cmd) > WALK_TH:
                    home_xy = base_xy.copy()
                else:
                    e = home_xy - base_xy
                    if float(np.linalg.norm(e)) > HOLD_DB:
                        q = robot.data.root_quat_w[0].cpu().numpy()
                        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                        loco_cmd[0] = float(np.clip(HOLD_KP * (np.cos(yaw) * e[0] + np.sin(yaw) * e[1]), -HOLD_VMAX, HOLD_VMAX))
                        loco_cmd[1] = float(np.clip(HOLD_KP * (-np.sin(yaw) * e[0] + np.cos(yaw) * e[1]), -HOLD_VMAX, HOLD_VMAX))

                # SONIC runs EVERY frame -- balance is not optional. (Gating this on `out` made the
                # free-base robot collapse whenever the headset wasn't streaming.)
                obs_hist.append(build_obs())
                flat = np.concatenate(obs_hist).astype(np.float32)[None]
                sess = walk_sess if np.linalg.norm(loco_cmd) > 0.05 else bal_sess
                last_action = sess.run(None, {in_name: flat})[0][0]
                leg_target = torch.as_tensor(last_action * ACTION_SCALE + DEFAULT_15, device=dev)
                robot.set_joint_position_target(leg_target.unsqueeze(0), joint_ids=act_idx)
                env.step(last_arm.repeat(env.num_envs, 1))   # arms via the real env action manager

                step_i += 1
                if step_i % (20 if step_i <= 200 else 100) == 0:
                    p = robot.data.root_pos_w[0].cpu().numpy()
                    print(f"[sonic-insert] step {step_i} pelvis=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) "
                          f"loco_cmd={np.round(loco_cmd, 2)} {'FELL' if p[2] < 0.4 else 'ok'}", flush=True)
        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001
            print(f"[sonic-insert] step skipped while view recovers: {e}", flush=True)
            try:
                env.sim.render()
            except Exception:  # noqa: BLE001
                pass

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
