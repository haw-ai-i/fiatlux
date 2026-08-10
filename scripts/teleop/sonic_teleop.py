"""Whole-body loco-manipulation teleop for any FIATLUX-*-Teleop task.

Runs a real teleop gym env (``--task`` -- Insert / Carry / LadderGallery / ...), FREES the base, and
drives legs+waist with the pre-trained NVIDIA SONIC policy so the operator can walk the robot around
the scene and manipulate -- true whole-body teleop, on any task.

Two input modes (``--input``):
  * vr        -- Pico controllers over CloudXR: LEFT stick walks, the env's own bimanual arm teleop
                 (controller pose -> IK, trigger -> grip) drives the arms.
  * keyboard  -- desktop, no headset: arrow keys walk; TAB picks the active arm; W/S A/D Q/E move it,
                 U/O I/K J/L rotate the wrist, G grips -- both arms + wrist rotation, i.e. VR parity.

  * arms  = the env's action manager (differential-IK EE pose + binary grip)
  * legs  = SONIC (balance + walk)
The env is retimed to SONIC's 200 Hz / 50 Hz. VR appends a walk retargeter to the env's teleop device
so ``advance()`` returns ``[arm_action..., vx,vy,wz,stop,lean]``; keyboard builds the same
``[arm_action..., walk]`` from key state. Either way the loop hands the arm part to ``env.step`` and
the walk part to SONIC.
"""
import argparse
import contextlib
import os

from isaaclab.app import AppLauncher

_POLICY_DIR = os.path.expanduser(
    "~/robotica_project/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/policy")

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="FIATLUX-Insert-Teleop-v0")
parser.add_argument("--input", choices=["vr", "keyboard"], default="vr",
                    help="vr = Pico controllers over CloudXR; keyboard = desktop keys (no headset)")
parser.add_argument("--teleop_device", default="controller_rel")
parser.add_argument("--hand", default="dex3", choices=["dex3", "inspire"])
parser.add_argument("--walk_onnx", default=f"{_POLICY_DIR}/GR00T-WholeBodyControl-Walk.onnx")
parser.add_argument("--balance_onnx", default=f"{_POLICY_DIR}/GR00T-WholeBodyControl-Balance.onnx")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.xr = (args.input == "vr")   # CloudXR stereo render only in VR mode (keyboard = desktop GUI)
args.device = "cuda:0"           # env physics on GPU (xr otherwise defaults to cpu)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import collections  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import fiatlux_task  # noqa: F401,E402
import fiatlux_teleop  # noqa: F401,E402  -- registers the FIATLUX-*-Teleop gym ids
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from isaaclab.devices.device_base import DeviceBase  # noqa: E402
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg  # noqa: E402
from isaaclab.devices.teleop_device_factory import create_teleop_device  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# ---------------------------------------------------------------------------
# SONIC contract (29-joint obs / 15-action legs+waist; from NVIDIA's GR00T-WholeBodyControl)
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


def main():  # noqa: C901  (one long orchestration: env setup + settle/resettle + the teleop loop)
    # --- env: real Insert-Teleop scene, retimed for SONIC, base freed ---
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError("expected a ManagerBasedRLEnv task")
    # Insert-Teleop defaults to Inspire (swap to Dex3 on request); Carry-Teleop is Dex3-native
    # (swap to Inspire on request). Each env's patch handles the robot + hand-action repoint.
    if args.hand.lower() == "dex3" and "Insert" in args.task:
        from fiatlux_teleop.insert_teleop_env_cfg import apply_dex3_hands
        apply_dex3_hands(env_cfg)
    elif args.hand.lower() == "inspire" and "Carry" in args.task:
        from fiatlux_teleop.carry_teleop_env_cfg import apply_inspire_hands
        apply_inspire_hands(env_cfg)

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
    # SONIC leg stance (both envs need it). The ARM spawn pose is env-dependent because the two envs'
    # IK behaves differently: Insert's redundant IK RELAXES the arm to a natural low rest regardless of
    # spawn (so a bent-elbow spawn settles to ~0.17), but Carry's IK HOLDS whatever it spawns in -- so
    # Carry must spawn directly in the natural pose or it stays tucked at the spawn angle. Keep Insert's
    # spawn exactly as it was so its settled pose is unchanged.
    _legs = {".*_hip_pitch_joint": -0.1, ".*_knee_joint": 0.3, ".*_ankle_pitch_joint": -0.2}
    # LadderGallery is a Carry-derived task -- its IK HOLDS the spawn pose too, so it needs the same
    # natural low spawn as Carry (spawning it at Insert's 1.57 would leave the arm tucked up at 90 deg).
    if "Carry" in args.task or "Gallery" in args.task:
        # Drop the SHOULDER so the arm hangs low. The elbow drifts up to ~1.1 on its own (redundant IK),
        # so we don't fight it -- a low/back shoulder points the upper arm down so the bent forearm sits
        # low instead of up at the chest. (Per operator: change the joint above the 90-deg elbow.)
        _arm_spawn = {".*_shoulder_pitch_joint": -0.35, ".*_elbow_joint": 0.35}
    else:
        _arm_spawn = {".*_elbow_joint": 1.57}  # Insert etc: IK relaxes this to ~0.17 (unchanged from before)
    env_cfg.scene.robot.init_state.joint_pos = {**_legs, **_arm_spawn}
    # Stiffen the arms so the ready pose (elbow ~90) HOLDS against gravity + the heavy Dex3 hand. The
    # RL-tuned arm gains (~50 at the elbow) are too soft, so the arm droops toward straight before the
    # operator connects. A firmer PD tracks the IK's joint targets, keeping the elbow bent, and also
    # makes teleop feel more precise. (Arms are commanded externally, separate from SONIC's legs.)
    _arms = env_cfg.scene.robot.actuators["arms"]
    _arms.stiffness = {".*_(shoulder|elbow|wrist).*_joint": 200.0}
    _arms.damping = {".*_(shoulder|elbow|wrist).*_joint": 20.0}

    if args.xr:
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    dev = env.device

    reset_flag = {"do": False}

    def _reset():
        reset_flag["do"] = True
        print("[sonic] reset requested", flush=True)

    # VR only: append the walk retargeter + unify ALL retargeter devices to the env device (so
    # advance()'s torch.cat over the retargeter outputs doesn't mix cpu/cuda). advance() = [arm..., walk(5)].
    teleop = None
    if args.input == "vr":
        dev_cfg = env_cfg.teleop_devices.devices[args.teleop_device]
        for rt in dev_cfg.retargeters:
            rt.sim_device = str(dev)
        dev_cfg.retargeters = list(dev_cfg.retargeters) + [WalkRetargeterCfg(sim_device=str(dev))]
        teleop = create_teleop_device(args.teleop_device, env_cfg.teleop_devices.devices,
                                      {"R": _reset, "RESET": _reset})
        print(f"[sonic] VR teleop device: {args.teleop_device} (+walk)", flush=True)

    # --- SONIC policy + joint maps ---
    jn = robot.joint_names
    sidx = [jn.index(n) for n in SONIC_JOINTS]
    act_idx = torch.tensor(sidx[:15], device=dev)
    walk_sess = ort.InferenceSession(args.walk_onnx, providers=["CPUExecutionProvider"])
    bal_sess = ort.InferenceSession(args.balance_onnx, providers=["CPUExecutionProvider"])
    in_name = walk_sess.get_inputs()[0].name
    print(f"[sonic] loaded Walk + Balance ONNX (input '{in_name}', 516->15)", flush=True)

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

    # Capture the "hold arms still" IK target ONCE, here at spawn, while the elbows are at the
    # commanded 90 deg. The settle used to call rest_arm_action() fresh every frame, which re-anchored
    # the target to the arm's just-sagged pose each step -- so under gravity the elbow ratcheted DOWN
    # (1.57 -> ~0.84) before "ready". Holding ONE fixed target preserves the 90 deg pose (a small PD
    # sag aside) and also stops the redundant-7-DOF null-space from drifting while it "holds still".
    last_arm = rest_arm_action()

    # SETTLE: PIN the base perfectly upright at the spawn while the legs/feet plant + the arms reach
    # their hold pose, THEN hand a level, zero-velocity robot to SONIC. Without this the free base
    # settles slightly pitched and SONIC lurches forward catching it (a near-faceplant). ~0.35 s.
    # Arm target is re-read each frame here (stable "hold where it is"): a FIXED elbow-90 wrist target
    # is redundant, so gravity drifts the spare DOF and the elbow snaps STRAIGHT to the far limit --
    # worse than a gentle sag. Stiffer arms (set above) keep this hold much closer to the 90 deg spawn.
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
    # Capture the hold-arms IK target ONCE, now, after the settle -- re-solving it every main-loop
    # frame lets the redundant null-space drift the arm; one fixed target keeps it steady.
    rest_arm = rest_arm_action()
    # Even with a fixed EE target the redundant IK's null-space slowly bows the elbow away from the
    # settle pose (UP in Carry, down in Insert). Capture the settle JOINT pose so we can pin the arm
    # there each IDLE frame (released the instant the controller streams), stopping that drift.
    _hold_idx = torch.tensor([i for i, n in enumerate(robot.joint_names)
                              if any(k in n for k in ("shoulder", "elbow", "wrist"))], device=dev)
    _hold_pose = robot.data.joint_pos[:, _hold_idx].clone()
    _hold_zero = torch.zeros_like(_hold_pose)

    if args.input == "vr":
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
            _ee_w = robot.data.body_state_w[0, _eeb, 0:3].cpu().numpy().astype(np.float32)
            _ee = (_R.as_matrix().T @ (_ee_w - _rpos)).astype(np.float32)  # EE in the BASE frame (target lives there)
            _rt._init_pos = _ee.copy()
            _rt._pos = _ee.copy()
            _rt._lo = _ee - np.array([0.45, 0.45, 0.45], dtype=np.float32)
            _rt._hi = _ee + np.array([0.45, 0.45, 0.45], dtype=np.float32)
            _rt._prev = None
            _rt._smooth = None

    # ---- keyboard input (desktop, no headset): full VR parity -- BOTH arms (position + wrist
    # rotation + grip), walk, and lean. TAB switches which arm the manipulation keys drive. ----
    kb = None
    if args.input == "keyboard":
        import math  # noqa: E402
        _pressed = collections.deque()
        try:
            import carb  # noqa: E402
            import omni.appwindow  # noqa: E402
            _kbd_iface = carb.input.acquire_input_interface()
            _kbd = omni.appwindow.get_default_app_window().get_keyboard()

            def _on_key(e):
                if e.type == carb.input.KeyboardEventType.KEY_PRESS:
                    _pressed.append(e.input.name)
                return True
            _kbd_iface.subscribe_to_keyboard_events(_kbd, _on_key)
        except Exception as _e:  # noqa: BLE001  (headless / no window -> just run SONIC with no input)
            print(f"[sonic] keyboard listener unavailable ({_e}); running with no input", flush=True)

        _box = torch.tensor([0.30, 0.30, 0.35], device=dev)                    # per-arm reach half-extent
        kb = {
            "active": "R",                                                      # arm the manip keys drive
            "R_ee": rest_arm[0:7].clone(), "L_ee": rest_arm[8:15].clone(),      # per-arm EE target (root frame)
            "R_lo": rest_arm[0:3] - _box, "R_hi": rest_arm[0:3] + _box,
            "L_lo": rest_arm[8:11] - _box, "L_hi": rest_arm[8:11] + _box,
            "R_grip_open": True, "L_grip_open": True, "lean": 0.0, "quit": False,
        }
        _POS_KEYS = {"W": (0, 0.02), "S": (0, -0.02), "A": (1, 0.02),           # key -> (xyz axis, dpos m)
                     "D": (1, -0.02), "Q": (2, 0.02), "E": (2, -0.02)}
        _ROT_KEYS = {"U": (0, 0.10), "O": (0, -0.10), "I": (1, 0.10),           # key -> (axis, dangle rad)
                     "K": (1, -0.10), "J": (2, 0.10), "L": (2, -0.10)}
        _WALK_KEYS = {"UP": (0, 0.1), "DOWN": (0, -0.1), "LEFT": (2, 0.1),      # key -> (vx/vy/wz idx, step)
                      "RIGHT": (2, -0.1), "COMMA": (1, 0.1), "PERIOD": (1, -0.1)}
        _AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

        def _qmul(a, b):                                                       # wxyz Hamilton product (single quats)
            return torch.stack([
                a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])

        def _rotate(q, axis_i, ang):                                          # pre-multiply q by a root-axis rotation
            ax = _AXES[axis_i]
            s = math.sin(ang / 2.0)
            dq = torch.tensor([math.cos(ang / 2.0), ax[0] * s, ax[1] * s, ax[2] * s], device=dev)
            out = _qmul(dq, q)
            return out / torch.linalg.norm(out)

        def kb_drain():
            while _pressed:
                k = _pressed.popleft()
                a = kb["active"]
                if k in _POS_KEYS:
                    i, d = _POS_KEYS[k]
                    kb[a + "_ee"][i] += d
                elif k in _ROT_KEYS:
                    i, d = _ROT_KEYS[k]
                    kb[a + "_ee"][3:7] = _rotate(kb[a + "_ee"][3:7], i, d)
                elif k in _WALK_KEYS:
                    i, d = _WALK_KEYS[k]
                    loco_cmd[i] += d
                elif k == "SPACE":
                    loco_cmd[:] = 0.0
                elif k == "G":
                    kb[a + "_grip_open"] = not kb[a + "_grip_open"]
                elif k == "TAB":
                    kb["active"] = "L" if a == "R" else "R"
                    print(f"[sonic] active arm -> {kb['active']}", flush=True)
                elif k == "T":
                    kb["lean"] = min(kb["lean"] + 0.2, 1.0)
                elif k == "Y":
                    kb["lean"] = max(kb["lean"] - 0.2, -1.0)
                elif k == "R":
                    _reset()
                elif k == "ESCAPE":
                    kb["quit"] = True
            kb["R_ee"][0:3] = torch.clamp(kb["R_ee"][0:3], kb["R_lo"], kb["R_hi"])
            kb["L_ee"][0:3] = torch.clamp(kb["L_ee"][0:3], kb["L_lo"], kb["L_hi"])
            np.clip(loco_cmd, [-0.8, -0.5, -1.0], [1.0, 0.5, 1.0], out=loco_cmd)
            rpy_cmd[1] = kb["lean"] * LEAN_MAG

        def kb_arm_action():
            # env action = [R pose(7), R grip(1), L pose(7), L grip(1)] -- both arms driven from keys.
            rg = torch.ones(1, device=dev) if kb["R_grip_open"] else -torch.ones(1, device=dev)
            lg = torch.ones(1, device=dev) if kb["L_grip_open"] else -torch.ones(1, device=dev)
            return torch.cat([kb["R_ee"], rg, kb["L_ee"], lg])
        print("KEYBOARD ready (bimanual).  TAB = switch active arm (R/L).  active arm: W/S A/D Q/E = move "
              "X/Y/Z, U/O I/K J/L = roll/pitch/yaw, G = grip.  walk: arrows (UP/DOWN fwd, LEFT/RIGHT turn), "
              ",/. strafe, T/Y lean, SPACE stop.  R reset, ESC quit.", flush=True)

    _elb_i = robot.joint_names.index("right_elbow_joint")
    print(f"[sonic] settled+centered at pelvis=({spawn_root[0,0]:.2f},{spawn_root[0,1]:.2f},"
          f"{spawn_root[0,2]:.2f}) right_elbow={float(robot.data.joint_pos[0, _elb_i]):.2f}rad "
          f"(target 1.57 = 90deg)", flush=True)
    # DIAGNOSTIC: full arm joint values, right vs left, to see the asymmetric IK elbow resolution.
    _dbg_j = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
              "right_elbow_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
              "left_shoulder_yaw_joint", "left_elbow_joint"]
    _dbg_v = robot.data.joint_pos[0, [robot.joint_names.index(j) for j in _dbg_j]].cpu().numpy()
    print("[sonic] arm joints  R[sp,sr,sy,elb]=" + ",".join(f"{v:+.2f}" for v in _dbg_v[:4]) +
          "  L[sp,sr,sy,elb]=" + ",".join(f"{v:+.2f}" for v in _dbg_v[4:]), flush=True)
    if args.input == "vr":
        print("Teleop ready. In the Isaac Sim UI: AR panel -> Start AR, then connect the Pico. "
              "LEFT stick = walk, RIGHT stick X = turn, RIGHT btn = stop, LEFT X/Y = lean. "
              "Arms: the usual controller_rel teleop (grip-clutch + move, trigger to grasp).", flush=True)
    else:
        print("Teleop ready (keyboard). Click the Isaac Sim viewport to focus it, then use the keys "
              "listed above (TAB switches arm; W/S A/D Q/E move + U/O I/K J/L rotate; arrows walk).", flush=True)

    def resettle():
        """Re-plant the free base after a re-home, using the SAME pin-upright + SONIC settle-in as
        startup. A reset teleports the root back to the spawn, but if we then hand the robot straight
        to SONIC from a cold (zeroed) obs history it lurches to "catch" itself and flies/faceplants.
        Pinning it level + zero-velocity while the feet re-plant, then warming SONIC in place, hands
        balance a level, still robot -- the same reason startup settles before "Teleop ready"."""
        nonlocal obs_hist, last_action, home_xy, last_arm
        last_action = np.zeros(N_ACT, dtype=np.float32)  # start SONIC's action history clean, like startup
        pin = spawn_root.clone()
        # Pin the pelvis at the RAISED spawn height (>=0.80), not the settled ~0.74, so the feet
        # re-plant WITH clearance -- exactly what startup does (init_state z is raised to 0.80). Pinning
        # at the settled height drops the feet onto/through the floor and the depenetration kick, plus a
        # cold SONIC catch, is what tipped the robot over "randomly" on reset.
        pin[:, 2] = max(float(spawn_root[0, 2]), 0.80)
        zv = torch.zeros((env.num_envs, 6), device=dev)
        ld = torch.as_tensor(DEFAULT_15, device=dev).unsqueeze(0)
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
        for _ in range(40):                                  # pin level while feet plant
            robot.set_joint_position_target(ld, joint_ids=act_idx)
            robot.write_root_pose_to_sim(pin)
            robot.write_root_velocity_to_sim(zv)
            env.step(rest_arm_action().repeat(env.num_envs, 1))  # re-read (like startup): symmetric elbows
        robot.write_root_pose_to_sim(pin)
        robot.write_root_velocity_to_sim(zv)
        obs_hist = collections.deque([build_obs()] * HIST_LEN, maxlen=HIST_LEN)  # warm w/ real state
        for _ in range(50):                                  # SONIC settles its weight in place
            obs_hist.append(build_obs())
            flat = np.concatenate(obs_hist).astype(np.float32)[None]
            # CRITICAL: update the *nonlocal* last_action every step (a throwaway local left the obs's
            # "previous action" channel stale, so SONIC's history was incoherent, it never balanced, and
            # it fell the instant the pin released). This is what startup does.
            last_action = bal_sess.run(None, {in_name: flat})[0][0]
            lt = torch.as_tensor(last_action * ACTION_SCALE + DEFAULT_15, device=dev)
            robot.set_joint_position_target(lt.unsqueeze(0), joint_ids=act_idx)
            env.step(rest_arm_action().repeat(env.num_envs, 1))  # re-read (like startup): symmetric elbows
        home_xy = robot.data.root_pos_w[0, 0:2].cpu().numpy().copy()  # hold where it actually stands
        last_arm = rest_arm

    n_walk = 5
    step_i = 0
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                if reset_flag["do"]:
                    reset_flag["do"] = False
                    env.reset()                              # reset the task (bulb/socket, episode buffers)
                    # A FREE base isn't re-homed by env.reset() -> teleport the root back to the good
                    # centered spawn (zero velocity), THEN re-plant with the same pin + SONIC settle-in
                    # as startup. Handing a cold/zeroed-history robot straight to SONIC made it lurch
                    # and fly; resettle() lands it level, still, and balanced before control resumes.
                    robot.write_root_pose_to_sim(spawn_root)
                    robot.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=dev))
                    resettle()
                    if args.input == "vr":
                        for _rt in getattr(teleop, "_retargeters", None) or []:
                            if hasattr(_rt, "reset"):
                                _rt.reset()                   # re-reference arm targets to the re-homed robot
                    elif kb is not None:                      # re-home both keyboard EE targets + grips
                        kb["R_ee"] = rest_arm[0:7].clone()
                        kb["L_ee"] = rest_arm[8:15].clone()
                        kb["R_grip_open"] = kb["L_grip_open"] = True
                        kb["lean"] = 0.0
                    loco_cmd[:] = 0.0
                    rpy_cmd[:] = 0.0
                    print("[sonic] reset done", flush=True)

                if args.input == "vr":
                    # Sync each arm retargeter's base ROTATION to the LIVE base every frame, but KEEP its
                    # baked base POSITION. The retargeter maps its world-frame EE target into the root frame
                    # the IK expects; the transform was baked ONCE assuming a bolted/static base, but SONIC
                    # now sways + walks the base.
                    #  - Live rotation: a stale root rotation aims the command the wrong way as the base
                    #    yaws/sways, so the arm lunges. Keeping it live kills that swing.
                    #  - Baked position (NOT updated to live): the world target then rides the base's
                    #    TRANSLATION, so the hand FOLLOWS the body when you walk instead of hanging in world.
                    # Never touch _pos/_init_pos: those hold the target.
                    _rq = robot.data.root_quat_w[0].cpu().numpy()  # w, x, y, z
                    _rR = _Rot.from_quat([_rq[1], _rq[2], _rq[3], _rq[0]])
                    for _rt in getattr(teleop, "_retargeters", None) or []:
                        if hasattr(_rt, "_root_pos"):
                            _rt._root_R = _rR
                            _rt._root_R_T = _rR.as_matrix().T.astype(np.float32)

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
                        last_arm = rest_arm          # no controller -> FIXED rest pose (no IK re-solve jitter)
                        loco_cmd[:] = 0.0
                        rpy_cmd[:] = 0.0
                else:                                # keyboard: keys persist loco_cmd + move the arm EE target
                    kb_drain()
                    if kb["quit"]:
                        break
                    last_arm = kb_arm_action()

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
                        loco_cmd[0] = float(np.clip(
                            HOLD_KP * (np.cos(yaw) * e[0] + np.sin(yaw) * e[1]), -HOLD_VMAX, HOLD_VMAX))
                        loco_cmd[1] = float(np.clip(
                            HOLD_KP * (-np.sin(yaw) * e[0] + np.cos(yaw) * e[1]), -HOLD_VMAX, HOLD_VMAX))

                # SONIC runs EVERY frame -- balance is not optional. (Gating this on `out` made the
                # free-base robot collapse whenever the headset wasn't streaming.)
                obs_hist.append(build_obs())
                flat = np.concatenate(obs_hist).astype(np.float32)[None]
                sess = walk_sess if np.linalg.norm(loco_cmd) > 0.05 else bal_sess
                last_action = sess.run(None, {in_name: flat})[0][0]
                leg_target = torch.as_tensor(last_action * ACTION_SCALE + DEFAULT_15, device=dev)
                robot.set_joint_position_target(leg_target.unsqueeze(0), joint_ids=act_idx)
                env.step(last_arm.repeat(env.num_envs, 1))   # arms via the real env action manager
                # Pin the arm at its settle joints whenever it isn't being actively moved -- i.e. when the
                # commanded arm EE is still ~at the rest target. (The retargeters return HELD, non-None
                # values even with the headset off, so gating on `out is None` never fired.) This stops the
                # redundant IK's slow null-space drift; the instant the operator moves the arm it releases.
                if bool(torch.allclose(last_arm, rest_arm, atol=0.05)):
                    robot.write_joint_state_to_sim(_hold_pose, _hold_zero, joint_ids=_hold_idx)

                step_i += 1
                if step_i % (20 if step_i <= 200 else 100) == 0:
                    p = robot.data.root_pos_w[0].cpu().numpy()
                    _re = float(robot.data.joint_pos[0, robot.joint_names.index("right_elbow_joint")])
                    _tag = "FELL" if p[2] < 0.4 else "ok"
                    print(f"[sonic] step {step_i} pelvis=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) "
                          f"r_elbow={_re:.2f} loco_cmd={np.round(loco_cmd, 2)} {_tag}", flush=True)
        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001
            print(f"[sonic] step skipped while view recovers: {e}", flush=True)
            with contextlib.suppress(Exception):
                env.sim.render()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
