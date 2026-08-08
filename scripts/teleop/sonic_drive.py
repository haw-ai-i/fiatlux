"""Drive the G1 in **Isaac Sim** with NVIDIA's pre-trained SONIC / GR00T-WholeBodyControl policy.

This is a sim-to-sim port: the ONNX policy (released by NVIDIA for MuJoCo deployment) is run
*inside Isaac Sim*. No MuJoCo. The MuJoCo reference script only told us the exact contract the
network expects; we reproduce that contract from Isaac's articulation state.

    SONIC contract (from GEAR-SONIC g1_gear_wbc.yaml + run_mujoco_gear_wbc.py)
    -------------------------------------------------------------------------
    * 29 policy-observed joints, Unitree G1 canonical order: [12 leg, 3 waist, 14 arm].
    * 15 actions  = legs + waist only (arms are commanded externally = your teleop).
    * single_obs (86) = cmd(7) | base_ang_vel_b*0.5 (3) | projected_gravity_b (3)
                        | (q-q_default)*1.0 (29) | qd*0.05 (29) | last_action (15)
      -> stacked over 6 frames = 516-dim policy input.
    * target = action*0.25 + default_angles, tracked by PD (kps/kds already match the
      fiatlux G1 actuator gains -> implicit PD in Isaac reproduces it faithfully).
    * two nets: Walk when |loco_cmd| > 0.05, else Balance (stand-in-place).
    * 200 Hz sim, 50 Hz policy (decimation 4).

You send the locomotion command; the policy lifts the legs + keeps balance:
  keyboard (run WITHOUT --headless): W/S +/- forward, A/D +/- turn, Q/E +/- strafe, X stop, ESC quit
  scripted --headless: walks forward on a schedule and renders a validation video.
"""
import argparse
import collections
import os

import numpy as np
from isaaclab.app import AppLauncher

_POLICY_DIR = "/home/yujin-chen/robotica_project/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/policy"

parser = argparse.ArgumentParser()
parser.add_argument("--input", choices=["keyboard", "scripted", "vr"], default="scripted")
parser.add_argument("--walk_onnx", default=f"{_POLICY_DIR}/GR00T-WholeBodyControl-Walk.onnx")
parser.add_argument("--balance_onnx", default=f"{_POLICY_DIR}/GR00T-WholeBodyControl-Balance.onnx")
parser.add_argument("--terrain", choices=["flat", "steps"], default="flat")
parser.add_argument("--rise", type=float, default=0.10, help="step riser height (m) for --terrain steps")
parser.add_argument("--n_steps", type=int, default=4)
parser.add_argument("--arm", action="store_true",
                    help="enable bimanual arm teleop IK (Insert-style) on top of SONIC walking = loco-manip")
parser.add_argument("--insert", action="store_true",
                    help="spawn the Insert task props (table + socket + graspable bulb) in front to walk up to")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.input == "scripted":
    args.headless = True
    args.enable_cameras = True
elif args.input == "vr":
    args.xr = True   # stereo OpenXR render streamed to the Pico via CloudXR
app = AppLauncher(args).app

import torch  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
import fiatlux_task  # noqa: F401,E402  (registers assets)
import fiatlux_teleop  # noqa: F401,E402  -- registers the FIATLUX-*-Teleop gym ids
from fiatlux_task.robots.g1 import (  # noqa: E402
    G1_INSPIRE_CFG, G1_ARM_JOINTS, G1_EE_BODY, G1_HAND_JOINTS, G1_HAND_OPEN, G1_HAND_GRASP,
)

# ---------------------------------------------------------------------------
# SONIC contract
# ---------------------------------------------------------------------------
SONIC_JOINTS = [
    # left leg (0-5)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # right leg (6-11)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # waist (12-14)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # left arm (15-21)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # right arm (22-28)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# default_angles for the 15 actuated (legs+waist); arms observed relative to 0.
DEFAULT_15 = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                       -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                       0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_29 = np.zeros(29, dtype=np.float32)
DEFAULT_29[:15] = DEFAULT_15

ANG_VEL_SCALE = 0.5
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
HEIGHT_CMD = 0.74
RPY_CMD = np.zeros(3, dtype=np.float32)
OBS_DIM, HIST_LEN, N_ACT = 86, 6, 15
SIM_DT, DECIM = 0.005, 4
FR = "/tmp/fiatlux-xr/sonic_frames"

# ---------------------------------------------------------------------------
# Scene (standalone -- no RL env machinery needed to run a policy)
# ---------------------------------------------------------------------------
sim = SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=1200.0).func("/World/Dome", sim_utils.DomeLightCfg(intensity=1200.0))

# Optional staircase in front (blind step-up test). Each step is a solid box whose top
# sits at i*rise; front face at X0+(i-1)*TREAD -> a climbable staircase.
if args.terrain == "steps":
    RISE, TREAD, WIDTH, X0 = args.rise, 0.32, 1.4, 1.0
    for i in range(1, args.n_steps + 1):
        h = i * RISE
        c = sim_utils.CuboidCfg(
            size=(TREAD, WIDTH, h),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.56)),
            collision_props=sim_utils.CollisionPropertiesCfg())
        c.func(f"/World/step_{i}", c, translation=(X0 + (i - 0.5) * TREAD, 0.0, h / 2))
    print(f"[sonic] staircase: {args.n_steps} steps, rise={RISE:.2f}m, tread={TREAD}m, front at x={X0}m", flush=True)

# Optional Insert task props: a table with the Omniverse socket (kinematic, upright) + a graspable
# bulb, placed in front so you can walk up and insert. Reuses the exact Insert assets + spawners.
if args.insert:
    from fiatlux_task.assets import OMNI_BULB_USD, OMNI_SOCKET_USD  # noqa: E402
    from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import (  # noqa: E402
        _quat_x_deg, _spawn_usd_as_rigid_body,
    )
    from fiatlux_teleop.insert_teleop_env_cfg import _spawn_omni_rigid  # noqa: E402

    TX, TTOP = 1.35, 0.80                       # table front distance + top height (reachable)
    _up = _quat_x_deg(90.0)                      # stand the Y-up assets upright, bulb up
    _tbl = sim_utils.CuboidCfg(
        size=(0.6, 1.2, TTOP), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.25, 0.18)),
        collision_props=sim_utils.CollisionPropertiesCfg())
    _tbl.func("/World/table", _tbl, translation=(TX, 0.0, TTOP / 2))
    _sock = sim_utils.UsdFileCfg(
        usd_path=OMNI_SOCKET_USD, func=_spawn_usd_as_rigid_body, scale=(0.007, 0.007, 0.007),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))
    _sock.func("/World/socket", _sock, translation=(TX - 0.08, -0.12, TTOP + 0.02), orientation=_up)
    _bulb = sim_utils.UsdFileCfg(
        usd_path=OMNI_BULB_USD, func=_spawn_omni_rigid, scale=(0.006, 0.006, 0.006),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, solver_position_iteration_count=64,
            solver_velocity_iteration_count=4, max_depenetration_velocity=0.5))
    _bulb.func("/World/bulb", _bulb, translation=(TX - 0.08, 0.14, TTOP + 0.06), orientation=_up)
    print(f"[sonic] Insert props: table+socket+bulb at x~{TX}m (walk forward to reach)", flush=True)

# spawn at SONIC's nominal standing pose (legs+waist at default_angles; arms/hands 0)
_init = {f"{s}_hip_pitch_joint": -0.1 for s in ("left", "right")}
_init.update({f"{s}_knee_joint": 0.3 for s in ("left", "right")})
_init.update({f"{s}_ankle_pitch_joint": -0.2 for s in ("left", "right")})
robot_cfg = G1_INSPIRE_CFG.replace(
    prim_path="/World/Robot",
    init_state=G1_INSPIRE_CFG.init_state.replace(pos=(0.0, 0.0, 0.80), joint_pos=_init),
)
robot = Articulation(robot_cfg)

cam = None
if args.input == "scripted":
    cam = Camera(CameraCfg(prim_path="/World/cam", height=600, width=900, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.01, 60.0)),
                offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), convention="world")))

sim.reset()
dev = sim.device

# joint index maps (resolve SONIC order -> Isaac articulation order, by name)
jn = robot.joint_names
missing = [n for n in SONIC_JOINTS if n not in jn]
assert not missing, f"G1 is missing SONIC joints: {missing}"
sidx = [jn.index(n) for n in SONIC_JOINTS]          # 29 Isaac indices, SONIC order
act_isaac_idx = torch.tensor(sidx[:15], device=dev)  # the 15 leg/waist joints
default_target = robot.data.default_joint_pos.clone()  # arms+hands stay here (=0)

# --- BIMANUAL differential IK (reuses the Insert task's exact controller: DLS, pose command).
# SONIC drives legs+waist (15); both arms ride on top = loco-manipulation. Each arm is a dict of
# its Isaac indices / IK / hand presets / nudge state; the loop iterates both. ---
arms = []
if args.arm:
    # left-arm names are just the right names with the side prefix flipped (as Insert derives them).
    _LEFT_ARM = [j.replace("right_", "left_", 1) for j in G1_ARM_JOINTS]
    _LEFT_HAND = [j.replace("R_", "L_", 1) for j in G1_HAND_JOINTS]
    _LEFT_OPEN = dict.fromkeys(_LEFT_HAND, 0.0)
    _LEFT_GRASP = {k.replace("R_", "L_", 1): v for k, v in G1_HAND_GRASP.items()}

    def make_arm(joints, ee_body, hand_joints, hopen, hgrasp):
        ids = [jn.index(n) for n in joints]
        return {
            "ids": ids, "ids_t": torch.tensor(ids, device=dev),
            "hand_ids_t": torch.tensor([jn.index(n) for n in hand_joints], device=dev),
            "hand_open": torch.tensor([hopen[n] for n in hand_joints], device=dev),
            "hand_grasp": torch.tensor([hgrasp[n] for n in hand_joints], device=dev),
            "ee_body_id": robot.body_names.index(ee_body),
            "jac_joints": [i + 6 for i in ids],                # floating base -> +6 base-DOF offset
            "ik": DifferentialIKController(
                DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
                num_envs=1, device=dev),
            "prev_rc": None, "quat_ref": None, "ee_quat_ref": None, "grip": False,
        }

    arms = [
        make_arm(G1_ARM_JOINTS, G1_EE_BODY, G1_HAND_JOINTS, G1_HAND_OPEN, G1_HAND_GRASP),
        make_arm(_LEFT_ARM, "left_wrist_yaw_link", _LEFT_HAND, _LEFT_OPEN, _LEFT_GRASP),
    ]

    def ee_pose_b(a):
        """Current EE pose in the (moving) base frame -> (pos(3), quat(4)) tensors."""
        ee_w = robot.data.body_state_w[:, a["ee_body_id"], 0:7]
        return mu.subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, ee_w[:, 0:3], ee_w[:, 3:7])

    def compute_arm_targets(a):
        """Solve arm a's joint targets so its EE reaches a['ee_cmd'] (base frame)."""
        a["ik"].set_command(torch.as_tensor(a["ee_cmd"], device=dev, dtype=torch.float32).unsqueeze(0))
        jac_w = robot.root_physx_view.get_jacobians()[:, a["ee_body_id"], :, a["jac_joints"]]
        base_rot_m = mu.matrix_from_quat(mu.quat_inv(robot.data.root_quat_w))
        jac_b = jac_w.clone()
        jac_b[:, :3, :] = torch.bmm(base_rot_m, jac_w[:, :3, :])
        jac_b[:, 3:, :] = torch.bmm(base_rot_m, jac_w[:, 3:, :])
        p_b, q_b = ee_pose_b(a)
        return a["ik"].compute(p_b, q_b, jac_b, robot.data.joint_pos[:, a["ids"]])[0]

    def _q(np4):
        return torch.as_tensor(np4, device=dev, dtype=torch.float32).unsqueeze(0)

    def nudge_arm(a, rc, rq, squeeze, trigger, arm_scale):
        """Clutch-gated body-relative nudge: squeeze + move controller -> pos delta + wrist rotation."""
        if squeeze > 0.5 and rc[2] > 0.3:                      # clutch held + controller tracked
            if a["prev_rc"] is None:                           # clutch just engaged -> set references
                a["quat_ref"] = rq.copy()
                a["ee_quat_ref"] = a["ee_cmd"][3:7].copy()
            else:
                d = rc - a["prev_rc"]
                n = float(np.linalg.norm(d))
                if 0.004 < n < 0.20:                           # deadzone .. glitch-reject
                    d = d * arm_scale
                    m = float(np.linalg.norm(d))
                    if m > 0.03:
                        d *= 0.03 / m                          # cap per-frame step
                    r_t = mu.matrix_from_quat(mu.quat_inv(robot.data.root_quat_w))[0].cpu().numpy()
                    a["ee_cmd"][:3] = np.clip(a["ee_cmd"][:3] + r_t @ d, a["reach_lo"], a["reach_hi"])
                # wrist: apply the controller's rotation-since-clutch to the ref EE quat (in base frame)
                root_q = robot.data.root_quat_w
                rel_w = mu.quat_mul(_q(rq), mu.quat_inv(_q(a["quat_ref"])))
                rel_b = mu.quat_mul(mu.quat_mul(mu.quat_inv(root_q), rel_w), root_q)
                qn = mu.quat_mul(rel_b, _q(a["ee_quat_ref"]))
                a["ee_cmd"][3:7] = (qn / torch.norm(qn, dim=-1, keepdim=True))[0].cpu().numpy()
            a["prev_rc"] = rc.copy()
        else:
            a["prev_rc"] = None                                # clutch released -> hold, drop reference
        a["grip"] = trigger > 0.5

# ---------------------------------------------------------------------------
# ONNX policies
# ---------------------------------------------------------------------------
walk_sess = ort.InferenceSession(args.walk_onnx, providers=["CPUExecutionProvider"])
bal_sess = ort.InferenceSession(args.balance_onnx, providers=["CPUExecutionProvider"])
_in_name = walk_sess.get_inputs()[0].name
print(f"[sonic] loaded Walk + Balance ONNX (input '{_in_name}', 516->15)", flush=True)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
loco_cmd = np.zeros(3, dtype=np.float32)   # [vx, vy, wz]
keys_once = collections.deque()
if args.input == "keyboard":
    import carb  # noqa
    import omni.appwindow  # noqa
    _iface = carb.input.acquire_input_interface()
    _kbd = omni.appwindow.get_default_app_window().get_keyboard()

    def _on_key(e):
        if e.type == carb.input.KeyboardEventType.KEY_PRESS:
            keys_once.append(e.input.name)
        return True
    _iface.subscribe_to_keyboard_events(_kbd, _on_key)
    print("KEYBOARD: W/S +/-forward,  A/D +/-turn,  Q/E +/-strafe,  X stop,  ESC quit", flush=True)


def drain_keys():
    """Increment-on-press command (dial in a speed; it keeps walking until X)."""
    quit_now = False
    while keys_once:
        k = keys_once.popleft()
        if k == "W":
            loco_cmd[0] += 0.1
        elif k == "S":
            loco_cmd[0] -= 0.1
        elif k == "A":
            loco_cmd[2] += 0.1
        elif k == "D":
            loco_cmd[2] -= 0.1
        elif k == "Q":
            loco_cmd[1] += 0.1
        elif k == "E":
            loco_cmd[1] -= 0.1
        elif k == "X":
            loco_cmd[:] = 0.0
        elif k == "ESCAPE":
            quit_now = True
    np.clip(loco_cmd, [-0.8, -0.5, -1.0], [1.0, 0.5, 1.0], out=loco_cmd)
    return quit_now


# Pico controller over CloudXR. Custom retargeter reads BOTH thumbsticks (walk) and buttons
# (stop + upper-body lean), returning [vx, vy, wz, stop, lean]. Signs baked for the Pico
# (left stick fwd -> +vx). advance() returns zeros until a controller streams (stands / Balance).
vr_device = None
if args.input == "vr":
    from dataclasses import dataclass  # noqa: E402
    from isaaclab.devices.device_base import DeviceBase  # noqa: E402
    from isaaclab.devices.openxr import XrCfg  # noqa: E402
    from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg  # noqa: E402
    from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg  # noqa: E402
    from isaaclab.devices.teleop_device_factory import create_teleop_device  # noqa: E402

    _POSE = DeviceBase.MotionControllerDataRowIndex.POSE.value
    _ROW = DeviceBase.MotionControllerDataRowIndex.INPUTS.value
    _IDX = DeviceBase.MotionControllerInputIndex
    _TL = DeviceBase.TrackingTarget.CONTROLLER_LEFT
    _TR = DeviceBase.TrackingTarget.CONTROLLER_RIGHT

    class SonicVRRetargeter(RetargeterBase):
        """LEFT stick=walk, buttons=stop+lean; each controller pose/squeeze/trigger=that-side arm+grip.
        Output [walk(5), R_pos(3), R_quat(4,wxyz), R_sq, R_trig, L_pos(3), L_quat(4,wxyz), L_sq, L_trig] (23)."""

        def __init__(self, cfg):
            super().__init__(cfg)
            self.cfg = cfg

        @staticmethod
        def _read(data, target):
            cd = data.get(target) if data else None
            if cd is None or len(cd) <= _ROW:
                return np.array([0, 0, 0, 1, 0, 0, 0], np.float32), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            pose = np.asarray(cd[_POSE], dtype=np.float32).reshape(-1)[:7]   # pos(3) + quat(4, wxyz)
            inp = cd[_ROW]
            return (pose,
                    float(inp[_IDX.THUMBSTICK_X.value]), float(inp[_IDX.THUMBSTICK_Y.value]),
                    float(inp[_IDX.TRIGGER.value]), float(inp[_IDX.SQUEEZE.value]),
                    float(inp[_IDX.BUTTON_0.value]), float(inp[_IDX.BUTTON_1.value]))

        def retarget(self, data):
            lp, lx, ly, ltrig, lsq, lb0, lb1 = self._read(data, _TL)
            rp, rx, ry, rtrig, rsq, rb0, rb1 = self._read(data, _TR)
            ms = self.cfg.movement_scale
            vx = ly * ms                                   # left stick fwd -> walk fwd
            vy = -lx * ms                                  # left stick right -> strafe
            wz = -rx                                       # right stick X -> turn
            stop = 1.0 if rb0 > 0.5 else 0.0               # RIGHT button A -> stop
            lean = (1.0 if lb0 > 0.5 else 0.0) - (1.0 if lb1 > 0.5 else 0.0)  # LEFT X fwd / Y back
            return torch.tensor([vx, vy, wz, stop, lean,
                                 rp[0], rp[1], rp[2], rp[3], rp[4], rp[5], rp[6], rsq, rtrig,
                                 lp[0], lp[1], lp[2], lp[3], lp[4], lp[5], lp[6], lsq, ltrig],
                                device=self.cfg.sim_device, dtype=torch.float32)

        def get_requirements(self):
            return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    @dataclass
    class SonicVRRetargeterCfg(RetargeterCfg):
        movement_scale: float = 0.5
        retargeter_type: type = SonicVRRetargeter

    _rt = SonicVRRetargeterCfg(movement_scale=0.5, sim_device=str(dev))
    _xr_cfg = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(0.0, 0.0, 0.0, 1.0))
    vr_device = create_teleop_device(
        "vrsonic", {"vrsonic": OpenXRDeviceCfg(retargeters=[_rt], sim_device=str(dev), xr_cfg=_xr_cfg)})
    print("VR: LEFT stick = walk (fwd/back/strafe),  RIGHT stick X = turn.", flush=True)
    print("    RIGHT button = STOP,  LEFT X = lean upper body fwd,  LEFT Y = lean back.", flush=True)
    if args.arm:
        print("    ARMS (bimanual): hold SQUEEZE on a controller to clutch that arm, then move + TWIST "
              "your hand to drive it; TRIGGER = grip. Release squeeze to hold.", flush=True)
    print("In the Isaac Sim UI: open the AR panel (OpenXR / System OpenXR Runtime), click 'Start AR', "
          "then connect the Pico CloudXR web client.", flush=True)


# ---------------------------------------------------------------------------
# Observation (reproduce compute_observation exactly, from Isaac state)
# ---------------------------------------------------------------------------
def build_single_obs(last_action, height_cmd, rpy):
    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[0:3] = loco_cmd * CMD_SCALE
    o[3] = height_cmd
    o[4:7] = rpy
    o[7:10] = robot.data.root_ang_vel_b[0].cpu().numpy() * ANG_VEL_SCALE
    o[10:13] = robot.data.projected_gravity_b[0].cpu().numpy()
    q = robot.data.joint_pos[0, sidx].cpu().numpy()
    qd = robot.data.joint_vel[0, sidx].cpu().numpy()
    o[13:42] = (q - DEFAULT_29) * DOF_POS_SCALE
    o[42:71] = qd * DOF_VEL_SCALE
    o[71:86] = last_action
    return o


os.makedirs(FR, exist_ok=True)
for f in os.listdir(FR):
    os.remove(os.path.join(FR, f))

obs_hist = collections.deque([np.zeros(OBS_DIM, dtype=np.float32)] * HIST_LEN, maxlen=HIST_LEN)
last_action = np.zeros(N_ACT, dtype=np.float32)
target15 = torch.tensor(DEFAULT_15, device=dev)
height_cmd = float(HEIGHT_CMD)              # frozen (crouch disabled -- see VR notes)
rpy_cmd = np.zeros(3, dtype=np.float32)     # [roll, pitch, yaw]; VR lean buttons drive pitch
LEAN_MAG = 0.30                             # rad of upper-body pitch per lean button (~17 deg)

ARM_SCALE = 1.0
if args.arm:
    for a in arms:
        p0, q0 = ee_pose_b(a)                                   # rest EE pose in base frame
        a["ee_cmd"] = torch.cat([p0[0], q0[0]]).cpu().numpy()   # (7,) hold target, base frame
        a["ee_cmd0"] = a["ee_cmd"].copy()
        a["targets"] = robot.data.joint_pos[0, a["ids_t"]].clone()
        a["reach_lo"] = a["ee_cmd0"][:3] - np.array([0.30, 0.30, 0.35], dtype=np.float32)
        a["reach_hi"] = a["ee_cmd0"][:3] + np.array([0.30, 0.30, 0.35], dtype=np.float32)

step = 0
if args.input != "scripted":
    MAXF = 10 ** 9
else:
    MAXF = 3200 if args.terrain == "steps" else 1600   # steps: ~16 s to reach + climb
while app.is_running() and step < MAXF:
    if args.input == "keyboard":
        if drain_keys():
            break
    elif args.input == "vr":
        out = vr_device.advance()               # [vx, vy, wz, stop, lean] (zeros until connected)
        if out is not None:
            o = out.detach().cpu().numpy()
            loco_cmd[:] = o[:3]
            if o[3] > 0.5:                       # STOP button -> zero all velocity
                loco_cmd[:] = 0.0
            rpy_cmd[1] = o[4] * LEAN_MAG         # upper-body lean (pitch): LEFT X fwd / LEFT Y back
            # height_cmd stays frozen at 0.74 (crouch integrator drifted/squatted on the Pico).
            if args.arm:                         # right controller -> arms[0], left -> arms[1]
                nudge_arm(arms[0], o[5:8], o[8:12], o[12], o[13], ARM_SCALE)
                nudge_arm(arms[1], o[14:17], o[17:21], o[21], o[22], ARM_SCALE)
            if step % 100 == 0 and (np.linalg.norm(loco_cmd) > 0.05 or abs(rpy_cmd[1]) > 0.01):
                print(f"[vr] vxyz={np.round(loco_cmd, 2)} lean_pitch={rpy_cmd[1]:+.2f}", flush=True)
    elif args.input == "scripted" and step == 150:   # stand, then walk forward
        loco_cmd[0] = 0.5

    # policy tick @ 50 Hz
    if step % DECIM == 0:
        obs_hist.append(build_single_obs(last_action, height_cmd, rpy_cmd))
        flat = np.concatenate(obs_hist).astype(np.float32)[None]   # (1, 516) oldest..newest
        sess = walk_sess if np.linalg.norm(loco_cmd) > 0.05 else bal_sess
        last_action = sess.run(None, {_in_name: flat})[0][0]
        target15 = torch.as_tensor(last_action * ACTION_SCALE + DEFAULT_15, device=dev)
        if args.arm:
            if args.input == "scripted":
                # headless IK test: reach the RIGHT hand +0.15 m fwd + up (base frame), then grip --
                # validates the arm tracks a target while SONIC balances.
                s = min(max((step - 300) / 200.0, 0.0), 1.0)
                arms[0]["ee_cmd"][0] = arms[0]["ee_cmd0"][0] + 0.15 * s
                arms[0]["ee_cmd"][2] = arms[0]["ee_cmd0"][2] + 0.15 * s
                arms[0]["grip"] = step > 800
            for a in arms:
                a["targets"] = compute_arm_targets(a)

    # PD tracks target every physics step
    tgt = default_target.clone()
    tgt[0, act_isaac_idx] = target15
    if args.arm:
        for a in arms:
            tgt[0, a["ids_t"]] = a["targets"]
            tgt[0, a["hand_ids_t"]] = a["hand_grasp"] if a["grip"] else a["hand_open"]
    robot.set_joint_position_target(tgt)
    robot.write_data_to_sim()
    sim.step()
    robot.update(SIM_DT)

    # follow camera
    p = robot.data.root_pos_w[0].cpu().numpy()
    if args.input == "keyboard":
        sim.set_camera_view(eye=(p[0] - 2.2, p[1] - 2.6, p[2] + 1.4), target=(p[0], p[1], p[2] + 0.3))
    elif cam is not None:
        cam.set_world_poses_from_view(
            torch.tensor([[p[0] - 2.2, p[1] - 2.6, p[2] + 1.3]], device=dev, dtype=torch.float32),
            torch.tensor([[p[0], p[1], p[2] + 0.2]], device=dev, dtype=torch.float32))
        cam.update(SIM_DT)
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(rgb).save(f"{FR}/f{step:05d}.png")
    step += 1

px_f = robot.data.root_pos_w[0, 0].item()
h = robot.data.root_pos_w[0, 2].item()
print(f"steps={step}  final_pelvis x={px_f:.2f} z={h:.3f}  loco_cmd={loco_cmd}", flush=True)
if args.arm:
    _pb, _ = ee_pose_b(arms[0])
    err = float(torch.norm(_pb[0] - torch.as_tensor(arms[0]["ee_cmd"][:3], device=dev)))
    print(f"ARM: right EE pos error vs target = {err * 1000:.0f} mm  (target reach = +0.15m fwd/up)", flush=True)
if h < 0.4:
    print("FELL", flush=True)
elif args.terrain == "steps" and h > 0.74 + 0.6 * args.rise:
    print(f"CLIMBED (pelvis rose ~{h - 0.74:.2f}m above flat)", flush=True)
else:
    print("UPRIGHT", flush=True)
print("SONIC_DONE", flush=True)
sim.stop()
app.close()
