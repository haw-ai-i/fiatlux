# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Insert-task teleop WITH WALKING (whole-body) for the G1 in Isaac Sim (issue #51).

One env, one robot, both capabilities at once:
  * **Legs + waist** are driven by NVIDIA GR00T's pretrained ONNX policy (Balance/Walk) so the robot
    actually **steps and balances** -- arrow keys command base velocity.
  * **Right arm** is driven by the human via **differential IK** (absolute EE target) -- W/S A/D Q/E
    move the hand, Z/X T/G C/V rotate it; **K** toggles the Inspire-hand grip.
So you can walk up to the bench, then reach in and grab / seat the bulb -- true whole-body manipulation.

    export GR00T_POLICY_DIR=/path/to/gr00t_wbc/.../g1/policy
    python scripts/insert_with_walk_teleop.py           # GUI
    python scripts/insert_with_walk_teleop.py --test     # headless self-test (walk + reach, no viewer)

The arm IK mirrors scripts/insert_teleop.py.

EXTERNAL DEPENDENCY: the Balance/Walk ONNX policies ship with NVIDIA's GR00T whole-body-control
project (``gr00t_wbc``), which is NOT part of this repo, plus ``onnxruntime``. Point the loader at
that policy folder via ``--policy_dir`` or the ``GR00T_POLICY_DIR`` env var.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Insert-task teleop with walking (GR00T policy).")
parser.add_argument("--fwd_speed", type=float, default=0.9)
parser.add_argument("--turn_speed", type=float, default=0.6)
parser.add_argument("--test", action="store_true", help="Headless scripted self-test (walk + reach).")
parser.add_argument("--hold_arm", action="store_true", help="Debug: hold arm at default (bypass IK).")
parser.add_argument("--no_bench", action="store_true", help="Debug: skip the table/bulb/socket bench.")
parser.add_argument(
    "--policy_dir",
    type=str,
    default=os.environ.get("GR00T_POLICY_DIR", ""),
    help="Folder with GR00T-WholeBodyControl-{Balance,Walk}.onnx (or set $GR00T_POLICY_DIR).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.test:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import collections

import numpy as np
import onnxruntime as ort
import torch
from fiatlux_task.assets import OMNI_BULB_USD, OMNI_SOCKET_USD, ROOM_USD, SKY_HDRI, TABLE_USD
from fiatlux_task.robots.g1 import (
    G1_ARM_JOINTS,
    G1_EE_BODY,
    G1_HAND_GRASP,
    G1_HAND_JOINTS,
    G1_HAND_OPEN,
    G1_INSPIRE_CFG,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.insert_teleop_env_cfg import _spawn_omni_rigid
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import _quat_x_deg, _spawn_usd_as_rigid_body

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import normalize, quat_apply, quat_from_angle_axis, quat_mul, subtract_frame_transforms

# --- GR00T walk-policy interface (obs/action spec, joint order, gains ported from gr00t_wbc) ------
OBS_MAP = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
ACT_MAP = OBS_MAP[:15]
DEFAULT_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
)
DEFAULT_29 = np.concatenate([DEFAULT_ANGLES, np.zeros(14, dtype=np.float32)])
CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
ANG_VEL_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.5, 0.05, 0.25
HEIGHT_CMD, SIM_DT, DECIMATION, SINGLE_OBS, HIST = 0.74, 0.005, 4, 86, 6

# Fixed "raised" pose for the LEFT arm so it stays up and out of the way (it isn't teleoperated and
# otherwise hangs at its default and clips the table as the robot walks in). Big negative shoulder
# pitch lifts the whole arm up; a little elbow keeps the forearm clear. Keyed by left-arm joint name.
LEFT_ARM_UP = {
    "left_shoulder_pitch_joint": -2.6,
    "left_shoulder_roll_joint": 0.25,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.4,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
}

# bench placement. The packing table is ~2.5 m long (+/-1.24 m from its origin), so it must sit far
# enough that the robot (spawned at the origin) starts clear of it and *walks up* to reach the props.
# bench placement. The packing table is ~2.5 m long (+/-1.24 m from its origin), so it must sit far
# enough that the robot (spawned at the origin) starts clear of it and *walks up* to reach the props.
TABLE_POS = (2.4, 0.0, 0.0)
TABLE_SCALE = (1.0, 1.0, 0.9)
BULB_POS = (1.25, 0.0, 0.95)
SOCKET_POS = (1.35, 0.18, 0.90)
LEASH = 0.12


def _leg_gain(v):
    return {".*_hip_pitch_joint": v[0], ".*_hip_roll_joint": v[0], ".*_hip_yaw_joint": v[0],
            ".*_knee_joint": v[1], ".*_ankle_pitch_joint": v[2], ".*_ankle_roll_joint": v[2]}


class _ComboKeyboard:
    """arrows -> base velocity [vx,vy,yaw]; W/S A/D Q/E move hand, Z/X T/G C/V rotate; K grip; R reset."""

    def __init__(self, fwd, turn, strafe=0.3, pos_s=0.006, rot_s=0.02):
        import carb
        import omni

        self._carb = carb
        self._input = carb.input.acquire_input_interface()
        self._kb = omni.appwindow.get_default_app_window().get_keyboard()
        self._pressed: set = set()
        self._reset = False
        self._grip_open = True
        self._prev_k = False
        self._prev_b = False
        # Camera: default FOLLOW (0) which tracks the robot; [ / ] keys zoom it in/out (mouse zoom
        # doesn't grab in the standalone viewport). B cycles follow -> chest -> free.
        self._cam_mode = 0
        self._cam_dist = 2.8
        self._sub = self._input.subscribe_to_keyboard_events(self._kb, self._on)
        K = carb.input.KeyboardInput
        self._base = {K.UP: (fwd, 0, 0), K.DOWN: (-fwd * 0.7, 0, 0), K.LEFT: (0, 0, turn),
                      K.RIGHT: (0, 0, -turn), K.COMMA: (0, strafe, 0), K.PERIOD: (0, -strafe, 0)}
        self._hand = {K.W: (pos_s, 0, 0), K.S: (-pos_s, 0, 0), K.A: (0, pos_s, 0), K.D: (0, -pos_s, 0),
                      K.Q: (0, 0, pos_s), K.E: (0, 0, -pos_s)}
        self._rot = {K.Z: (rot_s, 0, 0), K.X: (-rot_s, 0, 0), K.T: (0, rot_s, 0), K.G: (0, -rot_s, 0),
                     K.C: (0, 0, rot_s), K.V: (0, 0, -rot_s)}
        self._K = K

    def _on(self, e, *a):
        if e.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed.add(e.input)
            if e.input == self._K.K and not self._prev_k:
                self._grip_open = not self._grip_open
                self._prev_k = True
            if e.input == self._K.B and not self._prev_b:
                self._cam_mode = (self._cam_mode + 1) % 3
                self._prev_b = True
                print(f"[camera] {['FOLLOW', 'CHEST', 'FREE (mouse zoom/orbit)'][self._cam_mode]}")
            if e.input == self._K.R:
                self._reset = True
        elif e.type == self._carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(e.input)
            if e.input == self._K.K:
                self._prev_k = False
            if e.input == self._K.B:
                self._prev_b = False
        return True

    def _sum(self, table):
        out = np.zeros(3, dtype=np.float32)
        for key in self._pressed:
            if key in table:
                out += np.array(table[key], dtype=np.float32)
        return out

    def base_cmd(self):
        return self._sum(self._base)

    def hand_dpos(self):
        return self._sum(self._hand)

    def hand_drot(self):
        return self._sum(self._rot)

    def grip_open(self):
        return self._grip_open

    def cam_mode(self):
        return self._cam_mode

    def cam_dist(self):
        # Zoom IN with [ or = , OUT with ] or - (held). Clamp + print feedback so it's obvious.
        k = self._K
        before = self._cam_dist
        if k.LEFT_BRACKET in self._pressed or k.EQUAL in self._pressed:
            self._cam_dist = max(0.8, self._cam_dist - 0.06)
        if k.RIGHT_BRACKET in self._pressed or k.MINUS in self._pressed:
            self._cam_dist = min(9.0, self._cam_dist + 0.06)
        if abs(self._cam_dist - before) > 1e-6 and int(self._cam_dist * 5) != int(before * 5):
            print(f"[camera] zoom dist = {self._cam_dist:.1f} m")
        return self._cam_dist

    def take_reset(self):
        r = self._reset
        self._reset = False
        return r


def _robot_cfg():
    cfg = G1_INSPIRE_CFG.replace(prim_path="/World/Robot")
    cfg.spawn.articulation_props.fix_root_link = False
    cfg.spawn.articulation_props.enabled_self_collisions = False
    cfg.init_state = cfg.init_state.replace(
        pos=(0.0, 0.0, 0.80),
        joint_pos={".*_hip_pitch_joint": -0.1, ".*_knee_joint": 0.3, ".*_ankle_pitch_joint": -0.2},
    )
    cfg.actuators = {
        "legs": ImplicitActuatorCfg(joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"],
                                    stiffness=_leg_gain([150.0, 200.0, 40.0]), damping=_leg_gain([2.0, 4.0, 2.0]),
                                    effort_limit_sim=300.0),
        "waist": ImplicitActuatorCfg(joint_names_expr=["waist_.*_joint"], stiffness=250.0, damping=5.0,
                                     effort_limit_sim=200.0),
        "arms": ImplicitActuatorCfg(joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
                                    stiffness=350.0, damping=15.0, effort_limit_sim=88.0),
        "hands": ImplicitActuatorCfg(joint_names_expr=["[LR]_.*_joint"], stiffness=20.0, damping=1.0,
                                     effort_limit_sim=2.0),
    }
    return cfg


def main() -> None:
    if not args_cli.policy_dir or not os.path.isdir(args_cli.policy_dir):
        raise SystemExit(
            "GR00T policy folder not found. Pass --policy_dir <dir> or set $GR00T_POLICY_DIR to the "
            "gr00t_wbc folder holding GR00T-WholeBodyControl-{Balance,Walk}.onnx (external dependency)."
        )
    sim = SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device))
    sim.set_camera_view(eye=(2.2, 2.2, 1.8), target=(1.0, 0.0, 0.8))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    # Room dressing, matching the original Insert scene (DressedSceneCfg): HDRI sky dome + the
    # Simple Room backdrop (collision off so the ground keeps owning the floor physics).
    dome = sim_utils.DomeLightCfg(texture_file=SKY_HDRI, texture_format="latlong", intensity=1000.0)
    dome.func("/World/DomeLight", dome)
    room = sim_utils.UsdFileCfg(usd_path=ROOM_USD,
                                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
    room.func("/World/Room", room)
    # bench: kinematic table + kinematic socket + dynamic grabbable bulb
    upright = _quat_x_deg(90.0)
    socket = bulb = None
    if not args_cli.no_bench:
        table_cfg = sim_utils.UsdFileCfg(usd_path=TABLE_USD, scale=TABLE_SCALE,
                                         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))
        table_cfg.func("/World/Table", table_cfg, translation=TABLE_POS)
        socket = RigidObject(RigidObjectCfg(
            prim_path="/World/Socket",
            spawn=sim_utils.UsdFileCfg(usd_path=OMNI_SOCKET_USD, func=_spawn_usd_as_rigid_body,
                                       scale=(0.007, 0.007, 0.007),
                                       rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=SOCKET_POS, rot=upright)))
        bulb = RigidObject(RigidObjectCfg(
            prim_path="/World/Bulb",
            spawn=sim_utils.UsdFileCfg(usd_path=OMNI_BULB_USD, func=_spawn_omni_rigid, scale=(0.007, 0.007, 0.007),
                                       rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False,
                                                                                    solver_position_iteration_count=16),
                                       mass_props=sim_utils.MassPropertiesCfg(mass=0.10)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=BULB_POS, rot=upright)))
    robot = Articulation(_robot_cfg())
    sim.reset()

    prov = ["CPUExecutionProvider"]
    balance = ort.InferenceSession(f"{args_cli.policy_dir}/GR00T-WholeBodyControl-Balance.onnx", providers=prov)
    walk = ort.InferenceSession(f"{args_cli.policy_dir}/GR00T-WholeBodyControl-Walk.onnx", providers=prov)
    in_name = balance.get_inputs()[0].name

    dev = robot.device
    obs_idx = torch.tensor(OBS_MAP, device=dev)
    act_idx = torch.tensor(ACT_MAP, device=dev)
    arm_ids = [robot.joint_names.index(j) for j in G1_ARM_JOINTS]
    hand_ids = [robot.joint_names.index(j) for j in G1_HAND_JOINTS]
    arm_ids_t = torch.tensor(arm_ids, device=dev)
    hand_ids_t = torch.tensor(hand_ids, device=dev)
    left_names = list(LEFT_ARM_UP.keys())
    left_ids_t = torch.tensor([robot.joint_names.index(n) for n in left_names], device=dev)
    left_up = torch.tensor([LEFT_ARM_UP[n] for n in left_names], device=dev)
    hand_open = torch.tensor([G1_HAND_OPEN[j] for j in G1_HAND_JOINTS], device=dev)
    hand_grasp = torch.tensor([G1_HAND_GRASP[j] for j in G1_HAND_JOINTS], device=dev)
    ee_idx = robot.body_names.index(G1_EE_BODY)
    arm_jac_cols = torch.tensor([j + 6 for j in arm_ids], device=dev)  # +6: floating-base root DOFs

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1, device=dev)

    obs_hist = collections.deque(maxlen=HIST)
    action = np.zeros(15, dtype=np.float32)
    default_full = robot.data.default_joint_pos.clone()
    # Arm joint limits, to clamp the IK output: differential IK ignores limits, so an out-of-range
    # target makes the DLS solver jump near the limit/singularity and the stiff arm slams -- which
    # shoves the balancing robot. Clamping keeps the arm inside its range and the whole body stable.
    arm_lim = robot.data.joint_pos_limits[0, arm_ids_t]
    arm_lo, arm_hi = arm_lim[:, 0], arm_lim[:, 1]

    def ee_in_base():
        p, q = subtract_frame_transforms(robot.data.root_pos_w[:1], robot.data.root_quat_w[:1],
                                         robot.data.body_pos_w[:1, ee_idx], robot.data.body_quat_w[:1, ee_idx])
        return p[0], q[0]

    tgt_pos, tgt_quat = ee_in_base()

    def build_obs(cmd):
        q = robot.data.joint_pos[0, obs_idx].cpu().numpy()
        dq = robot.data.joint_vel[0, obs_idx].cpu().numpy()
        omega = robot.data.root_ang_vel_b[0].cpu().numpy()
        grav = robot.data.projected_gravity_b[0].cpu().numpy()
        s = np.zeros(SINGLE_OBS, dtype=np.float32)
        s[0:3] = cmd * CMD_SCALE
        s[3:4] = HEIGHT_CMD
        s[7:10] = omega * ANG_VEL_SCALE
        s[10:13] = grav
        s[13:42] = q - DEFAULT_29
        s[42:71] = dq * DOF_VEL_SCALE
        s[71:86] = action
        return s

    kbd = None if args_cli.test else _ComboKeyboard(args_cli.fwd_speed, args_cli.turn_speed)
    print("[insert-walk] started. Arrows walk/turn; W/S A/D Q/E hand, Z/X T/G C/V rotate; K grip; "
          "[ / ] zoom; B camera (follow/chest/free); R reset.")

    i = 0
    log = []
    while simulation_app.is_running():
        # --- command sources -----------------------------------------------------------------
        if args_cli.test:
            t = i * SIM_DT
            base_cmd = np.array([0.35, 0, 0], np.float32) if 1.0 < t < 3.5 else np.zeros(3, np.float32)
            dpos = np.array([0.004, 0, 0.002], np.float32) if t > 4.0 else np.zeros(3, np.float32)
            drot = np.zeros(3, np.float32)
            grip_open = True
            do_reset = False
        else:
            base_cmd = kbd.base_cmd()
            dpos, drot = kbd.hand_dpos(), kbd.hand_drot()
            grip_open = kbd.grip_open()
            do_reset = kbd.take_reset()

        if do_reset:
            robot.write_root_pose_to_sim(torch.tensor([[0, 0, 0.80, 1, 0, 0, 0]], device=dev, dtype=torch.float32))
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
            robot.write_joint_state_to_sim(default_full, torch.zeros_like(default_full))
            if bulb is not None:
                bulb.write_root_pose_to_sim(torch.tensor([[*BULB_POS, *upright]], device=dev, dtype=torch.float32))
            obs_hist.clear()
            tgt_pos, tgt_quat = ee_in_base()

        # --- walk policy: legs + waist (every DECIMATION steps) -------------------------------
        if i % DECIMATION == 0:
            obs_hist.append(build_obs(base_cmd))
            while len(obs_hist) < HIST:
                obs_hist.appendleft(np.zeros(SINGLE_OBS, dtype=np.float32))
            buf = np.concatenate(list(obs_hist)).reshape(1, SINGLE_OBS * HIST)
            pol = balance if np.linalg.norm(base_cmd) < 0.05 else walk
            action = pol.run(None, {in_name: buf})[0].squeeze().astype(np.float32)

        # --- arm IK: integrate hand deltas into an absolute base-frame EE target --------------
        cur_pos, _ = ee_in_base()
        tgt_pos = tgt_pos + torch.tensor(dpos, device=dev)
        off = tgt_pos - cur_pos
        dist = torch.norm(off)
        if dist > LEASH:
            tgt_pos = cur_pos + off * (LEASH / dist)
        ang = float(np.linalg.norm(drot))
        if ang > 1e-6:
            axis = torch.tensor(drot / ang, device=dev).unsqueeze(0)
            dq = quat_from_angle_axis(torch.tensor([ang], device=dev), axis)
            tgt_quat = normalize(quat_mul(dq, tgt_quat.unsqueeze(0)))[0]
        ik.set_command(torch.cat([tgt_pos, tgt_quat]).unsqueeze(0))
        jac = robot.root_physx_view.get_jacobians()[:, ee_idx, :, :][:, :, arm_jac_cols]
        ee_p, ee_q = ee_in_base()
        arm_des = ik.compute(ee_p.unsqueeze(0), ee_q.unsqueeze(0), jac, robot.data.joint_pos[:, arm_ids_t])
        arm_des = torch.clamp(arm_des, arm_lo, arm_hi)  # respect joint limits (IK ignores them)

        # --- assemble full joint target ------------------------------------------------------
        target = default_full.clone()
        target[0, act_idx] = torch.tensor(action * ACTION_SCALE + DEFAULT_ANGLES, device=dev)
        target[0, left_ids_t] = left_up  # keep the (un-teleoped) left arm raised, clear of the table
        if not args_cli.hold_arm:
            target[0, arm_ids_t] = arm_des[0]
        target[0, hand_ids_t] = hand_open if grip_open else hand_grasp
        robot.set_joint_position_target(target)

        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)
        if bulb is not None:
            socket.update(SIM_DT)
            bulb.update(SIM_DT)
        # Camera modes (toggle with B): 0=follow (3rd person), 1=robot-eye (1st person), 2=free.
        # In free mode we never touch the camera, so mouse zoom/orbit works normally.
        if kbd is not None:
            mode = kbd.cam_mode()
            if mode == 0 and i % 3 == 0:  # follow behind-and-above; [ / ] zoom via cam_dist
                b = robot.data.root_pos_w[0].cpu().numpy()
                d = kbd.cam_dist()
                sim.set_camera_view(eye=(b[0] - 0.8 * d, b[1] - 0.55 * d, b[2] + 0.5 * d),
                                    target=(b[0] + 0.5, b[1], b[2] + 0.1))
            elif mode == 1 and i % 3 == 0:  # over-shoulder view: above + behind the body, angled
                rp, rq = robot.data.root_pos_w[:1], robot.data.root_quat_w[:1]  # down at the arm/bench
                eye = (rp + quat_apply(rq, torch.tensor([[-0.25, 0.0, 0.75]], device=dev)))[0]
                tgt = (rp + quat_apply(rq, torch.tensor([[0.85, 0.0, -0.10]], device=dev)))[0]
                sim.set_camera_view(eye=tuple(eye.cpu().numpy()), target=tuple(tgt.cpu().numpy()))
        i += 1

        if args_cli.test and i % 40 == 0:
            bp = robot.data.root_pos_w[0]
            ep = robot.data.body_pos_w[0, ee_idx]
            log.append(f"t={i*SIM_DT:4.2f} base=({bp[0]:.2f},{bp[1]:.2f},{bp[2]:.2f}) "
                       f"{'UP' if bp[2] > 0.5 else 'FALLEN'} ee=({ep[0]:.2f},{ep[1]:.2f},{ep[2]:.2f})")
        if args_cli.test and i * SIM_DT > 6.0:
            break

    if args_cli.test:
        print("[insert-walk test]\n" + "\n".join(log))


if __name__ == "__main__":
    main()
    simulation_app.close()
