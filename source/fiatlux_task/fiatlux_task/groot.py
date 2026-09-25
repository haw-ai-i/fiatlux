# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T N1.7 whole-body baseline adapter.

Three policy specs, two in-process whole-body-controller decoders:

- ``groot`` -- THE baseline: NVIDIA's Isaac-GR00T PolicyServer serving the *base*
  N1.7-3B checkpoint with the ``REAL_G1`` embodiment predicts 40-step chunks of
  upper-body targets + ``navigate_command`` + ``base_height_command`` from two
  ego-view RGB frames, proprioception, and a language instruction; the *decoupled*
  GEAR whole-body controller (Balance/Walk ONNX pair, :class:`GearWbcDecoder`)
  turns the navigation/height/torso commands into 50 Hz leg+waist control while
  the VLA's arm joint targets pass through directly.
- ``wbc_stand`` -- the decoupled WBC holding zero commands (no VLA): the sim2sim
  gate for the ``groot`` baseline's lower body.
- ``sonic_stand`` -- the GEAR-SONIC controller holding its standing latent: the
  same gate for the SONIC stack. SONIC's VLA path (``UNITREE_G1_SONIC``) needs a
  *finetuned* checkpoint -- the base N1.7 release ships no SONIC head -- so only
  its no-VLA gate is wired here; :class:`SonicDecoder` is ready for when a
  finetuned checkpoint exists.

Standard-mode note: everything consumed here is sensor-realizable -- raw ego-view
camera frames (head-mounted, on ``d435_link``), IMU (base angular velocity / projected gravity), joint state,
and wrist poses obtainable by forward kinematics from proprioception. The
adapter reads them from the scene handles rather than the flattened ``policy``
observation group because the models need them raw, not normalized and
feature-extracted; no privileged state is touched.

Both decoder contracts are reverse-engineered from GR00T-WholeBodyControl
(``gear_sonic_deploy`` C++ / ``decoupled_wbc`` Python) and verified against the
released ONNX graphs. The controllers command the 29 URDF body joints; our G1's
extra Inspire finger joints are held at their defaults (the VLA's hand channels
speak Dex3, which this robot does not have).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# SONIC plant constants (mirrors gear_sonic_deploy policy_parameters.hpp)
# ---------------------------------------------------------------------------

# Motor armature constants and the derived critically-damped PD model.
_NATURAL_FREQ = 10.0 * 2.0 * np.pi  # 10 Hz
_DAMPING_RATIO = 2.0
# (armature, effort limit) per motor type.
_MOTORS = {
    "5020": (0.003609725, 25.0),
    "7520_14": (0.010177520, 88.0),
    "7520_22": (0.025101925, 139.0),
    "4010": (0.00425, 5.0),
}


def _kp(motor: str) -> float:
    return _MOTORS[motor][0] * _NATURAL_FREQ**2


def _kd(motor: str) -> float:
    return 2.0 * _DAMPING_RATIO * _MOTORS[motor][0] * _NATURAL_FREQ


def _scale(motor: str) -> float:
    return 0.25 * _MOTORS[motor][1] / _kp(motor)


# The 29 SONIC body joints in MuJoCo/URDF tree order:
# (name, motor type, gain multiplier, default standing angle). The gain
# multiplier covers the ankle/waist-roll/pitch rows that double kp/kd but keep
# the base-stiffness action scale.
_SONIC_JOINTS_MUJOCO = [
    ("left_hip_pitch_joint", "7520_22", 1.0, -0.312),
    ("left_hip_roll_joint", "7520_22", 1.0, 0.0),
    ("left_hip_yaw_joint", "7520_14", 1.0, 0.0),
    ("left_knee_joint", "7520_22", 1.0, 0.669),
    ("left_ankle_pitch_joint", "5020", 2.0, -0.363),
    ("left_ankle_roll_joint", "5020", 2.0, 0.0),
    ("right_hip_pitch_joint", "7520_22", 1.0, -0.312),
    ("right_hip_roll_joint", "7520_22", 1.0, 0.0),
    ("right_hip_yaw_joint", "7520_14", 1.0, 0.0),
    ("right_knee_joint", "7520_22", 1.0, 0.669),
    ("right_ankle_pitch_joint", "5020", 2.0, -0.363),
    ("right_ankle_roll_joint", "5020", 2.0, 0.0),
    ("waist_yaw_joint", "7520_14", 1.0, 0.0),
    ("waist_roll_joint", "5020", 2.0, 0.0),
    ("waist_pitch_joint", "5020", 2.0, 0.0),
    ("left_shoulder_pitch_joint", "5020", 1.0, 0.2),
    ("left_shoulder_roll_joint", "5020", 1.0, 0.2),
    ("left_shoulder_yaw_joint", "5020", 1.0, 0.0),
    ("left_elbow_joint", "5020", 1.0, 0.6),
    ("left_wrist_roll_joint", "5020", 1.0, 0.0),
    ("left_wrist_pitch_joint", "4010", 1.0, 0.0),
    ("left_wrist_yaw_joint", "4010", 1.0, 0.0),
    ("right_shoulder_pitch_joint", "5020", 1.0, 0.2),
    ("right_shoulder_roll_joint", "5020", 1.0, -0.2),
    ("right_shoulder_yaw_joint", "5020", 1.0, 0.0),
    ("right_elbow_joint", "5020", 1.0, 0.6),
    ("right_wrist_roll_joint", "5020", 1.0, 0.0),
    ("right_wrist_pitch_joint", "4010", 1.0, 0.0),
    ("right_wrist_yaw_joint", "4010", 1.0, 0.0),
]

# IsaacLab (breadth-first) order expressed as MuJoCo indices -- the order the
# policy's observation and action vectors use (``mujoco_to_isaaclab`` in the
# deploy code).
_ISAAC_ORDER_MUJOCO_IDX = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]  # fmt: skip
assert sorted(_ISAAC_ORDER_MUJOCO_IDX) == list(range(29))

SONIC_JOINT_NAMES = [_SONIC_JOINTS_MUJOCO[i][0] for i in _ISAAC_ORDER_MUJOCO_IDX]
"""The 29 SONIC body joints in the policy's (IsaacLab) order."""

_CONTROL_DT = 0.02  # s; the GEAR controllers' training-time control rate (50 Hz)
_HISTORY_LEN = 10  # frames, oldest first, sampled at the 50 Hz control rate
_TOKEN_DIM = 64


def _require_control_rate(env, decoder: type) -> None:
    """The released GEAR ONNX controllers are 50 Hz policies: their observation
    histories, action scales, and (for GR00T) the frame-lookback/chunk timing all
    assume ``env.step_dt == 0.02``. Running off-rate does not crash -- it silently
    degrades the controller -- so the contract is enforced, not assumed."""
    assert abs(env.step_dt - _CONTROL_DT) < 1e-6, (
        f"{decoder.__name__} is a {1.0 / _CONTROL_DT:.0f} Hz controller; this env steps at "
        f"{env.step_dt:.6f} s (sim.dt * decimation) -- fix the task cfg's control rate"
    )


SONIC_OBS_DIM = _TOKEN_DIM + _HISTORY_LEN * (3 + 29 + 29 + 29 + 3)  # = 994

DEFAULT_SONIC_ONNX = os.path.expanduser("~/tools/sonic_models/policy/release/model_decoder.onnx")

# 64-d motion token for a stable standing pose (gear_sonic
# ``LATENT_INITIAL_MOTION_TOKEN``; checkpoint-specific).
STAND_TOKEN = np.array(
    [
        -0.0625,  0.0000, -0.0625, -0.1250, -0.1875, -0.0625,  0.1875,
         0.2500,  0.1875, -0.1250,  0.0625, -0.0625, -0.2500, -0.2500,
        -0.3125, -0.0625,  0.0000, -0.0625, -0.1250, -0.1875,  0.0000,
        -0.2500,  0.0000, -0.2500, -0.0625,  0.0625,  0.1250, -0.1250,
         0.2500,  0.1875,  0.2500, -0.1250,  0.1250,  0.1875, -0.0625,
         0.0000, -0.1875, -0.1875,  0.2500,  0.0000,  0.0000, -0.1250,
         0.0625,  0.0000, -0.0625, -0.0625,  0.1875, -0.0625,  0.0000,
         0.0625,  0.1250,  0.0625,  0.1250,  0.0625,  0.1250,  0.0000,
         0.1250,  0.1875,  0.0000,  0.0000,  0.0625,  0.0625,  0.1875,
         0.0625,
    ],
    dtype=np.float32,
)  # fmt: skip


def _isaac_order(field: int) -> torch.Tensor:
    """A per-joint constant from ``_SONIC_JOINTS_MUJOCO`` reordered to IsaacLab order."""
    per_mujoco = []
    for name, motor, gain_mult, default in _SONIC_JOINTS_MUJOCO:
        per_mujoco.append(
            {
                0: default,
                1: _kp(motor) * gain_mult,
                2: _kd(motor) * gain_mult,
                3: _scale(motor),
            }[field]
        )
    return torch.tensor([per_mujoco[i] for i in _ISAAC_ORDER_MUJOCO_IDX], dtype=torch.float32)


def retarget_torque_to_position(
    q: torch.Tensor,
    dq: torch.Tensor,
    tau: torch.Tensor,
    kp_ours: torch.Tensor,
    kd_ours: torch.Tensor,
) -> torch.Tensor:
    """Solve for the position target ``q_t`` that reproduces intended torque ``tau``
    under our own PD gains: ``kp_ours*(q_t - q) - kd_ours*dq == tau``.

    Used to carry a released ONNX decoder's torque intent (computed with its own
    training-time gains) through to the env's actuator model without touching it.
    If ``kp_ours``/``kd_ours`` equal the decoder's own gains exactly, this reduces to
    ``q_t == q_des`` (the decoder's own position target) -- see
    ``tests/test_groot_adapter.py::test_retarget_reduces_to_q_des_when_gains_match``.
    """
    return q + (tau + kd_ours * dq) / kp_ours


class SonicDecoder:
    """In-process GEAR-SONIC whole-body controller (the released ONNX decoder).

    Consumes a 64-d motion token per env per 50 Hz step plus proprioceptive
    histories read from the scene's robot, and produces actions for the env's
    whole-body ``JointPositionAction`` term. SONIC's intended torques are
    retargeted through the benchmark's own PD gains
    (``q_t = q + (tau_sonic + kd_ours*q_dot) / kp_ours``), so the commanded
    torque matches SONIC's intent at each control step without touching the
    env's actuator model. Histories reset (backfilled with the current frame)
    for any env whose episode counter is at 0.
    """

    def __init__(self, env, onnx_path: str | None = None):
        import onnxruntime as ort

        _require_control_rate(env, type(self))
        onnx_path = onnx_path or DEFAULT_SONIC_ONNX
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(
                f"SONIC decoder ONNX not found at {onnx_path} -- download it with "
                "GR00T-WholeBodyControl's download_from_hf.py"
            )
        # The graph is a small MLP ([1, 994] -> [1, 29]); CPU keeps it clear of the
        # simulator's GPU memory.
        self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        in_meta = self._session.get_inputs()[0]
        assert in_meta.shape[-1] == SONIC_OBS_DIM, f"decoder expects {in_meta.shape}, adapter builds {SONIC_OBS_DIM}"
        self._input_name = in_meta.name

        self.robot = env.scene["robot"]
        device = env.device
        # SONIC joints resolved by name; index tensors address the articulation's
        # full joint vectors (the Inspire fingers sit between body joints in BFS
        # order, so positions are not contiguous).
        ids, names = self.robot.find_joints(SONIC_JOINT_NAMES, preserve_order=True)
        assert list(names) == SONIC_JOINT_NAMES, (
            f"robot is missing SONIC body joints: {set(SONIC_JOINT_NAMES) - set(names)}"
        )
        self.joint_ids = torch.tensor(ids, dtype=torch.long, device=device)

        self.sonic_default = _isaac_order(0).to(device)
        self._kp_s = _isaac_order(1).to(device)
        self._kd_s = _isaac_order(2).to(device)
        self._scale_s = _isaac_order(3).to(device)

        # The env's whole-body action term: all joints in articulation order.
        action_cfg = env.cfg.actions.joint_pos
        assert env.action_manager.total_action_dim == self.robot.num_joints, (
            "SONIC adapter assumes the whole-body JointPositionAction term (one slot per joint)"
        )
        self.action_scale = float(action_cfg.scale)

        n = env.num_envs
        self._his_ang = torch.zeros(n, _HISTORY_LEN, 3, device=device)
        self._his_qrel = torch.zeros(n, _HISTORY_LEN, 29, device=device)
        self._his_dq = torch.zeros(n, _HISTORY_LEN, 29, device=device)
        self._his_act = torch.zeros(n, _HISTORY_LEN, 29, device=device)
        self._his_grav = torch.zeros(n, _HISTORY_LEN, 3, device=device)

    def step(self, env, tokens: torch.Tensor, hold_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Decode ``tokens`` (num_envs, 64) into a whole-body env action tensor.

        ``hold_mask`` marks envs the startup blend still owns: their real motion
        keeps feeding the state histories, but their action-history frames stay
        zero and their returned actions are ignored by the caller.
        """
        data = self.robot.data
        q = data.joint_pos[:, self.joint_ids]
        dq = data.joint_vel[:, self.joint_ids]
        qrel = q - self.sonic_default
        ang = data.root_ang_vel_b
        grav = data.projected_gravity_b

        # Fresh episodes: backfill the whole history with the current frame and
        # clear stale actions (mirrors the deploy stack starting from live state).
        # While the startup blend owns an env (``hold_mask``), state histories keep
        # accumulating the real blending motion, but the action history stays zero
        # (the deploy control loop does not run the policy in POSE mode).
        fresh = env.episode_length_buf == 0
        if fresh.any():
            self._his_ang[fresh] = ang[fresh, None]
            self._his_qrel[fresh] = qrel[fresh, None]
            self._his_dq[fresh] = dq[fresh, None]
            self._his_act[fresh] = 0.0
            self._his_grav[fresh] = grav[fresh, None]

        for buf, frame in (
            (self._his_ang, ang),
            (self._his_qrel, qrel),
            (self._his_dq, dq),
            (self._his_grav, grav),
        ):
            buf[:] = buf.roll(-1, dims=1)
            buf[:, -1] = frame

        # Histories are oldest-first; the action history is used as-is (its newest
        # frame is the previous step's action, matching the deploy tick order:
        # log state -> gather obs -> infer -> record action).
        obs = torch.cat(
            [
                tokens.to(q.device, torch.float32),
                self._his_ang.flatten(1),
                self._his_qrel.flatten(1),
                self._his_dq.flatten(1),
                self._his_act.flatten(1),
                self._his_grav.flatten(1),
            ],
            dim=1,
        )
        assert obs.shape[1] == SONIC_OBS_DIM
        obs_np = obs.cpu().numpy()

        act = np.empty((env.num_envs, 29), dtype=np.float32)
        for i in range(env.num_envs):
            act[i] = self._session.run(None, {self._input_name: obs_np[i : i + 1]})[0][0]
        act_t = torch.from_numpy(act).to(q.device)
        if hold_mask is not None:
            act_t[hold_mask] = 0.0
        self._his_act[:] = self._his_act.roll(-1, dims=1)
        self._his_act[:, -1] = act_t

        # SONIC's intended torque, retargeted through the env's own PD gains.
        q_des = self.sonic_default + act_t * self._scale_s
        tau = self._kp_s * (q_des - q) - self._kd_s * dq
        kp_ours = data.joint_stiffness[:, self.joint_ids].clamp_min(1e-6)
        kd_ours = data.joint_damping[:, self.joint_ids]
        q_t = retarget_torque_to_position(q, dq, tau, kp_ours, kd_ours)
        limits = data.soft_joint_pos_limits[:, self.joint_ids]
        q_t = q_t.clamp(limits[..., 0], limits[..., 1])

        env_action = torch.zeros(env.num_envs, self.robot.num_joints, device=q.device)
        default_ours = data.default_joint_pos[:, self.joint_ids]
        env_action[:, self.joint_ids] = (q_t - default_ours) / self.action_scale
        return env_action


class _StartupBlend:
    """Drives the given joints from the spawn pose to the controller's standing
    pose at each episode start, then hands control to the decoder. The window
    must stay short: the spawn pose is not statically stable, so the active
    balancer has to take over before tilt accumulates (a 1 s hold tips the
    robot past recovery)."""

    def __init__(self, env, robot, joint_ids, target, action_scale, duration_s: float = 0.02):
        self._robot = robot
        self._joint_ids = joint_ids
        self._target = target
        self._action_scale = action_scale
        self.steps = max(1, int(duration_s / env.step_dt))
        self._q0 = torch.zeros(env.num_envs, len(joint_ids), device=env.device)

    def mask(self, env) -> torch.Tensor:
        """Envs still in their startup window (also captures the spawn pose at t=0)."""
        t = env.episode_length_buf
        fresh = t == 0
        if fresh.any():
            self._q0[fresh] = self._robot.data.joint_pos[fresh][:, self._joint_ids]
        return t < self.steps

    def override(self, env, decoder_action: torch.Tensor, blending: torch.Tensor) -> torch.Tensor:
        """Replace ``decoder_action`` with the pose blend for envs in ``blending``."""
        if not blending.any():
            return decoder_action
        alpha = (env.episode_length_buf.float() / self.steps).clamp(max=1.0)[:, None]
        target = self._q0 * (1.0 - alpha) + self._target * alpha
        default_ours = self._robot.data.default_joint_pos[:, self._joint_ids]
        blend_action = torch.zeros_like(decoder_action)
        blend_action[:, self._joint_ids] = (target - default_ours) / self._action_scale
        return torch.where(blending[:, None], blend_action, decoder_action)


def make_sonic_stand_policy(env, onnx_path: str | None = None):
    """SONIC holding its standing latent -- the no-VLA sim2sim gate policy."""
    decoder = SonicDecoder(env, onnx_path)
    blend = _StartupBlend(env, decoder.robot, decoder.joint_ids, decoder.sonic_default, decoder.action_scale)
    tokens = torch.from_numpy(STAND_TOKEN).expand(env.num_envs, -1)

    def policy(obs):
        blending = blend.mask(env)
        action = decoder.step(env, tokens, hold_mask=blending)
        return blend.override(env, action, blending)

    return policy


# ---------------------------------------------------------------------------
# Decoupled GEAR whole-body controller (lower body of the `groot` baseline)
# ---------------------------------------------------------------------------

# Mirrors decoupled_wbc's g1_gear_wbc.yaml + G1GearWbcPolicy: the WBC controls
# the first 15 URDF body joints (legs + waist) from a 6-frame history of an
# 86-d observation; Balance serves |nav cmd| < 0.05, Walk the rest.
DEFAULT_WBC_DIR = os.path.expanduser("~/tools/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/policy")
_WBC_BALANCE = "GR00T-WholeBodyControl-Balance.onnx"
_WBC_WALK = "GR00T-WholeBodyControl-Walk.onnx"
_WBC_LOWER_DEFAULTS = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0, 0.0, 0.0]
_WBC_ACTION_SCALE = 0.25
_WBC_CMD_SCALE = (2.0, 2.0, 0.5)
_WBC_ANG_VEL_SCALE = 0.5
_WBC_DOF_VEL_SCALE = 0.05
_WBC_HIST = 6
_WBC_OBS_DIM = 86  # 3 cmd + 1 height + 3 rpy + 3 ang vel + 3 gravity + 29 q + 29 dq + 15 act
WBC_HEIGHT_INIT = 0.74
# g1_gear_wbc.yaml's own kps/kds (legs x2 + waist, matching _WBC_LOWER_DEFAULTS' order):
# the WBC's training-time gains, used to compute its intended torque before that torque
# is retargeted through the env's own PD gains (retarget_torque_to_position). Pinned to
# the yaml by tests/test_groot_adapter.py::test_gear_wbc_gains_match_reference_yaml.
_WBC_KP_LOWER = [150.0, 150.0, 150.0, 200.0, 40.0, 40.0] * 2 + [250.0, 250.0, 250.0]
_WBC_KD_LOWER = [2.0, 2.0, 2.0, 4.0, 2.0, 2.0] * 2 + [5.0, 5.0, 5.0]

_BODY_JOINT_NAMES = [n for n, *_ in _SONIC_JOINTS_MUJOCO]  # 29, URDF order
_ARM_JOINT_NAMES = _BODY_JOINT_NAMES[15:29]  # left arm 7 + right arm 7
_WAIST_JOINT_NAMES = _BODY_JOINT_NAMES[12:15]


class GearWbcDecoder:
    """In-process decoupled GEAR WBC (the released Balance/Walk ONNX pair).

    Turns per-env navigation velocity / base height / torso rpy commands into
    50 Hz leg+waist joint control, reading proprioception from the scene's
    robot. Like :class:`SonicDecoder`, the WBC's intended torques are
    retargeted through the benchmark's own PD gains, so the env's actuator
    model stays untouched.
    """

    def __init__(self, env, model_dir: str | None = None):
        import onnxruntime as ort

        _require_control_rate(env, type(self))
        model_dir = model_dir or DEFAULT_WBC_DIR
        paths = [os.path.join(model_dir, name) for name in (_WBC_BALANCE, _WBC_WALK)]
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"decoupled WBC ONNX not found at {path} -- clone GR00T-WholeBodyControl and "
                    "`git lfs pull --include 'gr00t_wbc/sim2mujoco/resources/robots/g1/policy/*'`"
                )
        self._balance, self._walk = (ort.InferenceSession(path, providers=["CPUExecutionProvider"]) for path in paths)
        self._input_name = self._balance.get_inputs()[0].name

        self.robot = env.scene["robot"]
        device = env.device
        ids, names = self.robot.find_joints(_BODY_JOINT_NAMES, preserve_order=True)
        assert list(names) == _BODY_JOINT_NAMES, f"robot is missing body joints: {set(_BODY_JOINT_NAMES) - set(names)}"
        self.body_ids = torch.tensor(ids, dtype=torch.long, device=device)
        self.lower_ids = self.body_ids[:15]
        self.lower_defaults = torch.tensor(_WBC_LOWER_DEFAULTS, device=device)
        self._kp_wbc = torch.tensor(_WBC_KP_LOWER, device=device)
        self._kd_wbc = torch.tensor(_WBC_KD_LOWER, device=device)
        # The WBC's q observation subtracts defaults for the lower 15 only (the
        # reference pads with zeros for the arms).
        self._defaults29 = torch.zeros(29, device=device)
        self._defaults29[:15] = self.lower_defaults
        self._cmd_scale = torch.tensor(_WBC_CMD_SCALE, device=device)

        action_cfg = env.cfg.actions.joint_pos
        assert env.action_manager.total_action_dim == self.robot.num_joints, (
            "WBC adapter assumes the whole-body JointPositionAction term (one slot per joint)"
        )
        self.action_scale = float(action_cfg.scale)

        n = env.num_envs
        self._history = torch.zeros(n, _WBC_HIST, _WBC_OBS_DIM, device=device)
        self._last_action = torch.zeros(n, 15, device=device)

    def step(
        self,
        env,
        cmd: torch.Tensor,
        height_cmd: torch.Tensor,
        rpy_cmd: torch.Tensor,
    ) -> torch.Tensor:
        """One 50 Hz WBC tick -> env action tensor (leg+waist slots filled).

        Args:
            cmd: (num_envs, 3) navigation velocity command (vx, vy, yaw rate).
            height_cmd: (num_envs,) base height command.
            rpy_cmd: (num_envs, 3) torso orientation command.
        """
        data = self.robot.data
        q29 = data.joint_pos[:, self.body_ids]
        dq29 = data.joint_vel[:, self.body_ids]

        # Fresh episodes start from an all-zeros history (the reference deque
        # left-pads with zeros) and cleared actions.
        fresh = env.episode_length_buf == 0
        if fresh.any():
            self._history[fresh] = 0.0
            self._last_action[fresh] = 0.0

        frame = torch.cat(
            [
                cmd * self._cmd_scale,
                height_cmd[:, None],
                rpy_cmd,
                data.root_ang_vel_b * _WBC_ANG_VEL_SCALE,
                data.projected_gravity_b,
                q29 - self._defaults29,
                dq29 * _WBC_DOF_VEL_SCALE,
                self._last_action,
            ],
            dim=1,
        )
        assert frame.shape[1] == _WBC_OBS_DIM
        self._history[:] = self._history.roll(-1, dims=1)
        self._history[:, -1] = frame
        obs_np = self._history.flatten(1).cpu().numpy().astype(np.float32)

        act = np.empty((env.num_envs, 15), dtype=np.float32)
        standing = cmd.norm(dim=1) < 0.05
        for i in range(env.num_envs):
            session = self._balance if bool(standing[i]) else self._walk
            act[i] = session.run(None, {self._input_name: obs_np[i : i + 1]})[0][0]
        act_t = torch.from_numpy(act).to(q29.device)
        self._last_action[:] = act_t

        # The WBC's intended torque, retargeted through the env's own PD gains (same
        # derivation as SonicDecoder; see retarget_torque_to_position). q29/dq29's first
        # 15 entries are the lower body, matching lower_ids/lower_defaults/_WBC_KP_LOWER.
        q_lower, dq_lower = q29[:, :15], dq29[:, :15]
        q_des = self.lower_defaults + act_t * _WBC_ACTION_SCALE
        tau = self._kp_wbc * (q_des - q_lower) - self._kd_wbc * dq_lower
        kp_ours = data.joint_stiffness[:, self.lower_ids].clamp_min(1e-6)
        kd_ours = data.joint_damping[:, self.lower_ids]
        q_t = retarget_torque_to_position(q_lower, dq_lower, tau, kp_ours, kd_ours)
        limits = data.soft_joint_pos_limits[:, self.lower_ids]
        q_t = q_t.clamp(limits[..., 0], limits[..., 1])

        env_action = torch.zeros(env.num_envs, self.robot.num_joints, device=q29.device)
        default_ours = data.default_joint_pos[:, self.lower_ids]
        env_action[:, self.lower_ids] = (q_t - default_ours) / self.action_scale
        return env_action


def make_wbc_stand_policy(env, model_dir: str | None = None):
    """The decoupled WBC holding zero commands -- the ``groot`` baseline's stand gate."""
    wbc = GearWbcDecoder(env, model_dir)
    blend = _StartupBlend(env, wbc.robot, wbc.lower_ids, wbc.lower_defaults, wbc.action_scale)
    cmd = torch.zeros(env.num_envs, 3, device=env.device)
    height = torch.full((env.num_envs,), WBC_HEIGHT_INIT, device=env.device)
    rpy = torch.zeros(env.num_envs, 3, device=env.device)

    def policy(obs):
        blending = blend.mask(env)
        action = wbc.step(env, cmd, height, rpy)
        return blend.override(env, action, blending)

    return policy


# ---------------------------------------------------------------------------
# GR00T N1.7 PolicyServer client (the full baseline)
# ---------------------------------------------------------------------------

DEFAULT_GROOT_ENDPOINT = "localhost:5555"
DEFAULT_INSTRUCTION = (
    "Replace the light bulb: take the fresh bulb from the table, insert it into the "
    "light fixture, then put the old bulb in the yellow crate."
)

# The REAL_G1 embodiment: two ego frames 0.4 s apart in, 40-step action chunks
# out. A fresh chunk is fetched every 20 env steps (the sim clock stops while
# the request blocks, so no latency compensation is needed).
# Execute the full chunk before re-querying: the sim pauses during inference, so a
# shorter interval would always execute the chunk prefix and discard the rest -- and
# the model schedules stand-to-walk transitions in the second half of its 0.8 s plan.
_QUERY_INTERVAL = 40
_CHUNK_LEN = 40
_FRAME_LOOKBACK = 20

# REAL_G1's state.* dims, from the checkpoint's own processor_config.json
# (processor_kwargs.modality_configs["real_g1_relative_eef_relative_joints"]["state"]),
# not just this file's own joint slicing -- guards against a slicing change silently
# sending the wrong-shaped state (see tests/test_groot_adapter.py).
_STATE_KEY_DIMS = {
    "left_wrist_eef_9d": 9,
    "right_wrist_eef_9d": 9,
    "left_hand": 7,
    "right_hand": 7,
    "left_arm": 7,
    "right_arm": 7,
    "waist": 3,
}


class _Gr00tClient:
    """Minimal client for Isaac-GR00T's PolicyServer (ZMQ REQ/REP + msgpack_numpy)."""

    def __init__(self, endpoint: str, timeout_ms: int = 30000):
        import msgpack
        import msgpack_numpy
        import zmq

        self._packb = lambda data: msgpack.packb(data, default=msgpack_numpy.encode)
        self._unpackb = lambda data: msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(f"tcp://{endpoint}")

    def call(self, endpoint: str, data: dict | None = None):
        request: dict = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self._socket.send(self._packb(request))
        message = self._socket.recv()
        if message == b"ERROR":
            raise RuntimeError("GR00T server error (wrong policy server?)")
        response = self._unpackb(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def get_action(self, observation: dict) -> dict:
        action, _info = self.call("get_action", {"observation": observation, "options": None})
        return action


class GrootPolicy:
    """GR00T N1.7 base model (``REAL_G1`` embodiment) + decoupled whole-body control.

    Per chunk boundary (every ``_QUERY_INTERVAL`` env steps once the startup
    blend has handed over): sends two ego-view RGB frames (t-0.4 s and t), wrist
    poses (FK from proprioception, pelvis frame, xyz + rot6d), arm/waist joint
    state, and the language instruction; receives a 40-step chunk of arm joint
    targets, waist targets, ``navigate_command`` and ``base_height_command``.
    Every 50 Hz step the chunk's navigation/height/torso commands drive
    :class:`GearWbcDecoder` (legs + waist) while the arm targets apply
    directly. On the Dex3 robot (``--robot dex3``, the exact REAL_G1
    embodiment) hand state/actions pass through in GR00T's hand-channel
    order; on the Inspire robot the hand channels are zeros and the fingers
    hold their defaults.

    Exposes the optional ``info`` telemetry dict (``policy/`` namespace):
    server round-trip and the current chunk's command magnitudes.
    """

    def __init__(self, env, endpoint: str | None = None, instruction: str | None = None):
        assert env.num_envs == 1, "the GR00T baseline adapter is single-env (server API is single-robot)"
        self._env = env
        self._wbc = GearWbcDecoder(env)
        self._client = _Gr00tClient(endpoint or DEFAULT_GROOT_ENDPOINT)
        self._instruction = instruction or DEFAULT_INSTRUCTION
        self._camera = env.scene["ego_camera"]
        self.robot = env.scene["robot"]
        device = env.device

        arm_ids, arm_names = self.robot.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
        assert list(arm_names) == _ARM_JOINT_NAMES
        self._arm_ids = torch.tensor(arm_ids, dtype=torch.long, device=device)
        waist_ids, _ = self.robot.find_joints(_WAIST_JOINT_NAMES, preserve_order=True)
        self._waist_ids = torch.tensor(waist_ids, dtype=torch.long, device=device)
        wrist_ids, wrist_names = self.robot.find_bodies(
            ["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True
        )
        assert list(wrist_names) == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
        self._wrist_ids = wrist_ids

        from .robots.g1 import G1_DEX3_LEFT_HAND_JOINTS, G1_DEX3_RIGHT_HAND_JOINTS

        # REAL_G1's action head emits 7 DoF per hand, i.e. the Dex3. On any other variant
        # there is nothing to map those onto, so the server's hand output is discarded and
        # the fingers hold their default pose for the whole episode -- the robot cannot
        # grasp at all. Say so: silently running a manipulation benchmark with dead hands
        # reads as the policy failing the task.
        dex3_names = G1_DEX3_LEFT_HAND_JOINTS + G1_DEX3_RIGHT_HAND_JOINTS
        self._hand_ids: torch.Tensor | None = None
        if set(dex3_names) <= set(self.robot.joint_names):
            hand_ids, _ = self.robot.find_joints(dex3_names, preserve_order=True)
            self._hand_ids = torch.tensor(hand_ids, dtype=torch.long, device=device)
        else:
            print(
                "\n[GR00T] WARNING: this robot is not the Dex3 variant, so REAL_G1's 7-DoF-per-hand\n"
                "        action has no mapping. The server's hand output is DISCARDED and the fingers\n"
                "        hold their default pose -- grasping is impossible. Pass --robot dex3.\n",
                file=sys.stderr,
                flush=True,
            )

        self._blend = _StartupBlend(
            env, self._wbc.robot, self._wbc.lower_ids, self._wbc.lower_defaults, self._wbc.action_scale
        )
        self._frames: list[np.ndarray] = []  # rolling ego-view buffer, newest last
        self._chunk: dict[str, np.ndarray] | None = None
        self._chunk_step = 0
        self._steps_since_query = _QUERY_INTERVAL
        # Command state the WBC consumes every step (held between chunks).
        self._cmd = torch.zeros(env.num_envs, 3, device=device)
        self._height = torch.full((env.num_envs,), WBC_HEIGHT_INIT, device=device)
        self._rpy = torch.zeros(env.num_envs, 3, device=device)
        self.info: dict[str, float] = {}

        endpoint_str = endpoint or DEFAULT_GROOT_ENDPOINT
        try:
            self._client.call("ping")
        except Exception as exc:
            raise RuntimeError(
                f"no Isaac-GR00T PolicyServer reachable at {endpoint_str}; "
                "start it with scripts/groot/serve.sh"
            ) from exc

    def _capture_frame(self) -> None:
        frame = self._camera.data.output["rgb"][0]
        if frame.dtype != torch.uint8:
            frame = frame.clamp(0, 255).to(torch.uint8)
        self._frames.append(frame.cpu().numpy())
        if len(self._frames) > _FRAME_LOOKBACK + 1:
            self._frames.pop(0)

    def _wrist_eef_9d(self, wrist_id: int) -> np.ndarray:
        """Wrist pose in the pelvis frame: xyz + rot6d (first two rotation columns)."""
        from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul

        data = self.robot.data
        root_q = data.root_quat_w
        rel_pos = quat_apply(quat_inv(root_q), data.body_pos_w[:, wrist_id] - data.root_pos_w)
        rel_rot = matrix_from_quat(quat_mul(quat_inv(root_q), data.body_quat_w[:, wrist_id]))
        rot6d = rel_rot[:, :, :2].transpose(1, 2).flatten(1)  # columns, flattened
        return torch.cat([rel_pos, rot6d], dim=1)[0].cpu().numpy().astype(np.float32)

    def _observation(self) -> dict:
        # delta_indices [-20, 0]: the frame from 0.4 s ago and the current one.
        video = np.stack([self._frames[0], self._frames[-1]])[None]  # [B=1, T=2, H, W, 3]
        assert video.dtype == np.uint8 and video.shape[:2] == (1, 2), (
            f"REAL_G1 video.ego_view must be (1, 2, H, W, 3) uint8; got shape={video.shape} dtype={video.dtype}"
        )
        q = self.robot.data.joint_pos

        def joints(ids) -> np.ndarray:
            return q[:, ids][0].cpu().numpy().astype(np.float32)[None, None]

        if self._hand_ids is not None:
            left_hand = joints(self._hand_ids[:7])
            right_hand = joints(self._hand_ids[7:])
        else:
            # Dex3 hand-state channels; the Inspire robot sends neutral (open) hands.
            left_hand = np.zeros((1, 1, 7), dtype=np.float32)
            right_hand = np.zeros((1, 1, 7), dtype=np.float32)
        state = {
            "left_wrist_eef_9d": self._wrist_eef_9d(self._wrist_ids[0])[None, None],
            "right_wrist_eef_9d": self._wrist_eef_9d(self._wrist_ids[1])[None, None],
            "left_hand": left_hand,
            "right_hand": right_hand,
            "left_arm": joints(self._arm_ids[:7]),
            "right_arm": joints(self._arm_ids[7:]),
            "waist": joints(self._waist_ids),
        }
        for key, arr in state.items():
            expected = (1, 1, _STATE_KEY_DIMS[key])
            assert arr.shape == expected and arr.dtype == np.float32 and np.isfinite(arr).all(), (
                f"REAL_G1 state.{key} must be {expected} float32, finite; "
                f"got shape={arr.shape} dtype={arr.dtype} finite={np.isfinite(arr).all()}"
            )
        return {
            "video": {"ego_view": video},
            "state": state,
            "language": {"annotation.human.task_description": [[self._instruction]]},
        }

    def _query_server(self) -> None:
        import time

        t0 = time.perf_counter()
        action = self._client.get_action(self._observation())
        latency_ms = (time.perf_counter() - t0) * 1000.0

        def chunk(key: str, dim: int) -> np.ndarray:
            value = action.get(key, action.get(f"action.{key}"))
            assert value is not None, f"server response missing action key {key!r} (or action.{key!r})"
            arr = np.array(value, dtype=np.float32, copy=True).reshape(-1, dim)
            # .reshape(-1, dim) silently accepts any N whose total size factors into
            # (N, dim); pin N to _CHUNK_LEN too, or a truncated/duplicated response
            # would pass the reshape and only surface as a chunk-index-out-of-range
            # bug much later (or not at all, since __call__ clamps the index).
            assert arr.shape == (_CHUNK_LEN, dim), f"server action.{key} must be ({_CHUNK_LEN}, {dim}); got {arr.shape}"
            return arr

        self._chunk = {
            "left_arm": chunk("left_arm", 7),
            "right_arm": chunk("right_arm", 7),
            "waist": chunk("waist", 3),
            "navigate_command": chunk("navigate_command", 3),
            "base_height_command": chunk("base_height_command", 1),
        }
        if self._hand_ids is not None:
            self._chunk["left_hand"] = chunk("left_hand", 7)
            self._chunk["right_hand"] = chunk("right_hand", 7)
        self._chunk_step = 0
        self.info = {
            "server_latency_ms": latency_ms,
            "nav_cmd_norm": float(np.linalg.norm(self._chunk["navigate_command"][0])),
            "base_height_cmd": float(self._chunk["base_height_command"][0, 0]),
        }

    def __call__(self, obs) -> torch.Tensor:
        env = self._env
        if bool((env.episode_length_buf == 0).any()):
            self._frames.clear()
            self._chunk = None
            self._chunk_step = 0
            self._steps_since_query = _QUERY_INTERVAL
            self._cmd.zero_()
            self._height.fill_(WBC_HEIGHT_INIT)
            self._rpy.zero_()
        self._capture_frame()

        # The VLA only runs once the startup blend has handed over (its first
        # query then sees the settled standing pose, like the deploy workflow).
        blend_mask = self._blend.mask(env)
        blending = bool(blend_mask.any())
        device = env.device
        arm_action: torch.Tensor | None = None
        hand_action: torch.Tensor | None = None
        if not blending:
            if self._steps_since_query >= _QUERY_INTERVAL:
                self._steps_since_query = 0
                self._query_server()
            assert self._chunk is not None
            k = min(self._chunk_step, _CHUNK_LEN - 1)
            self._cmd[0] = torch.from_numpy(self._chunk["navigate_command"][k]).to(device)
            self._height[0] = float(self._chunk["base_height_command"][k, 0])
            # The VLA's waist joint targets stand in for the WBC's torso rpy
            # command (yaw/roll/pitch joints ~ torso rpy relative to the pelvis
            # for small angles; the reference pipeline computes this via FK).
            waist = self._chunk["waist"][k]
            self._rpy[0] = torch.tensor([waist[1], waist[2], waist[0]], device=device)
            arm_targets = torch.from_numpy(
                np.concatenate([self._chunk["left_arm"][k], self._chunk["right_arm"][k]])
            ).to(device)
            arm_action = (arm_targets - self.robot.data.default_joint_pos[0, self._arm_ids]) / self._wbc.action_scale
            if self._hand_ids is not None:
                hand_targets = torch.from_numpy(
                    np.concatenate([self._chunk["left_hand"][k], self._chunk["right_hand"][k]])
                ).to(device)
                hand_action = (
                    hand_targets - self.robot.data.default_joint_pos[0, self._hand_ids]
                ) / self._wbc.action_scale
            self._chunk_step += 1
            self._steps_since_query += 1

        action = self._wbc.step(env, self._cmd, self._height, self._rpy)
        if arm_action is not None:
            action[:, self._arm_ids] = arm_action
        if hand_action is not None:
            action[:, self._hand_ids] = hand_action
        return self._blend.override(env, action, blend_mask)


def make_groot_policy(env, endpoint: str | None = None, instruction: str | None = None):
    """The zero-shot GR00T N1.7 + decoupled-WBC baseline policy."""
    return GrootPolicy(env, endpoint=endpoint, instruction=instruction)
