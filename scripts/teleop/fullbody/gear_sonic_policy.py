"""GEAR-SONIC policy runner for an Isaac Lab articulation.

Builds the decoder's 994-dim observation from a live Isaac G1, runs the released
encoder+decoder ONNX, and converts the 29-dim output into joint position targets.

Semantics verified against NVIDIA's deployment source
(`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/`, 2026-08-24):

* history is **oldest-first**, laid out frame-major: `[frame0 all-joints][frame1 ...]`
* `body_joint_positions` = ``q - DEFAULT_ANGLES`` (offset from the crouch stance), per joint
* `body_joint_velocities` = raw ``dq``
* `gravity_dir` = world -Z rotated into the base frame (Isaac's ``projected_gravity_b``)
* `base_angular_velocity` = IMU gyro, i.e. base-frame angular velocity (``root_ang_vel_b``)
* `last_actions` = the RAW policy output of the previous step (before scaling)
* joint target = ``action * ACTION_SCALE + DEFAULT_ANGLES``

Ordering note: joints are matched to the articulation **by name** (`policy_to_robot_index`),
because our G1 carries 43 joints (29 body + 14 hand) while the vendor's positional index maps
assume a 29-DOF robot.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np

from gear_sonic_contract import (  # noqa: F401  (module lives beside this file)
    ACTION_DIM,
    ACTION_SCALE,
    DECODER_DIM,
    DECODER_OFFSETS,
    DEFAULT_ANGLES,
    ENCODER_DIM,
    ENCODER_MODES,
    ENCODER_OFFSETS,
    HISTORY_FRAMES,
    KD,
    KP,
    TOKEN_DIM,
    policy_to_robot_index,
)

_RELEASE = os.path.expanduser(
    "~/robotica_project/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release")


class GearSonicPolicy:
    """Drive an Isaac G1 with the released GEAR-SONIC encoder/decoder."""

    def __init__(self, robot, release_dir: str = _RELEASE, device: str = "cpu"):
        import onnxruntime as ort

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device.startswith("cuda") else ["CPUExecutionProvider"])
        self.enc = ort.InferenceSession(os.path.join(release_dir, "model_encoder.onnx"),
                                        providers=providers)
        self.dec = ort.InferenceSession(os.path.join(release_dir, "model_decoder.onnx"),
                                        providers=providers)
        self.enc_in = self.enc.get_inputs()[0].name
        self.dec_in = self.dec.get_inputs()[0].name

        self.idx = policy_to_robot_index(robot.joint_names)
        self.default = np.asarray(DEFAULT_ANGLES, dtype=np.float32)
        self.scale = np.asarray(ACTION_SCALE, dtype=np.float32)
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)

        n = HISTORY_FRAMES
        self._q = deque(maxlen=n)      # oldest .. newest
        self._dq = deque(maxlen=n)
        self._act = deque(maxlen=n)
        self._omega = deque(maxlen=n)
        self._grav = deque(maxlen=n)

    # -- state capture --------------------------------------------------------------
    def record(self, robot) -> None:
        """Append this control step's robot state. Call once per policy step."""
        d = robot.data
        q = d.joint_pos[0, self.idx].detach().cpu().numpy().astype(np.float32) - self.default
        dq = d.joint_vel[0, self.idx].detach().cpu().numpy().astype(np.float32)
        omega = d.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
        grav = d.projected_gravity_b[0].detach().cpu().numpy().astype(np.float32)
        self._q.append(q)
        self._dq.append(dq)
        self._omega.append(omega)
        self._grav.append(grav)
        self._act.append(self.last_action.copy())
        while len(self._q) < HISTORY_FRAMES:      # prime the buffers on the first steps
            self._q.appendleft(q); self._dq.appendleft(dq)
            self._omega.appendleft(omega); self._grav.appendleft(grav)
            self._act.appendleft(self.last_action.copy())

    # -- observation assembly -------------------------------------------------------
    def decoder_obs(self, token: np.ndarray) -> np.ndarray:
        obs = np.zeros(DECODER_DIM, dtype=np.float32)

        def put(name, frames):
            s, e = DECODER_OFFSETS[name]
            obs[s:e] = np.concatenate(list(frames)).astype(np.float32)

        s, e = DECODER_OFFSETS["token_state"]
        obs[s:e] = token.reshape(-1)[:TOKEN_DIM]
        put("his_base_angular_velocity_10frame_step1", self._omega)
        put("his_body_joint_positions_10frame_step1", self._q)
        put("his_body_joint_velocities_10frame_step1", self._dq)
        put("his_last_actions_10frame_step1", self._act)
        put("his_gravity_dir_10frame_step1", self._grav)
        return obs

    @staticmethod
    def encoder_obs_clip(motion_q_hist, motion_dq_hist, anchor_orn_hist,
                         anchor_orn, root_z, root_z_hist, wrists_hist=None,
                         lowerbody_q_hist=None, lowerbody_dq_hist=None) -> np.ndarray:
        """Encoder input for mode 0 ('g1'): imitate a G1 motion clip.

        All history arguments are oldest-first, 10 frames at step 5.
        Slots for the other modes stay zero -- ``encoder_mode_4`` selects this one.
        """
        obs = np.zeros(ENCODER_DIM, dtype=np.float32)

        def put(name, value):
            s, e = ENCODER_OFFSETS[name]
            v = np.asarray(value, dtype=np.float32).reshape(-1)
            obs[s:s + min(len(v), e - s)] = v[:e - s]

        # NOT a one-hot: their GatherEncoderMode writes the mode ID as a SCALAR in slot 0
        # and zero-fills the rest (fill_zeros_num=3). Writing 1.0 at index 0 would select
        # mode 1 (teleop) instead of mode 0 (g1) -- verified in g1_deploy_onnx_ref.cpp.
        mode = np.zeros(4, dtype=np.float32)
        mode[0] = float(ENCODER_MODES["g1"])
        put("encoder_mode_4", mode)
        put("motion_joint_positions_10frame_step5", np.concatenate(list(motion_q_hist)))
        put("motion_joint_velocities_10frame_step5", np.concatenate(list(motion_dq_hist)))
        put("motion_anchor_orientation_10frame_step5", np.concatenate(list(anchor_orn_hist)))
        put("motion_anchor_orientation", anchor_orn)
        put("motion_root_z_position", [root_z])
        put("motion_root_z_position_10frame_step5", root_z_hist)
        # Their runtime fills these every step from the motion, even outside their own modes;
        # leaving them zero reads as "wrists/lower body at the origin" and drags the arms
        # (observed: mode-0 tracking error dominated by elbow/shoulder joints).
        if wrists_hist is not None:
            put("motion_joint_positions_wrists_10frame_step1", np.concatenate(list(wrists_hist)))
        if lowerbody_q_hist is not None:
            put("motion_joint_positions_lowerbody_10frame_step5",
                np.concatenate(list(lowerbody_q_hist)))
        if lowerbody_dq_hist is not None:
            put("motion_joint_velocities_lowerbody_10frame_step5",
                np.concatenate(list(lowerbody_dq_hist)))
        return obs

    # -- one control step -----------------------------------------------------------
    def step(self, robot, encoder_obs: np.ndarray) -> np.ndarray:
        """record -> encode -> decode. Returns joint POSITION TARGETS (29, policy order)."""
        self.record(robot)
        token = self.enc.run(None, {self.enc_in: encoder_obs[None].astype(np.float32)})[0]
        obs = self.decoder_obs(token)
        action = self.dec.run(None, {self.dec_in: obs[None]})[0][0].astype(np.float32)
        self.last_action = action
        return action * self.scale + self.default

    # -- actuator alignment ---------------------------------------------------------
    def apply_policy_gains(self, robot) -> None:
        """Set the articulation's PD gains to the values the policy was trained with.

        Our benchmark G1 is much stiffer (knee 200 vs 99, ankle 40 vs 14, waist 250 vs 40);
        running a trained policy under different closed-loop dynamics is a prime suspect for
        instability/skating. Call once after the env is built.
        """
        import torch

        dev = robot.data.joint_pos.device
        ids = torch.as_tensor(self.idx, device=dev, dtype=torch.long)
        kp = torch.as_tensor(KP, device=dev, dtype=torch.float32).unsqueeze(0)
        kd = torch.as_tensor(KD, device=dev, dtype=torch.float32).unsqueeze(0)
        robot.write_joint_stiffness_to_sim(kp, joint_ids=ids)
        robot.write_joint_damping_to_sim(kd, joint_ids=ids)
