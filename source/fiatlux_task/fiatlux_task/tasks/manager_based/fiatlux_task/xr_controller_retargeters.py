# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-controller retargeters for XR teleoperation (issue #51).

Isaac Lab's stock SE(3)/gripper retargeters read **hand tracking** (thumb/index/wrist joints). The
Meta Quest / Pico CloudXR *web client*, however, streams **motion controllers**, not optical hand
joints -- with the controllers down, hand tracking falls back to a useless head-locked placeholder
(``'Right Head Device Hand'``). These retargeters mirror :class:`Se3AbsRetargeter` /
:class:`GripperRetargeter` but read the controller instead: the right controller's grip pose drives
the absolute end-effector target, and its analog trigger drives the binary grip.

They declare :attr:`RetargeterBase.Requirement.MOTION_CONTROLLER`, which makes
:class:`~isaaclab.devices.OpenXRDevice` populate ``data[TrackingTarget.CONTROLLER_RIGHT]`` -- a 2x7
array where **row 0** is the pose ``[x, y, z, w, x, y, z]`` and **row 1** is the inputs
``[thumbstick_x, thumbstick_y, trigger, squeeze, button_0, button_1, pad]``.

Use via the ``controller`` teleop device (see ``insert_teleop_env_cfg.py``):
``scripts/xr_teleop.py --teleop_device controller``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg

# Layout of ``data[CONTROLLER_*]`` (see module docstring / OpenXRDevice._query_controller).
_CTRL_ELEMS = 14  # 2 rows x 7
_TRIGGER_IDX = 2  # index of the analog trigger within the inputs row


def _controller_target(bound_hand: DeviceBase.TrackingTarget) -> DeviceBase.TrackingTarget:
    """Map a HAND_LEFT/RIGHT target to the matching CONTROLLER_LEFT/RIGHT target."""
    if bound_hand == DeviceBase.TrackingTarget.HAND_LEFT:
        return DeviceBase.TrackingTarget.CONTROLLER_LEFT
    return DeviceBase.TrackingTarget.CONTROLLER_RIGHT


class Se3AbsControllerRetargeter(RetargeterBase):
    """Map a motion controller's grip pose to an absolute end-effector command (pos + quaternion).

    Controller counterpart of :class:`Se3AbsRetargeter`. Returns a 7D tensor
    ``[x, y, z, qw, qx, qy, qz]`` matching a ``DifferentialInverseKinematicsActionCfg`` in
    ``command_type="pose", use_relative_mode=False``.
    """

    def __init__(self, cfg: Se3AbsControllerRetargeterCfg):
        super().__init__(cfg)
        self._target = _controller_target(cfg.bound_hand)
        self._zero_out_xy_rotation = cfg.zero_out_xy_rotation
        # Cache the last valid command so a dropped controller frame holds pose instead of snapping.
        self._last: np.ndarray | None = None

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def retarget(self, data: dict) -> torch.Tensor:
        arr = data.get(self._target)
        if arr is None or np.size(arr) < _CTRL_ELEMS:
            # No controller this frame -> hold last (or a neutral pose in front of the base at first).
            if self._last is None:
                self._last = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return torch.tensor(self._last, dtype=torch.float32, device=self._sim_device)

        arr = np.asarray(arr, dtype=np.float32).reshape(2, 7)
        position = arr[0, :3]
        quat_wxyz = arr[0, 3:7]  # [w, x, y, z]

        if self._zero_out_xy_rotation:
            # Keep only yaw (Z), like Se3AbsRetargeter's stabilized mode. scipy wants [x, y, z, w].
            rot = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            yaw, _, _ = rot.as_euler("ZYX")
            q = Rotation.from_euler("Z", yaw).as_quat()  # [x, y, z, w]
            quat_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)

        cmd = np.concatenate([position, quat_wxyz]).astype(np.float32)
        self._last = cmd
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)


@dataclass
class Se3AbsControllerRetargeterCfg(RetargeterCfg):
    """Configuration for :class:`Se3AbsControllerRetargeter`."""

    bound_hand: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.HAND_RIGHT
    zero_out_xy_rotation: bool = True
    retargeter_type: type[RetargeterBase] = Se3AbsControllerRetargeter


class ControllerGripperRetargeter(RetargeterBase):
    """Map a motion controller's analog trigger to a binary gripper command (-1 close / 1 open).

    Controller counterpart of :class:`GripperRetargeter`.
    """

    def __init__(self, cfg: ControllerGripperRetargeterCfg):
        super().__init__(cfg)
        self._target = _controller_target(cfg.bound_hand)
        self._threshold = cfg.trigger_threshold
        self._closed = False

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def retarget(self, data: dict) -> torch.Tensor:
        arr = data.get(self._target)
        if arr is not None and np.size(arr) >= _CTRL_ELEMS:
            trigger = float(np.asarray(arr, dtype=np.float32).reshape(2, 7)[1, _TRIGGER_IDX])
            self._closed = trigger > self._threshold
        gripper_value = -1.0 if self._closed else 1.0
        return torch.tensor([gripper_value], dtype=torch.float32, device=self._sim_device)


@dataclass
class ControllerGripperRetargeterCfg(RetargeterCfg):
    """Configuration for :class:`ControllerGripperRetargeter`."""

    bound_hand: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.HAND_RIGHT
    trigger_threshold: float = 0.5
    retargeter_type: type[RetargeterBase] = ControllerGripperRetargeter


class Se3RelControllerRetargeter(RetargeterBase):
    """Map a controller's **incremental** motion to the end-effector (relative / clutch-free).

    Unlike :class:`Se3AbsControllerRetargeter` (hand position *is* the EE target, which needs the
    operator's hand physically in the robot's workspace), this integrates the controller's
    frame-to-frame position delta into an accumulated absolute EE target, starting from
    ``initial_position``. So the operator can hold the controller anywhere and *nudge* the EE around
    -- far more forgiving. Output is still an absolute pose command matching the absolute IK action;
    orientation is held at ``initial_orientation`` (position control first; add rotation later). The
    accumulated target is clamped to ``[workspace_min, workspace_max]`` so it can't drift out of reach.
    """

    def __init__(self, cfg: Se3RelControllerRetargeterCfg):
        super().__init__(cfg)
        self._target = _controller_target(cfg.bound_hand)
        self._scale = cfg.position_scale
        self._init_pos = np.array(cfg.initial_position, dtype=np.float32)
        self._init_quat = np.array(cfg.initial_orientation, dtype=np.float32)
        self._lo = np.array(cfg.workspace_min, dtype=np.float32)
        self._hi = np.array(cfg.workspace_max, dtype=np.float32)
        self._max_step = cfg.max_step
        self._deadzone = cfg.deadzone
        self._min_valid_z = cfg.min_valid_z
        self._clutch_idx = cfg.clutch_button_idx
        self._clutch_threshold = cfg.clutch_threshold
        self._alpha = cfg.smoothing_alpha
        self._max_jump = cfg.max_jump
        self._max_reject = cfg.max_reject
        # World->root transform. The IK action interprets its command in the robot ROOT (pelvis) frame
        # (isaaclab task_space_actions._compute_frame_pose -> subtract_frame_transforms(root, ee)), but
        # controller poses / init / workspace here are all WORLD frame. The G1 base is effectively
        # static (torso drift < 1 mm), so bake the transform as a constant: pos_root = R^T (pos_w - t).
        # Without this the IK reads a world z~0.83 as a root z, demanding the hand ~0.79 m too high ->
        # unreachable -> the arm hunts upward and oscillates even for a dead-constant command.
        rq = cfg.root_quat  # (w, x, y, z)
        self._root_pos = np.array(cfg.root_pos, dtype=np.float32)
        self._root_R = Rotation.from_quat([rq[1], rq[2], rq[3], rq[0]])  # root (pelvis) orientation in world
        self._root_R_T = self._root_R.as_matrix().T.astype(np.float32)
        # Orientation tracking (clutch-gated): the controller's rotation drives the wrist. The IK command
        # orientation is in the ROOT frame but the controller quat is WORLD, so the incremental rotation is
        # conjugated into the root frame before being applied. Starts from the fixed root-frame rest quat
        # and "ratchets" like position (release clutch -> hold; re-grip -> continue from where it is).
        self._enable_rot = cfg.enable_rotation
        self._rot_deadzone = cfg.rot_deadzone
        self._rot_max_step = cfg.rot_max_step
        iq = cfg.initial_orientation  # (w, x, y, z)
        self._init_R = Rotation.from_quat([iq[1], iq[2], iq[3], iq[0]])
        self._quat_R = self._init_R  # current EE orientation command (root frame)
        self._prev_cq: Rotation | None = None
        self._pos = self._init_pos.copy()
        self._prev: np.ndarray | None = None
        self._smooth: np.ndarray | None = None
        self._reject = 0

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def reset(self) -> None:
        self._pos = self._init_pos.copy()
        self._prev = None
        self._smooth = None
        self._reject = 0
        self._quat_R = self._init_R
        self._prev_cq = None

    def retarget(self, data: dict) -> torch.Tensor:
        arr = data.get(self._target)
        _rawcur = None
        _moved = 0.0
        _jump = 0.0
        _clutched = False
        _rej = False
        if arr is not None and np.size(arr) >= _CTRL_ELEMS:
            a2 = np.asarray(arr, dtype=np.float32).reshape(2, 7)
            cur = a2[0, :3]
            _clutched = float(a2[1, self._clutch_idx]) >= self._clutch_threshold
            _rawcur = cur.round(3).tolist()
            if cur[2] < self._min_valid_z or not _clutched:
                # Don't accumulate when (a) the pose is untracked/frozen -- an untracked controller
                # reads the anchor origin (z~0) while a held one is ~1 m up -- or (b) the clutch button
                # is released. Either way HOLD the target and drop all references, so when tracking + a
                # held clutch resume, motion re-references cleanly instead of jumping/drifting. The
                # clutch is the "moves when I'm not moving it" fix: set the controller down (release
                # clutch) and the arm freezes no matter how much the CloudXR pose wanders.
                self._prev = None
                self._smooth = None
                self._reject = 0
                self._prev_cq = None
            else:
                # EMA-smooth the raw controller pose first: CloudXR controller tracking jitters/spikes
                # hard under motion (~0.9 m frame-to-frame swings observed), and following it raw makes
                # the arm whack around. Smoothing tames high-frequency noise; deltas come from the
                # smoothed signal.
                if self._smooth is None:
                    self._smooth = cur.copy()
                else:
                    self._smooth = self._alpha * cur + (1.0 - self._alpha) * self._smooth
                sm = self._smooth
                if self._prev is None:
                    self._prev = sm.copy()
                else:
                    raw = sm - self._prev
                    _jump = float(np.linalg.norm(raw))
                    if _jump > self._max_jump:
                        # Implausible one-frame teleport = tracking glitch. Reject it (hold target, keep
                        # the last good reference) so a spike can't fling the arm. If it persists for many
                        # frames the controller genuinely relocated -> re-reference to resume control.
                        self._reject += 1
                        _rej = True
                        if self._reject >= self._max_reject:
                            self._prev = sm.copy()
                            self._reject = 0
                    elif _jump >= self._deadzone:
                        # Deadzone: only deliberate motion past the threshold moves the arm.
                        self._reject = 0
                        delta = raw * self._scale
                        n = float(np.linalg.norm(delta))
                        if n > self._max_step:  # cap per-frame step
                            delta = delta * (self._max_step / n)
                        self._pos = np.clip(self._pos + delta, self._lo, self._hi)
                        _moved = float(np.linalg.norm(delta))
                        self._prev = sm.copy()
                    else:
                        self._reject = 0  # inside deadzone: hold, keep the reference
                # Orientation: track the controller's rotation (clutch-gated), applied in the root frame.
                if self._enable_rot:
                    cq = Rotation.from_quat([a2[0, 4], a2[0, 5], a2[0, 6], a2[0, 3]])  # ctrl quat wxyz->xyzw
                    if self._prev_cq is None:
                        self._prev_cq = cq
                    else:
                        dqw = cq * self._prev_cq.inv()  # world-frame incremental rotation since last frame
                        ang = float(dqw.magnitude())
                        if ang >= self._rot_deadzone:  # ignore tiny rotation (jitter); hold ref below it
                            if ang > self._rot_max_step:  # cap per-frame rotation -> reject glitch spikes
                                dqw = Rotation.from_rotvec(dqw.as_rotvec() * (self._rot_max_step / ang))
                            dqr = self._root_R.inv() * dqw * self._root_R  # express the delta in root frame
                            self._quat_R = dqr * self._quat_R  # rotate the EE orientation
                            self._prev_cq = cq  # re-reference only on deliberate rotation
        # Transform the accumulated WORLD-frame target into the ROOT frame the IK expects; the EE
        # orientation is tracked from the controller (self._quat_R), both already in the root frame.
        pos_root = self._root_R_T @ (self._pos - self._root_pos)
        oq = self._quat_R.as_quat()  # scipy returns [x, y, z, w]
        out_quat = np.array([oq[3], oq[0], oq[1], oq[2]], dtype=np.float32)  # -> [w, x, y, z] for the cmd
        # DEBUG (temporary): raw pos, clutch, jump, reject, step, target, wrist rotation from rest (deg).
        self._dbg = getattr(self, "_dbg", 0) + 1
        if self._dbg % 5 == 0:
            _rotdeg = float((self._quat_R * self._init_R.inv()).magnitude()) * 57.29578
            print(f"[REL] raw={_rawcur} clutch={int(_clutched)} jump={_jump:.3f} rej={int(_rej)} "
                  f"moved={_moved:.4f} tgt_w={self._pos.round(3).tolist()} rot={_rotdeg:.1f}deg", flush=True)
        cmd = np.concatenate([pos_root, out_quat]).astype(np.float32)
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)


@dataclass
class Se3RelControllerRetargeterCfg(RetargeterCfg):
    """Configuration for :class:`Se3RelControllerRetargeter` (workspace defaults suit the Insert task)."""

    bound_hand: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.HAND_RIGHT
    position_scale: float = 1.0  # controller pose is in metres; 1:1 maps hand motion to EE motion 1-for-1
    max_step: float = 0.08  # max EE target move per frame (m); rejects controller tracking spikes
    deadzone: float = 0.004  # ignore controller moves below this (m) so idle jitter can't drift the arm
    min_valid_z: float = 0.3  # reject poses below this height (m): an untracked controller reads the
    # anchor origin (z~0); a real held controller is ~1 m up. Guards against the frozen/origin pose.
    # Clutch (dead-man's switch): the arm only tracks while this button is held, so setting the
    # controller down freezes the arm regardless of tracking jitter. idx 3 = grip/squeeze (idx 2 =
    # trigger, reserved for grasp). Threshold on the analog value.
    clutch_button_idx: int = 3
    clutch_threshold: float = 0.5
    # Input conditioning for jittery CloudXR controller tracking:
    smoothing_alpha: float = 0.35  # EMA weight on the newest reading (lower = smoother, more lag; 1.0 = off)
    max_jump: float = 0.20  # per-frame smoothed jump (m) above which the reading is a glitch -> reject
    max_reject: int = 15  # consecutive rejects before assuming a real relocation and re-referencing
    # Wrist rotation: controller twist -> EE orientation (clutch-gated, in the root frame). Off = the
    # orientation stays locked to ``initial_orientation`` (position-only, the original stable behaviour).
    enable_rotation: bool = True
    rot_deadzone: float = 0.02  # rad (~1.1 deg) per frame; ignore smaller rotation (jitter)
    rot_max_step: float = 0.10  # rad (~5.7 deg) per-frame cap; rejects rotation glitch spikes
    # Robot ROOT (pelvis) pose in WORLD, probed from the env: base of the world->root transform applied
    # to the output. G1 base is static, so these are constants. quat is (w, x, y, z).
    root_pos: tuple[float, float, float] = (0.5, 0.7, 0.75)
    root_quat: tuple[float, float, float, float] = (0.7071, 0.0, 0.0, -0.7071)
    # G1 right-hand (right_wrist_yaw_link) REST pose. ``initial_position``/``workspace_*`` are WORLD
    # frame (the accumulator + clamp work in world, matching the controller poses). ``initial_orientation``
    # is the fixed EE orientation expressed in the ROOT frame (what the IK command wants), probed as the
    # right-hand rest orientation in root frame -> a feasible wrist so the IK settles instead of flailing.
    initial_position: tuple[float, float, float] = (0.351, 0.499, 0.833)
    initial_orientation: tuple[float, float, float, float] = (0.998, -0.006, 0.059, 0.001)  # w,x,y,z (root)
    workspace_min: tuple[float, float, float] = (0.15, 0.05, 0.78)  # spans rest + the tabletop props
    workspace_max: tuple[float, float, float] = (0.65, 0.60, 1.10)
    retargeter_type: type[RetargeterBase] = Se3RelControllerRetargeter
