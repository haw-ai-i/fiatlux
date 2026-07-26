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
