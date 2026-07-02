# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video capture for Fiatlux environments (one RTX sensor camera + one MP4 writer).

A single implementation serves both recording purposes in the benchmark:

- **scene inspection** -- an orbiting turntable video (:func:`record_orbit`; used by
  ``scripts/verify_scene.py --record``), and
- **episode documentation** -- frames captured inside an existing rollout loop, so the video
  and the trajectory bag describe the *same* run (:class:`VideoRecorder`; used by
  ``scripts/record_run.py``).

The camera is a real RTX sensor (``CameraCfg``) attached to the scene *before* env
construction (``env_cfg.scene.video_cam = make_video_camera_cfg()``). That makes capture
work identically for the non-RL ``ManagerBasedEnv`` scaffolds and the RL envs, allows any
per-frame pose (orbit or fixed), and requires camera rendering (``--enable_cameras``).
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg


def make_video_camera_cfg(width: int = 1280, height: int = 720) -> CameraCfg:
    """A pinhole RGB video camera to attach to a scene; aimed per frame by the recorder."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/video_cam",
        update_period=0.0,  # refresh every render
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.05, 1.0e4)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, -4.0, 2.8), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )


def orbit_pose(
    i: int,
    n_frames: int,
    *,
    center: tuple[float, float, float],
    radius: float,
    height: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(eye, lookat) for frame ``i`` of an ``n_frames``-frame 360-degree turntable orbit."""
    theta = 2.0 * math.pi * i / max(n_frames, 1)
    eye = (center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta), height)
    return eye, center


class VideoRecorder:
    """Buffers RGB frames from a scene camera; writes an MP4 plus a mid-video poster PNG.

    The caller owns the env stepping loop: pose the camera (:meth:`set_pose` or the ``pose``
    argument), :meth:`capture` once per rendered step, then :meth:`write`. Frame 0 of env 0
    is recorded; all env cameras are posed together.
    """

    def __init__(self, env, camera, out_path: str, fps: int = 30):
        self._env = env
        self._cam = camera
        self._out_path = out_path
        self._fps = fps
        self._frames: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._frames)

    def set_pose(self, eye: tuple[float, float, float], lookat: tuple[float, float, float]) -> None:
        """Aim every env's camera from ``eye`` toward ``lookat`` (world frame)."""
        n_cam, device = self._env.num_envs, self._env.device
        eye_t = torch.tensor(eye, dtype=torch.float32, device=device).expand(n_cam, 3)
        look_t = torch.tensor(lookat, dtype=torch.float32, device=device).expand(n_cam, 3)
        self._cam.set_world_poses_from_view(eye_t, look_t)

    def capture(self, pose: tuple | None = None) -> None:
        """Grab env 0's current RGB frame; optionally :meth:`set_pose` first."""
        if pose is not None:
            self.set_pose(*pose)
        rgb = self._cam.data.output["rgb"][0, ..., :3]  # (H, W, 3) uint8 on device
        self._frames.append(rgb.detach().cpu().numpy().astype(np.uint8))

    def write(self) -> str:
        """Encode the buffered frames (libx264) and drop a poster PNG alongside."""
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(self._out_path) or ".", exist_ok=True)
        imageio.mimwrite(self._out_path, self._frames, fps=self._fps, codec="libx264", quality=8)
        # also drop a poster PNG: PNGs preview inline in most editors, MP4s do not
        poster = self._out_path.rsplit(".", 1)[0] + "_poster.png"
        imageio.imwrite(poster, self._frames[len(self._frames) // 2])
        print(f"[viz] poster frame : {poster}")
        return self._out_path


def record_orbit(
    env,
    camera,
    actions,
    *,
    n_steps: int,
    fps: int,
    out_path: str,
    center: tuple[float, float, float],
    radius: float,
    height: float,
) -> str:
    """Step ``env`` under constant ``actions`` while orbiting the camera 360 degrees.

    A turntable orbit makes a (policy-less, near-static) scene watchable and shows it in 3D.
    Writes the MP4 (plus poster) to ``out_path`` and returns the path.
    """
    rec = VideoRecorder(env, camera, out_path, fps=fps)
    for i in range(n_steps):
        rec.set_pose(*orbit_pose(i, n_steps, center=center, radius=radius, height=height))
        env.step(actions)
        rec.capture()
    return rec.write()
