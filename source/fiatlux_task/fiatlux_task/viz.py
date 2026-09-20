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
    sweep_deg: float = 360.0,
    phase_deg: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(eye, lookat) for frame ``i`` of an ``n_frames``-frame turntable orbit.

    ``sweep_deg``/``phase_deg`` restrict it to an arc, for a subject that cannot be viewed
    from every side.
    """
    theta = math.radians(phase_deg + sweep_deg * i / max(n_frames, 1))
    eye = (center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta), height)
    return eye, center


# Orbit geometry for the fixture view: low enough to be a genuine upward look at a mount at
# 2.2 m (wall) or 2.37 m (ceiling), close enough that the fixture is more than a speck.
FIXTURE_VIEW_RADIUS = 2.2
FIXTURE_VIEW_HEIGHT = 1.5
FIXTURE_VIEW_MIN_RADIUS = 0.6  # closer than this and the fixture overflows the frame
FIXTURE_VIEW_WALL_CLEARANCE = 0.3  # keep the camera off the wall it would otherwise clip into


def _radius_inside(
    center: tuple[float, float, float],
    sweep_deg: float,
    phase_deg: float,
    bounds_min: tuple[float, float],
    bounds_max: tuple[float, float],
) -> float:
    """Largest orbit radius whose whole arc stays inside the room, capped at the nominal one.

    Derived per layout: a fixture may be sampled within one ladder zone of a wall, where the
    nominal radius does not fit. A camera outside the room returns flat grey, which looks
    exactly like a missing fixture.
    """
    lo = (bounds_min[0] + FIXTURE_VIEW_WALL_CLEARANCE, bounds_min[1] + FIXTURE_VIEW_WALL_CLEARANCE)
    hi = (bounds_max[0] - FIXTURE_VIEW_WALL_CLEARANCE, bounds_max[1] - FIXTURE_VIEW_WALL_CLEARANCE)
    best = FIXTURE_VIEW_RADIUS
    n = 72
    for k in range(n + 1):
        theta = math.radians(phase_deg + sweep_deg * k / n)
        for axis, comp in ((0, math.cos(theta)), (1, math.sin(theta))):
            if abs(comp) < 1e-9:  # travels parallel to this pair of walls; never crosses them
                continue
            edge = hi[axis] if comp > 0 else lo[axis]
            best = min(best, (edge - center[axis]) / comp)
    return max(best, FIXTURE_VIEW_MIN_RADIUS)


def fixture_orbit(env_cfg) -> dict:
    """Orbit kwargs (for :func:`orbit_pose` / :func:`record_orbit`) that look UP at the fixture.

    The only camera here that can see an overhead mount; every other one is aimed at the
    floor and the bench. Aimed at where the **cfg** says the fixture is, so empty air in the
    render means the fixture is not where the scene claims.

    A wall mount gets a 180-degree arc centred on the direction the socket opening faces --
    read off the fixture's own orientation, since the opening points into the room by
    construction and is the only side it can be filmed from. The radius is then clamped to
    keep the arc inside the room (:func:`_radius_inside`).

    Raises:
        ValueError: if the scene mounts no fixture at all, rather than silently orbiting the
            origin and producing a video that looks like a successful check.
    """
    for name in ("socket", "fixture"):
        entity = getattr(env_cfg.scene, name, None)
        if entity is None or getattr(entity, "init_state", None) is None:
            continue
        from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import ROOM_FLOOR_MAX, ROOM_FLOOR_MIN

        center = tuple(entity.init_state.pos)
        sweep_deg, phase_deg = 360.0, 0.0
        # the socket's opening is its local +Z; rotate it by the mount quaternion (w, x, y, z)
        w, x, y, z = entity.init_state.rot
        open_x, open_y = 2.0 * (x * z + w * y), 2.0 * (y * z - w * x)
        if math.hypot(open_x, open_y) > 0.5:  # points sideways -> wall mount
            sweep_deg = 180.0
            phase_deg = math.degrees(math.atan2(open_y, open_x)) - 90.0
        return {
            "center": center,
            "radius": _radius_inside(center, sweep_deg, phase_deg, ROOM_FLOOR_MIN, ROOM_FLOOR_MAX),
            "height": FIXTURE_VIEW_HEIGHT,
            "sweep_deg": sweep_deg,
            "phase_deg": phase_deg,
        }
    raise ValueError("the fixture view needs a 'socket' or 'fixture' scene entity; this scene has neither")


def _draw_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a few lines of text into the top-left of a frame, over a dark panel.

    Some state a recording needs to show has no visual signature at all. The bulb's retention
    wrench is the case in point: a seated bulb is held by a continuously applied force
    (``mdp.bulb_attachment``), not a visible mechanism, so a frame cannot tell a bulb held
    under that force apart from one merely resting in the same pose. The event is real, and
    the camera cannot show it. Printing the state machine's own numbers is honest where
    implying visible motion would not be.

    Falls back to the unannotated frame if PIL is missing, because a recording without a caption
    is still worth having.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:  # Pillow < 10.1 takes no size
        font = ImageFont.load_default()

    lines = text.splitlines()
    pad, line_h = 10, 24
    box_h = pad * 2 + line_h * len(lines)
    box_w = pad * 2 + max((int(draw.textlength(ln, font=font)) for ln in lines), default=0)
    draw.rectangle([(0, 0), (box_w, box_h)], fill=(0, 0, 0, 150))
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, fill=(255, 255, 255, 255), font=font)
    return np.asarray(image, dtype=np.uint8)


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

    def capture(self, pose: tuple | None = None, overlay: str | None = None) -> None:
        """Grab env 0's current RGB frame; optionally :meth:`set_pose` first.

        ``overlay`` burns text into the frame, for state a camera cannot show on its own.
        """
        if pose is not None:
            self.set_pose(*pose)
        rgb = self._cam.data.output["rgb"][0, ..., :3]  # (H, W, 3) uint8 on device
        frame = rgb.detach().cpu().numpy().astype(np.uint8)
        if overlay:
            frame = _draw_overlay(frame, overlay)
        self._frames.append(frame)

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
    sweep_deg: float = 360.0,
    phase_deg: float = 0.0,
) -> str:
    """Step ``env`` under constant ``actions`` while orbiting the camera around ``center``.

    A turntable orbit makes a (policy-less, near-static) scene watchable and shows it in 3D.
    ``sweep_deg``/``phase_deg`` restrict it to an arc (see :func:`orbit_pose`).
    Writes the MP4 (plus poster) to ``out_path`` and returns the path.
    """
    rec = VideoRecorder(env, camera, out_path, fps=fps)
    for i in range(n_steps):
        rec.set_pose(
            *orbit_pose(
                i, n_steps, center=center, radius=radius, height=height, sweep_deg=sweep_deg, phase_deg=phase_deg
            )
        )
        env.step(actions)
        rec.capture()
    return rec.write()
