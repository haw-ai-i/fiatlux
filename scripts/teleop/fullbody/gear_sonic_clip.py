"""Reference-motion source for GEAR-SONIC encoder mode 0 ("g1" clip imitation).

Feeds the encoder from one of NVIDIA's robot motion clips (`sample_data/robot_filtered/**.pkl`:
`dof` (T,29), `root_rot` (T,4) wxyz, `root_trans_offset` (T,3), `fps`).

Conventions verified in `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/`:
* the 10-frame reference windows look FORWARD: frames `t, t+5, t+10, ... t+45`
  (`target_frame = current_frame + frame_idx * step_size`), clamped at the clip end
* `motion_anchor_orientation` is a 6D rotation = first two COLUMNS of the rotation matrix,
  emitted row-major `[m00,m01, m10,m11, m20,m21]`, and it is RELATIVE:
  `quat_to_6d(inv(robot_heading) * reference_heading)`
* motion joint positions are taken raw from the clip
"""

from __future__ import annotations

import numpy as np

CONTROL_HZ = 50   # policy rate; clips are resampled to this before windowing
STEP = 5          # reference window stride, in CONTROL steps (0.1 s)
FRAMES = 10       # reference window length


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2], dtype=np.float64)


def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def heading_quat(q: np.ndarray) -> np.ndarray:
    """Yaw-only part of a wxyz quaternion (the 'heading')."""
    yaw = np.arctan2(2.0 * (q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


def quat_to_6d(q: np.ndarray) -> np.ndarray:
    """First two columns of the rotation matrix, row-major pairs (their layout)."""
    w, x, y, z = q
    m = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return np.array([m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[2, 0], m[2, 1]], dtype=np.float32)


class MotionClip:
    """Sample a clip and build encoder-mode-0 observations."""

    def __init__(self, path: str):
        """Accepts a .npz (preferred) or a joblib .pkl.

        The benchmark venv is kept byte-identical to `uv sync` (no joblib), so clips are
        pre-converted to npz with the vendor's own env:
        `.venv_sim/bin/python` + `joblib.load` -> `np.savez_compressed` (see
        `~/gear_sonic_clips/`).
        """
        if path.endswith(".npz"):
            clip = dict(np.load(path))
            clip["fps"] = int(clip["fps"])
        else:
            import joblib
            raw = joblib.load(path)
            clip = next(iter(raw.values())) if len(raw) == 1 and "dof" not in raw else raw
        self.dof = np.asarray(clip["dof"], dtype=np.float32)                 # (T, 29)
        self.root_rot = np.asarray(clip["root_rot"], dtype=np.float64)       # (T, 4)
        self.root_z = np.asarray(clip["root_trans_offset"], dtype=np.float32)[:, 2]
        self.fps = int(clip.get("fps", 30))
        # Resample to the CONTROL rate (50 Hz). Their pipeline holds motions at 50 Hz
        # (`planner_motion_50hz_`), so a "10 frames, step 5" window means 5 *control steps*
        # (=0.1 s) apart -- not 5 clip frames. Sampling a 30 fps clip directly stretches the
        # reference window by 1.67x and advances the reference at the wrong rate.
        if self.fps != CONTROL_HZ:
            t_src = np.arange(len(self.dof)) / float(self.fps)
            t_dst = np.arange(0.0, t_src[-1], 1.0 / CONTROL_HZ)
            self.dof = np.stack([np.interp(t_dst, t_src, self.dof[:, j])
                                 for j in range(self.dof.shape[1])], axis=1).astype(np.float32)
            self.root_z = np.interp(t_dst, t_src, self.root_z).astype(np.float32)
            rr = np.stack([np.interp(t_dst, t_src, self.root_rot[:, c]) for c in range(4)], axis=1)
            self.root_rot = rr / np.linalg.norm(rr, axis=1, keepdims=True)
            self.fps = CONTROL_HZ
        self.T = len(self.dof)
        # finite-difference joint velocities (clip rate)
        self.dof_vel = np.zeros_like(self.dof)
        self.dof_vel[1:] = (self.dof[1:] - self.dof[:-1]) * self.fps   # rad/s at control rate
        if self.root_rot.shape[1] == 4 and abs(np.linalg.norm(self.root_rot[0]) - 1) > 1e-3:
            self.root_rot /= np.linalg.norm(self.root_rot, axis=1, keepdims=True)

    def _idx(self, frame: int) -> list[int]:
        return [min(frame + k * STEP, self.T - 1) for k in range(FRAMES)]

    def encoder_obs(self, frame: int, robot_quat_wxyz: np.ndarray) -> np.ndarray:
        """Build the 1762-dim encoder input for this clip frame (mode 0)."""
        from gear_sonic_policy import GearSonicPolicy

        idx = self._idx(frame)
        robot_head_inv = quat_inv(heading_quat(np.asarray(robot_quat_wxyz, dtype=np.float64)))
        orn_hist = [quat_to_6d(quat_mul(robot_head_inv, heading_quat(self.root_rot[i]))) for i in idx]
        # wrists = the 3 wrist joints per arm (roll/pitch/yaw), 6 values -> 60 over 10 frames
        wr = [19, 20, 21, 26, 27, 28]
        legs = list(range(12))
        return GearSonicPolicy.encoder_obs_clip(
            wrists_hist=[self.dof[i][wr] for i in idx],
            lowerbody_q_hist=[self.dof[i][legs] for i in idx],
            lowerbody_dq_hist=[self.dof_vel[i][legs] for i in idx],
            motion_q_hist=[self.dof[i] for i in idx],
            motion_dq_hist=[self.dof_vel[i] for i in idx],
            anchor_orn_hist=orn_hist,
            anchor_orn=orn_hist[0],
            root_z=float(self.root_z[idx[0]]),
            root_z_hist=[self.root_z[i] for i in idx],
        )

    def frame_at(self, t_seconds: float) -> int:
        return int(min(t_seconds * self.fps, self.T - 1))
