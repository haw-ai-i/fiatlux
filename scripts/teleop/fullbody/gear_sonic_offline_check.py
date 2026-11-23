"""Fast offline sanity check for the GEAR-SONIC observation conventions (no Isaac needed).

Idea: pretend the robot is tracking the clip PERFECTLY -- its joint positions/velocities are
the clip's, its base is upright, gravity is (0,0,-1) in the base frame. Feed that through
encoder+decoder and compare the commanded joint targets against the clip's own next pose.

If our conventions are right, the policy should command something close to the reference
(small, structured residuals). A large or systematically offset command means an encoder /
reference convention is still wrong -- which is far cheaper to discover here (seconds) than in
a full Isaac rollout (minutes).

    python gear_sonic_offline_check.py ~/gear_sonic_clips/<clip>.npz
"""

from __future__ import annotations

import sys

import numpy as np
from gear_sonic_clip import MotionClip
from gear_sonic_contract import ACTION_SCALE, DEFAULT_ANGLES, HISTORY_FRAMES, JOINT_ORDER
from gear_sonic_policy import GearSonicPolicy


class _FakeData:
    pass


class _FakeRobot:
    """Minimal stand-in for an Isaac articulation: perfect tracking of the clip."""

    joint_names = list(JOINT_ORDER)

    def __init__(self):
        self.data = _FakeData()

    def set(self, q, dq):
        import torch

        self.data.joint_pos = torch.as_tensor(q, dtype=torch.float32).unsqueeze(0)
        self.data.joint_vel = torch.as_tensor(dq, dtype=torch.float32).unsqueeze(0)
        self.data.root_ang_vel_b = torch.zeros(1, 3)
        self.data.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]])


def main(path: str, steps: int = 120) -> None:
    clip = MotionClip(path)
    robot = _FakeRobot()
    pol = GearSonicPolicy(robot, device="cpu")
    upright = np.array([1.0, 0.0, 0.0, 0.0])
    scale = np.asarray(ACTION_SCALE, dtype=np.float32)
    default = np.asarray(DEFAULT_ANGLES, dtype=np.float32)

    print(f"clip: {clip.T} frames @ {clip.fps} Hz")
    print(f"{'step':>5} {'|cmd-ref|':>10} {'max|err|':>9} {'worst joint':>26} {'act range':>16}")
    for k in range(steps):
        robot.set(clip.dof[k], clip.dof_vel[k])
        enc = clip.encoder_obs(k, upright)
        tgt = pol.step(robot, enc)
        ref_next = clip.dof[min(k + 1, clip.T - 1)]
        err = tgt - ref_next
        if k % 20 == 0 or k == steps - 1:
            j = int(np.argmax(np.abs(err)))
            print(
                f"{k:5d} {np.linalg.norm(err):10.3f} {np.abs(err).max():9.3f} "
                f"{JOINT_ORDER[j]:>26} [{pol.last_action.min():+.2f},{pol.last_action.max():+.2f}]"
            )
    # a policy that agrees with the reference should also leave the DEFAULT pose alone when
    # the reference IS the default pose -- check that separately
    flat = np.tile(default, (HISTORY_FRAMES, 1))
    robot.set(default, np.zeros(29, dtype=np.float32))
    clip_default = clip
    enc0 = GearSonicPolicy.encoder_obs_clip(
        motion_q_hist=list(flat),
        motion_dq_hist=list(np.zeros_like(flat)),
        anchor_orn_hist=[np.array([1, 0, 0, 1, 0, 0], np.float32)] * HISTORY_FRAMES,
        anchor_orn=np.array([1, 0, 0, 1, 0, 0], np.float32),
        root_z=0.78,
        root_z_hist=[0.78] * HISTORY_FRAMES,
    )
    for _ in range(HISTORY_FRAMES):
        tgt0 = pol.step(robot, enc0)
    print(
        f"\nSTATIC TEST (reference == default stance): max|target-default| = "
        f"{np.abs(tgt0 - default).max():.3f} rad   (want small)"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/yujin-chen/gear_sonic_clips/walk_forward_amateur_001__A001_M.npz")
