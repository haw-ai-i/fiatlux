"""RobotTracking with the start of an episode randomised the way the fiatlux benchmark does it.

Select with ``task.command._target_=active_adaptation.envs.mdp.commands.hdmi.command_spawn.RobotTrackingSpawn``.

Why. LadderMan starts each training episode at a random phase of the clip with a small root
offset (+-0.02 m) that is safe on a tread. The benchmark always starts at the clip's FIRST frame,
on the floor, and re-poses the robot on every reset: root x and y by +-0.05 m and yaw by +-0.1 rad
about the spawn pose, each joint by +-0.05 rad (fiatlux ``SubtaskEventCfg``). A teacher that has
seldom started at frame 0, and never 5 cm off, meets both on every benchmark episode.

Two additions, both off unless configured:

  start_zero_prob   fraction of training episodes forced to start at step 0
  floor_steps       clip steps that are a walk-in on the floor; an episode starting inside them
  floor_pose_range  takes its root offset from this range instead of ``pose_range``
  reset_lift        metres added to the root's height at every reset (default 0)

Mid-climb starts keep LadderMan's own ranges: 5 cm off a 68 mm tread is not a start the
benchmark ever produces.
"""
from __future__ import annotations

import torch
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

from .command import RobotTracking


class RobotTrackingSpawn(RobotTracking):
    def __init__(self, env, *args, start_zero_prob: float = 0.0, floor_steps: int = 0,
                 floor_pose_range: dict | None = None, reset_lift: float = 0.0, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.reset_lift = float(reset_lift)
        self.start_zero_prob = float(start_zero_prob)
        self.floor_steps = int(floor_steps)
        rng = dict(floor_pose_range or {})
        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        self.floor_pose_range = torch.tensor(
            [tuple(rng[k]) if k in rng else tuple(self.pose_range[i].tolist()) for i, k in enumerate(keys)],
            device=self.device)
        print(f"[command] spawn randomisation: {self.start_zero_prob:.0%} of training episodes start at step 0; "
              f"starts inside the first {self.floor_steps} steps use root range {self.floor_pose_range.tolist()}")

    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        super()._sample_motions(env_ids)
        if self.env.training and self.start_zero_prob > 0 and not self.replay_motion:
            zero = torch.rand(len(env_ids), device=self.device) < self.start_zero_prob
            t = self.t[env_ids]
            t[zero] = 0
            self.t[env_ids] = t

    def sample_init(self, env_ids: torch.Tensor) -> None:
        # reset_lift: every reset places the root this much higher (training and evaluation), so the robot
        # settles onto its feet instead of starting in the surface it stands on. LadderMan adds +-0.1 rad to
        # every joint at a reset, in evaluation too; with the clip's soles 1 mm above the platform that puts a
        # sole centimetres inside it, and a start on the PLATFORM reads up to 28 kN on an ankle in the first
        # step (the floor start: 0.5 kN). The benchmark does the same thing: its on-ladder spawns are 28 mm high.
        if self.reset_lift > 0:
            self.env.scene.env_origins[env_ids, 2] += self.reset_lift
        try:
            self._sample_init_spawn(env_ids)
        finally:
            if self.reset_lift > 0:
                self.env.scene.env_origins[env_ids, 2] -= self.reset_lift

    def _sample_init_spawn(self, env_ids: torch.Tensor) -> None:
        super().sample_init(env_ids)
        if not self.env.training or self.floor_steps <= 0 or self.replay_motion:
            return
        on_floor = self.t[env_ids] < self.floor_steps
        if not bool(on_floor.any()):
            return
        ids = env_ids[on_floor]
        motion = self._motion_reset[on_floor]
        r = sample_uniform(self.floor_pose_range[:, 0], self.floor_pose_range[:, 1], (len(ids), 6), device=self.device)
        pos = motion.body_pos_w[:, self.root_body_idx_motion] + self.env.scene.env_origins[ids] + r[:, 0:3]
        quat = quat_mul(motion.body_quat_w[:, self.root_body_idx_motion], quat_from_euler_xyz(r[:, 3], r[:, 4], r[:, 5]))
        self.asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
