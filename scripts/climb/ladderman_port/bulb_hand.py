"""Reward and termination terms for a hand that CARRIES something (the bulb hand in S10 / S04).

Deployed to LadderMan as ``active_adaptation/envs/mdp/commands/hdmi/bulb_hand.py`` and registered by
one added import in ``active_adaptation/envs/mdp/commands/__init__.py``. Every term is opt-in: a
config that does not name them trains exactly as before.

Why. LadderMan's rewards look at pose only, and loosely for the arms: 0.1 m and 0.1 rad are free,
the fall-off is 1.0 m / 0.5 rad, and each term averages over six links or eight joints. A 24 deg
elbow error costs about 8 % of one term. Its trainer has no bulb. Measured in the benchmark (S10,
free ladder, Dex3) with the bulb in the hand, a teacher trained that way:

* snaps the arm to its own posture on step 1 (wrist 1.2 m/s) -- the bulb is thrown;
* lands the bulb wrist on tread 4 at 1.8 m/s where the clip has 0.8 -- 88 N on the hand in one
  step, ``fresh_bulb_struck``;
* leans 20-70 N on that wrist while stepping up -- ``bulb_crushed``.

The simulator already measures what is needed: ``contact_forces`` covers every wrist link, and the
link velocities are there. These terms read them. They are stand-ins, not the bulb: a force limit on
the wrist for "the glass would be crushed", a velocity jump of the hand for "the bulb was struck".

Rewards (add them to an existing group; a NEW group changes the critic's output size and a
checkpoint no longer loads):

    carry_hand_contact_force_l2   penalty on wrist contact force above a free amount (a light rest)
    carry_hand_accel_l2           penalty on wrist linear acceleration above a free amount
    carry_arm_joint_pos_tracking  the clip's arm joints, one arm, tight
    carry_hand_pos_tracking       the clip's wrist position (local), tight
    carry_hand_ori_tracking       the clip's wrist orientation (local), tight
    carry_hand_linvel_tracking    the clip's wrist velocity, tight

    tight_arm_joint_pos_tracking  both arms' joints, tight (any climb, bulb or not)
    tight_arm_pos_tracking        arm links' position (local), tight
    tight_arm_ori_tracking        arm links' orientation (local), tight

    tight_feet_pos_tracking       the two ankle links' WORLD position, tight (where the robot stands)
    tight_feet_ori_tracking       the two ankle links' WORLD orientation

Terminations:

    carry_hand_force_exceeded     wrist contact force over a limit (benchmark: 50 N on the glass)
    carry_hand_velocity_jump      wrist velocity changes by more than a limit in one control step
                                  (benchmark: 2.4 m/s on the bulb)

The four tracking terms are plain subclasses of LadderMan's own, only so that a config can name a
second, tighter instance without the ``name(class)`` key syntax, which a command-line override
cannot carry.
"""
from __future__ import annotations

from typing import List

import torch

from active_adaptation.envs.mdp.base import Reward, Termination

from .rewards import (
    joint_pos_tracking_product,
    keypoint_ori_tracking_product,
    keypoint_pos_tracking_product,
    keypoint_lin_vel_tracking_product,
    keypoint_ori_tracking_local_product,
    keypoint_pos_tracking_local_product,
)

DEFAULT_HAND_BODIES = "right_wrist_.*_link"
DEFAULT_HAND_LINK = "right_wrist_yaw_link"


def _hand_force(sensor, body_ids) -> torch.Tensor:
    """[N] hardest contact force on any of the hand's links over the sensor's history (the substeps of this step)."""
    f = sensor.data.net_forces_w_history[:, :, body_ids].norm(dim=-1)       # [N, history, links]
    return f.amax(dim=(1, 2))


class carry_hand_contact_force_l2(Reward):
    """-(max(0, F - free_force) / scale)^2, clipped. F is the hardest contact force on the hand's links."""

    def __init__(self, env, weight: float, body_names: str | List[str] = DEFAULT_HAND_BODIES,
                 free_force: float = 15.0, scale: float = 25.0, clip: float = 4.0, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.sensor = self.env.scene["contact_forces"]
        self.body_ids, names = self.sensor.find_bodies(body_names)
        self.free_force, self.scale, self.clip = free_force, scale, clip
        print(f"[bulb_hand] contact force over {free_force:.0f} N penalised on {names}")

    def compute(self) -> torch.Tensor:
        over = (_hand_force(self.sensor, self.body_ids) - self.free_force).clamp_min(0.0) / self.scale
        return -over.square().clamp_max(self.clip).unsqueeze(1)


class carry_hand_accel_l2(Reward):
    """-(max(0, |a| - free_accel) / scale)^2, clipped. a is the hand link's velocity change over one control step."""

    def __init__(self, env, weight: float, body_name: str = DEFAULT_HAND_LINK,
                 free_accel: float = 5.0, scale: float = 10.0, clip: float = 4.0, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.env.scene["robot"]
        self.body_id = self.asset.find_bodies(body_name)[0][0]
        self.free_accel, self.scale, self.clip = free_accel, scale, clip
        self.prev = torch.zeros(self.num_envs, 3, device=self.device)
        self.fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.accel = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self.fresh[env_ids] = True

    def update(self):
        v = self.asset.data.body_com_lin_vel_w[:, self.body_id]
        self.accel[:] = torch.where(self.fresh, torch.zeros_like(self.accel), (v - self.prev).norm(dim=-1) / self.env.step_dt)
        self.prev[:] = v
        self.fresh[:] = False

    def compute(self) -> torch.Tensor:
        over = (self.accel - self.free_accel).clamp_min(0.0) / self.scale
        return -over.square().clamp_max(self.clip).unsqueeze(1)


class carry_arm_joint_pos_tracking(joint_pos_tracking_product):
    pass


class carry_hand_pos_tracking(keypoint_pos_tracking_local_product):
    pass


class carry_hand_ori_tracking(keypoint_ori_tracking_local_product):
    pass


class carry_hand_linvel_tracking(keypoint_lin_vel_tracking_product):
    pass


# ---- both arms, any climb -----------------------------------------------------------------------
# The benchmark robot has shoulder colliders the stock trainer robot lacks (make_fxbody_urdf.py), and
# the clip's arm angles are the ones checked clear of the torso. These hold the arms to the clip.
class tight_arm_joint_pos_tracking(joint_pos_tracking_product):
    pass


class tight_arm_pos_tracking(keypoint_pos_tracking_local_product):
    pass


class tight_arm_ori_tracking(keypoint_ori_tracking_local_product):
    pass


# ---- the feet, in the WORLD ----------------------------------------------------------------------
# LadderMan's leg terms are local (relative to the pelvis) or average over several links with a 0.5 m
# fall-off, so a robot that ends the climb standing 75 mm short of the clip's stance loses almost nothing.
# In the benchmark that is the one condition S02 still fails on every seed: both heels over the platform's
# open edge (49-114 mm, `soles_within_ladder_top`). These hold the two ankle links to the clip's WORLD pose.
class tight_feet_pos_tracking(keypoint_pos_tracking_product):
    pass


class tight_feet_ori_tracking(keypoint_ori_tracking_product):
    pass


class carry_hand_force_exceeded(Termination):
    """True once the hardest contact force on the hand's links has been over ``limit`` for ``min_steps`` control steps.

    The first ``grace_steps`` steps of an episode are exempt. Most episodes start at a random phase of the
    clip, many of them with the wrist already on a tread, and the settling of such a start is not a crush:
    without the grace the mean episode at 4,096 environments was 4.7 steps (30 without the rule).
    """

    def __init__(self, env, body_names: str | List[str] = DEFAULT_HAND_BODIES, limit: float = 50.0, min_steps: int = 1,
                 grace_steps: int = 25):
        super().__init__(env)
        self.sensor = self.env.scene["contact_forces"]
        self.body_ids, _ = self.sensor.find_bodies(body_names)
        self.limit, self.min_steps, self.grace_steps = limit, min_steps, grace_steps
        self.count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.age = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def reset(self, env_ids):
        self.count[env_ids] = 0
        self.age[env_ids] = 0

    def update(self):
        self.age += 1
        over = (_hand_force(self.sensor, self.body_ids) > self.limit) & (self.age > self.grace_steps)
        self.count[over] += 1
        self.count[~over] = 0

    def __call__(self) -> torch.Tensor:
        return (self.count >= self.min_steps).unsqueeze(-1)


class carry_hand_velocity_jump(Termination):
    """True where the hand link's velocity changed by more than ``limit`` (m/s) in one control step.

    The first ``grace_steps`` steps of an episode are exempt: a random-phase start writes the clip's
    pose and velocity onto the robot with noise, and its settling is not an impact (LadderMan's own
    terminations wait 25-60 steps for the same reason).
    """

    def __init__(self, env, body_name: str = DEFAULT_HAND_LINK, limit: float = 2.4, grace_steps: int = 25):
        super().__init__(env)
        self.asset = self.env.scene["robot"]
        self.body_id = self.asset.find_bodies(body_name)[0][0]
        self.limit, self.grace_steps = limit, grace_steps
        self.prev = torch.zeros(self.num_envs, 3, device=self.device)
        self.age = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.jump = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids):
        self.age[env_ids] = 0

    def update(self):
        v = self.asset.data.body_com_lin_vel_w[:, self.body_id]
        self.jump[:] = (v - self.prev).norm(dim=-1)
        self.prev[:] = v
        self.age += 1

    def __call__(self) -> torch.Tensor:
        return ((self.jump > self.limit) & (self.age > self.grace_steps)).unsqueeze(-1)
