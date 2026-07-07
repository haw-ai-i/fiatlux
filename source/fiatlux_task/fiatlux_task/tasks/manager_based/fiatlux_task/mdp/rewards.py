# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward / success functions for the Fiatlux G1 tasks.

Bulb insertion (``FIATLUX-Insert-v0``) — the task signal is the pose error between
the grasped *bulb* and the *socket*:
- distance kernels (L2 / tanh / exponential) for coarse-to-fine reaching,
- an orientation-alignment kernel,
- a sparse "seated" bonus (also reused as the success termination),
- a contact-force penalty for compliant insertion,
plus generic smoothness / joint-limit penalties.

Ladder climb (``FIATLUX-Climb-v0``) — ascent terms:
- a progressive best-height reward (each centimetre of new height paid once),
- a limb-on-ladder contact fraction (filtered contact sensor),
- a whole-body CoM sway penalty,
- an at-the-top success predicate (also the success termination).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude

from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_OFFSET

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


# ---------------------------------------------------------------------------
# Bulb -> socket pose error helpers
# ---------------------------------------------------------------------------


def _seat_point_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World position of the lamp's socket seat (bulblampF metalink)."""
    socket: RigidObject = env.scene["socket"]
    offset = torch.tensor(SOCKET_SEAT_OFFSET, device=env.device).expand(env.num_envs, 3)
    return socket.data.root_pos_w + quat_apply(socket.data.root_quat_w, offset)


def _plug_point_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World position of the bulb's plug (bulblampM metalink)."""
    bulb: RigidObject = env.scene["bulb"]
    offset = torch.tensor(BULB_PLUG_OFFSET, device=env.device).expand(env.num_envs, 3)
    return bulb.data.root_pos_w + quat_apply(bulb.data.root_quat_w, offset)


def _bulb_socket_pos_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Euclidean distance (m) between the bulb's plug point and the lamp's socket seat.

    Both are the OmniGibson attachment metalinks (bulblampM / bulblampF), not the object
    origins: seating means the plug reaches the socket, and the two origins are offset by
    the plug geometry even when fully mated (issue #29)."""
    return torch.norm(_plug_point_w(env) - _seat_point_w(env), dim=1)


def _bulb_socket_ori_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Shortest-path angular distance (rad) between bulb and socket frames."""
    bulb: RigidObject = env.scene["bulb"]
    socket: RigidObject = env.scene["socket"]
    return quat_error_magnitude(bulb.data.root_quat_w, socket.data.root_quat_w)


def object_socket_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 distance penalty (use with a negative weight)."""
    return _bulb_socket_pos_error(env)


def object_socket_distance_tanh(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Dense reaching reward via ``1 - tanh(d / std)``."""
    return 1.0 - torch.tanh(_bulb_socket_pos_error(env) / std)


def object_socket_distance_exp(env: ManagerBasedRLEnv, sigma: float) -> torch.Tensor:
    """Sharp seating reward via a Gaussian kernel (strong only very close-in)."""
    d = _bulb_socket_pos_error(env)
    return torch.exp(-torch.square(d) / (sigma**2))


def object_socket_orientation_tanh(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Orientation-alignment reward via ``1 - tanh(angle / std)``."""
    return 1.0 - torch.tanh(_bulb_socket_ori_error(env) / std)


# ---------------------------------------------------------------------------
# Success / sparse bonus (also used as the success termination)
# ---------------------------------------------------------------------------


def bulb_seated(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.015,
    ori_threshold: float = 0.2,
) -> torch.Tensor:
    """True where the bulb is within position *and* orientation tolerance of the socket."""
    return (_bulb_socket_pos_error(env) < pos_threshold) & (
        _bulb_socket_ori_error(env) < ori_threshold
    )


def object_dropped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float,
) -> torch.Tensor:
    """True where the object's height has fallen below ``min_height`` (m)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < min_height


# ---------------------------------------------------------------------------
# Contact / compliance
# ---------------------------------------------------------------------------


def hand_contact_force_l2(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize squared net contact force on the grasping hand (compliance)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net = sensor.data.net_forces_w  # (N, B, 3)
    return torch.sum(torch.square(net), dim=(1, 2))


# ---------------------------------------------------------------------------
# Generic smoothness / safety penalties
# ---------------------------------------------------------------------------


def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations (L2 squared) for smoother motion."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joints that exceed their soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# ---------------------------------------------------------------------------
# Ladder ascent (FIATLUX-Climb-v0)
# ---------------------------------------------------------------------------


def _root_pos_env(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root position in the env-local frame (world minus per-env origin)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


class climb_height_progress(ManagerTermBase):
    """Progressive-ascent reward: the positive increment of the episode's best root height.

    Each centimetre of *new* height is paid exactly once (episode total = metres
    gained), so standing still earns nothing and oscillating/jumping cannot farm the
    term — unlike a distance-to-target kernel (constant pay for standing anywhere)
    or a z-velocity reward (pays launch transients). The per-env best-height buffer
    is re-seeded on reset from the post-randomization root state (reset events run
    before the reward-manager reset).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best_z = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        asset: Articulation = self._env.scene["robot"]
        z = asset.data.root_pos_w[ids, 2] - self._env.scene.env_origins[ids, 2]
        self._best_z[ids] = z

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        z = _root_pos_env(env, asset_cfg)[:, 2]
        gain = (z - self._best_z).clamp(min=0.0)
        self._best_z = torch.maximum(self._best_z, z)
        return gain


def ladder_contact_fraction(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Fraction of the sensor's bodies in contact with its filtered prim (the ladder).

    Reads the filtered ``force_matrix_w`` of one multi-body contact sensor (feet +
    palms vs the single-rigid-body ladder), so ground reaction and self-contact do
    not count — only genuine limb-on-ladder force above ``threshold`` (N).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.force_matrix_w.sum(dim=2)  # (N, B, M, 3) -> (N, B, 3)
    in_contact = force.norm(dim=-1) > threshold
    return in_contact.float().mean(dim=1)


class com_sway_l2(ManagerTermBase):
    """Squared horizontal velocity of the whole-body centre of mass (sway penalty).

    A controlled ascent moves the CoM mostly vertically, so this penalizes lunging
    and lateral wobble without fighting the sustained forward lean that climbing an
    A-frame requires (which a CoM-offset-from-support formulation would punish).
    Mass fractions are precomputed once; ``default_mass`` lives on the CPU.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene["robot"]
        masses = asset.data.default_mass.to(env.device)  # (N, B)
        self._mass_frac = (masses / masses.sum(dim=-1, keepdim=True)).unsqueeze(-1)

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        com_vel = (asset.data.body_com_lin_vel_w * self._mass_frac).sum(dim=1)  # (N, 3)
        return torch.sum(torch.square(com_vel[:, :2]), dim=1)


def fall_terminated(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True on the step the robot falls (root below ``minimum_height`` or tilt beyond
    ``limit_angle``), i.e. exactly the ``fell_*`` termination step — the env resets
    right after, so the penalty fires once per fall.

    Recomputes the same predicates as the built-in ``root_height_below_minimum`` /
    ``bad_orientation`` terminations instead of using ``mdp.is_terminated_term``:
    that helper reads ``TerminationManager.get_term``, whose backing buffer is the
    *sticky* which-term-ended-the-last-episode log (``compute()`` never clears rows
    for envs where no term fired), so a per-step reward built on it keeps paying the
    penalty every step after an env's first fall.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    below = asset.data.root_pos_w[:, 2] < minimum_height
    tilted = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle
    return (below | tilted).float()


def climbed_to_target(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    xy_center: tuple[float, float],
    xy_radius: float,
    max_speed: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True where the root has climbed above ``minimum_height`` at the ladder top.

    The horizontal gate (env-local xy within ``xy_radius`` of the upper steps) and
    the ``max_speed`` cap reject ballistic trajectories (a solver-kicked robot flying
    through the success region must not score). Also the ``success`` termination.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = _root_pos_env(env, asset_cfg)
    high = pos[:, 2] > minimum_height
    dx = pos[:, 0] - xy_center[0]
    dy = pos[:, 1] - xy_center[1]
    near = (dx.square() + dy.square()) < xy_radius**2
    calm = asset.data.root_lin_vel_w.norm(dim=-1) < max_speed
    return high & near & calm
