# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward / success functions for the Fiatlux G1 tasks.

Bulb seating (used by ``FIATLUX-Replace-v0`` and the seat/insert subtasks) — the task signal
is the pose error between the grasped *bulb* and the *socket*:
- distance kernels (L2 / tanh / exponential) for coarse-to-fine reaching,
- an orientation-alignment kernel,
- a sparse "seated" bonus (also reused as the success termination),
- a contact-force penalty for compliant insertion,
plus generic smoothness / joint-limit penalties.

Ladder climb (used by the on-the-ladder subtasks and Replace, via
``subtask_tiers/balance.py``'s ``ClimbSubtaskCfg``) — ascent terms:
- a progressive best-height reward (each centimetre of new height paid once),
- a limb-on-ladder contact fraction (filtered contact sensor),
- a whole-body CoM sway penalty,
- an at-the-top success predicate (also the success termination).

Ladder descent (used by the same tier's ``DescendSubtaskCfg``) — the mirror image of the climb
terms: a progressive best-*lowest*-height reward (``descend_height_progress``) and an at-the-
bottom success predicate (``descended_to_target``); reuses climb's contact/sway terms
and fall gate unchanged.

Full replacement (``FIATLUX-Replace-v0``) — the scored full task:
- named distance channels (ladder top → fixture, fresh bulb → fixture, old bulb clearance
  from the fixture, old bulb → disposal crate),
- a generic normalized-progress term (``(d0 - d) / d0`` clamped to [0, 1], so randomized
  spawn distances cannot dominate the score),
- sparse completion predicates (ladder in range, old bulb removed / disposed, full success),
- a ladder-tipped predicate (penalty + termination for the dynamic ladder).

Old-bulb removal/disposal (used by the remove-old-bulb subtask and Replace) — the
``old_bulb_*`` functions take an ``asset_cfg`` (default Replace's ``old_bulb``) so a
single-bulb scene can point them at its own ``bulb`` entity.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_mul

from fiatlux_task.assets import (
    BULB_BODY_CENTRE_OFFSET,
    BULB_PLUG_AXIS,
    BULB_PLUG_OFFSET,
    G1_WORKING_SHOULDER_OFFSET,
    SOCKET_SEAT_AXIS,
    SOCKET_SEAT_OFFSET,
    STEP_LADDER_TOP_OFFSET,
)

from ..scene_cfg import LADDER_READY_MIN_BEARING, TOP_STANCE_PELVIS_OFFSET, TOP_STANCE_YAW_OFFSET_DEG
from .place_terms import old_bulb_in_bin

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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


def _plug_point_w(env: ManagerBasedRLEnv, entity: str = "fresh_bulb") -> torch.Tensor:
    """World position of the bulb's plug (bulblampM metalink).

    ``entity`` exists for the task-generic writers in ``recording.py``: Remove and Carry build
    no ``fresh_bulb``, so they pass ``old_bulb``. Every reward term keeps the default.
    """
    bulb: RigidObject = env.scene[entity]
    offset = torch.tensor(BULB_PLUG_OFFSET, device=env.device).expand(env.num_envs, 3)
    return bulb.data.root_pos_w + quat_apply(bulb.data.root_quat_w, offset)


def _bulb_socket_pos_error(env: ManagerBasedRLEnv, entity: str = "fresh_bulb") -> torch.Tensor:
    """Euclidean distance (m) between the bulb's plug point and the lamp's socket seat.

    Both are the OmniGibson attachment metalinks (bulblampM / bulblampF), not the object
    origins: seating means the plug reaches the socket, and the two origins are offset by
    the plug geometry even when fully mated (issue #29)."""
    return torch.norm(_plug_point_w(env, entity) - _seat_point_w(env), dim=1)


def _bulb_socket_axis_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Angle (rad) between the bulb's plug axis and the socket's seat axis.

    Unlike :func:`_bulb_socket_ori_error` this ignores rotation ABOUT the mating axis,
    which ``assets.BULB_PLUG_AXIS`` defines as free -- that rotation *is* the screwing
    motion. A full-frame comparison makes rotation about the mating axis read as
    misalignment, so it is not appropriate during the rotation stage (issue #54).
    """
    bulb: RigidObject = env.scene["fresh_bulb"]
    socket: RigidObject = env.scene["socket"]
    n = env.num_envs
    plug = torch.tensor(BULB_PLUG_AXIS, device=env.device).expand(n, 3)
    seat = torch.tensor(SOCKET_SEAT_AXIS, device=env.device).expand(n, 3)
    a = quat_apply(bulb.data.root_quat_w, plug)
    b = quat_apply(socket.data.root_quat_w, seat)
    cos = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-9)
    return torch.acos(cos.clamp(-1.0, 1.0))


def _bulb_socket_ori_error(env: ManagerBasedRLEnv, entity: str = "fresh_bulb") -> torch.Tensor:
    """Shortest-path angular distance (rad) between bulb and socket frames."""
    bulb: RigidObject = env.scene[entity]
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
    return (_bulb_socket_pos_error(env) < pos_threshold) & (_bulb_socket_ori_error(env) < ori_threshold)


def bulb_unseated(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.015,
    ori_threshold: float = 0.2,
) -> torch.Tensor:
    """The negation of :func:`bulb_seated`, for a gate that must END if the bulb comes loose.

    A subtask that starts with the bulb already installed measures keeping it there, so leaving
    the socket is a termination rather than an unmet success conjunct.
    """
    return ~bulb_seated(env, pos_threshold, ori_threshold)


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


def hand_contact_force_l2(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize squared contact force on the grasping hand (compliance).

    Force on the OBJECT, not the sensor's net force: the net also carries whatever scenery the
    arm rests against, which would charge the policy for standing near the bench.
    """
    from .observations import object_contact_forces

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.sum(torch.square(object_contact_forces(sensor)), dim=(1, 2))


# ---------------------------------------------------------------------------
# Generic smoothness / safety penalties
# ---------------------------------------------------------------------------


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations (L2 squared) for smoother motion."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joints that exceed their soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# ---------------------------------------------------------------------------
# Ladder ascent
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

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        z = _root_pos_env(env, asset_cfg)[:, 2]
        gain = (z - self._best_z).clamp(min=0.0)
        self._best_z = torch.maximum(self._best_z, z)
        return gain


class descend_height_progress(ManagerTermBase):
    """Progressive-descent reward: the positive increment of the episode's best (lowest)
    root height reached. Mirrors ``climb_height_progress`` exactly but pays height *lost*
    instead of gained, since Descend starts high and must come back down under control."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best_z = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        asset: Articulation = self._env.scene["robot"]
        z = asset.data.root_pos_w[ids, 2] - self._env.scene.env_origins[ids, 2]
        self._best_z[ids] = z

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        z = _root_pos_env(env, asset_cfg)[:, 2]
        drop = (self._best_z - z).clamp(min=0.0)
        self._best_z = torch.minimum(self._best_z, z)
        return drop


def descended_to_target(
    env: ManagerBasedRLEnv,
    maximum_height: float,
    xy_center: tuple[float, float],
    xy_radius: float,
    max_speed: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True where the root has come back down below ``maximum_height`` near the ladder base.

    Mirrors ``climbed_to_target``'s horizontal gate and speed cap (rejects a solver-kicked
    robot flying through the success region) with the height gate inverted. Also the
    ``success`` termination."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = _root_pos_env(env, asset_cfg)
    low = pos[:, 2] < maximum_height
    dx = pos[:, 0] - xy_center[0]
    dy = pos[:, 1] - xy_center[1]
    near = (dx.square() + dy.square()) < xy_radius**2
    calm = asset.data.root_lin_vel_w.norm(dim=-1) < max_speed
    return low & near & calm


def ladder_contact_fraction(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
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
    ladder requires (which a CoM-offset-from-support formulation would punish).
    Mass fractions are precomputed once; ``default_mass`` lives on the CPU.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene["robot"]
        masses = asset.data.default_mass.to(env.device)  # (N, B)
        self._mass_frac = (masses / masses.sum(dim=-1, keepdim=True)).unsqueeze(-1)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Full replacement task (FIATLUX-Replace-v0)
# ---------------------------------------------------------------------------
#
# Distance channels are module-level named functions (not lambdas/closures) so env cfgs can
# reference them in ``RewardTermCfg.params`` and stay serializable.


def _ladder_top_point_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World position of the step ladder's top platform (STEP_LADDER_TOP_OFFSET)."""
    ladder: RigidObject = env.scene["ladder"]
    offset = torch.tensor(STEP_LADDER_TOP_OFFSET, device=env.device).expand(env.num_envs, 3)
    return ladder.data.root_pos_w + quat_apply(ladder.data.root_quat_w, offset)


def _old_bulb_plug_point_w(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("old_bulb")
) -> torch.Tensor:
    """World position of the seated bulb's plug point.

    Every seated bulb is the ``old_bulb`` entity since #76 Step 1, so ``asset_cfg`` keeps its
    default everywhere. It stays a parameter only because the manager API passes one.
    """
    old_bulb: RigidObject = env.scene[asset_cfg.name]
    offset = torch.tensor(BULB_PLUG_OFFSET, device=env.device).expand(env.num_envs, 3)
    return old_bulb.data.root_pos_w + quat_apply(old_bulb.data.root_quat_w, offset)


def ladder_fixture_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance (m) from the ladder's top platform to the fixture's socket seat."""
    return torch.norm(_ladder_top_point_w(env) - _seat_point_w(env), dim=1)


def bulb_fixture_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance (m) from the fresh bulb's plug to the fixture's socket seat."""
    return _bulb_socket_pos_error(env)


def old_bulb_fixture_clearance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("old_bulb")
) -> torch.Tensor:
    """Distance (m) of the old bulb's plug from the fixture's socket seat (0 = still seated)."""
    return torch.norm(_old_bulb_plug_point_w(env, asset_cfg) - _seat_point_w(env), dim=1)


def old_bulb_disposal_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("old_bulb"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """Distance (m) from the old bulb to the disposal crate's origin."""
    old_bulb: RigidObject = env.scene[asset_cfg.name]
    crate: RigidObject = env.scene[bin_cfg.name]
    return torch.norm(old_bulb.data.root_pos_w - crate.data.root_pos_w, dim=1)


class distance_progress(ManagerTermBase):
    """Potential-based shaping on normalized distance progress: pays the *change* each step.

    Progress (the potential ``Phi``) is ``(d0 - d) / d0`` clamped to [0, 1], with ``d0`` the
    term's ``distance_fn`` captured at episode reset (event-manager reset terms run before
    the reward-manager reset, so this reads the post-randomization state). Normalizing by
    the *episode's own* start distance means a lucky spawn that starts close cannot outscore
    an unlucky far one -- both saturate at 1.0 on arrival -- the full-task plan's
    randomization-fairness requirement.

    With ``away_threshold`` set, the channel measures progress *away* from a point
    instead: ``d / away_threshold`` clamped to [0, 1]. The d0 normalization cannot apply
    there (the old bulb starts *at* the fixture, d0 ~ 0), so an absolute clearance
    threshold bounds it.

    Each step pays ``Phi(s') - Phi(s)`` (Ng, Harada & Russell 1999) rather than a
    permanently-banked best-so-far increment: approaching pays positive, retreating pays
    *negative*, and holding still nets ~0. That directly fixes the best-so-far scheme's
    exploit -- almost reaching the target and then fully leaving still banked full credit,
    since the increment can never go down. Summed over an episode this telescopes to
    ``Phi(final) - Phi(initial)``, so the total reflects final position, not a historical
    peak, while every single step still carries a real, dense gradient.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._initial = torch.ones(env.num_envs, device=env.device)
        self._prev_phi = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        fn: Callable[[ManagerBasedRLEnv], torch.Tensor] = self.cfg.params["distance_fn"]
        d = fn(self._env)[ids]
        # floor d0: a spawn already at the target must read "done" (1.0), not divide by ~0
        self._initial[ids] = d.clamp_min(1e-3)
        away = self.cfg.params.get("away_threshold")
        # seed Phi with the spawn's own potential so the first step's difference is well-formed
        self._prev_phi[ids] = 0.0 if away is None else (d / away).clamp(0.0, 1.0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        distance_fn: Callable[[ManagerBasedRLEnv], torch.Tensor],
        away_threshold: float | None = None,
    ) -> torch.Tensor:
        d = distance_fn(env)
        if away_threshold is None:
            phi = ((self._initial - d) / self._initial).clamp(0.0, 1.0)
        else:
            phi = (d / away_threshold).clamp(0.0, 1.0)
        shaped = phi - self._prev_phi
        self._prev_phi = phi
        return shaped


class signal_progress(ManagerTermBase):
    """Potential-based shaping on any ``[0, 1]`` signal: pays the *change* each step.

    Same scheme as :class:`distance_progress`, for channels whose potential is already a
    bounded fraction (contact coverage, grip force, axis alignment) rather than a distance.
    Each step pays ``Phi(s') - Phi(s)``, so the episode telescopes to
    ``Phi(final) - Phi(initial)`` and the channel's total is capped by ``weight * dt`` no
    matter how long the horizon is.

    Paying the raw signal per step instead makes holding a pose the dominant strategy on a
    long episode: the ``success`` termination ends the episode, so finishing would cut off
    an income stream worth more than the completion bonus (issue #169).

    ``signal_fn`` is named in ``params``; every other param is forwarded to it. They stay flat
    because ``ManagerBase`` resolves a ``SceneEntityCfg`` only at the top level of ``params`` --
    nested inside a sub-dict it would never be bound to the scene.

    ``__call__`` must name every param a call site passes, defaulted to ``None``, rather than
    catch them in ``**kwargs``: ``ManagerBase._resolve_common_term_cfg`` validates a term's
    ``params`` by matching cfg keys against ``inspect.signature(term.__call__)`` by name, and
    does not treat a ``**kwargs`` catch-all as "accepts anything" -- it takes the literal
    parameter name ``kwargs`` as one more required key, so any call site passing params other
    than exactly ``{"signal_fn", "kwargs"}`` fails at env construction (``ValueError: The term
    'X' expects mandatory parameters: [...'kwargs'] ... but received: [...]``). The values here
    are unused: ``_phi`` re-reads them from ``self.cfg.params`` (needed anyway, since ``reset``
    calls ``_phi`` with no per-call kwargs available), so the signature only has to name them.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_phi = torch.zeros(env.num_envs, device=env.device)

    def _phi(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        params = {k: v for k, v in self.cfg.params.items() if k != "signal_fn"}
        return self.cfg.params["signal_fn"](env, **params).clamp(0.0, 1.0)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._prev_phi[ids] = self._phi(self._env)[ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        signal_fn: Callable[..., torch.Tensor],
        sensor_cfg: SceneEntityCfg | None = None,
        threshold: float | None = None,
        std: float | None = None,
        saturation_force: float | None = None,
    ) -> torch.Tensor:
        del sensor_cfg, threshold, std, saturation_force  # forwarded to signal_fn via self.cfg.params in _phi
        phi = self._phi(env)
        shaped = phi - self._prev_phi
        self._prev_phi = phi
        return shaped


class completion_bonus(ManagerTermBase):
    """Pay 1.0 on the first step ``predicate_fn`` is true each episode (sparse completion).

    A completion state persists (a seated bulb stays seated), so paying the raw predicate
    per step would reward milking an achieved state until timeout instead of finishing;
    one-shot pay keeps ``Episode_Reward/<term>`` a clean did-it-happen flag.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._paid[ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        predicate_fn: Callable[..., torch.Tensor],
        predicate_params: dict | None = None,
    ) -> torch.Tensor:
        pred = predicate_fn(env, **(predicate_params or {}))
        fire = pred & ~self._paid
        self._paid |= pred
        return fire.float()


def _predicted_stance_shoulder_w(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """(shoulder position, stance forward) the working stance would have on THIS ladder pose.

    The stance is a pure function of the ladder pose -- ``stand_robot_on_ladder_top`` puts the
    pelvis over the tread and turns it a quarter turn -- so a gate judging a placement can work
    out where the shoulder would end up before the robot ever climbs.
    """
    ladder: RigidObject = env.scene["ladder"]
    n, device = env.num_envs, env.device
    quat = ladder.data.root_quat_w
    tread = torch.tensor(STEP_LADDER_TOP_OFFSET, device=device).expand(n, 3)
    pelvis = ladder.data.root_pos_w + quat_apply(quat, tread)
    pelvis = pelvis + torch.tensor((0.0, 0.0, TOP_STANCE_PELVIS_OFFSET), device=device).expand(n, 3)

    turn = math.radians(TOP_STANCE_YAW_OFFSET_DEG)
    half = torch.tensor((math.cos(turn / 2), 0.0, 0.0, math.sin(turn / 2)), device=device).expand(n, 4)
    stance_quat = quat_mul(quat, half)
    shoulder = pelvis + quat_apply(stance_quat, torch.tensor(G1_WORKING_SHOULDER_OFFSET, device=device).expand(n, 3))
    forward = quat_apply(stance_quat, torch.tensor((1.0, 0.0, 0.0), device=device).expand(n, 3))
    return shoulder, forward


def _fixture_grasp_point_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World position of the seated bulb's graspable body centre."""
    socket: RigidObject = env.scene["socket"]
    offset = torch.tensor(BULB_BODY_CENTRE_OFFSET, device=env.device).expand(env.num_envs, 3)
    return socket.data.root_pos_w + quat_apply(socket.data.root_quat_w, offset)


def ladder_ready(
    env: ManagerBasedRLEnv,
    reach: float,
    tilt_limit: float,
    facing_tolerance: float,
) -> torch.Tensor:
    """True where the ladder is placed so a stance on it could GRASP the seated bulb.

    Four things, where there used to be two (issue #147). Upright, as before. Then the bulb's
    graspable body centre within ``reach`` of the predicted stance's shoulder in 3-D, not a flat
    radius. The bulb must also sit at least ``LADDER_READY_MIN_BEARING`` out horizontally, or the
    fixture housing is inside the stance's torso -- and the facing angle below is undefined on a
    zero bearing. Finally the stance must FACE it.
    """
    shoulder, forward = _predicted_stance_shoulder_w(env)
    bearing = _fixture_grasp_point_w(env) - shoulder
    within = torch.norm(bearing, dim=1) < reach
    flat = bearing[:, :2]
    flat_norm = flat.norm(dim=1)
    clear = flat_norm >= LADDER_READY_MIN_BEARING
    cos = (forward[:, :2] * flat).sum(dim=1) / flat_norm.clamp(min=1e-9)
    facing = torch.acos(cos.clamp(-1.0, 1.0)) < facing_tolerance
    return within & clear & facing & ~ladder_tipped(env, tilt_limit)


def old_bulb_removed(
    env: ManagerBasedRLEnv, clearance_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("old_bulb")
) -> torch.Tensor:
    """True where the old bulb has cleared the fixture's socket by ``clearance_threshold``."""
    return old_bulb_fixture_clearance(env, asset_cfg) > clearance_threshold


def old_bulb_disposed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("old_bulb"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """True where the old bulb rests inside the disposal crate (#131)."""
    return old_bulb_in_bin(env, asset_cfg=asset_cfg, bin_cfg=bin_cfg)


def old_bulb_dropped(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("old_bulb"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """True where the old bulb lies at floor level *outside* the disposal crate.

    A plain height gate cannot work here: legitimately disposing the bulb also ends near
    the floor (resting inside the crate), so "dropped" additionally requires being outside
    the crate.
    """
    old_bulb: RigidObject = env.scene[asset_cfg.name]
    below = old_bulb.data.root_pos_w[:, 2] < min_height
    return below & ~old_bulb_disposed(env, asset_cfg, bin_cfg)


def full_replacement_success(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.015,
    ori_threshold: float = 0.2,
) -> torch.Tensor:
    """True where the fresh bulb is seated AND the old bulb is in the disposal crate.

    Disposal implies removal, so the removed predicate is not re-checked. Also the
    ``success`` termination.
    """
    return bulb_seated(env, pos_threshold, ori_threshold) & old_bulb_disposed(env)


LADDER_TILT_LIMIT = 0.6  # rad; the upright ladder stands at ~0


def ladder_tipped(
    env: ManagerBasedRLEnv,
    tilt_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ladder"),
) -> torch.Tensor:
    """True where the (dynamic) ladder's local up-axis tilts beyond ``tilt_limit`` (rad).

    Penalty + termination for the replace task; the episode ends on the tipping step, so
    (like ``fall_terminated``) the penalty fires once per tip.
    """
    ladder: RigidObject = env.scene[asset_cfg.name]
    up = torch.zeros(env.num_envs, 3, device=env.device)
    up[:, 2] = 1.0
    up_w = quat_apply(ladder.data.root_quat_w, up)
    return torch.acos(up_w[:, 2].clamp(-1.0, 1.0)) > tilt_limit


def base_ladder_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Horizontal distance (m) from the robot's root to the ladder's root.

    Bare ``(env) -> Tensor`` so it can be a ``distance_progress`` ``distance_fn``, which forbids
    lambdas and closures.
    """
    robot: Articulation = env.scene["robot"]
    ladder: RigidObject = env.scene["ladder"]
    return torch.norm((robot.data.root_pos_w - ladder.data.root_pos_w)[:, :2], dim=1)


def _base_yaw(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Robot root yaw (rad) about world +z."""
    q = env.scene["robot"].data.root_quat_w  # (N, 4) wxyz
    return torch.atan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
    )


def base_facing_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Absolute yaw error (rad) between the robot's heading and the bearing to an entity.

    Wrapped to [-pi, pi] before taking the magnitude, so a target directly behind reads pi
    rather than ~2*pi.
    """
    target: RigidObject = env.scene[asset_cfg.name]
    delta = (target.data.root_pos_w - env.scene["robot"].data.root_pos_w)[:, :2]
    err = torch.atan2(delta[:, 1], delta[:, 0]) - _base_yaw(env)
    return torch.abs(torch.atan2(torch.sin(err), torch.cos(err)))
