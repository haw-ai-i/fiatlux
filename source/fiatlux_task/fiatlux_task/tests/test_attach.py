# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guardrails for the bayonet state machine's contact bounds (issue #92, PR #128).

The 2026-09-01 sessions showed a hand closing on a locked bulb unlocking and ejecting it
through solver noise alone: depenetration spun the bulb at 30-50 rad/s, ``theta`` unwound a
quarter-turn lock in 14 control steps with no visible rotation, and the ejected bulb left at
up to 5.6 m/s. These tests drive ``mdp.bulb_attachment`` through that exact recorded unwind
(``bench-releases/ep06`` of ``gs://fiatlux/teleop-trajectory/
issue-evidence-2026-09-01-92-release-sessions/``) and through the legitimate flows around it.

The state machine is pure torch, but ``attach.py`` imports Isaac Lab symbols, so the module
is loaded standalone: ``isaaclab`` is stubbed (with real wxyz quaternion math) when it is not
installed, and the ``.rewards`` sibling -- needed only by the score-channel helpers, not the
state machine -- is replaced by an inert stand-in. No GPU, no ``isaacsim_ci`` marker.
"""

import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

# -- isaaclab stand-ins ---------------------------------------------------------------------


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv, w = q[..., 1:], q[..., :1]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w * t + torch.cross(qv, t, dim=-1)


def _install_isaaclab_stubs() -> None:
    try:
        import isaaclab.assets  # noqa: F401
        import isaaclab.managers  # noqa: F401
        import isaaclab.utils.math  # noqa: F401

        return  # the real thing is available; use it
    except ImportError:
        pass

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

    assets = types.ModuleType("isaaclab.assets")
    assets.RigidObject = type("RigidObject", (), {})
    assets.Articulation = type("Articulation", (), {})
    managers = types.ModuleType("isaaclab.managers")
    managers.ManagerTermBase = ManagerTermBase
    managers.EventTermCfg = SimpleNamespace
    managers.SceneEntityCfg = SimpleNamespace
    utils = types.ModuleType("isaaclab.utils")
    utils_math = types.ModuleType("isaaclab.utils.math")
    utils_math.quat_mul = _quat_mul
    utils_math.quat_inv = _quat_inv
    utils_math.quat_apply = _quat_apply
    utils.math = utils_math
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.assets, isaaclab.managers, isaaclab.utils = assets, managers, utils
    sys.modules.update(
        {
            "isaaclab": isaaclab,
            "isaaclab.assets": assets,
            "isaaclab.managers": managers,
            "isaaclab.utils": utils,
            "isaaclab.utils.math": utils_math,
        }
    )


def _load_attach():
    _install_isaaclab_stubs()
    import fiatlux_task  # light: pulls no Isaac Lab code of its own

    mdp_dir = Path(fiatlux_task.__file__).parent / "tasks" / "manager_based" / "fiatlux_task" / "mdp"
    pkg = types.ModuleType("_attach_pkg")
    pkg.__path__ = [str(mdp_dir)]
    rewards = types.ModuleType("_attach_pkg.rewards")
    for name in (
        "old_bulb_disposal_distance",
        "old_bulb_disposed",
        "old_bulb_dropped",
        "old_bulb_fixture_clearance",
    ):
        setattr(rewards, name, lambda *a, **k: None)
    sys.modules["_attach_pkg"], sys.modules["_attach_pkg.rewards"] = pkg, rewards
    spec = importlib.util.spec_from_file_location("_attach_pkg.attach", mdp_dir / "attach.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_attach_pkg.attach"] = module
    spec.loader.exec_module(module)
    return module


attach = _load_attach()

# -- a one-env fake scene the manager can write to ------------------------------------------

_IDENTITY = (1.0, 0.0, 0.0, 0.0)
LOCK_ANGLE = 0.5 * math.pi
DEPTH = 0.034
STEP_DT = 0.02
MAX_TWIST_RATE = 12.0  # the __call__ default; the per-step theta credit is this * STEP_DT
MAX_AXIAL_RATE = 1.0
MAX_TWIST_STEP = MAX_TWIST_RATE * STEP_DT


class _FakeBody:
    """The slice of ``RigidObject`` the manager touches: root state + the two sim writes."""

    def __init__(self, pos):
        self.data = SimpleNamespace(
            root_pos_w=torch.tensor([pos]),
            root_quat_w=torch.tensor([_IDENTITY]),
            root_lin_vel_w=torch.zeros(1, 3),
            root_ang_vel_w=torch.zeros(1, 3),
        )

    def write_root_pose_to_sim(self, pose, env_ids):
        self.data.root_pos_w[env_ids] = pose[:, :3]
        self.data.root_quat_w[env_ids] = pose[:, 3:]

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.data.root_lin_vel_w[env_ids] = velocity[:, :3]
        self.data.root_ang_vel_w[env_ids] = velocity[:, 3:]

    def twist_by(self, angle):
        """Rotate the body about world z, as a hand (or the solver) would."""
        half = 0.5 * angle
        q = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]])
        self.data.root_quat_w = _quat_mul(q, self.data.root_quat_w)


def _make_env():
    """Socket at origin; the old bulb seated on it (both halves author assembled at identity);
    the fresh bulb far away and FREE."""
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=STEP_DT,
        scene={
            "socket": _FakeBody([0.0, 0.0, 0.0]),
            "old_bulb": _FakeBody([0.0, 0.0, 0.0]),
            "fresh_bulb": _FakeBody([1.0, 0.0, 0.0]),
        },
    )
    cfg = SimpleNamespace(params={"insertion_depth": DEPTH, "rotation_angle": LOCK_ANGLE, "rotation_sign": -1.0})
    return env, attach.bulb_attachment(cfg, env)


def _step(mgr, env, **overrides):
    mgr(env, None, **overrides)


OLD, FRESH = attach._OLD, attach._FRESH
FREE, AXIAL, ROTATING = attach._FREE, attach._AXIAL, attach._ROTATING

# ``rotation_sign`` is -1: locking is a negative (clockwise) twist, so UNLOCKING twists
# positive. The recorded ep06 unwind below is stored as positive twist increments.

# Per-step twist of the crushed bulb over t=4.56-4.72 s of bench-releases/ep06 -- the solver
# spun a LOCKED bulb through its full quarter-turn lock in 8 control steps while the operator
# only squeezed. theta unwound 1.571 -> 0 and the bulb ejected at the following step.
EP06_UNWIND = [0.115, 0.086, 0.167, 0.496, 0.074, 0.144, 0.345, 0.144]


def test_spawn_resolution_locks_the_seated_bulb_and_frees_the_far_one():
    """Post-#140 contract: reset leaves both bulbs FREE and *pending*; the first step reads the
    scene and locks whichever bulb actually spawned at the seat (issues #108/#109). Here the old
    bulb spawns seated and the fresh one a metre away, so the first call locks exactly the old."""
    env, mgr = _make_env()
    assert mgr._phase[OLD, 0] == FREE  # pending until the first step
    assert mgr._phase[FRESH, 0] == FREE
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == ROTATING
    assert abs(mgr._theta[OLD, 0].item() - LOCK_ANGLE) < 1e-6
    assert mgr._phase[FRESH, 0] == FREE


def test_recorded_solver_unwind_does_not_fully_unlock():
    """Replaying ep06's measured spin leaves the lock still engaged, where it fully opened before.

    On the pre-clamp code these eight deltas summed to exactly 1.571 rad and unwound theta to 0 --
    the bulb left the socket. With the per-step credit capped at ``max_twist_rate * step_dt``, the
    same eight steps shed at most 0.24 rad each, so theta stays >~0.36 and the bulb stays ROTATING.

    This is a RATE cap, not an impossibility proof: sustaining that spin for more steps than the
    recording holds would still unlock. What stops the *sustained* spin is ``position_slack``
    removing the contact force that drives it -- untestable here without contact physics, so this
    asserts only what the clamp alone guarantees.
    """
    env, mgr = _make_env()
    bulb = env.scene["old_bulb"]
    for delta in EP06_UNWIND:
        bulb.twist_by(delta)
        _step(mgr, env)
        assert mgr._phase[OLD, 0] == ROTATING
    shed = sum(min(d, MAX_TWIST_STEP) for d in EP06_UNWIND)
    assert abs(mgr._theta[OLD, 0].item() - (LOCK_ANGLE - shed)) < 1e-5
    assert mgr._theta[OLD, 0] > 0.3


def test_clamp_bounds_theta_change_even_when_raw_twist_aliases():
    """A raw twist above pi folds through ``_wrap_to_pi``; the credit clamp bounds the theta
    change to ``max_twist_rate * step_dt`` regardless, so no single spin unwinds a large arc.

    Tested off the lock ceiling: at theta == angle the ``min(.., angle)`` saturation would hide a
    lock-direction change, so this first unwinds partway with sub-cap steps, then applies one
    >pi jump. Pre-clamp, the aliased delta moves theta by ~0.8 rad in that step; clamped, <= 0.24.
    """
    env, mgr = _make_env()
    bulb = env.scene["old_bulb"]
    for _ in range(6):  # sub-cap unlock, so theta sits mid-range
        bulb.twist_by(0.10)
        _step(mgr, env)
    assert 0.2 < mgr._theta[OLD, 0].item() < LOCK_ANGLE - 0.2
    before = mgr._theta[OLD, 0].item()
    bulb.twist_by(3.3)  # raw jump > pi: _wrap_to_pi folds it toward a large credit
    _step(mgr, env)
    assert abs(mgr._theta[OLD, 0].item() - before) <= MAX_TWIST_STEP + 1e-6


def test_axial_bulb_retains_only_clamped_axial_velocity():
    """A bulb in the insertion channel keeps at most ``max_axial_rate`` of axial velocity, so a
    violent solver step cannot fire it up the channel. Deleting the clamp on attach.py's
    ``axial_speed`` makes this fail while every other test still passes (review coverage gap)."""
    env, mgr = _make_env()
    bulb = env.scene["old_bulb"]
    for _ in range(30):  # unlock fully -> AXIAL
        bulb.twist_by(0.2)
        _step(mgr, env)
        if mgr._phase[OLD, 0] == AXIAL:
            break
    assert mgr._phase[OLD, 0] == AXIAL
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    # Inside the channel (axial < depth) so it does not eject; a violent last solver step.
    bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.005]])
    bulb.data.root_lin_vel_w = 5.0 * axis
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == AXIAL
    assert (bulb.data.root_lin_vel_w[0] * axis[0]).sum().abs().item() <= MAX_AXIAL_RATE + 1e-6


def test_slack_concedes_small_displacement_and_bounds_large():
    """The anti-ratchet property: the projection never drags the bulb back into a gap that
    contact resolution just closed -- it concedes up to ``position_slack`` and no more."""
    env, mgr = _make_env()
    bulb = env.scene["old_bulb"]
    # Within the slack: the bulb stays exactly where the solver parked it.
    bulb.data.root_pos_w = torch.tensor([[0.002, 0.0, 0.0]])
    _step(mgr, env)
    assert torch.allclose(bulb.data.root_pos_w, torch.tensor([[0.002, 0.0, 0.0]]), atol=1e-7)
    # Beyond it: pulled back exactly to the slack shell, in the displacement's direction.
    bulb.data.root_pos_w = torch.tensor([[0.010, 0.0, 0.0]])
    _step(mgr, env)
    assert torch.allclose(bulb.data.root_pos_w, torch.tensor([[0.003, 0.0, 0.0]]), atol=1e-6)
    assert mgr._phase[OLD, 0] == ROTATING


def test_rotating_bulb_keeps_only_clamped_twist_velocity():
    env, mgr = _make_env()
    bulb = env.scene["old_bulb"]
    bulb.data.root_lin_vel_w = torch.tensor([[2.0, -1.0, 3.0]])
    bulb.data.root_ang_vel_w = torch.tensor([[5.0, 5.0, 40.0]])
    _step(mgr, env)
    assert torch.allclose(bulb.data.root_lin_vel_w, torch.zeros(1, 3))
    assert torch.allclose(bulb.data.root_ang_vel_w, torch.tensor([[0.0, 0.0, MAX_TWIST_RATE]]))


def test_deliberate_unscrew_withdraw_and_bounded_eject():
    """The legitimate removal still works end to end, and the FREE handoff is capped.

    A 2 rad/s twist (well under the 12 rad/s cap) unwinds the lock; an 0.4 m/s pull travels
    the channel; the eject step hands physics a bulb at ``max_axial_rate``, not at whatever
    the last solver step produced (2.3-5.6 m/s in the 2026-09-01 pop-outs).
    """
    env, mgr = _make_env()
    bulb = env.scene["old_bulb"]
    for _ in range(41):  # 1.571 rad at 0.04 rad/step
        bulb.twist_by(2.0 * STEP_DT)
        _step(mgr, env)
        if mgr._phase[OLD, 0] == AXIAL:
            break
    assert mgr._phase[OLD, 0] == AXIAL
    assert mgr._interaction[OLD, 0, attach._C_GATE] == 3.0  # unlock fired this step (#163)
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    while mgr._phase[OLD, 0] == AXIAL:
        bulb.data.root_pos_w = bulb.data.root_pos_w + 0.008 * axis
        bulb.data.root_lin_vel_w = 3.0 * axis  # a violent last solver step
        _step(mgr, env)
    assert mgr._phase[OLD, 0] == FREE
    assert torch.linalg.norm(bulb.data.root_lin_vel_w) <= 1.0 + 1e-6
    # The eject step's interaction row records the raw and the written handoff (#163).
    assert mgr._interaction[OLD, 0, attach._C_GATE] == 4.0
    assert abs(mgr._interaction[OLD, 0, attach._C_AXIAL_RAW].item() - 3.0) < 1e-5
    assert mgr._interaction[OLD, 0, attach._C_AXIAL_WRITTEN].item() <= 1.0 + 1e-6


def test_engage_keeps_the_entry_clock_angle():
    """Issue #90's semantics survive the clamps: a bulb engages at whatever twist it arrived
    with, and is not teleported onto the socket's own clock angle."""
    env, mgr = _make_env()
    # Let the first step resolve the spawn phases (old locks at the seat, fresh is far and
    # FREE) -- placing the fresh bulb at the mouth BEFORE this step would spawn-lock it
    # directly to ROTATING (post-#140), bypassing the engage path this test exercises.
    _step(mgr, env)
    # Empty the socket so the fresh bulb may engage: the old bulb is long gone and FREE.
    # Both must hold -- a FREE bulb parked at the seat would simply re-engage first.
    mgr._phase[OLD, 0] = FREE
    env.scene["old_bulb"].data.root_pos_w = torch.tensor([[2.0, 0.0, 0.0]])
    fresh = env.scene["fresh_bulb"]
    fresh.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.01]])
    fresh.twist_by(0.3)
    _step(mgr, env)
    assert mgr._phase[FRESH, 0] == AXIAL
    # The written pose keeps the 0.3 rad entry twist (z-component of the wxyz quaternion).
    written = fresh.data.root_quat_w[0]
    assert abs(2.0 * math.atan2(written[3], written[0]) - 0.3) < 1e-5


def test_interaction_channels_report_the_constraint_reaction():
    """The #163 channels: raw is what the solver produced, written is what the projection kept,
    and their difference is the reaction the virtual socket applied. The raw side must survive
    even though the projection overwrites the bulb's state in the same call."""
    env, mgr = _make_env()
    bulb = env.scene["old_bulb"]
    _step(mgr, env)  # resolve spawn: old locks at the seat

    # A crush-like solver step: fast spin, junk linear velocity, a 10 mm shove off the seat,
    # and a half-radian twist of the pose.
    bulb.data.root_ang_vel_w = torch.tensor([[0.0, 0.0, 40.0]])
    bulb.data.root_lin_vel_w = torch.tensor([[2.0, 0.0, 0.0]])
    bulb.data.root_pos_w = torch.tensor([[0.010, 0.0, 0.0]])
    bulb.twist_by(0.5)
    _step(mgr, env)

    row = mgr._interaction[OLD, 0]
    assert abs(row[attach._C_TWIST_RAW].item() - 40.0) < 1e-5
    assert abs(row[attach._C_TWIST_WRITTEN].item() - MAX_TWIST_RATE) < 1e-5
    assert row[attach._C_AXIAL_WRITTEN].item() == 0.0  # ROTATING keeps no axial speed
    # 10 mm shove, 3 mm slack: the projection pulled the bulb back ~7 mm.
    assert abs(row[attach._C_SLACK].item() - 0.007) < 1e-4
    # +0.5 rad of pose twist reads as a -0.5 rad credit (rotation_sign = -1), clamped.
    assert abs(row[attach._C_CREDIT_RAW].item() + 0.5) < 1e-4
    assert abs(row[attach._C_CREDIT].item() + MAX_TWIST_STEP) < 1e-6
    assert row[attach._C_GATE].item() == 0.0  # no transition fired

    # The telemetry dict carries every channel for both bulbs, from the snapshot.
    telemetry = attach.bulb_lock_telemetry(env)
    for prefix in ("old_bulb", "fresh_bulb"):
        for name in attach._INTERACTION_CHANNELS:
            assert f"{prefix}_{name}" in telemetry, f"missing {prefix}_{name}"
    assert abs(telemetry["old_bulb_twist_rate_raw"][0].item() - 40.0) < 1e-5
    # A free bulb nothing writes keeps raw == written: a zero correction.
    assert telemetry["fresh_bulb_twist_rate_raw"][0] == telemetry["fresh_bulb_twist_rate_written"][0]


def test_engage_step_records_gate_and_approach_geometry():
    """The approach channels answer "did the socket capture the bulb, and if not, why" (#163)."""
    env, mgr = _make_env()
    mgr._phase[OLD, 0] = FREE
    env.scene["old_bulb"].data.root_pos_w = torch.tensor([[2.0, 0.0, 0.0]])
    _step(mgr, env)  # resolve spawns; old is away and FREE, fresh is far and FREE

    fresh = env.scene["fresh_bulb"]
    fresh.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.01]])
    _step(mgr, env)
    row = mgr._interaction[FRESH, 0]
    assert row[attach._C_GATE].item() == 1.0  # engage fired
    assert abs(row[attach._C_AXIAL].item() - 0.01) < 1e-6
    assert row[attach._C_LATERAL].item() < 1e-6
    assert row[attach._C_TILT].item() < 1e-6
