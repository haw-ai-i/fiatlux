# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guardrails for the axial-detent retention mechanism (issue #167).

An earlier version of this file tested a three-phase bayonet state machine (FREE/AXIAL/
ROTATING, twist tracking, PR #128's contact-rate clamps) that a per-step pose/velocity
overwrite enforced. That mechanism is gone: two independent diagnostics found real bulb-socket
contact geometry blocked both the release-twist and the axial-insertion motions it assumed
were unobstructed (see ``plans/bayonet-force-based-attachment.md``), and fixing that geometry
made the whole apparatus unnecessary -- with real collision enabled, the socket confines the
bulb laterally and angularly on its own. What replaced it is a continuous axial spring-damper
FORCE, applied alongside (never instead of) real contact, plus a much smaller FREE/SEATED
admission gate. These tests cover that: the spring's sign and saturation, the seat/release
gates, and that nothing here ever falls back to a pose/velocity overwrite.

The state machine is pure torch, but ``attach.py`` imports Isaac Lab symbols, so the module is
loaded standalone: ``isaaclab`` is stubbed (with real wxyz quaternion math) when it is not
installed, and the ``.rewards`` sibling -- needed only by the score-channel helpers, not the
state machine -- is replaced by an inert stand-in. No GPU, no ``isaacsim_ci`` marker.
"""

import importlib.util
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
SEAT_TOLERANCE = 0.004  # the __call__ default
RADIAL_TOLERANCE = 0.015
TILT_TOLERANCE = 0.2
RELEASE_THRESHOLD = 0.02
SPRING_K = 20.0
SPRING_D = 1.7
MAX_FORCE = 5.0


class _FakeBody:
    """The slice of ``RigidObject`` the manager touches: root state + the wrench call.

    No ``write_root_pose_to_sim``/``write_root_velocity_to_sim`` here on purpose -- the new
    mechanism never calls them. If a future change reintroduces a pose/velocity overwrite,
    every test below fails with an ``AttributeError`` rather than silently passing.
    """

    def __init__(self, pos, device="cpu"):
        self.device = device
        self.data = SimpleNamespace(
            root_pos_w=torch.tensor([pos]),
            root_quat_w=torch.tensor([_IDENTITY]),
            root_lin_vel_w=torch.zeros(1, 3),
            root_ang_vel_w=torch.zeros(1, 3),
        )
        self.last_force: torch.Tensor | None = None
        self.last_torque: torch.Tensor | None = None
        self.force_calls = 0

    def set_external_force_and_torque(self, forces, torques, is_global=False):
        assert is_global, "the retention spring must be applied in the world frame"
        self.last_force = forces.clone()
        self.last_torque = torques.clone()
        self.force_calls += 1

    def twist_by(self, angle):
        """Rotate the body about world z, as a hand (or the solver) would."""
        import math

        half = 0.5 * angle
        q = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]])
        self.data.root_quat_w = _quat_mul(q, self.data.root_quat_w)


def _make_env():
    """Socket at origin; the old bulb seated on it (both halves author assembled at identity);
    the fresh bulb far away and FREE."""
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        scene={
            "socket": _FakeBody([0.0, 0.0, 0.0]),
            "old_bulb": _FakeBody([0.0, 0.0, 0.0]),
            "fresh_bulb": _FakeBody([1.0, 0.0, 0.0]),
        },
    )
    cfg = SimpleNamespace(params={})
    return env, attach.bulb_attachment(cfg, env)


def _step(mgr, env, **overrides):
    mgr(env, None, **overrides)


OLD, FRESH = attach._OLD, attach._FRESH
FREE, SEATED = attach._FREE, attach._SEATED


def test_spawn_resolution_seats_the_seated_bulb_and_frees_the_far_one():
    """Reset leaves both bulbs FREE and *pending*; the first step reads the scene and seats
    whichever bulb actually spawned at the seat (issues #108/#109, carried over unchanged).
    Here the old bulb spawns seated and the fresh one a metre away, so the first call seats
    exactly the old one."""
    env, mgr = _make_env()
    assert mgr._phase[OLD, 0] == FREE  # pending until the first step
    assert mgr._phase[FRESH, 0] == FREE
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == SEATED
    assert mgr._phase[FRESH, 0] == FREE


def test_insert_only_scene_without_old_bulb_seats_fresh_bulb():
    """Insert-only scenes (the tabletop preset) never spawn an ``old_bulb`` -- there is nothing
    to remove. The manager must not crash indexing a scene entity that does not exist, and
    ``fresh_bulb`` must still seat normally: with no old bulb, ``self._phase[_OLD]`` never
    leaves its ``_FREE`` default, so the socket-empty check is satisfied on its own."""
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        scene={
            "socket": _FakeBody([0.0, 0.0, 0.0]),
            "fresh_bulb": _FakeBody([1.0, 0.0, 0.0]),
        },
    )
    cfg = SimpleNamespace(params={})
    mgr = attach.bulb_attachment(cfg, env)
    _step(mgr, env)
    fresh = env.scene["fresh_bulb"]
    assert mgr._phase[FRESH, 0] == FREE  # far away, unseated

    fresh.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.0]])
    _step(mgr, env)
    assert mgr._phase[FRESH, 0] == SEATED


def test_free_bulb_gets_no_force():
    """A FREE bulb is untouched: the wrench call still happens every step (so a stale force
    from a previous seat never lingers), but the applied force/torque are exactly zero."""
    env, mgr = _make_env()
    _step(mgr, env)  # resolves spawn phase; fresh bulb is FREE, far from the seat
    fresh = env.scene["fresh_bulb"]
    assert fresh.force_calls >= 1
    assert torch.allclose(fresh.last_force, torch.zeros_like(fresh.last_force))
    assert torch.allclose(fresh.last_torque, torch.zeros_like(fresh.last_torque))


def test_engage_requires_alignment_not_just_axial_proximity():
    """Reaching the seat axially is not enough: lateral offset or tilt past tolerance keeps
    the bulb FREE, exactly like the old bayonet's entry gate (issue #90's finding still
    applies -- alignment, not just depth, gates entry)."""
    env, mgr = _make_env()
    _step(mgr, env)  # old bulb seats; fresh bulb stays FREE and far away

    # Free the socket by withdrawing the old bulb past release_threshold (real, physics-driven
    # release, like test_release_past_threshold_frees_the_bulb_and_zeroes_the_force), so this
    # test isolates the alignment gate from the socket-empty gate, which
    # test_engage_requires_socket_empty covers on its own -- otherwise the fresh bulb's move
    # to the exact seat position below would correctly stay FREE just because the old bulb
    # still occupies it.
    old = env.scene["old_bulb"]
    old.data.root_pos_w = torch.tensor([[0.0, 0.0, 1.0]])  # withdrawn along the seat axis
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == FREE

    fresh = env.scene["fresh_bulb"]
    # At the seat axially, but well outside the lateral tolerance.
    fresh.data.root_pos_w = torch.tensor([[0.0, 0.05, 0.0]])
    _step(mgr, env)
    assert mgr._phase[FRESH, 0] == FREE

    # Move it onto the axis and it seats.
    fresh.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.0]])
    _step(mgr, env)
    assert mgr._phase[FRESH, 0] == SEATED


def test_engage_requires_socket_empty():
    """A bulb cannot seat while the other bulb already occupies the socket."""
    env, mgr = _make_env()
    _step(mgr, env)  # old bulb seats
    fresh = env.scene["fresh_bulb"]
    fresh.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.0]])  # perfectly at the seat too
    _step(mgr, env)
    assert mgr._phase[FRESH, 0] == FREE  # socket occupied by the old bulb
    assert mgr._phase[OLD, 0] == SEATED


def test_seated_bulb_spring_resists_small_outward_displacement():
    """A small axial displacement from the seat produces a restoring force back toward it,
    proportional to the error (the spring term), and does not release."""
    env, mgr = _make_env()
    _step(mgr, env)  # seats the old bulb
    bulb = env.scene["old_bulb"]
    bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.010]])  # 10 mm out, past seat_tolerance
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == SEATED
    # axis is +z (SOCKET_SEAT_AXIS in the real module is (0,0,1)); force should pull -z.
    applied_z = bulb.last_force[0, 0, 2].item()
    assert applied_z < 0.0
    assert abs(applied_z) <= MAX_FORCE + 1e-6
    expected = -SPRING_K * 0.010
    assert abs(applied_z - expected) < 1e-6  # zero velocity, so damping contributes nothing


def test_spring_saturates_at_max_force():
    """A displacement large enough that k*x would exceed max_force is clamped, not left to
    grow without bound -- this is what stands in for the bayonet's rate caps."""
    env, mgr = _make_env()
    _step(mgr, env)
    bulb = env.scene["old_bulb"]
    bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, 1.0]])  # absurdly far; still SEATED (< release)
    # release_threshold is 0.02 m by default, so first push it past that separately (below);
    # here we only check saturation while still within release_threshold.
    bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, RELEASE_THRESHOLD - 1e-4]])
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == SEATED
    assert abs(bulb.last_force[0, 0, 2].item()) <= MAX_FORCE + 1e-6


def test_release_past_threshold_frees_the_bulb_and_zeroes_the_force():
    """A real, physics-driven excursion past ``release_threshold`` releases the bulb -- not a
    force reading, a displacement one, so it is directly legible off real contact with nothing
    overwritten. The very next step applies zero force, since nothing is SEATED anymore."""
    env, mgr = _make_env()
    _step(mgr, env)
    bulb = env.scene["old_bulb"]
    bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, RELEASE_THRESHOLD + 0.001]])
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == FREE
    assert torch.allclose(bulb.last_force, torch.zeros_like(bulb.last_force))


def test_small_resting_sag_does_not_self_release():
    """release_threshold (2 cm) sits comfortably above what an unheld bulb sags under gravity
    at rest, so a bulb the spring is actively holding does not flicker in and out of SEATED
    from ordinary settling."""
    env, mgr = _make_env()
    _step(mgr, env)
    bulb = env.scene["old_bulb"]
    bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.002]])  # a couple mm of sag
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == SEATED


def test_damping_opposes_outward_velocity():
    """The damping term pulls against outward velocity even at zero displacement -- so a bulb
    given an outward kick, still at the seat, gets an inward-pulling force immediately."""
    env, mgr = _make_env()
    _step(mgr, env)
    bulb = env.scene["old_bulb"]
    bulb.data.root_lin_vel_w = torch.tensor([[0.0, 0.0, 1.0]])  # moving outward (+z)
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == SEATED
    applied_z = bulb.last_force[0, 0, 2].item()
    assert applied_z < 0.0  # opposes the outward velocity
