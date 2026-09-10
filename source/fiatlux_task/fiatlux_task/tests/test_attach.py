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
bulb laterally and angularly on its own. What replaced it is a continuous axial MAGNET force
pulling the plug to the bottom of the bore, applied alongside (never instead of) real contact,
plus a much smaller FREE/SEATED admission gate. These tests cover that: the magnet's shape
(strongest at the bottom, never outward-pushing), that it beats gravity across the whole range
it has to hold and that its gains respect the control rate's stability bounds, the seat/release
and in-the-bore gates, and that nothing here ever falls back to a pose/velocity overwrite.

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
RELEASE_THRESHOLD = 0.015
RELEASE_DEBOUNCE_STEPS = 3
HOLD_FORCE = 1.5
BORE_DEPTH = 0.025
SPRING_D = 1.5
MAX_FORCE = 5.0
BULB_MASS = 0.035
GRAVITY = 9.81
STEP_DT = 0.02  # the 50 Hz control rate the gain bounds below are derived against
LATERAL_K = 5.0
LATERAL_D = 0.85
MAX_LATERAL_FORCE = 1.0
TILT_K = 0.25
TILT_D = 0.003
MAX_TORQUE = 0.25
# No TWIST_* constants: there is no twist term. See test_no_twist_torque_at_any_spin_rate.


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
    # release_debounce_steps=1 here: this test isolates the ALIGNMENT gate, not the release
    # debounce (test_release_requires_consecutive_over_threshold_steps covers that on its own).
    _step(mgr, env, release_debounce_steps=1)
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


def test_seated_bulb_magnet_pulls_inward_at_a_small_displacement():
    """A small axial displacement from the seat produces an inward attraction, of exactly the
    magnitude the linear falloff predicts, and does not release (issue #171)."""
    env, mgr = _make_env()
    _step(mgr, env)  # seats the old bulb
    bulb = env.scene["old_bulb"]
    bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.005]])  # 5 mm out, past seat_tolerance
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == SEATED
    # axis is +z (SOCKET_SEAT_AXIS in the real module is (0,0,1)); attraction pulls -z.
    applied_z = bulb.last_force[0, 0, 2].item()
    assert applied_z < 0.0
    assert abs(applied_z) <= MAX_FORCE + 1e-6
    expected = -HOLD_FORCE * (1.0 - 0.005 / BORE_DEPTH)
    assert abs(applied_z - expected) < 1e-6  # zero velocity, so damping contributes nothing


def test_magnet_is_strongest_at_the_bore_bottom_and_fades_outward():
    """The defining shape of a magnet sunk in the bore bottom (issue #171): STRONGEST exactly
    at the bottom and monotonically weaker as the plug withdraws -- the opposite of a spring,
    which is weakest at the seat and grows with distance.

    Sampled AT the seat too, unlike the previous signed version: this law has no direction flip
    at zero, so the force there is the full peak rather than zero by construction."""
    env, mgr = _make_env()
    _step(mgr, env)  # seats the old bulb
    bulb = env.scene["old_bulb"]
    magnitudes = []
    for offset in (0.0, 0.002, 0.004, 0.006, 0.008):
        bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, offset]])
        _step(mgr, env)
        magnitudes.append(abs(bulb.last_force[0, 0, 2].item()))
    assert magnitudes[0] == max(magnitudes)  # strongest at the bore bottom
    assert abs(magnitudes[0] - HOLD_FORCE) < 1e-6  # and that peak IS hold_force
    assert magnitudes == sorted(magnitudes, reverse=True)  # fades monotonically outward
    assert magnitudes[-1] < magnitudes[0]  # genuinely faded, not flat


def test_magnet_never_pushes_outward_however_deep_the_plug_sits():
    """Being pressed PAST the seat reference must not weaken or reverse the attraction.

    A magnet has no outward-pushing side: what stops the plug going deeper is contact bottoming
    it out, not the field turning around. This is also why there is no sign flip left to
    chatter across -- the failure mode that made the twist term below eject the bulb.

    Stops at ``release_threshold``: release is ``|axial| > release_threshold``, symmetric, so
    an offset deeper than that reads as a release and zeroes the whole wrench. That symmetry is
    moot in practice (contact bottoms the plug out ~2 mm past the seat, nowhere near 8 mm
    deeper) and is left as it was; it just means this sweep has nothing to say beyond it."""
    env, mgr = _make_env()
    _step(mgr, env)
    bulb = env.scene["old_bulb"]
    for offset in (0.0, -0.001, -0.004, -0.007):  # strictly inside release_threshold (8 mm)
        bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, offset]])
        _step(mgr, env)
        applied_z = bulb.last_force[0, 0, 2].item()
        assert applied_z < 0.0, f"attraction reversed at axial={offset}"
        assert abs(applied_z - (-HOLD_FORCE)) < 1e-6  # full strength, not tapered by depth


def test_magnet_beats_gravity_everywhere_it_has_to_hold():
    """The property the previous version lacked, checked on the constants themselves (the fake
    harness integrates no dynamics).

    A magnitude that DECAYS with distance has an UNSTABLE balance point against constant
    gravity: inside it the bulb recovers, past it gravity wins and it accelerates out. So the
    requirement is not "the sag is small" but "the attraction exceeds the weight across the
    whole travel where the bulb is supposed to be held" -- otherwise there is a distance past
    which nothing catches it. The shipped 0.5 N / 0.01 m magnet crossed gravity at 4.6 mm with
    a peak of only 1.46x weight, and a ceiling-mounted bulb went over that cliff and fell out
    in 0.64 s; the docstring of the day mistook the 4.6 mm crossing for a resting sag."""
    weight = BULB_MASS * GRAVITY
    assert HOLD_FORCE / weight > 4.0  # firm breakaway at the bore bottom
    # Attraction at the release threshold -- the far end of the held range -- with real margin.
    # 1.75x, not the 2x this asserted while release_threshold was 8 mm: the threshold moved out
    # to 15 mm to keep the magnet engaged through a teleop finger-brush (issue #171, sixth
    # finding), which spends some of this margin on purpose. What may NOT be spent is the
    # crossing itself -- see below.
    at_release = HOLD_FORCE * (1.0 - RELEASE_THRESHOLD / BORE_DEPTH)
    assert at_release / weight > 1.5
    # The load-bearing property, and the real bound on how far release_threshold may move: the
    # gravity crossing must stay OUTSIDE it, so that everywhere the bulb still counts as held,
    # the attraction beats its weight and gravity alone cannot walk it out to release.
    crossing = BORE_DEPTH * (1.0 - weight / HOLD_FORCE)
    assert crossing > RELEASE_THRESHOLD
    assert RELEASE_THRESHOLD < BORE_DEPTH  # and the phase gate still fires before the bore gate


def test_axial_gains_respect_the_solver_stability_bounds():
    """Both axial gains are bounded by the 50 Hz control rate, not chosen freely (issue #171).

    The falloff slope is a stiffness and must stay under the semi-implicit bound
    ``mass/step_dt^2``. ``spring_d`` is bounded more tightly, by ``mass/step_dt``: one control
    step of damping must not be able to reverse the velocity it opposes. That second rule is
    the lesson of the twist-friction bug -- an impulse larger than the momentum it opposes does
    not dissipate, it chatters -- so it is asserted here rather than left to critical-damping
    arithmetic, which would allow 2.90 and be wrong."""
    stiffness = HOLD_FORCE / BORE_DEPTH
    assert stiffness < BULB_MASS / STEP_DT**2
    assert SPRING_D <= BULB_MASS / STEP_DT
    # Deliberately under-damped for the slope above; real contact supplies the rest.
    critically_damped = 2.0 * (stiffness * BULB_MASS) ** 0.5
    assert SPRING_D / critically_damped < 1.0


def test_wrench_is_zero_once_the_plug_has_left_the_bore():
    """The whole wrench is gated on the plug being IN the bore, not merely SEATED (issue #171).

    A magnet in the bore bottom acts on a plug inside that bore and on nothing else. Normally
    the phase gate fires first (release_threshold 8 mm is far inside bore_depth 25 mm), so this
    is exercised with an explicit release_threshold override deeper than the bore -- the
    misconfiguration the gate exists to contain."""
    env, mgr = _make_env()
    _step(mgr, env)
    bulb = env.scene["old_bulb"]
    # Past the bore mouth, but told not to release, so only the geometric gate can zero it.
    bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, BORE_DEPTH + 0.001]])
    _step(mgr, env, release_threshold=BORE_DEPTH + 0.01)
    assert mgr._phase[OLD, 0] == SEATED  # still nominally seated...
    assert torch.count_nonzero(bulb.last_force) == 0  # ...but out of the bore, so no force
    assert torch.count_nonzero(bulb.last_torque) == 0
    # Radially outside the bore is gated the same way, at the seat depth.
    bulb.data.root_pos_w = torch.tensor([[0.0, RADIAL_TOLERANCE + 0.001, 0.0]])
    _step(mgr, env)
    assert torch.count_nonzero(bulb.last_force) == 0


def test_seated_bulb_lateral_spring_resists_lateral_displacement():
    """A seated bulb pushed sideways (perpendicular to the seat axis, axial unchanged) gets a
    restoring lateral force -- the gentler lateral analogue of the axial spring (issue #171).
    A lateral offset alone does not release the bulb: release is axial-only, unchanged."""
    env, mgr = _make_env()
    _step(mgr, env)  # seats the old bulb
    bulb = env.scene["old_bulb"]
    bulb.data.root_pos_w = torch.tensor([[0.0, 0.010, 0.0]])  # 10 mm sideways; axial still 0
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == SEATED  # a lateral offset alone never releases it
    applied_y = bulb.last_force[0, 0, 1].item()
    assert applied_y < 0.0  # pulls back toward the seat axis
    assert abs(applied_y) <= MAX_LATERAL_FORCE + 1e-6
    expected = -LATERAL_K * 0.010
    assert abs(applied_y - expected) < 1e-6  # zero velocity, so damping contributes nothing


def test_seated_bulb_tilt_torque_resists_tilt():
    """A seated bulb tilted off-axis (rotated about an axis PERPENDICULAR to the seat axis, not
    twisted about it) gets a restoring torque aligning its plug axis back -- issue #171. Twist
    about the seat axis itself is untouched, matching the module's twist-is-free convention
    (issue #90): rotating about world X here tilts the bulb's local +z away from the socket's,
    which the correction opposes, but never rotates purely about z (there is no torque_z here to
    check against, unlike a rotation via ``twist_by`` about world z, which would produce none)."""
    import math

    env, mgr = _make_env()
    _step(mgr, env)  # seats the old bulb
    bulb = env.scene["old_bulb"]
    angle = 0.1  # rad, about world X -- tilts the bulb's +z toward -y (right-hand rule)
    half = 0.5 * angle
    bulb.data.root_quat_w = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == SEATED  # tilt alone never releases it either
    torque = bulb.last_torque[0, 0]
    assert torque[1].item() == 0.0
    assert torque[2].item() == 0.0
    expected_x = -TILT_K * math.sin(angle)
    assert expected_x < 0.0
    assert abs(torque[0].item() - expected_x) < 1e-6  # zero ang. velocity, so damping is zero
    assert abs(torque[0].item()) <= MAX_TORQUE + 1e-6


def test_no_twist_torque_at_any_spin_rate():
    """A seated bulb spinning about the seat axis itself gets NO torque about that axis, at any
    rate -- issue #171's third finding, after the ablation.

    A 0.5 N*m Coulomb twist friction used to live here. It was the CAUSE of the reported
    ceiling spin, not a fix for it: against this bulb's ~2.9e-05 kg*m^2 twist inertia one
    control step of it swings the twist rate by ~344 rad/s, while its sign flips at a 0.1 rad/s
    deadband -- so it overshot zero and reversed every step. The original teleop bag shows
    exactly that, twist alternating sign on 138 of 140 consecutive control steps at +-16 rad/s.
    Zeroing it (and no other term) removed the spin: 95-97 rad/s with it, 0.7-3.0 without.

    Rotation about the seat axis is now left entirely to the socket's real contact friction,
    which the asset carries (static 1.2 / dynamic 1.0, bound on every bulb and socket
    collider). Sweeping the rate here is the point, not incidental: any velocity-dependent
    twist term, viscous or Coulomb, would show up as a nonzero z-torque at one of these."""
    env, mgr = _make_env()
    _step(mgr, env)  # seats the old bulb
    bulb = env.scene["old_bulb"]
    for rate in (0.05, 5.0, 20.0, 100.0, -100.0):
        bulb.data.root_ang_vel_w = torch.tensor([[0.0, 0.0, rate]])
        _step(mgr, env)
        assert mgr._phase[OLD, 0] == SEATED  # spin alone never releases it
        torque = bulb.last_torque[0, 0]
        assert torque[2].item() == 0.0, f"twist torque reappeared at {rate} rad/s"
        # Pure twist is also not leaked into the tilt term: no misalignment, and the tilt
        # damping sees only the axis-perpendicular part of angular velocity, which is zero.
        assert torque[0].item() == 0.0
        assert torque[1].item() == 0.0


def test_tilt_damping_ignores_twist_so_it_keeps_its_budget():
    """Tilt damping reads only the axis-PERPENDICULAR angular velocity, so carried twist cannot
    consume tilt's deliberately tiny max_torque (issue #171).

    This is the one sound piece of the abandoned twist work, and it still matters with no twist
    term: damping the full angular velocity would put ``tilt_d * twist_rate`` into a 0.05 N*m
    budget, saturating it at any real spin and starving the alignment correction the budget
    exists for. Here a bulb is tilted (so alignment wants a definite torque) AND spinning fast
    about the seat axis; the torque must be exactly what tilt alone asks for."""
    import math

    env, mgr = _make_env()
    _step(mgr, env)
    bulb = env.scene["old_bulb"]
    angle = 0.1  # rad about world X -- same tilt as test_seated_bulb_tilt_torque_resists_tilt
    half = 0.5 * angle
    bulb.data.root_quat_w = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])
    bulb.data.root_ang_vel_w = torch.tensor([[0.0, 0.0, 50.0]])  # fast twist, no tilt rate
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == SEATED
    torque = bulb.last_torque[0, 0]
    # Identical to the un-spinning case: the 50 rad/s twist contributes nothing, and nothing
    # is clamped away, so the full alignment torque survives.
    assert abs(torque[0].item() - (-TILT_K * math.sin(angle))) < 1e-6
    assert abs(torque[0].item()) <= MAX_TORQUE + 1e-6
    assert torque[2].item() == 0.0


def test_axial_force_saturates_at_max_force():
    """The commanded force is clamped, not left to grow without bound, whatever drives it
    there. The attraction alone can never approach max_force -- it's bounded by hold_force
    (1.5 N vs. a 5 N cap) by construction, at any depth -- so this drives saturation through a
    large velocity (the damping term) instead, exercising the same clamp from the other side."""
    env, mgr = _make_env()
    _step(mgr, env)
    bulb = env.scene["old_bulb"]
    bulb.data.root_lin_vel_w = torch.tensor([[0.0, 0.0, 10.0]])  # large outward kick
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == SEATED  # still at the seat; velocity alone doesn't release it
    applied_z = bulb.last_force[0, 0, 2].item()
    assert abs(applied_z) <= MAX_FORCE + 1e-6
    assert abs(applied_z - (-MAX_FORCE)) < 1e-6  # actually pinned at the cap, not just under it


def test_release_past_threshold_frees_the_bulb_and_zeroes_the_force():
    """A real, physics-driven excursion past ``release_threshold`` releases the bulb -- not a
    force reading, a displacement one, so it is directly legible off real contact with nothing
    overwritten -- once it has persisted for ``release_debounce_steps`` consecutive steps (issue
    #171: real teleop found every observed release was a single-step contact jolt, not a
    sustained pull, so a bare one-frame test no longer matches what release means here). The
    step after release applies zero force, since nothing is SEATED anymore."""
    env, mgr = _make_env()
    _step(mgr, env)
    bulb = env.scene["old_bulb"]
    bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, RELEASE_THRESHOLD + 0.001]])
    for _ in range(RELEASE_DEBOUNCE_STEPS - 1):
        _step(mgr, env)
        assert mgr._phase[OLD, 0] == SEATED  # still within the debounce window
    _step(mgr, env)  # the Nth consecutive over-threshold step
    assert mgr._phase[OLD, 0] == FREE
    assert torch.allclose(bulb.last_force, torch.zeros_like(bulb.last_force))


def test_release_requires_consecutive_over_threshold_steps_not_one_frame():
    """A single-step excursion past ``release_threshold`` that reads back under it the very next
    step does NOT release the bulb -- issue #171's fifth finding: real S11 insert teleop showed
    every release was exactly this shape (one contact-jolt frame), which the bare pre-debounce
    check could not tell apart from a genuine sustained withdrawal. This is the "light knock"
    the module docstring always claimed ``release_threshold`` alone would resist but did not,
    until the debounce counter existed to actually enforce it."""
    env, mgr = _make_env()
    _step(mgr, env)
    bulb = env.scene["old_bulb"]
    for _ in range(5):  # several separate one-frame jolts, well past RELEASE_DEBOUNCE_STEPS calls
        bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, RELEASE_THRESHOLD + 0.001]])
        _step(mgr, env)
        assert mgr._phase[OLD, 0] == SEATED  # the jolt frame itself never releases on its own
        bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.0]])  # back at the seat the next frame
        _step(mgr, env)
        assert mgr._phase[OLD, 0] == SEATED  # streak reset -- no accumulation across jolts


def test_small_resting_sag_does_not_self_release():
    """release_threshold (15 mm) sits comfortably above what a held bulb sits at once contact
    has bottomed it out (~2 mm in the real sim), so a bulb the magnet is actively holding does
    not flicker in and out of SEATED from ordinary settling."""
    env, mgr = _make_env()
    _step(mgr, env)
    bulb = env.scene["old_bulb"]
    bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.002]])  # a couple mm of sag
    _step(mgr, env)
    assert mgr._phase[OLD, 0] == SEATED


def test_teleop_sized_finger_brush_stays_seated_and_is_pulled_home():
    """A brush the size real VR teleop actually produces neither releases the bulb nor loses the
    magnet (issue #171, sixth finding).

    The dex3 operator cannot let go cleanly -- across 33 s of seated time the fingers were fully
    off the bulb for 0.6 s -- so every hold ends in a brush. Measured across 14 knock-outs in the
    real bags, the excursion each brush caused peaked between 8.9 and 13.8 mm. At the old 8 mm
    threshold every one of those crossed it, the phase went FREE, and the wrench went to zero
    mid-excursion, leaving the bulb coasting outward with nothing to arrest it. The property
    that fixes that is checked here at both ends of the measured range: still SEATED, and still
    being pulled home by more than the bulb's own weight."""
    weight = BULB_MASS * GRAVITY
    for excursion in (0.0089, 0.0120, 0.0138):  # min, middle and max of the measured brushes
        env, mgr = _make_env()
        _step(mgr, env)
        bulb = env.scene["old_bulb"]
        bulb.data.root_pos_w = torch.tensor([[0.0, 0.0, excursion]])
        # Several steps, not one: a brush parks the bulb out there rather than bouncing it, so
        # the debounce must not be what is carrying this -- the threshold has to.
        for _ in range(RELEASE_DEBOUNCE_STEPS + 2):
            _step(mgr, env)
        assert mgr._phase[OLD, 0] == SEATED, f"a {excursion * 1000:.1f} mm brush released the bulb"
        applied_z = bulb.last_force[0, 0, 2].item()
        assert applied_z < -weight, f"a {excursion * 1000:.1f} mm brush is not pulled home against its own weight"


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
