# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Which of the retention wrench's terms causes the ceiling spin/ejection? (issue #171)

``verify_twist_damping.py`` asks "does the current force law hold?"; this asks the prior
question the answer to that kept begging -- *which* of the applied forces is responsible.
``mdp/attach.py`` applies SEVEN independent terms to a seated bulb (four force, three torque),
all summed into one ``set_external_force_and_torque`` call, so a run either holds or does not
and there is no way to attribute the failure. This script enumerates them, prints their live
gains, then re-runs the same forced-ceiling-mount scenario once per term with that term (and
only that term) zeroed, so the traces can be compared side by side.

Every gain is already a ``__call__`` keyword of ``mdp.bulb_attachment``, so an ablation is a
gain override in the event term's live ``params`` dict -- no code path is stubbed out or
branched on, and ``baseline`` runs the committed defaults untouched. One env is built and
reused across trials (Isaac Sim startup dominates the runtime); each trial re-resets it under
the same seed and re-writes the bulb exactly to the seat, and ``baseline`` is run again as the
LAST trial so any warm-start drift between the first and last identical trial is visible
rather than silent.

The bulb's real contact friction against the socket bore is a premise of the whole question
(if the bore were frictionless, no wrench term could be blamed for a spin nothing opposes), so
``--report_friction`` (on by default) resolves and prints the physics material actually bound
to both bodies' colliders, and the ``zero_wrench`` trial measures what real contact alone does
with no wrench at all.

Run with `uv run python` from the repo root, not a bare .venv/bin/python: a .venv shared across
checkouts can otherwise resolve fiatlux_task to the wrong one (unlike its sibling scripts, this
one has no ``verify_common.assert_right_checkout`` guard against that yet -- issue #238).

Examples
--------
    uv run python scripts/ablate_attach_forces.py --headless --seed 3
    uv run python scripts/ablate_attach_forces.py --headless --seed 3 --spin_rate 15
    uv run python scripts/ablate_attach_forces.py --headless --seed 3 --only baseline,no_axial_magnet
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Ablate each retention wrench term to find the one causing the spin.")
parser.add_argument("--seed", type=int, default=3, help="Env seed. 3 is the seed the bug was reported at.")
parser.add_argument("--seconds", type=float, default=5.0, help="Real simulated seconds to watch per trial.")
parser.add_argument(
    "--spin_rate",
    type=float,
    default=0.0,
    help="Initial spin about the seat axis, rad/s. 0 (default) reproduces the reported failure, "
    "which starts from rest -- the spin appears on its own. Nonzero injects a kick like "
    "verify_twist_damping.py does.",
)
parser.add_argument(
    "--only",
    type=str,
    default="",
    help="Comma-separated ablation names to run (default: all). See ABLATIONS.",
)
parser.add_argument(
    "--repeats",
    type=int,
    default=1,
    help="Run each trial this many times. Real contact at this seed has genuine run-to-run "
    "spread (identical baselines have ejected anywhere from 0.9s to 2.2s), so any claim that "
    "an ablation HOLDS needs more than one sample.",
)
parser.add_argument(
    "--sweep",
    type=str,
    default="",
    help="Instead of the on/off ablations, sweep ONE gain's magnitude: '<gain>=v1,v2,...' "
    "(e.g. twist_friction=0.5,0.05,0.005,0). Distinguishes a mis-SIZED term from a wrong one.",
)
parser.add_argument(
    "--report_friction",
    action="store_true",
    default=True,
    help="Resolve and print the physics material bound to the bulb/socket colliders (default on).",
)
parser.add_argument("--no_report_friction", dest="report_friction", action="store_false")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything else follows."""

import importlib
import inspect
import sys

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import torch
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import attach as task_attach
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import CEILING_FIXTURE_Z, _quat_y_deg, set_layout_seed

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg

# ---------------------------------------------------------------------------------------------
# The complete inventory of what mdp/attach.py applies to a SEATED bulb, one entry per
# independent term, in the order they appear in `_advance`. `gains` names the keyword(s) that
# scale that term alone -- zeroing them removes exactly that term and nothing else, which is
# what makes a gain override a faithful ablation rather than an approximation of one.
#
# Not listed because they are not terms: max_force / max_lateral_force / max_torque are
# saturation clamps on the sums below, and bore_depth is the magnet's range (it sets where the
# attraction reaches zero, not how strong it is).
TERMS = (
    (
        "axial_magnet",
        "force",
        "-hold_force * max(1 - max(axial,0)/bore_depth, 0) * seat_axis",
        "Primary retention: an inward attraction, strongest at the bore bottom, zero at its mouth.",
        ("hold_force",),
    ),
    (
        "axial_damping",
        "force",
        "-spring_d * axial_rate * seat_axis",
        "Viscous damping of motion along the seat axis; bounded by mass/step_dt, not critical damping.",
        ("spring_d",),
    ),
    (
        "lateral_spring",
        "force",
        "-lateral_k * lateral_vec",
        "Gentle centering of the plug onto the seat axis line (fights bore-clearance wobble).",
        ("lateral_k",),
    ),
    (
        "lateral_damping",
        "force",
        "-lateral_d * lateral_vel",
        "Viscous damping of sideways motion.",
        ("lateral_d",),
    ),
    (
        "tilt_alignment",
        "torque",
        "tilt_k * cross(plug_axis, seat_axis)",
        "Restoring torque aligning the plug axis to the seat axis (no twist component).",
        ("tilt_k",),
    ),
    (
        "tilt_damping",
        "torque",
        "-tilt_d * tilt_ang_vel",
        "Viscous damping of the tilt (axis-perpendicular) part of angular velocity only.",
        ("tilt_d",),
    ),
)
# There is no seventh, twist term any more. A 0.5 N*m Coulomb twist friction used to sit here
# and this very sweep is what identified it as the CAUSE of the reported ceiling spin rather
# than a fix for it (see mdp/attach.py's docstring); it was removed in the same change. Nothing
# to ablate, so nothing to list -- but the sweep still reports every term's per-step authority
# against the bulb's real inertia, which is the check that would have caught it up front.

# One ablation per term, plus the per-axis groupings (a pair of terms can only be blamed jointly
# if neither alone reproduces it) and the two controls: `baseline` unmodified, `zero_wrench` with
# nothing applied at all -- gravity and real contact only, which is the floor this mechanism has
# to improve on rather than a passing result in itself.
_ALL_GAINS = tuple(gain for _, _, _, _, gains in TERMS for gain in gains)


def _off(*gains: str) -> dict[str, float]:
    return {gain: 0.0 for gain in gains}


ABLATIONS: dict[str, dict[str, float]] = {
    "baseline": {},
    **{f"no_{name}": _off(*gains) for name, _, _, _, gains in TERMS},
    "no_axial": _off("hold_force", "spring_d"),
    "no_lateral": _off("lateral_k", "lateral_d"),
    "no_tilt": _off("tilt_k", "tilt_d"),
    "zero_wrench": _off(*_ALL_GAINS),
    # Determinism control: identical to `baseline`, run last. If these two disagree, trials are
    # not comparable and nothing below can be attributed to the ablation.
    "baseline_repeat": {},
}


def parse_sweep(spec: str) -> dict[str, dict[str, float]]:
    """``gain=v1,v2,...`` -> one trial per value, for following up on an implicated term.

    Once an ablation names the term, the question stops being "which one" and becomes "is it
    the term or its MAGNITUDE" -- a term whose failure disappears below some gain is mis-sized,
    while one that fails at every nonzero value is the wrong force law. That is a different
    experiment from on/off, so it gets its own flag rather than more ABLATIONS entries.
    """
    if not spec:
        return {}
    gain, _, values = spec.partition("=")
    gain = gain.strip()
    known = set(default_gains())
    if not values or gain not in known:
        raise SystemExit(f"--sweep must be '<gain>=v1,v2,...' with gain one of {sorted(known)}; got {spec!r}")
    trials = {}
    for raw in values.split(","):
        raw = raw.strip()
        if raw:
            trials[f"{gain}={raw}"] = {gain: float(raw)}
    return trials


def default_gains() -> dict[str, float]:
    """The gains `attach.py` actually ships, read off its signature rather than duplicated here."""
    sig = inspect.signature(task_attach.bulb_attachment.__call__)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty and isinstance(param.default, (int, float))
    }


def print_authority(env, params: dict) -> None:
    """What one control step of each term's PEAK magnitude does to the bulb, in real units.

    The gains in ``attach.py`` are absolute (newtons, newton-metres) and were chosen against the
    bulb's MASS, but the two torque terms act against its MOMENT OF INERTIA about the seat axis
    -- a plug ~18 mm in radius at 35 g, so a number that looks small as a torque can still be a
    very large angular acceleration. An external wrench set via
    ``set_external_force_and_torque`` persists across the whole control step, so the velocity
    change one application can produce is ``tau / I * step_dt`` (and ``F / m * step_dt``); when
    that exceeds the velocity it was meant to cancel, a sign-flipping (Coulomb) law overshoots
    through zero and reverses instead of settling, which is chatter rather than friction.
    """
    defaults = default_gains()
    live = {**defaults, **{k: v for k, v in params.items() if isinstance(v, (int, float))}}
    bulb = env.scene["old_bulb"]
    mass = float(bulb.root_physx_view.get_masses()[0].sum().item())
    inertia = bulb.root_physx_view.get_inertias()[0].reshape(3, 3)
    axis_l = torch.tensor(SOCKET_SEAT_AXIS, device=inertia.device, dtype=inertia.dtype)
    axis_l = axis_l / axis_l.norm()
    # Inertia is reported in the body frame, and the seat axis is a body-local constant here, so
    # a plain quadratic form gives the scalar inertia the twist torque actually works against.
    inertia_twist = float((axis_l @ inertia.to(axis_l.device) @ axis_l).item())
    dt = env.step_dt
    print("\n=== AUTHORITY OF EACH TERM OVER ONE CONTROL STEP ===", flush=True)
    print(
        f"  bulb mass={mass:.4f} kg  I(diag)={[round(float(inertia[i, i]), 9) for i in range(3)]} kg*m^2  "
        f"I about seat axis={inertia_twist:.3e} kg*m^2  step_dt={dt:.4f} s",
        flush=True,
    )
    for label, peak, unit, denom, vel_unit in (
        ("axial_magnet", live["hold_force"], "N", mass, "m/s"),
        ("axial_damping (at 1 m/s)", live["spring_d"], "N", mass, "m/s"),
        ("lateral_spring (clamp)", live["max_lateral_force"], "N", mass, "m/s"),
        ("tilt_alignment (clamp)", live["max_torque"], "N*m", inertia_twist, "rad/s"),
    ):
        delta = peak / denom * dt
        print(f"  {label:24s} peak {peak:>7.4g} {unit:4s} -> delta_v {delta:>12.4g} {vel_unit} per step", flush=True)
    print(
        "  A term whose delta_v per step exceeds the velocity it is meant to cancel does not\n"
        "  dissipate -- it overshoots and reverses. That is what the removed twist friction did\n"
        "  (0.5 N*m against 2.9e-05 kg*m^2 = 344 rad/s per step, sign-flipping at 0.1 rad/s), and\n"
        "  it is why spring_d is capped at mass/step_dt = "
        f"{mass / dt:.3g} rather than at critical damping.",
        flush=True,
    )


def print_force_inventory(params: dict) -> None:
    """List every term, its formula and its live gain values -- the "list them all" step."""
    defaults = default_gains()
    live = {**defaults, **{k: v for k, v in params.items() if isinstance(v, (int, float))}}
    print("\n=== FORCES/TORQUES APPLIED TO A SEATED BULB BY mdp/attach.py ===", flush=True)
    for i, (name, kind, formula, why, gains) in enumerate(TERMS, start=1):
        shown = " ".join(f"{g}={live.get(g)}" for g in gains)
        print(f"  {i}. [{kind:6s}] {name:16s} {formula}", flush=True)
        print(f"                              {why}", flush=True)
        print(f"                              ablated by: {shown}", flush=True)
    clamps = ("max_force", "max_lateral_force", "max_torque")
    gates = ("seat_tolerance", "release_threshold", "bore_depth", "radial_tolerance")
    print(f"  clamps (not terms): {' '.join(f'{c}={live.get(c)}' for c in clamps)}", flush=True)
    print(f"  gates  (not terms): {' '.join(f'{g}={live.get(g)}' for g in gates)}", flush=True)
    print(
        "  NOT from attach.py, always present: gravity; real PhysX bulb<->socket contact "
        "(normal + friction). No hand contact in this scenario (zero action).",
        flush=True,
    )


def _bound_physics_material(stage, prim, UsdShade):
    """The material bound to ``prim`` for the ``physics`` purpose, walking up ancestors.

    ``ComputeBoundMaterial`` is the right call but only accepts UsdShade's own purpose tokens
    in some USD builds, and ``physics`` is UsdPhysics' custom one -- so try it, then fall back
    to reading ``material:binding:physics`` directly and inheriting it from ancestors the way
    USD's own binding resolution does (the asset binds per-collider, but a bind could equally
    sit on a parent).
    """
    try:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        if material and material.GetPrim().IsValid():
            return material.GetPrim()
    except Exception:  # noqa: BLE001 -- purpose token rejected by this USD build; fall through
        pass
    node = prim
    while node and node.IsValid() and not node.IsPseudoRoot():
        rel = node.GetRelationship("material:binding:physics")
        if rel:
            targets = rel.GetTargets()
            if targets:
                return stage.GetPrimAtPath(targets[0])
        node = node.GetParent()
    return None


def report_friction(env) -> None:
    """Resolve the physics material actually bound to the bulb's and socket's colliders.

    The premise check: real contact friction between plug and bore is what any wrench term here
    is meant to *assist*, so if PhysX resolved these colliders to a frictionless default the
    whole attribution question would be malformed. Reads the resolved USD binding (purpose
    ``physics``) rather than trusting the asset layer, since the material lives in
    ``LightBulb_collision.usda`` and reaches the spawned prim through two levels of reference
    remapping -- exactly the kind of thing that can silently fail to carry over.
    """
    from pxr import Usd, UsdPhysics, UsdShade

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    print("\n=== REAL CONTACT FRICTION (bulb <-> socket bore) ===", flush=True)
    for label in ("old_bulb", "fresh_bulb", "socket"):
        if label not in env.scene.keys():
            continue
        # Ask the view for the concrete env_0 path rather than reconstructing it from the cfg's
        # {ENV_REGEX_NS} template, which is already regex-substituted by this point.
        prim_path = env.scene[label].root_physx_view.prim_paths[0]
        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            print(f"  {label:10s} prim {prim_path} not found", flush=True)
            continue
        seen: dict[str, int] = {}
        colliders = 0
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            colliders += 1
            mat_prim = _bound_physics_material(stage, prim, UsdShade)
            if mat_prim is None or not mat_prim.IsValid():
                key = "<none -> PhysX scene default>"
            else:
                api = UsdPhysics.MaterialAPI(mat_prim)
                static = api.GetStaticFrictionAttr().Get() if api.GetStaticFrictionAttr() else None
                dynamic = api.GetDynamicFrictionAttr().Get() if api.GetDynamicFrictionAttr() else None
                restitution = api.GetRestitutionAttr().Get() if api.GetRestitutionAttr() else None
                key = f"static={static} dynamic={dynamic} restitution={restitution} ({mat_prim.GetName()})"
            seen[key] = seen.get(key, 0) + 1
        print(f"  {label:10s} {colliders} collider(s) under {prim_path}:", flush=True)
        for key, count in seen.items():
            print(f"               x{count}  {key}", flush=True)
    print(
        "  A '<none>' here means PhysX falls back to its scene default material -- the bore "
        "would then be nearly frictionless and a free spin unopposed by anything real.",
        flush=True,
    )


def build_cfg():
    """Same forced-ceiling-mount FIATLUX-Replace-v0 setup as verify_twist_damping.py."""
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=1)
    assert hasattr(cfg.scene, "fresh_bulb"), (
        f"cfg.scene ({type(cfg.scene)} from {sys.modules[type(cfg.scene).__module__].__file__}) has no "
        "fresh_bulb -- fiatlux_task likely resolved to the wrong checkout again; check sys.path/pyrun"
    )
    cfg.seed = args_cli.seed
    for camera in ("ego_camera", "torso_camera", "wrist_camera"):
        if getattr(cfg.scene, camera, None) is not None:
            setattr(cfg.scene, camera, None)
    for group_name in ("policy", "privileged"):
        group = getattr(cfg.observations, group_name, None)
        for term in ("ego_rgb", "torso_rgb", "wrist_rgb"):
            if group is not None and getattr(group, term, None) is not None:
                setattr(group, term, None)
    for term in ("success", "old_bulb_dropped", "fresh_bulb_dropped"):
        if getattr(cfg.terminations, term, None) is not None:
            setattr(cfg.terminations, term, None)
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    # Force a CEILING mount, matching the reported failure (seed 3, S03, ceiling preset).
    ceiling_quat = _quat_y_deg(180.0)
    x, y, _ = cfg.scene.socket.init_state.pos
    cfg.scene.socket.init_state.pos = (x, y, CEILING_FIXTURE_Z)
    cfg.scene.socket.init_state.rot = ceiling_quat
    return cfg


class Probe:
    """Reads the seat-frame quantities straight off the contact-resolved sim, per step."""

    def __init__(self, env):
        self._env = env
        self.socket = env.scene["socket"]
        self.bulb = env.scene["old_bulb"]
        self.manager = task_attach.attachment_manager(env)
        device = env.device
        self._seat_axis = torch.tensor(SOCKET_SEAT_AXIS, device=device)
        self._seat_offset = torch.tensor(SOCKET_SEAT_OFFSET, device=device)
        self._plug_offset = torch.tensor(BULB_PLUG_OFFSET, device=device)

    def axis_w(self) -> torch.Tensor:
        return quat_apply(self.socket.data.root_quat_w, self._seat_axis.unsqueeze(0))

    def seated_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        quat = self.socket.data.root_quat_w[0]
        seat = self.socket.data.root_pos_w[0] + quat_apply(quat.unsqueeze(0), self._seat_offset.unsqueeze(0))[0]
        return seat - quat_apply(quat.unsqueeze(0), self._plug_offset.unsqueeze(0))[0], quat

    def axial(self) -> float:
        seat = self.socket.data.root_pos_w + quat_apply(self.socket.data.root_quat_w, self._seat_offset.unsqueeze(0))
        plug = self.bulb.data.root_pos_w + quat_apply(self.bulb.data.root_quat_w, self._plug_offset.unsqueeze(0))
        return float(((plug - seat) * self.axis_w()).sum(dim=1).item())

    def lateral(self) -> float:
        seat = self.socket.data.root_pos_w + quat_apply(self.socket.data.root_quat_w, self._seat_offset.unsqueeze(0))
        plug = self.bulb.data.root_pos_w + quat_apply(self.bulb.data.root_quat_w, self._plug_offset.unsqueeze(0))
        d = plug - seat
        axis = self.axis_w()
        return float(torch.norm(d - (d * axis).sum(dim=1, keepdim=True) * axis, dim=1).item())

    def twist_rate(self) -> float:
        return float((self.bulb.data.root_ang_vel_w * self.axis_w()).sum(dim=1).item())

    def tilt(self) -> float:
        return float(
            task_attach._tilt_error(
                self.socket.data.root_quat_w, self.bulb.data.root_quat_w, self._seat_axis.unsqueeze(0)
            ).item()
        )

    def phase(self) -> int:
        return int(self.manager._phase[task_attach._OLD, 0].item())


def run_trial(env, probe: Probe, name: str, overrides: dict[str, float], base_params: dict) -> dict:
    """Reset, seat the bulb, apply this trial's gain overrides, and watch under zero action."""
    term_cfg = env.event_manager.get_term_cfg("bulb_attachment")
    # Restore the shipped params, then apply only this trial's overrides, so trials never
    # inherit each other's.
    term_cfg.params.clear()
    term_cfg.params.update(base_params)
    term_cfg.params.update(overrides)

    # Re-seed every reset so the reset-time randomization terms draw identically per trial;
    # without this, trials would differ by more than the ablation.
    env.reset(seed=args_cli.seed)
    pos, quat = probe.seated_pose()
    probe.bulb.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))
    probe.bulb.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    zero_action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    env.step(zero_action)  # resolves the spawn phase -> SEATED (bulb is within tolerance of the seat)
    if probe.phase() != task_attach._SEATED:
        return {"name": name, "error": "did not seat at spawn -- setup wrong, not an ablation result"}

    if args_cli.spin_rate:
        spin = torch.zeros((1, 6), device=env.device)
        spin[:, 3:] = args_cli.spin_rate * probe.axis_w()
        probe.bulb.write_root_velocity_to_sim(spin)

    release_threshold = base_params.get("release_threshold", default_gains()["release_threshold"])
    steps = int(round(args_cli.seconds / env.step_dt))
    trace, twists = [], []
    max_abs_axial = max_abs_twist = max_tilt = max_lateral = 0.0
    dropped_at = None
    for i in range(steps):
        env.step(zero_action)  # no hand, no disturbance: gravity + real contact + (ablated) wrench
        a, w, t, lat, p = probe.axial(), probe.twist_rate(), probe.tilt(), probe.lateral(), probe.phase()
        twists.append(abs(w))
        max_abs_axial = max(max_abs_axial, abs(a))
        max_abs_twist = max(max_abs_twist, abs(w))
        max_tilt = max(max_tilt, t)
        max_lateral = max(max_lateral, lat)
        if p != task_attach._SEATED and dropped_at is None:
            dropped_at = round(i * env.step_dt, 3)
        if i % max(1, steps // 12) == 0 or i == steps - 1:
            trace.append((round(i * env.step_dt, 2), round(a, 4), round(w, 2), round(t, 3)))

    tail = twists[max(0, len(twists) - int(round(1.0 / env.step_dt))) :]
    return {
        "name": name,
        "overrides": overrides,
        "held": dropped_at is None and max_abs_axial <= release_threshold,
        "dropped_at": dropped_at,
        "max_abs_axial": round(max_abs_axial, 4),
        "max_lateral": round(max_lateral, 4),
        "max_abs_twist": round(max_abs_twist, 2),
        "mean_abs_twist_last_1s": round(sum(tail) / max(1, len(tail)), 2),
        "final_abs_twist": round(abs(probe.twist_rate()), 2),
        "max_tilt": round(max_tilt, 3),
        "final_phase": probe.phase(),
        "trace": trace,
    }


def main() -> int:
    cfg = build_cfg()
    spec = gym.spec("FIATLUX-Replace-v0")
    module_name, class_name = spec.entry_point.split(":")
    env = getattr(importlib.import_module(module_name), class_name)(cfg=cfg)
    env.reset(seed=args_cli.seed)

    if task_attach.attachment_manager(env) is None:
        print("FATAL no bulb_attachment term on FIATLUX-Replace-v0", flush=True)
        return 1
    probe = Probe(env)
    base_params = dict(env.event_manager.get_term_cfg("bulb_attachment").params)

    print_force_inventory(base_params)
    print_authority(env, base_params)
    if args_cli.report_friction:
        report_friction(env)

    sweep = parse_sweep(args_cli.sweep)
    if sweep:
        # A magnitude sweep replaces the on/off ablations (a different question), but keeps
        # baseline at both ends as the same determinism control.
        trials = {"baseline": {}, **sweep, "baseline_repeat": {}}
    else:
        names = [n.strip() for n in args_cli.only.split(",") if n.strip()] or list(ABLATIONS)
        unknown = [n for n in names if n not in ABLATIONS]
        if unknown:
            print(f"FATAL unknown ablation(s) {unknown}; known: {list(ABLATIONS)}", flush=True)
            return 1
        trials = {name: ABLATIONS[name] for name in names}

    print(
        f"\n=== ABLATION SWEEP seed={args_cli.seed} mount=ceiling spin_rate={args_cli.spin_rate} "
        f"seconds={args_cli.seconds} trials={len(trials)} ===",
        flush=True,
    )
    results = []
    for name, overrides in trials.items():
        for rep in range(args_cli.repeats):
            label = name if args_cli.repeats == 1 else f"{name}#{rep + 1}"
            result = run_trial(env, probe, label, overrides, base_params)
            result["group"] = name
            results.append(result)
            print(f"TRIAL {label}: {result}", flush=True)

    print(f"\n=== SUMMARY (seed {args_cli.seed}, ceiling, zero action) ===", flush=True)
    header = f"{'ablation':22s} {'held':6s} {'drop@':7s} {'max|ax|':8s} {'max|tw|':8s} {'tw last1s':10s} {'maxtilt':8s}"
    print(header, flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['name']:22s} ERROR {r['error']}", flush=True)
            continue
        print(
            f"{r['name']:22s} {str(r['held']):6s} {str(r['dropped_at']):7s} {r['max_abs_axial']:<8.4f} "
            f"{r['max_abs_twist']:<8.2f} {r['mean_abs_twist_last_1s']:<10.2f} {r['max_tilt']:<8.3f}",
            flush=True,
        )
    if args_cli.repeats > 1:
        print(f"\n=== HOLD RATE OVER {args_cli.repeats} REPEATS ===", flush=True)
        for name in trials:
            group = [r for r in results if r.get("group") == name and "error" not in r]
            holds = sum(1 for r in group if r["held"])
            spins = [r["max_abs_twist"] for r in group]
            print(
                f"  {name:22s} held {holds}/{len(group)}   max|twist| over repeats: "
                f"min={min(spins):.2f} max={max(spins):.2f}",
                flush=True,
            )
    print(
        "\nRead it as: a term is IMPLICATED where removing it makes the run BETTER than baseline "
        "(that term was driving the failure), and NECESSARY where removing it makes the run worse. "
        "Compare baseline vs baseline_repeat first -- if those two disagree, the run-to-run spread "
        "is larger than the effect and nothing else in this table is attributable.",
        flush=True,
    )
    env.close()
    return 0


if __name__ == "__main__":
    import os
    import traceback

    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code:
            os._exit(exit_code)
        simulation_app.close()
