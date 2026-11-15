# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guardrails for the three arrival gates (S05, S07, S09) and the layout that feeds them.

Two defects, both measured, both regressions this file is meant to catch a second time:

* **#267** -- S09's single-frame fix (#206) was never propagated to S05 or S07, which kept scoring
  on the one frame where every conjunct happened to line up. The tell was an unpropagated constant:
  ``ARRIVAL_SUSTAIN_SECONDS`` sat in the shared cfg with exactly one consumer.
* **#266** -- the Replace layout sampler could stand the robot inside S05's own success region
  (2 of 200 seeds; worst 0.695 m from the crate against a 0.5045 m arrival radius), so the take
  scored 1.00 after two steps and 2 mm of travel. A debounce window does NOT fix that: a
  stationary robot holds every conjunct for as long as you care to wait.

:func:`test_every_arrival_gate_is_debounced_and_armed` is the one that generalizes -- it reads the
cfg source with ``ast`` and so needs neither Kit nor a GPU, and it fails for a *fourth* arrival
gate added later without the fix, which is the shape both bugs actually had.
"""

from __future__ import annotations

import ast
import pathlib
import random

import pytest

SUBTASKS = pathlib.Path(__file__).resolve().parents[1] / "tasks" / "manager_based" / "fiatlux_task" / "subtasks"

# An "arrival gate" is a leaf scored on walking somewhere and standing there. They are exactly the
# leaves that import the shared arrival tolerances -- a membership test the source can answer, so a
# new one cannot be added without this file noticing.
ARRIVAL_MARKER = "ARRIVAL_FACING_TOLERANCE"
EXPECTED_ARRIVAL_LEAVES = {
    "s05_carry_bulb_to_disposal_env_cfg.py",
    "s07_approach_new_bulb_env_cfg.py",
    "s09_carry_bulb_to_ladder_env_cfg.py",
}


def _arrival_leaves() -> dict[str, ast.Module]:
    out = {}
    for path in sorted(SUBTASKS.glob("s*_env_cfg.py")):
        src = path.read_text()
        if ARRIVAL_MARKER in src:
            out[path.name] = ast.parse(src)
    return out


def _class_level_assignment(tree: ast.Module, name: str):
    """The value assigned to ``name`` at class level anywhere in the module, or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                targets = stmt.targets if isinstance(stmt, ast.Assign) else ([stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
                for t in targets:
                    if isinstance(t, ast.Name) and t.id == name and getattr(stmt, "value", None) is not None:
                        return stmt.value
    return None


def test_arrival_leaves_are_the_expected_three():
    """If this fails a leaf was added or renamed -- extend the set, then make it pass the next test."""
    assert set(_arrival_leaves()) == EXPECTED_ARRIVAL_LEAVES


@pytest.mark.parametrize("filename", sorted(EXPECTED_ARRIVAL_LEAVES))
def test_every_arrival_gate_is_debounced_and_armed(filename):
    """Every arrival gate: ``mdp.sustained`` over ``ARRIVAL_SUSTAIN_SECONDS``, ``require_transition``.

    Issues #266 and #267. Asserted on the source rather than on a built cfg so it runs without Kit.
    """
    tree = _arrival_leaves()[filename]

    predicate = _class_level_assignment(tree, "success_predicate")
    assert predicate is not None, f"{filename}: no success_predicate"
    assert isinstance(predicate, ast.Attribute) and predicate.attr == "sustained", (
        f"{filename}: success_predicate is {ast.unparse(predicate)}, not mdp.sustained -- a bare"
        " conjunction scores on one frame (#267)"
    )

    params = _class_level_assignment(tree, "success_params")
    assert isinstance(params, ast.Dict), f"{filename}: success_params is not a dict literal"
    keys = {k.value for k in params.keys if isinstance(k, ast.Constant)}
    by_key = {
        k.value: v for k, v in zip(params.keys, params.values) if isinstance(k, ast.Constant)
    }

    assert "seconds" in keys, f"{filename}: no sustain window (#267)"
    assert ast.unparse(by_key["seconds"]) == "ARRIVAL_SUSTAIN_SECONDS", (
        f"{filename}: sustain window is {ast.unparse(by_key['seconds'])}, not the shared"
        " ARRIVAL_SUSTAIN_SECONDS -- one constant, one meaning"
    )
    assert "require_transition" in keys, (
        f"{filename}: no require_transition -- a spawn inside the success region holds every"
        " conjunct forever and clears any window (#266)"
    )
    assert by_key["require_transition"].value is True, f"{filename}: require_transition is not True"


# ---------------------------------------------------------------------------------------------
# Gate semantics and layout geometry. These import Isaac Lab, so they skip without Kit.
# ---------------------------------------------------------------------------------------------


class _FakeEnv:
    """The three attributes ``sustained`` touches. Enough to drive it without a scene."""

    def __init__(self, num_envs: int = 2, step_dt: float = 0.02):
        self.num_envs = num_envs
        self.device = "cpu"
        self.step_dt = step_dt


def _make_sustained():
    pytest.importorskip("isaaclab.sim", reason="needs a bootstrapped Kit runtime")
    gates = pytest.importorskip("fiatlux_task.tasks.manager_based.fiatlux_task.mdp.gates")
    env = _FakeEnv()
    term = gates.sustained.__new__(gates.sustained)
    import torch

    term._held = torch.zeros(env.num_envs, dtype=torch.long)
    term._armed = torch.zeros(env.num_envs, dtype=torch.bool)
    return gates, term, env


@pytest.mark.isaacsim_ci
def test_sustained_still_debounces():
    """The #206 behaviour, unchanged: a dip resets the counter, so a flicker never reaches the end."""
    pytest.importorskip("isaaclab.sim", reason="needs a bootstrapped Kit runtime")
    import torch

    _, term, env = _make_sustained()
    window = 0.1  # 5 steps at step_dt 0.02
    pattern = [True, True, False, True, True, True, True, True]

    def step(value: bool) -> bool:
        const = torch.tensor([value, value])
        return bool(term(env, lambda e: const, window, require_transition=False)[0])

    fired = [step(v) for v in pattern]
    # two trues, a dip that resets, then five trues: only the fifth of those clears the window
    assert fired == [False, False, False, False, False, False, False, True], fired


@pytest.mark.isaacsim_ci
def test_require_transition_rejects_a_goal_held_from_step_zero():
    """#266: true from the first step and never false -- the window passes, the gate must not."""
    pytest.importorskip("isaaclab.sim", reason="needs a bootstrapped Kit runtime")
    import torch

    _, term, env = _make_sustained()
    always_true = lambda e: torch.tensor([True, True])  # noqa: E731
    fired = [bool(term(env, always_true, 0.04, require_transition=True)[0]) for _ in range(50)]
    assert not any(fired), "a spawn inside the success region scored without ever achieving it"


@pytest.mark.isaacsim_ci
def test_require_transition_allows_a_goal_that_is_actually_reached():
    """The other half: false first, then held for the window, still scores."""
    pytest.importorskip("isaaclab.sim", reason="needs a bootstrapped Kit runtime")
    import torch

    _, term, env = _make_sustained()
    false_fn = lambda e: torch.tensor([False, False])  # noqa: E731
    true_fn = lambda e: torch.tensor([True, True])  # noqa: E731
    term(env, false_fn, 0.04, require_transition=True)  # arrives at the goal having been away
    fired = [bool(term(env, true_fn, 0.04, require_transition=True)[0]) for _ in range(4)]
    assert fired == [False, True, True, True], fired


@pytest.mark.isaacsim_ci
def test_ladder_mount_spawn_sits_on_the_mount_radius():
    pytest.importorskip("isaaclab.sim", reason="needs a bootstrapped Kit runtime")
    import math

    sc = pytest.importorskip("fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg")
    spawn = sc._ladder_mount_spawn((1.0, 2.0), (5.0, 2.0))
    assert math.isclose(math.dist(spawn, (1.0, 2.0)), sc.LADDER_MOUNT_RADIUS, abs_tol=1e-9)
    # on the bearing toward the robot's own zone, not past it
    assert spawn[0] > 1.0 and math.isclose(spawn[1], 2.0, abs_tol=1e-9)


@pytest.mark.isaacsim_ci
def test_zone_edge_distance_is_zero_inside():
    pytest.importorskip("isaaclab.sim", reason="needs a bootstrapped Kit runtime")
    sc = pytest.importorskip("fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg")
    assert sc._zone_edge_distance((0.0, 0.0), (0.0, 0.0), 0.5) == 0.0
    assert sc._zone_edge_distance((0.4, 0.4), (0.0, 0.0), 0.5) == 0.0
    assert sc._zone_edge_distance((1.5, 0.0), (0.0, 0.0), 0.5) == pytest.approx(1.0)


@pytest.mark.isaacsim_ci
def test_no_sampled_layout_spawns_a_carry_leg_at_its_own_goal():
    """#266, as a sweep: the ladder-mount spawn always has somewhere to carry to.

    Before the fix this failed on 2 of 200 seeds; the worst put the spawn 0.349 m from the crate
    footprint against a 0.5045 m arrival radius.

    ``couple_ladder_to_fixture=True`` only, because that is what every Replace leaf passes --
    the uncoupled default is documented as a debug/curriculum aid, and there the ladder is not at
    the reserved anchor the sampler tests against.
    """
    pytest.importorskip("isaaclab.sim", reason="needs a bootstrapped Kit runtime")
    sc = pytest.importorskip("fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg")
    worst = None
    for seed in range(200):
        _, centers, _ = sc._sample_replace_layout(random.Random(seed), couple_ladder_to_fixture=True)
        robot_center, _, ladder_center, disposal_center = centers
        spawn = sc._ladder_mount_spawn(ladder_center, robot_center)
        d = sc._zone_edge_distance(spawn, disposal_center, sc.DISPOSAL_ZONE_HALF_SIZE)
        worst = d if worst is None else min(worst, d)
    assert worst >= sc.MIN_CARRY_CLEARANCE - 1e-9, f"closest ladder-mount spawn to the crate zone: {worst:.3f} m"
