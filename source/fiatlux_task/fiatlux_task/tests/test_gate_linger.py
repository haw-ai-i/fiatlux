# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``mdp.sustained``: the hold, and the linger that keeps an episode running past its success.

``gates`` imports ``isaaclab.managers``, which needs a running simulator, so that one module is
stubbed and ``gates.py`` is loaded from its file. The class under test only uses ``torch``.
"""

import importlib.util
import pathlib
import sys
import types

import pytest
import torch

N = 3


@pytest.fixture(scope="module")
def gates():
    stub = types.ModuleType("isaaclab.managers")

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env

    stub.ManagerTermBase = ManagerTermBase
    stub.RewardTermCfg = stub.TerminationTermCfg = object
    saved = {k: sys.modules.get(k) for k in ("isaaclab", "isaaclab.managers")}
    sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    sys.modules["isaaclab.managers"] = stub
    path = pathlib.Path(__file__).parents[1] / "tasks/manager_based/fiatlux_task/mdp/gates.py"
    spec = importlib.util.spec_from_file_location("_gates_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


class _Env:
    num_envs = N
    device = "cpu"
    step_dt = 0.02


def _run(gates, script, seconds, linger=None):
    """Feed ``script`` (one bool per step, same for every env) and return the term's output per step."""
    term = gates.sustained(None, _Env())
    state = {"i": 0}

    def predicate(env):
        return torch.full((N,), bool(script[state["i"]]))

    out = []
    for i in range(len(script)):
        state["i"] = i
        kwargs = {} if linger is None else {"linger_seconds": linger}
        out.append(bool(term(_Env(), predicate, seconds, **kwargs)[0]))
    return out, term


def test_without_a_linger_the_term_is_the_plain_hold(gates):
    """Default arguments must be the behaviour every existing gate was written against."""
    out, _ = _run(gates, [True] * 30, seconds=0.5)
    assert out.index(True) == 24  # 25 steps of 0.02 s, zero-based
    assert all(out[24:])


def test_a_zero_linger_is_identical_to_no_linger(gates):
    script = [True] * 10 + [False] + [True] * 40
    assert _run(gates, script, 0.5)[0] == _run(gates, script, 0.5, linger=0.0)[0]


def test_the_term_fires_exactly_linger_after_the_hold_is_met(gates):
    out, _ = _run(gates, [True] * 200, seconds=0.5, linger=2.0)
    assert out.index(True) == 24 + 100
    assert not any(out[: 24 + 100])


def test_the_decision_is_latched_through_the_linger(gates):
    """Once the hold is met the success stands, whatever the predicate does while the episode
    runs on -- a robot that stands up or steps on after a climb must not un-succeed."""
    script = [True] * 25 + [False] * 150
    out, _ = _run(gates, script, seconds=0.5, linger=2.0)
    assert out.index(True) == 24 + 100


def test_a_dip_before_the_hold_is_met_still_resets_it(gates):
    script = [True] * 20 + [False] + [True] * 200
    out, _ = _run(gates, script, seconds=0.5, linger=2.0)
    assert out.index(True) == 21 + 24 + 100


def test_a_predicate_that_never_holds_never_fires(gates):
    script = ([True] * 10 + [False]) * 30
    assert not any(_run(gates, script, seconds=0.5, linger=2.0)[0])


def test_a_stateless_gate_wrapped_with_a_zero_hold_decides_on_the_same_step(gates):
    """``SubtaskEnvCfg`` wraps bare ``all_of`` gates as ``sustained(seconds=0)`` to give them the
    linger; a zero hold has to mean one step, the decision the bare gate made."""
    script = [False] * 7 + [True] + [False] * 150
    out, _ = _run(gates, script, seconds=0.0, linger=2.0)
    assert out.index(True) == 7 + 100
    assert _run(gates, script, seconds=0.0)[0].index(True) == 7


def test_reset_clears_the_latch_for_the_reset_envs_only(gates):
    term = gates.sustained(None, _Env())
    always = lambda env: torch.ones(N, dtype=torch.bool)  # noqa: E731
    for _ in range(60):
        term(_Env(), always, 0.5, linger_seconds=2.0)
    term.reset([1])
    for _ in range(70):
        out = term(_Env(), always, 0.5, linger_seconds=2.0)
    # 130 calls in all: envs 0 and 2 passed call 125; env 1 started over at call 60 and needs 125 more
    assert out.tolist() == [True, False, True]


def test_envs_latch_independently(gates):
    term = gates.sustained(None, _Env())
    step = {"i": 0}

    def staggered(env):
        return torch.tensor([True, step["i"] >= 10, False])

    fired = [None] * N
    for i in range(200):
        step["i"] = i
        out = term(_Env(), staggered, 0.5, linger_seconds=2.0)
        for e in range(N):
            if fired[e] is None and bool(out[e]):
                fired[e] = i
    assert fired == [124, 134, None]


def test_conjuncts_of_sees_through_the_linger_wrapping(gates):
    """Partial credit and the recorder's gate columns read the conjunct list; wrapping a bare gate
    to add the linger must not change what they see."""

    def a(env):
        return None

    def b(env, x):
        return None

    bare = gates.conjuncts_of(gates.all_of, {"predicates": [(a, {}), (b, {"x": 1})]})
    wrapped = gates.conjuncts_of(
        gates.sustained,
        {
            "predicate_fn": gates.all_of,
            "seconds": 0.0,
            "predicate_params": {"predicates": [(a, {}), (b, {"x": 1})]},
            "linger_seconds": 2.0,
        },
    )
    assert wrapped == bare == [(a, {}), (b, {"x": 1})]
