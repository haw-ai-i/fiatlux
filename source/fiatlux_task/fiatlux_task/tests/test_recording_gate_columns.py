# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guardrails for the recorder's gate columns and its tracked-object list.

``recording`` imports the task's mdp package, which pulls in Isaac Lab, so the three modules it
needs are stubbed here. That buys the two pieces of logic below a test with no GPU and no
simulator; everything else in the recorder needs a live env and is not covered here.
"""

import sys
import types

import pytest


@pytest.fixture(scope="module")
def recording():
    for name in (
        "fiatlux_task.tasks",
        "fiatlux_task.tasks.manager_based",
        "fiatlux_task.tasks.manager_based.fiatlux_task",
        "fiatlux_task.tasks.manager_based.fiatlux_task.mdp",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    mdp = sys.modules["fiatlux_task.tasks.manager_based.fiatlux_task.mdp"]
    for sub in ("attach", "observations", "rewards"):
        mod = types.ModuleType(f"{mdp.__name__}.{sub}")
        sys.modules[mod.__name__] = mod
        setattr(mdp, sub, mod)

    import fiatlux_task.recording as rec

    return rec


class _Stub:
    """The slice of a recorder ``gate_fields`` touches, without an env to build one from."""

    def __init__(self, conjuncts):
        self.env = object()
        self._gate_conjuncts = conjuncts
        self._gate_resolved = False
        self._meta = {}


def _conjunct(name, value=True, raises=None):
    def fn(env, **params):
        if raises is not None:
            raise raises
        _conjunct.calls[name] = _conjunct.calls.get(name, 0) + 1
        return value

    fn.__name__ = name
    return fn


_conjunct.calls = {}


def test_a_conjunct_is_evaluated_once_on_the_probing_step(recording):
    """The first call settles which conjuncts are recordable by calling each one. Calling them a
    second time to fill the same row doubles the live tensor reads for no new information."""
    _conjunct.calls = {}
    stub = _Stub([(_conjunct("robot_standing"), None)])
    row = recording.TrajectoryRecorder.gate_fields(stub)
    assert row == {"gate_robot_standing": True}
    assert _conjunct.calls["robot_standing"] == 1


def test_an_unevaluable_conjunct_is_dropped_and_the_rest_recorded(recording):
    stub = _Stub(
        [
            (_conjunct("robot_standing"), None),
            (_conjunct("payload_held", raises=KeyError("grip_contact")), None),
        ]
    )
    assert recording.TrajectoryRecorder.gate_fields(stub) == {"gate_robot_standing": True}
    assert stub._meta["gate_conjuncts"] == ["robot_standing"]
    assert recording.TrajectoryRecorder.gate_fields(stub) == {"gate_robot_standing": True}


def test_a_gate_that_evaluates_to_nothing_aborts_the_run(recording):
    """Silently recording no columns from here on leaves a run that looks fine and turns out to be
    unscoreable only once it is over -- and in teleop, one whose success can never latch."""
    stub = _Stub([(_conjunct("payload_held", raises=KeyError("grip_contact")), None)])
    with pytest.raises(RuntimeError, match="cannot be scored"):
        recording.TrajectoryRecorder.gate_fields(stub)


def test_a_task_with_no_gate_records_no_columns(recording):
    """Distinct from the case above: nothing was declared, so there is nothing to fail."""
    stub = _Stub([])
    assert recording.TrajectoryRecorder.gate_fields(stub) == {}
    assert recording.TrajectoryRecorder.gate_fields(stub) == {}


class _Scene:
    def __init__(self, rigid):
        self.rigid_objects = {name: object() for name in rigid}


class _Env:
    def __init__(self, rigid, declared=None):
        self.scene = _Scene(rigid)
        self.cfg = types.SimpleNamespace()
        if declared is not None:
            self.cfg.record_objects = declared


def test_every_rigid_object_is_tracked_by_default(recording):
    env = _Env(["old_bulb", "fresh_bulb", "ladder"])
    assert recording._tracked_object_names(env) == ["old_bulb", "fresh_bulb", "ladder"]


def test_a_curated_list_that_names_a_non_rigid_object_says_so(recording, capsys):
    """A silently missing column reads offline as an object that never moved."""
    env = _Env(["old_bulb"], declared=["old_bulb", "robot"])
    assert recording._tracked_object_names(env) == ["old_bulb"]
    assert "robot" in capsys.readouterr().out
