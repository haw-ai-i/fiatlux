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


class _Data:
    def __init__(self, root):
        if root:
            self.root_pos_w = object()


class _Entity:
    def __init__(self, root=True):
        self.data = _Data(root)


class _Sensor:
    """A contact sensor: it has ``.data``, but nothing with a root pose on it."""

    def __init__(self):
        self.data = _Data(False)


class _Prop:
    """An AssetBaseCfg entity, which lands in ``scene.extras`` as an XformPrimView."""


class _Scene:
    def __init__(self, rigid, articulations=(), sensors=(), extras=()):
        self.rigid_objects = {n: _Entity() for n in rigid}
        self.articulations = {n: _Entity() for n in articulations}
        self.sensors = {n: _Sensor() for n in sensors}
        self.extras = {n: _Prop() for n in extras}

    def __getitem__(self, key):
        for family in (self.articulations, self.rigid_objects, self.sensors, self.extras):
            if key in family:
                return family[key]
        raise KeyError(key)


class _SensorScene(_Scene):
    def __init__(self, rigid, sensors):
        super().__init__(rigid, sensors=sensors)


class _Env:
    def __init__(self, scene, declared=None):
        self.scene = scene
        self.cfg = types.SimpleNamespace()
        if declared is not None:
            self.cfg.record_objects = declared


def test_every_rigid_object_is_tracked_by_default(recording):
    """The default deliberately does not sweep in the articulations: the robot's root state and
    joints already have their own columns, and recording it here would write them twice."""
    scene = _Scene(["old_bulb", "fresh_bulb", "ladder"], articulations=["robot"])
    assert recording._tracked_object_names(_Env(scene)) == ["old_bulb", "fresh_bulb", "ladder"]


def test_a_curated_list_may_name_an_articulation(recording):
    """The columns are a root pose and its two velocities, which an Articulation serves as well as
    a RigidObject does. Rigidity is not the thing being asked about."""
    scene = _Scene(["old_bulb"], articulations=["robot"])
    assert recording._tracked_object_names(_Env(scene, declared=["robot", "old_bulb"])) == ["robot", "old_bulb"]


def test_a_curated_name_with_no_root_pose_is_skipped_and_said_out_loud(recording, capsys):
    """A silently missing column reads offline as an object that never moved."""
    scene = _Scene(["old_bulb"], sensors=["hand_contact"], extras=["fixture"])
    env = _Env(scene, declared=["old_bulb", "hand_contact", "fixture", "nonesuch"])
    assert recording._tracked_object_names(env) == ["old_bulb"]
    warning = capsys.readouterr().out
    assert "hand_contact" in warning and "fixture" in warning and "nonesuch" in warning


def test_every_contact_sensor_without_a_named_column_is_discovered(recording):
    """The disposal and grasp gates read release_contact and grasp_contact. Their columns say WHY
    a gate stayed shut, and a hardcoded list left them out of policy bags entirely. The three that
    have a named column of their own are excluded, so no sensor is written twice."""
    env = types.SimpleNamespace(
        scene=_SensorScene(
            [], ["hand_contact", "left_hand_contact", "grip_contact", "grasp_contact", "release_contact"]
        )
    )
    assert recording._extra_contact_names(env) == ["grasp_contact", "release_contact"]


def test_a_scene_with_no_sensors_records_no_extra_contact_columns(recording):
    assert recording._extra_contact_names(types.SimpleNamespace(scene=_Scene([]))) == []
