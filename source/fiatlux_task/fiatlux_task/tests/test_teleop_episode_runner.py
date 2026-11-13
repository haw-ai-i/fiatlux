"""Unit tests for the teleop episode machine (``fiatlux_teleop.episode_runner``).

Loaded by path rather than imported as ``fiatlux_teleop.episode_runner``: that package's
``__init__`` registers gym ids and pulls in ``isaaclab``, which needs Kit. The module under
test deliberately imports nothing heavier than the standard library, so a path load keeps
these tests runnable in a bare checkout -- which is the point, since the behaviour being
tested (when a take ends and where it lands) is exactly what is expensive to check in sim.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "fiatlux_teleop", "fiatlux_teleop", "episode_runner.py")
)


def _load():
    spec = importlib.util.spec_from_file_location("_episode_runner_under_test", _MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


er = _load()


class _Vec:
    """Stands in for the (N,) bool tensor a termination term returns."""

    def __init__(self, value):
        self._value = value

    def reshape(self, *_):
        return [self._value]


class _Scene:
    pass


class _Env:
    def __init__(self):
        self.scene = _Scene()


class _Term:
    """Stands in for a ``DoneTerm`` cfg: a ``func`` (callable OR class) plus ``params``."""

    def __init__(self, func, params=None):
        self.func = func
        self.params = params or {}


def _watch(env, terms, verbose=False):
    return er.FailureWatch(env, terms, verbose=verbose)


def _staged(tmp_path, name="ep003"):
    """The in-progress take folder the DRIVER owns; TakeStore only sorts it on seal."""
    d = os.path.join(str(tmp_path), name)
    os.makedirs(d, exist_ok=True)
    return d


# -- FailureWatch ------------------------------------------------------------


def test_watch_reports_only_the_terms_that_are_true():
    env = _Env()
    flags = {"fell": False, "dropped": False}
    terms = {
        "fell": _Term(lambda e, **k: _Vec(flags["fell"])),
        "dropped": _Term(lambda e, **k: _Vec(flags["dropped"])),
    }
    w = _watch(env, terms)
    assert w.fired() == []
    flags["dropped"] = True
    assert w.fired() == ["dropped"]
    flags["fell"] = True
    assert sorted(w.fired()) == ["dropped", "fell"]


def test_watch_drops_a_term_it_cannot_evaluate_and_keeps_the_rest():
    def _broken(env, **kw):
        raise RuntimeError("sensor not in this scene")

    w = _watch(_Env(), {"broken": _Term(_broken), "ok": _Term(lambda e, **k: _Vec(True))})
    assert w.fired() == ["ok"]  # the broken one is gone, not silently False
    assert w.dropped == ["broken"]
    assert w.names == ["ok"]


def test_watch_resolves_scene_entity_params_before_use():
    """A nulled termination's SceneEntityCfg is never bound by the TerminationManager."""
    resolved = []

    class _EntityCfg:
        def resolve(self, scene):
            resolved.append(scene)

    cfg = _EntityCfg()
    w = _watch(_Env(), {"t": _Term(lambda e, **k: _Vec(False), {"asset_cfg": cfg, "limit": 1.0})})
    w.fired()
    assert len(resolved) == 1


def test_watch_accepts_plain_bools_as_well_as_tensors():
    w = _watch(_Env(), {"t": _Term(lambda e, **k: True)})
    assert w.fired() == ["t"]


# -- EpisodeRunner -----------------------------------------------------------


def _runner(fired=(), timeout_steps=10, warn_every=0):
    state = {"fired": list(fired)}
    terms = {name: _Term(lambda e, _n=name, **k: _Vec(_n in state["fired"])) for name in ("fell", "dropped")}
    r = er.EpisodeRunner(_watch(_Env(), terms), timeout_steps=timeout_steps, step_dt=0.02, warn_every_steps=warn_every)
    return r, state


def test_idle_never_ends_an_episode_even_when_a_predicate_is_true():
    r, _ = _runner(fired=["fell"])
    assert not r.armed
    for _ in range(50):
        assert r.step(success_fired=False) is None
    assert r.steps == 0  # idle does not consume the budget


def test_idle_warns_once_per_predicate():
    r, _ = _runner(fired=["fell"], warn_every=5)
    for _ in range(40):
        r.step(success_fired=False)
    assert r._warned["fell"] >= 1


def test_arming_starts_the_clock_at_zero():
    r, _ = _runner()
    for _ in range(100):
        r.step(success_fired=False)
    r.arm()
    assert r.armed and r.steps == 0
    r.step(success_fired=False)
    assert r.steps == 1 and r.seconds == pytest.approx(0.02)


def test_success_ends_the_episode_as_success():
    r, _ = _runner()
    r.arm()
    r.step(success_fired=False)
    v = r.step(success_fired=True)
    assert v is not None and v.ok and v.outcome == "success" and v.reason == "success"
    assert v.steps == 2


def test_a_failure_predicate_ends_the_episode_and_names_itself():
    r, state = _runner()
    r.arm()
    assert r.step(success_fired=False) is None
    state["fired"] = ["dropped"]
    v = r.step(success_fired=False)
    assert v is not None and not v.ok
    assert v.outcome == "failure" and v.reason == "dropped"


def test_timeout_fires_exactly_at_the_budget_and_counts_as_failure():
    r, _ = _runner(timeout_steps=5)
    r.arm()
    for _ in range(4):
        assert r.step(success_fired=False) is None
    v = r.step(success_fired=False)
    assert v is not None and v.outcome == "failure" and v.reason == "timeout"
    assert v.steps == 5 and v.seconds == pytest.approx(0.10)


def test_success_beats_a_failure_predicate_on_the_same_step():
    r, state = _runner()
    r.arm()
    state["fired"] = ["fell"]
    v = r.step(success_fired=True)
    assert v.outcome == "success"


def test_timeout_is_counted_from_the_arm_not_the_reset():
    """A long idle wait must not eat the operator's budget."""
    r, _ = _runner(timeout_steps=3)
    for _ in range(500):
        r.step(success_fired=False)
    r.arm()
    assert r.step(success_fired=False) is None
    assert r.step(success_fired=False) is None
    assert r.step(success_fired=False).reason == "timeout"


def test_disarm_returns_to_idle_and_clears_the_clock():
    r, _ = _runner()
    r.arm()
    r.step(success_fired=False)
    r.disarm()
    assert not r.armed and r.steps == 0


def test_remaining_seconds_counts_down():
    r, _ = _runner(timeout_steps=10)
    r.arm()
    assert r.remaining_s == pytest.approx(0.2)
    r.step(success_fired=False)
    assert r.remaining_s == pytest.approx(0.18)


# -- TakeStore ---------------------------------------------------------------


def test_take_lands_under_success_with_a_verdict_and_a_manifest_line(tmp_path):
    store = er.TakeStore(root=str(tmp_path), task_id="FIATLUX-S06-DisposeBulb-Teleop-v0", session="S", operator="yj")
    take = _staged(tmp_path)
    open(os.path.join(take, "run.h5"), "w").close()

    dest = store.seal(take, er.Verdict("success", "success", 271, 5.42), {"score": {"mean_score": 1.0}, "seed": 42})

    assert os.path.basename(os.path.dirname(dest)) == "success"
    assert not os.path.exists(take)  # moved, not copied
    assert os.path.isfile(os.path.join(dest, "run.h5"))
    with open(os.path.join(dest, "verdict.json")) as fh:
        verdict = json.load(fh)
    assert verdict["outcome"] == "success" and verdict["reason"] == "success"
    assert verdict["steps"] == 271 and verdict["score"] == 1.0 and verdict["seed"] == 42
    assert verdict["operator"] == "yj"

    with open(store.manifest) as fh:
        lines = [json.loads(ln) for ln in fh]
    assert len(lines) == 1 and lines[0]["take"] == os.path.basename(dest)


def test_failure_lands_under_failure_and_keeps_the_reason(tmp_path):
    store = er.TakeStore(root=str(tmp_path), task_id="T", session="S")
    dest = store.seal(_staged(tmp_path), er.Verdict("failure", "timeout", 6000, 120.0))
    assert os.path.basename(os.path.dirname(dest)) == "failure"
    with open(os.path.join(dest, "verdict.json")) as fh:
        assert json.load(fh)["reason"] == "timeout"


def test_manifest_appends_across_takes(tmp_path):
    store = er.TakeStore(root=str(tmp_path), task_id="T", session="S")
    store.seal(_staged(tmp_path, "ep000"), er.Verdict("success", "success", 10, 0.2))
    store.seal(_staged(tmp_path, "ep001"), er.Verdict("failure", "robot_fell", 20, 0.4))
    with open(store.manifest) as fh:
        rows = [json.loads(ln) for ln in fh]
    assert [r["outcome"] for r in rows] == ["success", "failure"]
    assert rows[1]["reason"] == "robot_fell"


def test_task_sits_above_session_in_the_tree(tmp_path):
    store = er.TakeStore(root=str(tmp_path), task_id="FIATLUX-S01-MoveLadder-Teleop-v0", session="2026-11-13_090000")
    dest = store.seal(_staged(tmp_path), er.Verdict("success", "success", 1, 0.02))
    rel = os.path.relpath(dest, tmp_path).split(os.sep)
    assert rel[0] == "FIATLUX-S01-MoveLadder-Teleop-v0"
    assert rel[1] == "2026-11-13_090000"
    assert rel[2] == "success"


# -- failure_terms_of --------------------------------------------------------


class _RLTerminations:
    def __init__(self):
        self.time_out = _Term(lambda e, **k: False)
        self.success = _Term(lambda e, **k: False)
        self.fell_below = _Term(lambda e, **k: False, {"minimum_height": 0.5})
        self.payload_lost = _Term(lambda e, **k: False)


class _RLCfg:
    def __init__(self):
        self.terminations = _RLTerminations()


class _TeleopCfg(_RLCfg):
    def __init__(self):
        super().__init__()
        self.terminations = _RLTerminations()
        for name in list(vars(self.terminations)):  # what apply_subtask_teleop does
            if name != "success":
                setattr(self.terminations, name, None)


def test_failure_terms_come_from_the_rl_base_not_the_nulled_teleop_cfg():
    terms = er.failure_terms_of(_TeleopCfg())
    assert sorted(terms) == ["fell_below", "payload_lost"]
    assert terms["fell_below"].params == {"minimum_height": 0.5}


def test_failure_terms_excludes_success_and_time_out():
    terms = er.failure_terms_of(_TeleopCfg())
    assert "success" not in terms and "time_out" not in terms


def test_failure_terms_is_empty_rather_than_raising_for_an_unrecognised_cfg():
    class _Odd:
        pass

    assert er.failure_terms_of(_Odd()) == {}


# -- stateful (class-based) terms --------------------------------------------


class _StatefulTerm:
    """A ManagerTermBase-shaped term: constructed with (cfg, env), called with (env, **params)."""

    instances: list = []

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.hits = 0
        self.resets = 0
        _StatefulTerm.instances.append(self)

    def __call__(self, env, **params):
        self.hits += 1
        return _Vec(self.hits > 2)  # fires only after it has accumulated state

    def reset(self, env_ids=None):
        self.resets += 1
        self.hits = 0


def test_a_class_based_term_is_instantiated_not_called_as_a_class():
    """payload_lost / payload_struck are classes; calling the class raises TypeError."""
    _StatefulTerm.instances.clear()
    w = _watch(_Env(), {"payload_lost": _Term(_StatefulTerm, {"force_threshold": 1.0})})
    assert w.names == ["payload_lost"], "the class-based term must be watched, not dropped"
    assert len(_StatefulTerm.instances) == 1
    assert _StatefulTerm.instances[0].cfg.params == {"force_threshold": 1.0}


def test_a_class_based_term_accumulates_then_fires():
    _StatefulTerm.instances.clear()
    w = _watch(_Env(), {"payload_lost": _Term(_StatefulTerm)})
    assert w.fired() == []  # probe + this call
    assert w.fired() == ["payload_lost"]


def test_arming_clears_stateful_term_history():
    """Otherwise the previous take's sustain counter fires on the next take's first steps."""
    _StatefulTerm.instances.clear()
    r = er.EpisodeRunner(_watch(_Env(), {"payload_lost": _Term(_StatefulTerm)}), timeout_steps=100, step_dt=0.02)
    r.arm()
    for _ in range(5):
        r.step(success_fired=False)
    term = _StatefulTerm.instances[0]
    assert term.hits > 0
    r.arm()
    assert term.resets >= 1 and term.hits == 0


def test_reset_survives_a_term_that_cannot_reset():
    class _NoReset:
        def __init__(self, cfg, env):
            pass

        def __call__(self, env, **params):
            return _Vec(False)

    w = _watch(_Env(), {"t": _Term(_NoReset)})
    w.reset()  # must not raise
    assert w.names == ["t"]
