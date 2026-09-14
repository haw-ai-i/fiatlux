# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guardrails for deriving the benchmark score from a recorded bag.

Import-light: ``scripts/score.py`` deliberately depends on nothing from Isaac Lab, which is what
makes offline re-scoring possible at all. This runs with no ``isaacsim_ci`` marker and no GPU.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCORE_PY = Path(__file__).resolve().parents[4] / "scripts" / "score.py"


@pytest.fixture(scope="module")
def score():
    spec = importlib.util.spec_from_file_location("_fiatlux_score_test", _SCORE_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _episode(**cols) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=bool) for k, v in cols.items()}


def test_no_gate_columns_reports_missing_not_zero(score):
    """ "Not recorded" and "got nowhere" are different claims, so a bag without the columns
    must not silently score 0.0."""
    assert score.episode_gate_progress({"bulb_pos": np.zeros((3, 3))}) is None


def test_firing_the_whole_gate_scores_one(score):
    ep = _episode(gate_a=[False, True], gate_b=[False, True], gate_c=[False, True])
    assert score.episode_gate_progress(ep) == 1.0


def test_credit_is_normalized_against_what_held_at_the_start(score):
    """Conjuncts like robot_standing are true at t=0 on every subtask. Counting them would hand
    out free credit, so the baseline is subtracted from both sides."""
    ep = _episode(gate_a=[True, True], gate_b=[False, True], gate_c=[False, False])
    # one of the two conjuncts that were not already satisfied ever came true
    assert score.episode_gate_progress(ep) == pytest.approx(0.5)


def test_no_progress_at_all_scores_zero(score):
    ep = _episode(gate_a=[True, True], gate_b=[False, False])
    assert score.episode_gate_progress(ep) == 0.0


def test_credit_is_best_ever_held_at_once_not_a_final_reading(score):
    """The channel measures how far the episode got; letting go at the end does not erase it."""
    ep = _episode(gate_a=[False, True, False], gate_b=[False, True, False])
    assert score.episode_gate_progress(ep) == 1.0


def test_conjuncts_must_hold_simultaneously(score):
    """The gate is a conjunction, so credit counts conditions held AT ONCE. Two conditions that
    each came true on their own turn, never together, is half a gate -- not a whole one."""
    apart = _episode(gate_a=[False, True, False], gate_b=[False, False, True])
    together = _episode(gate_a=[False, True, False], gate_b=[False, True, True])
    assert score.episode_gate_progress(apart) == pytest.approx(0.5)
    assert score.episode_gate_progress(together) == 1.0


def test_a_start_state_that_already_satisfies_everything_scores_one(score):
    ep = _episode(gate_a=[True], gate_b=[True])
    assert score.episode_gate_progress(ep) == 1.0


def test_bag_score_carries_the_gym_id_and_the_weighted_subtask_score(score):
    """score_subtasks.py keys difficulty weights on the gym id and requires gate_progress, so a
    bag's own score JSON has to be readable by it directly."""
    task = "FIATLUX-S06-DisposeBulb-v0"
    episodes = [
        {
            "success_term": np.array([False, True]),
            "gate_a": np.array([False, True]),
            "gate_b": np.array([False, True]),
        }
    ]
    out = score.score_bag(episodes, {"task_id": task, "task": "S06DisposeBulbEnvCfg"}, score.ScoreConfig())
    assert out["task"] == task
    assert out["success_rate"] == 1.0
    assert out["gate_progress"] == 1.0
    assert out["subtask_score"] == 1.0
    assert out["subtask_weight"] == pytest.approx(1.8)


def test_an_empty_episode_reports_missing_not_zero(score):
    """A bag can carry the gate columns and an episode with no rows in them; reading a baseline
    off row 0 there is an IndexError, and reporting 0.0 would be a claim about nothing."""
    assert score.episode_gate_progress({"gate_a": np.zeros((0,), dtype=bool)}) is None


def test_multi_dimensional_gate_columns_are_flattened(score):
    """Columns arrive from the bag with whatever singleton dims the writer left on them."""
    ep = {"gate_a": np.array([[False], [True]]), "gate_b": np.array([[False], [True]])}
    assert score.episode_gate_progress(ep) == 1.0


def test_a_teleop_bag_scores_as_the_subtask_it_is_a_take_of(score):
    """fiatlux_teleop registers a twin per subtask and a teleop bag records the twin's id, which
    the difficulty weights are not keyed on."""
    episodes = [{"success_term": np.array([True]), "gate_a": np.array([False, True])}]
    out = score.score_bag(episodes, {"task_id": "FIATLUX-S06-DisposeBulb-Teleop-v0"}, score.ScoreConfig())
    assert out["subtask_score"] == 1.0
    assert out["subtask_weight"] == pytest.approx(1.8)


def test_scoring_two_bags_does_not_grow_sys_path(score):
    """score.py resolves fiatlux_task by path; a batch driver scoring many bags in one process
    must not append a duplicate entry per bag."""
    meta = {"task_id": "FIATLUX-S06-DisposeBulb-v0"}
    episodes = [{"success_term": np.array([True]), "gate_a": np.array([True])}]
    score.score_bag(episodes, meta, score.ScoreConfig())
    before = list(sys.path)
    score.score_bag(episodes, meta, score.ScoreConfig())
    assert sys.path == before


def test_a_bag_from_a_non_subtask_gets_no_subtask_score(score):
    episodes = [{"success_term": np.array([True]), "gate_a": np.array([True])}]
    out = score.score_bag(episodes, {"task_id": "FIATLUX-Replace-v0"}, score.ScoreConfig())
    assert "subtask_score" not in out
    assert out["gate_progress"] == 1.0


def test_a_bag_with_no_task_id_or_empty_meta_scores_cleanly(score):
    """A bag with no task_id or an empty meta dict must score cleanly and not raise AttributeError."""
    episodes = [{"success_term": np.array([True]), "gate_a": np.array([True])}]
    out_empty = score.score_bag(episodes, {}, score.ScoreConfig())
    assert out_empty["task"] is None
    assert "subtask_score" not in out_empty
    assert out_empty["gate_progress"] == 1.0

    out_none = score.score_bag(episodes, {"task_id": None}, score.ScoreConfig())
    assert out_none["task"] is None
    assert "subtask_score" not in out_none
    assert out_none["gate_progress"] == 1.0



# --------------------------------------------------------------------------- #
# Issue #202: the penalty channels followed ONE bulb, and in S03-S06 it was the
# wrong one -- a fresh bulb parked on the bench while the operator manipulated
# the old one. Dropping is therefore scored per bulb, against a floor the task
# sets, because the height that means "dropped" depends on where the task wants
# that bulb to END: S05 carries the old bulb and losing it is a fault, while S06
# disposes of the same bulb and a success leaves it on the crate floor.
# --------------------------------------------------------------------------- #


def _two_bulb_episode(*, fresh_z, old_z, seated=True, steps=20):
    """A bag-shaped episode carrying both bulbs' position columns."""

    def column(z):
        pos = np.zeros((steps, 3), dtype=np.float32)
        pos[:, 2] = np.linspace(z[0], z[1], steps) if isinstance(z, tuple) else z
        return pos

    return {
        "success_term": np.full((steps,), seated, dtype=bool),
        "fresh_bulb_pos": column(fresh_z),
        "old_bulb_pos": column(old_z),
        # ``bulb_pos`` is the legacy single column; in S03-S06 it aliases the FRESH bulb,
        # which is exactly why it could not see the old one fall.
        "bulb_pos": column(fresh_z),
    }


def test_old_bulb_falling_is_penalised_even_though_the_legacy_column_is_clean(score):
    """The reported bug: S03 drops the old bulb, ``bulb_pos`` never moves, score says clean."""
    ep = _two_bulb_episode(fresh_z=0.958, old_z=(2.1, 0.005))
    floors = {"fresh_bulb": 0.4, "old_bulb": 0.15}

    legacy = score.score_episode(ep, score.ScoreConfig(drop_min_height=0.15))
    assert not legacy["dropped"], "precondition: the single-column rule misses it"

    scored = score.score_episode(ep, score.ScoreConfig(drop_floors_by_bulb=floors))
    assert scored["dropped"], "a dropped old bulb must register"
    assert scored["score"] < legacy["score"]


def test_disposing_of_the_old_bulb_is_not_a_drop(score):
    """S06 puts the old bulb on the crate floor on purpose; ``None`` means exempt."""
    ep = _two_bulb_episode(fresh_z=0.958, old_z=(0.82, -0.006))
    floors = {"fresh_bulb": 0.4, "old_bulb": None}

    scored = score.score_episode(ep, score.ScoreConfig(drop_floors_by_bulb=floors))
    assert not scored["dropped"], "a successful disposal is not a dropped bulb"
    assert scored["score"] == 1.0


def test_a_carried_old_bulb_still_has_a_floor(score):
    """S05 carries the same bulb, so there the fall IS the fault."""
    floors = {"fresh_bulb": 0.4, "old_bulb": 0.4}
    cfg = score.ScoreConfig(drop_floors_by_bulb=floors)
    assert score.score_episode(_two_bulb_episode(fresh_z=0.958, old_z=(0.97, 0.26)), cfg)["dropped"]
    assert not score.score_episode(_two_bulb_episode(fresh_z=0.958, old_z=0.95), cfg)["dropped"]


def test_either_bulb_can_trip_the_penalty(score):
    """Both are scored, not whichever one the recorder happened to guess."""
    floors = {"fresh_bulb": 0.4, "old_bulb": 0.15}
    cfg = score.ScoreConfig(drop_floors_by_bulb=floors)
    assert score.score_episode(_two_bulb_episode(fresh_z=(1.0, 0.02), old_z=2.1), cfg)["dropped"]
    assert score.score_episode(_two_bulb_episode(fresh_z=0.958, old_z=(2.1, 0.02)), cfg)["dropped"]


def test_bags_recorded_before_the_per_bulb_floors_still_score(score):
    """No floors in the metadata: fall back to the single column rather than scoring nothing."""
    ep = _two_bulb_episode(fresh_z=(1.0, 0.02), old_z=2.1)
    assert score.score_episode(ep, score.ScoreConfig(drop_min_height=0.4))["dropped"]


def test_floors_naming_a_column_the_bag_lacks_fall_back_instead_of_passing_silently(score):
    """A penalty that cannot be evaluated must not read as 'no penalty' -- that is this bug."""
    ep = _two_bulb_episode(fresh_z=(1.0, 0.02), old_z=2.1)
    del ep["fresh_bulb_pos"]
    del ep["old_bulb_pos"]
    cfg = score.ScoreConfig(drop_floors_by_bulb={"fresh_bulb": 0.4}, drop_min_height=0.4)
    scored = score.score_episode(ep, cfg)
    assert scored["dropped"], "fell back to bulb_pos rather than silently scoring clean"


def test_an_exempt_bulb_does_not_suppress_the_other_one(score):
    """S11 shape: old bulb binned and exempt, fresh bulb dropped -- still a penalty."""
    ep = _two_bulb_episode(fresh_z=(2.1, 0.03), old_z=0.016)
    floors = {"fresh_bulb": 0.4, "old_bulb": None}
    assert score.score_episode(ep, score.ScoreConfig(drop_floors_by_bulb=floors))["dropped"]
