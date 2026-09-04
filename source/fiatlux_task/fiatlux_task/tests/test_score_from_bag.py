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
