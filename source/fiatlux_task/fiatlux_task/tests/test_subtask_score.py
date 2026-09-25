# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guardrails for the weighted subtask score.

Import-light: ``subtask_score`` pulls in nothing from Isaac Lab, so this runs with no
``isaacsim_ci`` marker and no GPU.
"""

import math

import pytest

from fiatlux_task.subtask_score import (
    FACTOR_MULTIPLIERS,
    SUBTASK_FACTORS,
    aggregate,
    base_subtask_id,
    subtask_score,
    subtask_weight,
    subtask_weights,
)

WALK_ONLY = [
    "FIATLUX-S05-CarryBulbToDisposal-v0",
    "FIATLUX-S07-ApproachNewBulb-v0",
    "FIATLUX-S09-CarryBulbToLadder-v0",
]


def _perfect(task_ids):
    return {t: {"success_rate": 1.0, "gate_progress": 1.0} for t in task_ids}


def _zero(task_ids):
    return {t: {"success_rate": 0.0, "gate_progress": 0.0} for t in task_ids}


def test_every_factor_is_a_known_multiplier():
    for task_id, factors in SUBTASK_FACTORS.items():
        for factor in factors:
            assert factor in FACTOR_MULTIPLIERS, (task_id, factor)


def test_only_the_folded_leg_spans_modes():
    """``span`` applies to S01 alone; a second one would be a decomposition change."""
    assert [t for t, f in SUBTASK_FACTORS.items() if "span" in f] == ["FIATLUX-S01-MoveLadder-v0"]


def test_the_bare_walk_is_the_floor():
    assert subtask_weight("FIATLUX-S07-ApproachNewBulb-v0") == pytest.approx(1.0)
    assert SUBTASK_FACTORS["FIATLUX-S07-ApproachNewBulb-v0"] == ()


def test_screwing_in_on_the_ladder_is_the_hardest():
    weights = subtask_weights()
    assert max(weights, key=weights.get) == "FIATLUX-S11-ScrewInBulb-v0"
    assert weights["FIATLUX-S11-ScrewInBulb-v0"] > weights["FIATLUX-S03-RemoveOldBulb-v0"]


def test_release_outranks_grasp():
    """Installation must outrank removal, as the mate tier's two sustain windows already say."""
    assert FACTOR_MULTIPLIERS["release"] > FACTOR_MULTIPLIERS["grasp"]


def test_loaded_ladder_traverse_outranks_every_floor_subtask():
    weights = subtask_weights()
    floor = [
        "FIATLUX-S05-CarryBulbToDisposal-v0",
        "FIATLUX-S07-ApproachNewBulb-v0",
        "FIATLUX-S08-GrabNewBulb-v0",
        "FIATLUX-S09-CarryBulbToLadder-v0",
        "FIATLUX-S06-DisposeBulb-v0",
        "FIATLUX-S01-MoveLadder-v0",
    ]
    assert weights["FIATLUX-S10-ClimbWithBulb-v0"] > max(weights[t] for t in floor)


def test_a_walk_only_policy_scores_far_below_the_unweighted_share():
    """Three of twelve subtasks are "walk to X", so an unweighted mean would pay 25%."""
    summary = aggregate({**_perfect(WALK_ONLY), **_zero(set(SUBTASK_FACTORS) - set(WALK_ONLY))})
    walk_weight = sum(subtask_weight(t) for t in WALK_ONLY)
    assert summary["weighted_score"] == pytest.approx(walk_weight / summary["weight_total"])
    assert summary["weighted_score"] < 0.15


def test_partial_credit_caps_below_success():
    """An episode that touches every condition without holding them together must not tie with
    one that finished."""
    assert subtask_score(success_rate=0.0, gate_progress=1.0) == pytest.approx(0.5)
    assert subtask_score(success_rate=1.0, gate_progress=1.0) == pytest.approx(1.0)
    assert subtask_score(success_rate=0.0, gate_progress=0.0) == pytest.approx(0.0)


def test_partial_credit_is_monotone_in_both_inputs():
    assert subtask_score(0.5, 0.2) < subtask_score(0.5, 0.8)
    assert subtask_score(0.2, 0.5) < subtask_score(0.8, 0.5)


def test_missing_subtasks_are_excluded_not_zeroed():
    """Scoring only the easy subtasks at 100% reports 1.0 over the weight it covered, plus what
    it did not cover."""
    summary = aggregate(_perfect(WALK_ONLY))
    assert summary["weighted_score"] == pytest.approx(1.0)
    assert summary["weight_covered"] == pytest.approx(sum(subtask_weight(t) for t in WALK_ONLY))
    assert summary["weight_covered"] < summary["weight_total"]
    assert len(summary["subtasks_missing"]) == 9


def test_full_marks_scores_one():
    summary = aggregate(_perfect(SUBTASK_FACTORS))
    assert summary["weighted_score"] == pytest.approx(1.0)
    assert summary["subtasks_missing"] == []


def test_weights_are_recomputable_from_the_factor_table():
    """Each weight is the product of its stated factors, not a free-floating number."""
    for task_id, weight in subtask_weights().items():
        expected = math.prod(FACTOR_MULTIPLIERS[f] for f in SUBTASK_FACTORS[task_id])
        assert weight == pytest.approx(expected)


def test_unknown_task_ids_are_rejected():
    """The coarse tier measures a different capability and must not be mixed in."""
    with pytest.raises(KeyError):
        aggregate({"FIATLUX-Replace-v0": {"success_rate": 1.0, "gate_progress": 1.0}})
    with pytest.raises(KeyError):
        subtask_weight("FIATLUX-NotASubtask-v0")


def test_score_subtasks_script_print_weights(capsys):
    """Verify scripts/score_subtasks.py print_weights() runs cleanly and formats factors as products."""
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import score_subtasks

    score_subtasks.print_weights()
    captured = capsys.readouterr().out
    assert "factors: {" in captured
    assert "FIATLUX-S01-MoveLadder-v0" in captured
    assert "carry * span" in captured
    assert "FIATLUX-S11-ScrewInBulb-v0" in captured
    assert "release * balance * mate" in captured
    assert "34.05" in captured


def test_score_subtasks_script_load_results_and_main(tmp_path, monkeypatch, capsys):
    """Verify scripts/score_subtasks.py CLI loads eval JSONs and produces weighted aggregate scores."""
    import json
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import score_subtasks

    # Create mock eval results
    f1 = tmp_path / "s01.json"
    f1.write_text(json.dumps({"task": "FIATLUX-S01-MoveLadder-v0", "success_rate": 1.0, "gate_progress": 1.0}))
    f2 = tmp_path / "s02.json"
    f2.write_text(json.dumps({"task": "FIATLUX-S02-ClimbLadder-v0", "success_rate": 0.5, "gate_progress": 0.8}))

    results = score_subtasks.load_results([f1, f2])
    assert len(results) == 2
    assert results["FIATLUX-S01-MoveLadder-v0"]["success_rate"] == 1.0

    out_json = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["score_subtasks.py", str(tmp_path), "--output", str(out_json)])
    ret = score_subtasks.main()
    assert ret == 0
    assert out_json.exists()
    summary = json.loads(out_json.read_text())
    assert summary["subtasks_scored"] == 2
    assert len(summary["subtasks_missing"]) == 10


def test_score_subtasks_script_load_results_errors(tmp_path):
    """Verify score_subtasks.load_results rejects invalid / duplicate JSON files."""
    import json
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import score_subtasks

    no_task = tmp_path / "no_task.json"
    no_task.write_text(json.dumps({"success_rate": 1.0, "gate_progress": 1.0}))
    with pytest.raises(ValueError, match="no 'task' field"):
        score_subtasks.load_results([no_task])

    no_progress = tmp_path / "no_progress.json"
    no_progress.write_text(json.dumps({"task": "FIATLUX-S01-MoveLadder-v0", "success_rate": 1.0}))
    with pytest.raises(ValueError, match="no partial credit"):
        score_subtasks.load_results([no_progress])

    # A bag with no gate columns writes the key as null. That is as unscoreable as a missing key,
    # and reaching aggregate() with it would raise TypeError on float(None) instead.
    null_progress = tmp_path / "null_progress.json"
    null_progress.write_text(
        json.dumps({"task": "FIATLUX-S01-MoveLadder-v0", "success_rate": 1.0, "gate_progress": None})
    )
    with pytest.raises(ValueError, match="no partial credit"):
        score_subtasks.load_results([null_progress])

    # A teleop take of a subtask is a take of that subtask, and collides with the policy run.
    teleop = tmp_path / "teleop.json"
    teleop.write_text(
        json.dumps({"task": "FIATLUX-S01-MoveLadder-Teleop-v0", "success_rate": 0.5, "gate_progress": 0.5})
    )
    assert score_subtasks.load_results([teleop]) == {
        "FIATLUX-S01-MoveLadder-v0": {"success_rate": 0.5, "gate_progress": 0.5}
    }


def test_training_twin_scores_as_its_subtask():
    assert base_subtask_id("FIATLUX-S11-ScrewInBulb-Training-v0") == "FIATLUX-S11-ScrewInBulb-v0"
    assert subtask_weight("FIATLUX-S11-ScrewInBulb-Training-v0") == subtask_weight("FIATLUX-S11-ScrewInBulb-v0")


def test_none_task_id_handled_gracefully():
    assert base_subtask_id(None) is None
    with pytest.raises(KeyError):
        subtask_weight(None)
