# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Difficulty weights and the aggregate score for the subtask tier.

The twelve subtasks are not equal difficulty, so their success rates are not averaged. Each
subtask's weight is the PRODUCT of the multipliers below, one per property of the task:

    carry     1.5   a payload must be retained while the whole body moves
    grasp     1.4   a grip must be acquired on an object at rest
    release   1.8   the hand comes off and the object has to stay where it was put
    balance   2.0   performed on the ladder
    traverse  1.5   climbs or descends the ladder rather than working from a fixed stance
    mate      1.5   bulb/socket work at the fixture, under a fragility bound
    span      1.5   the episode crosses mode boundaries instead of staying in one

``release`` outranks ``grasp`` for the same reason the mate tier gives them different sustain
windows (1.0 s vs 0.5 s): proving something stays put unaided is a stronger claim than proving a
grasp was not a glancing contact. ``span`` applies only to S01, where approach, grasp, transport
and place are one episode.

A subtask's score combines its success rate with the mean of its ``gate_progress`` channel
(``mdp.gates``), which carries the best fraction of the success gate's conjuncts an episode ever
satisfied at once:

    SUCCESS_SHARE * success_rate + (1 - SUCCESS_SHARE) * mean_gate_progress

At ``SUCCESS_SHARE`` = 0.5 an episode that touches every condition without ever holding them
together caps at 0.5; only the gate firing reaches 1.0.

Scores may not be compared across layout seeds, and must not be mixed with the coarse tier
(Carry / Climb / Descend / Replace), which measures a different capability.
"""

from __future__ import annotations

import math

FACTOR_MULTIPLIERS: dict[str, float] = {
    "carry": 1.5,
    "grasp": 1.4,
    "release": 1.8,
    "balance": 2.0,
    "traverse": 1.5,
    "mate": 1.5,
    "span": 1.5,
}

SUBTASK_FACTORS: dict[str, tuple[str, ...]] = {
    "FIATLUX-S01-MoveLadder-v0": ("carry", "span"),
    "FIATLUX-S02-ClimbLadder-v0": ("balance", "traverse"),
    "FIATLUX-S03-RemoveOldBulb-v0": ("grasp", "balance", "mate"),
    "FIATLUX-S04-DescendWithBulb-v0": ("carry", "balance", "traverse"),
    "FIATLUX-S05-CarryBulbToDisposal-v0": ("carry",),
    "FIATLUX-S06-DisposeBulb-v0": ("release",),
    "FIATLUX-S07-ApproachNewBulb-v0": (),
    "FIATLUX-S08-GrabNewBulb-v0": ("grasp",),
    "FIATLUX-S09-CarryBulbToLadder-v0": ("carry",),
    "FIATLUX-S10-ClimbWithBulb-v0": ("carry", "balance", "traverse"),
    "FIATLUX-S11-ScrewInBulb-v0": ("release", "balance", "mate"),
    "FIATLUX-S12-ClimbDown-v0": ("balance", "traverse"),
}

SUCCESS_SHARE = 0.5
TELEOP_SUFFIX = "-Teleop-v0"


def base_subtask_id(task_id: str | None) -> str | None:
    """The benchmark id a teleop twin belongs to (``...-Teleop-v0`` -> ``...-v0``).

    ``fiatlux_teleop`` registers one twin per subtask, and a teleop bag records that id. The
    weights are keyed on the benchmark ids alone, so an operator's take of S06 is still S06.
    """
    if not task_id or not isinstance(task_id, str):
        return task_id
    if task_id.endswith(TELEOP_SUFFIX):
        return task_id[: -len(TELEOP_SUFFIX)] + "-v0"
    return task_id


def subtask_weight(task_id: str | None) -> float:
    """The subtask's difficulty weight: the product of its factor multipliers.

    A subtask with no factors weighs 1.0 -- the floor, a bare walk to a target.
    """
    base = base_subtask_id(task_id)
    if base is None or base not in SUBTASK_FACTORS:
        raise KeyError(f"{task_id} is not a subtask; the coarse tier is not weighted here")
    factors = SUBTASK_FACTORS[base]
    return math.prod(FACTOR_MULTIPLIERS[f] for f in factors)


def subtask_weights() -> dict[str, float]:
    """Every subtask's weight, in chain order."""
    return {task_id: subtask_weight(task_id) for task_id in SUBTASK_FACTORS}


def subtask_score(success_rate: float, gate_progress: float, success_share: float = SUCCESS_SHARE) -> float:
    """One subtask's score in [0, 1] from its success rate and its mean partial credit."""
    return success_share * success_rate + (1.0 - success_share) * gate_progress


def aggregate(results: dict[str, dict[str, float]], success_share: float = SUCCESS_SHARE) -> dict:
    """Roll per-subtask results up into the weighted benchmark score.

    Args:
        results: ``{task_id: {"success_rate": float, "gate_progress": float}}``.
        success_share: share of a subtask's score carried by finishing rather than getting there.

    A subtask absent from ``results`` is reported missing and excluded from the denominator,
    never scored zero: "did not run" and "ran and failed" are different claims.
    """
    canonical: dict[str, dict[str, float]] = {}
    for task_id, entry in results.items():
        base = base_subtask_id(task_id)
        if base is None or base in canonical:
            raise KeyError(f"{base} appears twice (as {task_id}); scoring it once is ambiguous")
        canonical[base] = entry
    results = canonical

    unknown = sorted(set(results) - set(SUBTASK_FACTORS))
    if unknown:
        raise KeyError(f"not subtask ids: {unknown}")

    per_task: dict[str, dict[str, float]] = {}
    weighted_sum = 0.0
    weight_total = 0.0
    for task_id in SUBTASK_FACTORS:
        if task_id not in results:
            continue
        weight = subtask_weight(task_id)
        entry = results[task_id]
        success_rate = float(entry.get("success_rate", 0.0))
        progress = float(entry.get("gate_progress", 0.0))
        score = subtask_score(success_rate, progress, success_share)
        per_task[task_id] = {
            "weight": weight,
            "success_rate": success_rate,
            "gate_progress": progress,
            "score": score,
            "weighted_score": score * weight,
        }
        weighted_sum += score * weight
        weight_total += weight

    return {
        "weighted_score": weighted_sum / weight_total if weight_total else 0.0,
        "success_share": success_share,
        "weight_covered": weight_total,
        "weight_total": sum(subtask_weights().values()),
        "subtasks_scored": len(per_task),
        "subtasks_missing": [t for t in SUBTASK_FACTORS if t not in results],
        "per_subtask": per_task,
    }
