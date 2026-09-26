# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll per-subtask ``score.py`` results up into the weighted benchmark score.

Reads the JSON ``score.py --output`` writes for each subtask and applies ``fiatlux_task.subtask_score``'s
difficulty weights and partial credit. No simulator, so re-weighting a set of results is free.

A subtask with no result is reported missing, not zero.

Examples
--------
    # one JSON per subtask, named however you like; the task id is read from inside
    python scripts/score_subtasks.py logs/eval/*.json

    # a directory of them
    python scripts/score_subtasks.py logs/eval/

    # just show the weight table
    python scripts/score_subtasks.py --weights
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "fiatlux_task"))

from fiatlux_task.subtask_score import (  # noqa: E402
    FACTOR_MULTIPLIERS,
    SUBTASK_FACTORS,
    SUCCESS_SHARE,
    aggregate,
    base_subtask_id,
    subtask_weights,
)


def load_results(paths: list[Path]) -> dict[str, dict[str, float]]:
    """Read ``score.py`` JSONs, keyed by the ``task`` each one records."""
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])

    results: dict[str, dict[str, float]] = {}
    for f in files:
        blob = json.loads(f.read_text())
        task = blob.get("task")
        if task is None:
            raise ValueError(f"{f} has no 'task' field; it is not a score.py result")
        # A teleop take of a subtask is a take of that subtask, and records the twin's id.
        task = base_subtask_id(task)
        if task in results:
            raise ValueError(f"{task} appears twice (second: {f}); scoring it once is ambiguous")
        # A result recorded before partial credit existed carries only the raw per-horizon value,
        # which is not comparable across subtasks. A bag recorded after, but with no gate columns
        # in it, writes the key as null -- also unscoreable, and it has to be caught here too.
        if blob.get("gate_progress") is None:
            raise ValueError(f"{f} ({task}) has no partial credit -- re-record the bag and re-run score.py")
        results[task] = {"success_rate": blob["success_rate"], "gate_progress": blob["gate_progress"]}
    return results


def print_weights() -> None:
    print(f"factors: {FACTOR_MULTIPLIERS}\n")
    print(f"{'subtask':<38} {'weight':>6}  factors")
    for task_id, weight in subtask_weights().items():
        print(f"{task_id:<38} {weight:>6}  {' * '.join(SUBTASK_FACTORS[task_id])}")
    print(f"\n{'total':<38} {sum(subtask_weights().values()):>6}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Weighted subtask benchmark score.")
    parser.add_argument("results", nargs="*", type=Path, help="score.py JSON files, or directories of them.")
    parser.add_argument("--weights", action="store_true", help="Print the weight table and exit.")
    parser.add_argument(
        "--success-share",
        type=float,
        default=SUCCESS_SHARE,
        help=f"Share of a subtask's score carried by finishing vs getting there (default {SUCCESS_SHARE}).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write the full result as JSON here.")
    args = parser.parse_args()

    if args.weights or not args.results:
        print_weights()
        return 0

    summary = aggregate(load_results(args.results), success_share=args.success_share)

    print(f"{'subtask':<38} {'w':>2} {'success':>8} {'partial':>8} {'score':>7} {'w*score':>8}")
    for task_id, row in summary["per_subtask"].items():
        print(
            f"{task_id:<38} {row['weight']:>2} {row['success_rate']:>8.3f} "
            f"{row['gate_progress']:>8.3f} {row['score']:>7.3f} {row['weighted_score']:>8.3f}"
        )
    for task_id in summary["subtasks_missing"]:
        print(f"{task_id:<38} {'--':>2} {'MISSING':>8}")

    print(
        f"\nweighted score: {summary['weighted_score']:.4f}"
        f"   (success share {summary['success_share']},"
        f" covering {summary['weight_covered']}/{summary['weight_total']} weight,"
        f" {summary['subtasks_scored']}/{len(SUBTASK_FACTORS)} subtasks)"
    )
    if summary["subtasks_missing"]:
        print("NOTE: this is a partial evaluation -- missing subtasks are excluded, not scored zero.")

    if args.output:
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
