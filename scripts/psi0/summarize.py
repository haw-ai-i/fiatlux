# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize a ``scripts/psi0/sweep.sh`` output tree: ``<out>/<task>/seed<N>/score.json``.

Prints per-subtask mean +- std over seeds (subtask_score, success_rate, gate_progress, episode
length) and one weighted roll-up per seed through ``fiatlux_task.subtask_score.aggregate`` --
the same per-seed-then-mean convention the zero/random/groot comparison tables use. A seed with
a missing subtask is rolled up over the subtasks it has and says so.

    python scripts/psi0/summarize.py logs/runs/psi0_zeroshot [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "fiatlux_task"))

from fiatlux_task.subtask_score import SUBTASK_FACTORS, aggregate  # noqa: E402


def _mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.3f}+-{std:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="sweep output directory")
    parser.add_argument("--json", type=Path, default=None, help="also write the summary here")
    args = parser.parse_args()

    by_task: dict[str, dict[int, dict]] = defaultdict(dict)
    for f in sorted(args.root.glob("*/seed*/score.json")):
        blob = json.loads(f.read_text())
        by_task[blob["task"]][int(f.parent.name.removeprefix("seed"))] = blob

    print(f"{'subtask':36} {'n':>2} {'subtask_score':>14} {'success':>12} {'gate_progress':>14} {'ep_len':>10}")
    table = {}
    for task in SUBTASK_FACTORS:
        runs = by_task.get(task, {})
        if not runs:
            print(f"{task:36} {0:>2} {'(not run)':>14}")
            continue
        cols = {k: [r[k] for r in runs.values()] for k in ("subtask_score", "success_rate", "gate_progress")}
        lengths = [r["mean_episode_length"] for r in runs.values()]
        table[task] = {"seeds": sorted(runs), **{k: v for k, v in cols.items()}, "episode_length": lengths}
        print(
            f"{task:36} {len(runs):>2} {_mean_std(cols['subtask_score']):>14} {_mean_std(cols['success_rate']):>12} "
            f"{_mean_std(cols['gate_progress']):>14} {statistics.mean(lengths):>10.0f}"
        )

    seeds = sorted({s for runs in by_task.values() for s in runs})
    totals = {}
    print()
    for seed in seeds:
        results = {t: runs[seed] for t, runs in by_task.items() if seed in runs}
        agg = aggregate(results)
        totals[seed] = agg["weighted_score"]
        missing = [t for t in SUBTASK_FACTORS if t not in results]
        note = f" (missing {len(missing)}: {', '.join(m.split('-')[1] for m in missing)})" if missing else ""
        print(f"seed{seed}: weighted_score={agg['weighted_score']:.4f} over {len(results)} subtasks{note}")
    if totals:
        print(f"weighted_score mean over seeds: {_mean_std(list(totals.values()))}")

    if args.json is not None:
        args.json.write_text(json.dumps({"per_subtask": table, "per_seed_weighted": totals}, indent=2))


if __name__ == "__main__":
    main()
