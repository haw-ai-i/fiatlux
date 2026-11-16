# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Side-by-side subtask table for several sweep output trees (``<root>/<task>/seed<N>/score.json``).

Each ``label=path`` column is one policy's sweep (``scripts/psi0/sweep.sh`` layout). Prints the mean
``subtask_score`` per subtask (with the seed count) and one weighted roll-up per seed through
``fiatlux_task.subtask_score.aggregate`` -- the tables in docs/psi0_baseline.md and
docs/psi0_finetune.md.

    python scripts/psi0/compare_baselines.py zero=logs/runs/zero_dex3 psi0=logs/runs/psi0_zeroshot \
        psi0_ft=logs/runs/psi0_ft_s06
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "fiatlux_task"))

from fiatlux_task.subtask_score import SUBTASK_FACTORS, aggregate, subtask_weight  # noqa: E402


def load(root: Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for f in sorted(root.glob("*/seed*/score.json")):
        blob = json.loads(f.read_text())
        runs.setdefault(blob["task"], {})[int(f.parent.name.removeprefix("seed"))] = blob
    return runs


def main() -> None:
    cols = [arg.split("=", 1) for arg in sys.argv[1:]]
    if not cols or any(len(c) != 2 for c in cols):
        raise SystemExit(__doc__)
    data = {label: load(Path(path)) for label, path in cols}
    print(f"{'subtask':36} {'weight':>6} " + " ".join(f"{label:>15}" for label, _ in cols))
    for task in SUBTASK_FACTORS:
        cells = []
        for label, _ in cols:
            runs = data[label].get(task, {})
            if runs:
                mean = statistics.mean(r["subtask_score"] for r in runs.values())
                cells.append(f"{mean:>8.3f} (n={len(runs)})")
            else:
                cells.append(f"{'-':>15}")
        print(f"{task:36} {subtask_weight(task):>6.2f} " + " ".join(cells))
    print()
    for label, _ in cols:
        seeds = sorted({s for runs in data[label].values() for s in runs})
        totals = []
        for seed in seeds:
            res = {t: runs[seed] for t, runs in data[label].items() if seed in runs}
            totals.append((seed, aggregate(res)["weighted_score"], len(res)))
        txt = ", ".join(f"seed{s}={w:.3f} ({n} subtasks)" for s, w, n in totals)
        mean = statistics.mean(w for _, w, _ in totals) if totals else float("nan")
        std = statistics.pstdev([w for _, w, _ in totals]) if len(totals) > 1 else 0.0
        print(f"{label:10} weighted: {txt}  -> {mean:.3f} +- {std:.3f}")


if __name__ == "__main__":
    main()
