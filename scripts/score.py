# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline scoring for a recorded Fiatlux run.

Scoring is decoupled from the simulator: it reads a trajectory bag produced by
``scripts/record_run.py`` and derives the score from recorded state. The *same*
bag can be re-scored under different thresholds / penalties without re-running the
sim -- that is the point of recording everything.

Scoring model (the task is "seat the bulb without violating constraints"):

- An episode **succeeds** if the bulb was seated at the end (the recorded
  ``success`` termination, using the pos/ori tolerances in ``meta.json``).
- Constraint **violations are penalties**, not separate axes:
  - *broken*  -- peak hand contact force exceeded a fragility threshold.
  - *dropped* -- the bulb fell (recorded drop termination / below min height).
- Control effort and episode length are **not** scored (reported as diagnostics).

Subtask scoring (the benchmark headline) is derived here too, from the bag's recorded
``gate_*`` conjunct columns: ``success_rate`` and ``gate_progress``, combined by
``fiatlux_task.subtask_score``. Those two numbers used to exist only inside a live reward
manager, which meant the headline score could not be reproduced from a recording. The output
is shaped so ``scripts/score_subtasks.py`` can read it directly and roll bags up by difficulty
weight.

Run it with no simulator:
    python scripts/score.py logs/runs/random0
    python scripts/score.py logs/runs/random0 --fragility-threshold 30 --output score.json

    # bags in, weighted benchmark score out -- still no simulator
    python scripts/score.py logs/runs/s06 --output logs/eval/s06.json
    python scripts/score_subtasks.py logs/eval/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ScoreConfig:
    """Tunable scoring rules. Thresholds default from the bag's ``meta.json``."""

    fragility_threshold: float = 50.0  # N; peak hand contact force => "broken"
    broken_penalty: float = 1.0  # subtracted from a seated episode's score
    dropped_penalty: float = 1.0  # subtracted from any episode's score
    min_score: float = 0.0  # floor for a per-episode score
    drop_min_height: float = 0.4  # m; bulb below this counts as dropped
    # Per-bulb floors from the bag (issue #202). A scene can hold two bulbs and only one is
    # the one the leg manipulates, so a single ``bulb_pos`` column scored the wrong one:
    # S03-S06 watched a fresh bulb parked on the bench while the operator dropped the old
    # one. ``None`` for a bulb means exempt -- the task puts that bulb down on purpose.
    drop_floors_by_bulb: dict[str, float | None] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Bag loading (no Isaac/torch dependency)                                      #
# --------------------------------------------------------------------------- #
def load_bag(path: str | Path) -> tuple[list[dict[str, np.ndarray]], dict]:
    """Return ``(episodes, meta)`` from a run directory or bag file."""
    path = Path(path)
    if path.is_dir():
        meta = json.loads((path / "meta.json").read_text())
        bag = path / meta.get("bag_file", "run.h5")
    else:
        bag = path
        meta_path = bag.with_name("meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if bag.suffix == ".h5":
        return _load_hdf5(bag), meta
    if bag.suffix == ".npz":
        return _load_npz(bag), meta
    raise ValueError(f"unknown bag file: {bag}")


def _load_hdf5(path: Path) -> list[dict[str, np.ndarray]]:
    import h5py

    episodes = []
    with h5py.File(path, "r") as f:
        data = f["data"]
        for name in sorted(data, key=lambda n: int(n.split("_")[-1])):
            g = data[name]
            episodes.append({k: g[k][()] for k in g})
    return episodes


def _load_npz(path: Path) -> list[dict[str, np.ndarray]]:
    payload = dict(np.load(path))
    ends = payload.pop("episode_ends")
    starts = np.concatenate([[0], ends[:-1]])
    episodes = []
    for s, e in zip(starts, ends):
        episodes.append({k: v[s:e] for k, v in payload.items()})
    return episodes


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def _peak_contact_force(ep: dict[str, np.ndarray]) -> float:
    """Max contact-force magnitude over steps and hand bodies, BOTH hands (N).

    Reading the right hand alone scored a left-handed crush as clean (issue #89): the operator
    uses whichever hand is convenient, and the 2026-08-21 session drove the old bulb left-handed
    throughout. A missed break inflates ``clean_success_rate``, which is the wrong direction for
    a fragility channel to be wrong in.

    Bags recorded before both hands existed carry no ``contact_force_left`` and fall back to the
    right-hand peak, so their scores do not move.
    """
    peak = 0.0
    for key in ("contact_force", "contact_force_left"):  # (T, B, 3) each
        f = ep.get(key)
        if f is None or not getattr(f, "size", 0):
            continue
        peak = max(peak, float(np.linalg.norm(f, axis=-1).max()))
    return peak


def gate_columns(ep: dict[str, np.ndarray]) -> list[str]:
    """The episode's success-gate conjunct columns, in a stable order."""
    return sorted(k for k in ep if k.startswith("gate_"))


def episode_gate_progress(ep: dict[str, np.ndarray]) -> float | None:
    """Partial credit in [0, 1]: how much of the success gate the episode ever held at once.

    The offline twin of ``mdp.gates.gate_progress``, and computed the same way -- the best number
    of conjuncts simultaneously true, normalized against how many were ALREADY true at the start,
    because conditions like ``robot_standing`` hold at t=0 on every subtask and would otherwise
    hand out free credit.

    One documented difference from the live term: it captures its baseline in the reward manager's
    reset, before the first step, while the bag's first row is after it. ``at_reset`` is the
    DENOMINATOR here (``n - at_reset``), not counted additively, so a conjunct that flips during
    step 0 does not move this by a flat "one conjunct's worth" -- it rescales the whole headroom.
    Example: a 5-conjunct gate with ``at_reset`` 3 vs 4 turns one remaining conjunct from worth
    0.50 into worth 1.00. See issue #239 for a case (S06) where the resulting live/offline
    disagreement is not yet explained by this alone.

    ``None`` when the bag carries no gate columns (recorded before they existed), which is
    reported as missing rather than as a zero.
    """
    cols = gate_columns(ep)
    if not cols or len(np.asarray(ep[cols[0]])) == 0:
        return None
    counts = np.sum([np.asarray(ep[c]).astype(bool).reshape(-1) for c in cols], axis=0)
    n, at_reset = len(cols), float(counts[0])
    if at_reset >= n:
        return 1.0
    return float(np.clip((float(counts.max()) - at_reset) / (n - at_reset), 0.0, 1.0))


def apply_bag_metadata(cfg: ScoreConfig, meta: dict) -> ScoreConfig:
    """Take the thresholds the bag itself recorded, and return ``cfg``.

    A bag knows what its own task counts as a dropped bulb; a caller scoring it with bare
    defaults silently substitutes the family numbers instead. That is not academic -- the
    per-bulb floors (issue #202) live only here, so a caller that skips this scores S06's
    disposal against a height the task never asked for, or misses an old bulb that fell.
    Call it wherever a bag is scored, and apply any explicit overrides AFTER it.
    """
    if "drop_min_height" in meta:
        cfg.drop_min_height = float(meta["drop_min_height"])
    if isinstance(meta.get("drop_min_height_by_bulb"), dict):
        cfg.drop_floors_by_bulb = {
            name: (None if floor is None else float(floor)) for name, floor in meta["drop_min_height_by_bulb"].items()
        }
    if "fragility_threshold" in meta:
        cfg.fragility_threshold = float(meta["fragility_threshold"])
    elif "glass_contact_limit_n" in meta:
        cfg.fragility_threshold = float(meta["glass_contact_limit_n"])
    return cfg


def score_episode(ep: dict[str, np.ndarray], cfg: ScoreConfig) -> dict:
    success = ep.get("success_term")
    seated = bool(success[-1]) if success is not None and getattr(success, "size", 0) else False

    dropped_term = ep.get("dropped_term")
    dropped = bool(dropped_term[-1]) if dropped_term is not None and getattr(dropped_term, "size", 0) else False
    # Every bulb the bag names, each against its own floor. Bags recorded before the per-bulb
    # floors existed carry neither, and fall back to the single ``bulb_pos`` column.
    columns = [(f"{name}_pos", floor) for name, floor in cfg.drop_floors_by_bulb.items()]
    checked = False
    for column, floor in columns:
        if floor is None:  # exempt: the task requires this bulb to end up low
            continue
        pos = ep.get(column)
        if pos is None or not getattr(pos, "size", 0):
            continue
        checked = True
        dropped = dropped or bool(pos[:, 2].min() < floor)
    # Fall back only when the per-bulb pass checked NOTHING -- an older bag with no per-bulb
    # floors, or floors naming columns this bag does not carry. Staying silent there would be a
    # penalty that never fires, which is the bug this metadata exists to fix.
    if not checked and not dropped:
        bulb_pos = ep.get("bulb_pos")
        if bulb_pos is not None and getattr(bulb_pos, "size", 0):
            dropped = bool(bulb_pos[:, 2].min() < cfg.drop_min_height)

    peak_force = _peak_contact_force(ep)
    broken = peak_force > cfg.fragility_threshold

    score = 1.0 if seated else 0.0
    if broken:
        score -= cfg.broken_penalty
    if dropped:
        score -= cfg.dropped_penalty
    score = max(cfg.min_score, score)

    pos_error = ep.get("pos_error")
    min_pos_error = float(pos_error.min()) if pos_error is not None and getattr(pos_error, "size", 0) else 0.0
    done = ep.get("done")
    length = int(done.shape[0]) if done is not None and getattr(done, "size", 0) else 0

    return {
        "seated": seated,
        "broken": broken,
        "dropped": dropped,
        "peak_contact_force": peak_force,
        "min_pos_error": min_pos_error,
        "length": length,
        "score": score,
        "gate_progress": episode_gate_progress(ep),
    }


def score_bag(
    episodes: list[dict[str, np.ndarray]], meta: dict, cfg: ScoreConfig
) -> dict:
    per_ep = [score_episode(ep, cfg) for ep in episodes]
    n = max(len(per_ep), 1)

    def rate(key):
        return sum(1 for e in per_ep if e[key]) / n

    def mean(key):
        return float(np.mean([e[key] for e in per_ep])) if per_ep else 0.0

    clean = sum(
        1 for e in per_ep if e["seated"] and not e["broken"] and not e["dropped"]
    )
    # The gym id keys the difficulty weights; meta's `task` is a cfg class name, which does not.
    task = meta.get("task_id") or meta.get("task")
    progress = [e["gate_progress"] for e in per_ep if e["gate_progress"] is not None]
    results = {
        "task": task,
        "benchmark_version": meta.get("benchmark_version"),
        "policy": meta.get("policy"),
        "seed": meta.get("seed"),
        "episodes": len(per_ep),
        # --- score (headline) ---
        "success_rate": rate("seated"),
        "clean_success_rate": clean / n,
        "mean_score": mean("score"),
        # --- constraint violations ---
        "broken_rate": rate("broken"),
        "dropped_rate": rate("dropped"),
        "peak_contact_force": (
            max((e["peak_contact_force"] for e in per_ep), default=0.0)
        ),
        # --- non-scoring diagnostics ---
        "mean_episode_length": mean("length"),
        "mean_min_pos_error": mean("min_pos_error"),
        "score_config": asdict(cfg),
    }
    # Partial credit, and the weighted subtask score it feeds. A bag with no gate columns says
    # so rather than reporting 0.0 -- "not recorded" and "got nowhere" are different claims.
    if progress:
        results["gate_progress"] = float(np.mean(progress))
        results["gate_conjuncts"] = meta.get("gate_conjuncts")
        subtask = _subtask_score(task, results["success_rate"], results["gate_progress"])
        if subtask is not None:
            results["subtask_score"], results["subtask_weight"] = subtask
    else:
        results["gate_progress"] = None
    return results


def _subtask_score(task: str | None, success_rate: float, gate_progress: float) -> tuple[float, float] | None:
    """``(score, difficulty weight)`` for a subtask id, or ``None`` for anything else."""
    if not task:
        return None
    pkg = str(Path(__file__).resolve().parents[1] / "source" / "fiatlux_task")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    try:
        from fiatlux_task.subtask_score import subtask_score, subtask_weight

        return subtask_score(success_rate, gate_progress), subtask_weight(task)
    except (ImportError, KeyError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Score a recorded Fiatlux run.")
    parser.add_argument("bag", type=str, help="Run directory or bag file (.h5/.npz).")
    parser.add_argument("--fragility-threshold", type=float, default=None)
    parser.add_argument("--broken-penalty", type=float, default=None)
    parser.add_argument("--dropped-penalty", type=float, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output.")
    args = parser.parse_args()

    episodes, meta = load_bag(args.bag)

    cfg = apply_bag_metadata(ScoreConfig(), meta)
    for name, value in [
        ("fragility_threshold", args.fragility_threshold),
        ("broken_penalty", args.broken_penalty),
        ("dropped_penalty", args.dropped_penalty),
        ("min_score", args.min_score),
    ]:
        if value is not None:
            setattr(cfg, name, value)

    results = score_bag(episodes, meta, cfg)
    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"[INFO] wrote {args.output}")


if __name__ == "__main__":
    main()
