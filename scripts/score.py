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

Run it with no simulator:
    python scripts/score.py logs/runs/random0
    python scripts/score.py logs/runs/random0 --fragility-threshold 30 --output score.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
    """Max contact-force magnitude over steps and hand bodies (N)."""
    f = ep.get("contact_force")  # (T, B, 3)
    if f is None or not getattr(f, "size", 0):
        return 0.0
    return float(np.linalg.norm(f, axis=-1).max())


def score_episode(ep: dict[str, np.ndarray], cfg: ScoreConfig) -> dict:
    success = ep.get("success_term")
    seated = bool(success[-1]) if success is not None and getattr(success, "size", 0) else False

    dropped_term = ep.get("dropped_term")
    dropped = bool(dropped_term[-1]) if dropped_term is not None and getattr(dropped_term, "size", 0) else False
    bulb_pos = ep.get("bulb_pos")
    if not dropped and bulb_pos is not None and getattr(bulb_pos, "size", 0):
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
    return {
        "task": meta.get("task"),
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

    cfg = ScoreConfig()
    # Thresholds default from the bag's metadata when present.
    if "drop_min_height" in meta:
        cfg.drop_min_height = float(meta["drop_min_height"])
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
