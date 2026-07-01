# Scoring & Evaluation Protocol

Evaluation is a single command, `scripts/eval.py`, run from a fixed seed for a
fixed number of episodes. Same `--task`, `--seed`, `--policy` (and checkpoint)
→ same numbers.

```bash
python scripts/eval.py --task FIATLUX-Insert-v0 --policy random --episodes 20 --seed 0
python scripts/eval.py --task FIATLUX-Insert-v0 --policy rsl_rl --checkpoint <model.pt> \
    --episodes 50 --seed 0 --output results.json
```

## Metrics

| Metric | Meaning | Primary? |
| --- | --- | --- |
| `success_rate` | fraction of episodes ending with the bulb seated | ✅ headline |
| `mean_episode_length` | avg steps per episode | |
| `mean_final_pos_error` | avg bulb→socket distance at episode end (m) | |
| `mean_control_effort` | avg Σ(action²) per step | |
| `peak_contact_force` | max net hand contact force (N) | safety |

The benchmark is **multi-dimensional on purpose**: report success rate alongside
control effort and peak contact force — a policy that succeeds by slamming the
bulb in is not a good policy.

## Reporting convention

- Default protocol: `--episodes 50 --seed 0`.
- Report all five metrics, the policy type, and the checkpoint.
- For learned policies, also report seeds `0,1,2` and their mean ± std.

## Baselines

- `zero` — no action (sanity floor).
- `random` — uniform random actions (sanity floor).
- `rsl_rl` — a trained PPO checkpoint (`scripts/rsl_rl/train.py`).

## Offline scoring (`scripts/score.py`)

`eval.py` computes metrics live during a rollout. For repeatable, no-simulator
scoring instead, record a run once and score the bag as many times as needed
under different rules:

```bash
python scripts/record_run.py --task FIATLUX-Insert-v0 --policy random \
    --episodes 20 --record bag --headless --enable_cameras --out logs/runs/random0
python scripts/score.py logs/runs/random0
python scripts/score.py logs/runs/random0 --fragility-threshold 30 --output score.json
```

The task is framed as "seat the bulb without violating constraints" — success
is necessary but not sufficient; a policy that seats the bulb by slamming it in
or drops it mid-transit is still penalized.

**Per-episode scoring** (`score_episode` in `score.py`):
- **`seated`** — the bulb was seated at episode end (the recorded `success`
  termination flag).
- **`broken`** — the peak hand contact force over the episode exceeded the
  **fragility threshold** (`--fragility-threshold`, default `50.0` N; the bulb
  is modeled as fragile, so contact force is a safety proxy for "handled too
  roughly," not just a task-completion signal).
- **`dropped`** — the bulb fell (the recorded `dropped` termination flag, or its
  height dropped below `drop_min_height`, default `0.4` m — this default is
  overridden by the bag's own `meta.json` if present, so it stays consistent
  with whatever threshold the env used at record time).
- **`score`** — starts at `1.0` if seated else `0.0`, then subtracts
  `--broken-penalty` (default `1.0`) if broken and `--dropped-penalty`
  (default `1.0`) if dropped, floored at `--min-score` (default `0.0`). With
  the defaults, any violation zeroes out an otherwise-successful episode's
  score — violations are penalties on top of success, not a separate axis.

**Aggregate metrics** (`score_bag`):
- **`success_rate`** — fraction of episodes with `seated = True`, regardless
  of whether they also violated a constraint.
- **`clean_success_rate`** — fraction of episodes that were seated *and*
  neither broken nor dropped. This is the stricter, safety-aware headline
  number: `success_rate` alone can hide a policy that succeeds but frequently
  also breaks or drops the bulb along the way.
- **`mean_score`** — mean of the per-episode `score` above.
- **`broken_rate` / `dropped_rate`** — fraction of episodes hitting each
  violation independently (not mutually exclusive, and not mutually exclusive
  with `seated`).
- **`peak_contact_force`** — max peak contact force across all episodes in the
  bag (diagnostic, not itself scored).
- `mean_episode_length` / `mean_min_pos_error` are non-scoring diagnostics.

All thresholds/penalties are `ScoreConfig` fields, overridable via CLI flags
(`--fragility-threshold`, `--broken-penalty`, `--dropped-penalty`,
`--min-score`) — the same bag can be re-scored under stricter or looser rules
without re-running the simulator.
