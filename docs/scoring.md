# Scoring & Evaluation Protocol

Two things are scored, on two different models. **`FIATLUX-Replace-v0`** (the full
replacement; see `docs/task_spec.md`) is scored by the metrics below. The **twelve subtasks**
are scored by `subtask_score` — see [Subtask score](#subtask-score--the-benchmark-headline)
— and are also evaluable with the tooling below.

Evaluation is two commands, `scripts/record_run.py` then
`scripts/score.py` (see "Offline scoring" below), run from a fixed seed for a
fixed number of episodes. Same `--task`, `--seed`, `--policy` (and checkpoint)
→ same numbers.

```bash
python scripts/record_run.py --task FIATLUX-Replace-v0 --policy basic_standard \
    --episodes 20 --seed 0 --record bag --enable_cameras --out logs/runs/basic_standard0
python scripts/score.py logs/runs/basic_standard0
python scripts/record_run.py --task FIATLUX-Replace-v0 --policy rsl_rl --checkpoint <model.pt> \
    --episodes 50 --seed 0 --record bag --enable_cameras --out logs/runs/rsl_rl0
python scripts/score.py logs/runs/rsl_rl0 --output results.json
```

## Metrics

| Metric | Meaning | Primary? |
| --- | --- | --- |
| `success_rate` | fraction of episodes ending in the task's `success` termination | ✅ headline |
| `score_breakdown` | per-term episode means from the env's reward/termination managers | ✅ the breakdown |
| `mean_episode_length` | avg steps per episode | |
| `mean_control_effort` | avg Σ(action²) per step | |
| `peak_contact_force` | max net hand contact force (N) | safety |

`score_breakdown` reports every named reward term (`Episode_Reward/<term>`: the
episodic sum averaged per second of episode time) and termination term
(`Episode_Termination/<term>`: the fraction of episodes that term ended). For
`FIATLUX-Replace-v0` this is the benchmark's score breakdown by construction —
dense normalized progress per subgoal, sparse completions, penalties, and full
success are all separate named channels, so partial progress maps onto the
subtasks without making the subtasks separate benchmark targets. The dense terms
use per-episode normalized progress (`(d0 − d) / d0`, clamped to [0, 1]), so
randomized spawn distances do not skew scores.

The benchmark is **multi-dimensional on purpose**: report success rate alongside
the breakdown, control effort, and peak contact force — a policy that succeeds
by slamming the bulb in is not a good policy.

## Reporting convention

- Default protocol: `--episodes 20 --seed 0`.
- Report all five metrics, the policy type, and the checkpoint. **Known gap (issue
  #248):** `scripts/score.py` -- the authoritative, scored path -- does not currently
  produce `score_breakdown` or `mean_control_effort` at all. (`mean_episode_length` is
  fine from `score.py`, computed independently from the bag's own step count;
  `record_run.py`'s *live* wandb summary is separately broken for it and for
  `mean_control_effort`, but that summary was never the scored number to begin with.)
- For learned policies, also report seeds `0,1,2` and their mean ± std.

## Protocol contract

`FIATLUX-Replace-v0` runs `episode_length_s = 1440` and each subtask env runs `120`. A
submission may not shorten or lengthen either — the horizon caps how long a policy may take,
so changing it changes what `success_rate` means. Report a run against the horizon it used.

## Telemetry (Weights & Biases)

The benchmark ships its own logging abstraction (`fiatlux_task.telemetry.ScoreLogger`,
issue #16): pass `--wandb` to `scripts/record_run.py` to stream the score breakdown
live — one wandb chart per named channel (`Episode_Reward/<term>`,
`Episode_Termination/<term>`) plus the running `success_rate`, x-axis = completed
episodes. If `--record` includes video (`video` or `both`), `--wandb` also attaches
the rollout MP4 to the run; the final aggregate results land in the run summary
either way.

**The run summary's `gate_progress` is a live diagnostic, not the benchmark
score.** It is computed the same way as the deleted live scorer was
(`ScoreLogger.gate_progress`, unchanged by this consolidation) and inherits the
same `at_reset`-before-step-0 bias documented in `scripts/score.py`'s
`episode_gate_progress` docstring and issue #239 -- treat it as a rough
during-the-run signal only. The scored number is always `scripts/score.py`
run on the bag.

```bash
python scripts/record_run.py --task FIATLUX-Replace-v0 --policy basic_standard \
    --episodes 20 --seed 0 --record bag --enable_cameras --out logs/runs/basic_standard0 \
    --wandb --wandb_project fiatlux
```

This is *benchmark-side* telemetry: the channels are defined by the task's own
reward/termination managers, so every submission logs the same channel names no
matter how the policy was produced. Use `WANDB_MODE=offline` without an account;
`--wandb_entity/--wandb_project/--wandb_run_name` control the destination.
*Training-side* telemetry is a policy concern and already has a path — e.g.
`scripts/rsl_rl/train.py --logger wandb` for the RSL-RL baseline.

Extending it: all metric definitions live in one module,
`fiatlux_task/telemetry.py` — new channels go in `ScoreLogger.step`/`results`
(they then appear in wandb and the run summary at once), new backends implement
the small `Sink` protocol next to `WandbSink`. Policies may optionally expose
per-step diagnostics (e.g. a critic value estimate) via an `info` dict
attribute; these stream as running means under the `policy/` namespace, kept
apart from the score channels (see `fiatlux_task/policy.py`).

Determinism note: the replace preset's room layout is drawn at scene-build time from a
private `random.Random` (not the global `random` module), seeded via
`scene_cfg.set_layout_seed`. `record_run.py`, `rsl_rl/train.py`, and `verify_scene.py` all
call it from `--seed` before the env is built — so the same-seed-same-numbers contract
covers the layout too.

## Baselines

- `zero` — no action (sanity floor).
- `random` — uniform random actions (sanity floor).
- `basic_standard` — zero-action smoke test bound to the **standard** observation
  mode (the sensor-realizable `policy` group only; never touches privileged
  state). Its acceptance bar is valid episode execution + score artifact
  generation, not task success.
- `basic_cheatcode` — the same, but additionally asserts and reads the
  **privileged** observation group (exact simulator state) every step.
- `rsl_rl` — a trained PPO checkpoint (`scripts/rsl_rl/train.py`).
- `wbc_stand` / `sonic_stand` — the decoupled GEAR whole-body controller holding
  zero commands, and the GEAR-SONIC controller holding its standing latent (no
  VLA in either): the sim2sim stand gates for the model baselines and a "can
  anything keep this robot upright" reference (`fiatlux_task/groot.py`).
- `groot` — **the reference model baseline**: zero-shot NVIDIA GR00T N1.7 (base
  checkpoint, `REAL_G1` embodiment) + the decoupled GEAR WBC, standard mode only
  (torso RGB + proprioception + a language instruction, set via
  `--instruction`; the default is the task's canonical sentence in
  `fiatlux_task/groot.py`). Requires the external PolicyServer:
  `scripts/groot/serve.sh` (setup notes in its header).
  Fine-tuned GR00T submissions evaluate through the same spec — point the
  server at the fine-tuned checkpoint (`GROOT_MODEL=<path> GROOT_EMBODIMENT=<tag>
  scripts/groot/serve.sh`); a `UNITREE_G1_SONIC` finetune plugs into the
  ready `SonicDecoder` path.

## Subtask score — the benchmark headline

The twelve subtasks are scored on their own model, separate from the metrics above. Each one
combines how often the success gate fired with how much of it was ever held at once
(`fiatlux_task/subtask_score.py`):

```
subtask_score = SUCCESS_SHARE * success_rate + (1 - SUCCESS_SHARE) * gate_progress
```

with `SUCCESS_SHARE` = **0.5**, so an episode that touches every condition without ever holding
them together caps at 0.5; only the gate firing reaches 1.0.

- **`success_rate`** — fraction of episodes whose gate latched.
- **`gate_progress`** — partial credit in [0, 1]: the most conjuncts ever true *simultaneously*,
  normalized against how many were already true at reset, so conditions like `robot_standing`
  that hold at t=0 everywhere hand out no free credit. `None` in `score.py`'s own JSON output
  for a bag recorded before the `gate_*` columns existed, never a fabricated 0.0.

Both come from the bag's recorded `gate_*` conjunct columns, so the headline is reproducible from
a recording rather than existing only inside a live reward manager. `scripts/score_subtasks.py`
rolls bags up across subtasks, weighting each by its difficulty — the product of its factor
multipliers (`FACTOR_MULTIPLIERS`: balance 2.0, release 1.8, grasp 1.4, carry/traverse/mate/span
1.5), with a factorless subtask weighing 1.0. A subtask absent from the results is reported
missing and excluded from the denominator ("did not run" and "ran and failed" are different
claims); a `gate_progress: null` result for a subtask that DID run is a different case again --
not comparable, so `score_subtasks.py` raises loudly rather than silently dropping it, and the
bag needs re-scoring with `score.py` after the run that produced it recorded the `gate_*`
columns.

Subtask scores may not be compared across layout seeds.
`FIATLUX-Replace-v0`'s own score (above) is a separate number:
the whole flat episode, not one subtask.

## Offline scoring (`scripts/score.py`)

Scoring is decoupled from the simulator: record a run once and score the bag
as many times as needed under different rules, with no re-run of the sim
required:

```bash
python scripts/record_run.py --task FIATLUX-S08-GrabNewBulb-v0 --policy random \
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
