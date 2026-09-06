# Full-Task Benchmark Plan

## Summary

Issues 33-48 are currently over-split around subtask completion and per-subtask baseline scoring. The benchmark should instead center on `FIATLUX-Replace-v0`: one full randomized replacement task, two rollout modes, and scoring that measures normalized progress rather than rewarding lucky randomized starts.

GitHub issues 33-48 should remain unchanged for now. This file records the intended implementation allocation.

## Benchmark Task

Promote `FIATLUX-Replace-v0` from scene-only scaffold to the primary scored full-task environment.

The full task starts with:

- robot in a randomized valid location,
- ladder in a randomized valid location,
- fixture/socket in a randomized valid ceiling or wall location,
- fresh bulb available for insertion,
- old bulb initially seated in the fixture,
- disposal target box/crate for the old bulb.

The goal is:

- insert the fresh bulb into the fixture,
- remove the old bulb from the fixture,
- place the old bulb in the disposal target.

The ladder must not be coupled to the fixture by default. Any ladder-fixture coupling should be an explicit debug or curriculum option only.

### Horizon

Every subtask env runs **120 s**; `FIATLUX-Replace-v0` runs their sum, **1440 s**. Set
2026-09-05 (issue #136), replacing twelve hand-set per-stage values (435 s in total) and a flat
40 s on Replace that was shorter than the stages it contains.

Storage, measured on `FIATLUX-Replace-v0` with the recorder attached over 200 steps:
**12,884 B per step** on disk (29,635 B uncompressed). So a full-length Replace bag is
**0.93 GB** per episode, 46 GB for the standard 50, and a 120 s subtask bag is 77 MB.

`scripts/eval.py` writes a JSON summary only, so the scored protocol stores no bags. Bags come
from `scripts/record_run.py` and teleop takes, and reach full length only when an episode runs
to timeout rather than ending on a failure termination. 94% of each step is `policy_obs`
(27,920 B) -- narrowing or subsampling that field is worth more than shortening the horizon.

## Scoring

Scene randomization must not directly improve or worsen score just because objects spawned closer together.

Use normalized progress for distance-based terms:

```text
progress = (initial_distance - current_distance) / initial_distance
```

Clamp normalized progress to `[0, 1]`.

Use normalized progress for:

- ladder top progress toward the fixture,
- fresh bulb progress toward the fixture,
- old bulb progress toward the disposal target.

Also score old-bulb removal as clearance from the fixture, with a completion threshold.

Add sparse completion terms for:

- ladder in usable range/alignment near fixture,
- fresh bulb inserted,
- old bulb removed from fixture,
- old bulb placed in disposal target,
- full replacement success.

Add penalties for:

- robot fall,
- ladder fall or tip,
- fresh bulb drop,
- old bulb drop,
- unsafe collisions or excessive contact forces.

Report a score breakdown, not just a scalar total, so partial progress maps onto the subtasks without making the subtasks separate benchmark targets.

## Observation And Policy Modes

Add two rollout/evaluation modes.

`standard` mode uses only regular real-world inputs:

- RGB or camera-derived features,
- robot proprioception,
- IMU / estimated base state,
- contact or force sensors,
- previous action.

`cheatcode` mode may additionally use privileged simulator-only state:

- exact robot pose,
- exact object poses,
- exact fixture pose,
- ladder target geometry,
- disposal target pose,
- exact score-relevant distances.

The observation contract should make it clear which group is sensor-realizable and which group is privileged.

## Basic Policies

A basic policy is a smoke-test policy, not a solver.

Its purpose is to prove that the environment, observations, action interface, resets, episode loop, recording, and scoring run end-to-end.

`basic_standard`:

- consumes only the standard policy observation group,
- must not read privileged object or world poses,
- can be zero-action, posture-hold, or simple reactive behavior,
- acceptance is valid episode execution and score artifact generation.

`basic_cheatcode`:

- explicitly consumes the privileged observation group,
- may inspect exact sim-state observations,
- still acts through the normal action space,
- acceptance is valid episode execution and score artifact generation, not task success.

## Acceptance Checks

Implementation should verify:

- `FIATLUX-Replace-v0` instantiates as an RL environment,
- the default ladder placement is independent of fixture placement,
- `basic_standard` runs complete episodes without privileged observations,
- `basic_cheatcode` runs complete episodes with privileged observations available,
- randomized starts use normalized distance progress so lucky spawn distances do not dominate score,
- score output includes dense normalized components, sparse completion terms, penalties, and full success.

## Assumptions

- GitHub issues 33-48 are not edited in this pass.
- Existing subtask environments may remain as scaffolds or development aids.
- The benchmark allocation moves to full-task `FIATLUX-Replace-v0`.
- The old bulb uses a target disposal box/crate.
- Ladder-fixture coupling remains available only as an explicit non-default option.
