# S04 — Place the ladder

`FIATLUX-S04-PlaceLadder-v0` · mode **grasping (release)** · object **ladder**
Read `00-foundation.md`, `ABSTRACTIONS.md` and `CONTINUITY.md` first.

## Objective

Set the carried ladder down on its feet at the fixture, upright and stable, and let go of it.

## Start state

`START_STATE["S04"]` = S03's success state: robot standing beside the fixture target, ladder held
and lifted, ladder top within `LADDER_READY_XY_RADIUS` of the seat point.

## Success gate

`ladder_placed` — all of:

| Condition | Value |
|---|---|
| `ladder_ready(xy_radius=0.9, tilt_limit=0.6)` | existing predicate |
| ladder feet on the floor | root z within 0.02 m of `LADDER_POSITION[2]` |
| ladder at rest | lin vel < 0.05 m/s, ang vel < 0.10 rad/s |
| **released** — zero hand↔ladder contact | < 1 N |
| robot standing | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` |
| sustained | **1.0 s** (`sustained`) |

Release plus at-rest, sustained for a full second, is what distinguishes *placed* from *held in
the right place*. The 1 s debounce is deliberately longer than elsewhere: a ladder that is stable
only while a hand steadies it is not placed, and a 0.5 s window would pass one that topples
just after release.

## Rewards

- `placement_progress` = `distance_progress(distance_fn=ladder_fixture_distance)` — carried over
  from S03 so the last centimetres of positioning are still shaped.
- `settle_progress` = `distance_progress(distance_fn=ladder_feet_height)` — the ladder root's
  height above the floor, driving it down onto its feet.
- `placement_bonus` = `completion_bonus(predicate_fn=ladder_placed)`.
- `ladder_tipped` penalty, `termination_penalty`, base shaping.

## Terminations

`time_out`, `success=ladder_placed`, `fell_below`, `fell_over`, `ladder_tipped`.
`episode_length_s = 20.0`.

No `ladder_dropped` termination here — unlike S03, letting go is the *goal*. A drop that leaves
the ladder tipped is already caught by `ladder_tipped`; one that leaves it standing at the target
is a success, however inelegant. Say so at the cfg site so a future reader does not "fix" it by
adding the term back.

## Reuse

`ladder_ready`, `ladder_fixture_distance`, `ladder_tipped`, `LADDER_POSITION`,
`distance_progress`, `completion_bonus`, `sustained`. New: `ladder_placed`,
`ladder_feet_height`.

## Visual start-state validation

Frames must show the ladder held, upright, its feet a few centimetres above the floor, positioned
at the fixture target — and crucially, that **the floor beneath the target is clear**: the
disposal crate, table, and robot's own feet must not occupy the ladder's 0.68 × 1.11 m footprint.
The layout's zone sampler reserves a ladder zone, so this should hold; confirm it rather than
assume, because the fixture mount is sampled and a wall mount pushes the target toward the wall.

Also confirm the ladder is not already at rest on the floor at t=0 — if it is, S04 succeeds
trivially and the real defect is in S03's handoff.

## Acceptance

As foundation, plus: a scripted release from the start state leaves the ladder standing (prove
the target footprint is actually stable ground before scoring placement on it) · settle soak
shows the *placed* ladder drifts < 2 mm · `handoff:S04->S05` passes — S05's start state must put
the robot at the base of the ladder *where S04 left it*, not at `LADDER_POSITION`.

## Blockers

**PR #64** — the released condition reads the filtered hand↔ladder channel.

## Note

The `handoff:S04->S05` check is the one that matters most in the whole chain. S05's ascent target
is derived from the ladder's live pose, and the ladder's pose here is wherever S03/S04 put it,
which follows the sampled fixture mount. Any subtask downstream of S04 that hardcodes
`LADDER_POSITION` or `TOP_ROBOT_POSITION` is wrong — those constants describe the *default*
workshop layout, not this chain's.
