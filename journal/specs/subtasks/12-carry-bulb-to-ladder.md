# S12 — Walk to the ladder with the new bulb

`FIATLUX-S12-CarryBulbToLadder-v0` · mode **navigation (loaded)** · payload **bulb**
Read `00-foundation.md`, `ABSTRACTIONS.md`, `CONTINUITY.md` and `CRITIQUE.md` first.

## Objective

Carry the fresh bulb from the bench to the foot of the placed ladder, arriving standing, still
holding it, unbroken.

## Start state

`START_STATE["S12"]` = S11's success state:

- Robot standing at the bench, fresh bulb in hand — root pose from `BULB_IN_ROOT_STANDING` (S11's
  deliverable). Grasping hand pinned by `mdp.hold_grasp_pose` for step 0.
- Ladder standing at the fixture target, at rest; fixture empty.
- Old bulb at rest in the disposal crate.

## Success gate

`arrived_at_ladder_with_bulb` — all of:

| Condition | Value |
|---|---|
| robot root within xy radius of the ladder root | `LADDER_MOUNT_RADIUS` — **not** the grasp radius |
| facing error to ladder bearing | < 0.5 rad |
| root speed | < 1.0 m/s |
| **bulb still held** | hand↔bulb contact > 2 N |
| bulb not dropped | `object_dropped`, `FRESH_BULB_DROP_HEIGHT = 0.4` |
| ladder not tipped | `LADDER_TILT_LIMIT = 0.6` |
| robot standing | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` |
| sustained | 0.5 s |

`ladder_tipped` is back as a condition and a termination, unlike S08: the robot is walking *toward*
the ladder and can knock it over on arrival — which is exactly what happened in the WBC walk probe,
where every episode ended on `ladder_tipped` because the robot walked into the ladder. Arriving by
demolishing the thing you are about to climb must not score.

## Rewards

- `approach_progress` = `distance_progress(distance_fn=base_ladder_distance)` — S01's function,
  reused unchanged.
- `arrival_bonus` = `completion_bonus(predicate_fn=arrived_at_ladder_with_bulb)`.
- `contact_penalty` = `hand_contact_force_l2` on the filtered channel.
- `bulb_dropped` penalty, `ladder_tipped` penalty, `termination_penalty`, base shaping.

## Terminations

`time_out`, `success=arrived_at_ladder_with_bulb`, `fell_below`, `fell_over`, `bulb_dropped`,
`ladder_tipped`. `episode_length_s = 25.0`.

## Reuse

`base_ladder_distance`, `base_facing_entity` (both from the approach-the-ladder subtask),
`BULB_IN_ROOT_STANDING` (S11's), `object_dropped`, `FRESH_BULB_DROP_HEIGHT`, `ladder_tipped`,
`hand_contact_force_l2`, `distance_progress`, `completion_bonus`.
New: `arrived_at_ladder_with_bulb`, `LADDER_MOUNT_RADIUS`.

**Do not reuse `LADDER_APPROACH_RADIUS`.** That constant answers "close enough to reach out and
grasp a rail", which is an arm's-reach question. This subtask ends about to *climb*, which is a
foot-placement question — the robot must be at the ladder's step-facing side within stepping
distance, which is a different and probably shorter standoff. An earlier draft borrowed the grasp
constant without noticing they are different requirements.

Almost nothing new — S12 is S01's navigation with a payload condition and a drop termination
bolted on. That is the intended shape: the shared logic lives in `mdp/rewards.py` and the two cfgs
differ only in their gates, so S01 and S12 can be versioned independently while sharing the
implementation.

## Visual start-state validation

Frames must show: robot standing at the bench with the bulb enclosed in the hand, clear of the
table's collision footprint; the ladder standing at the fixture target with all four feet on the
floor; a walkable path between them. Report whether the table sits between the bench position and
the ladder — the layout guarantees non-overlapping zones with `ZONE_MARGIN = 0.5`, not a clear
straight line.

Also confirm the ladder is in the ego camera frustum at t=0 from the bench-side stance and report
if not.

## Acceptance

As foundation, plus the **retention gate** over the full 25 s horizon (this is the third loaded
navigation leg; if S03 and S08 both retained their payloads, this should too, but the fresh bulb is
the lightest and most fragile of the three and its grip has no curl headroom left) ·
`handoff:S12->S13` passes.

## Blockers

**PR #64** — the still-held condition and the fragility penalty read the filtered channel.
