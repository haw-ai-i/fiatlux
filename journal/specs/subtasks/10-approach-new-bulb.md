# S10 — Walk to the new bulb

`FIATLUX-S10-ApproachNewBulb-v0` · mode **navigation** · payload **none**
Read `00-foundation.md` and `CONTINUITY.md` first.

## Objective

From the disposal crate, walk to the bench where the fresh bulb sits and end standing within
reach of it, facing it.

## Start state

`START_STATE["S10"]` = S09's success state:

- Robot standing at the disposal crate, **hands free**, old bulb at rest inside the crate.
- Table at its sampled zone with the fresh bulb on it: `TABLETOP_BULB_POSITION` offset from the
  table centre, standing on its cap (`TABLETOP_SURFACE_Z - BULB_STAND_Z_OFFSET`), exactly as
  `apply_replace_preset` places it.
- Ladder still standing at the fixture target; fixture still empty.

## Success gate

`arrived_at_bulb` — all of:

| Condition | Value |
|---|---|
| robot root within xy radius of the fresh bulb | `BENCH_APPROACH_RADIUS`, **probe** |
| facing error to bulb bearing | < 0.5 rad |
| root speed | < 1.0 m/s |
| fresh bulb undisturbed | still on the table: root z within 0.02 m of its start z |
| robot standing | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` |
| sustained | 0.5 s |

The undisturbed condition matters more here than in S01: the target is a 35 g object standing on
its cap at the edge of a bench, and walking into the table can knock it to the floor. Arriving
next to a bulb that is no longer there must not score.

**Probe:** `BENCH_APPROACH_RADIUS` — measure against the table's collision footprint, not the
bulb's position. The robot cannot stand where the table is, so the usable standoff is
(table half-extent + a stance clearance), and the bulb must be inside the arm's horizontal reach
*from there*. `apply_tabletop_preset` already solves this problem for the bench tasks —
`TABLETOP_ROBOT_POSITION = (0.60, 0.73, ...)` with `TABLETOP_ROBOT_YAW_DEG = -90` places the robot
on the table's +y side "clear of the table's collision footprint". Derive the radius from that
validated offset rather than probing from scratch.

## Rewards

- `approach_progress` = `distance_progress(distance_fn=base_bulb_distance)`.
- `arrival_bonus` = `completion_bonus(predicate_fn=arrived_at_bulb)`.
- `bulb_dropped` penalty (`object_dropped`, `FRESH_BULB_DROP_HEIGHT = 0.4`) — the fresh bulb's
  working heights are table level and above, so a 0.4 m gate reads "knocked off the bench".
- `termination_penalty`, base shaping.

## Terminations

`time_out`, `success=arrived_at_bulb`, `fell_below`, `fell_over`, `bulb_dropped`.
`episode_length_s = 25.0`.

## Reuse

`apply_tabletop_preset`'s validated robot-beside-bench offset, `TABLETOP_ROBOT_POSITION`,
`TABLETOP_ROBOT_YAW_DEG`, `TABLETOP_SURFACE_Z = 0.9941`, `TABLETOP_BULB_POSITION`,
`BULB_STAND_Z_OFFSET`, `object_dropped`, `FRESH_BULB_DROP_HEIGHT`, `distance_progress`,
`completion_bonus`. New: `base_bulb_distance`, `arrived_at_bulb`, `BENCH_APPROACH_RADIUS`.

## Visual start-state validation

Frames must show: robot standing at the crate with empty hands; the old bulb visible **inside** the
crate (the S09 handoff, rendered); the table at its zone with all legs on the floor; the fresh bulb
standing upright on its cap on the tabletop, not sunk into it and not floating. Confirm the fresh
bulb is on the table's *reachable* side — the bulb offset is applied relative to the table centre
and the table's yaw is not randomized, so this should hold, but the robot approaches from wherever
the crate zone is, which is sampled.

Confirm the table and bulb are in the ego camera frustum at t=0; report if not.

## Acceptance

As foundation, plus: `handoff:S09->S10` and `handoff:S10->S11` pass · the settle soak leaves the
fresh bulb standing on its cap with drift < 2 mm (a bulb that topples on its own during the soak
makes S11's start state wrong, and an upright bulb on a cap is the marginal case — the calibration
notes record that the *old* B1K bulb toppled when stood upright, which is why `BULB_LYING_QUAT`
exists; the Omniverse bulb's cap is its stable base, so confirm rather than assume).

## Blockers

None of its own. Inherits the provisional S06–S09 start states only in the sense that the chain
must reach it; S10 can be authored and validated standalone.
