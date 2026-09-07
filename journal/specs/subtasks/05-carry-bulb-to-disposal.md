# S05 — Walk to the disposal box with the bulb in hand

`FIATLUX-S05-CarryBulbToDisposal-v0` · mode **navigation (loaded)** · payload **old_bulb**
Read `00-foundation.md`, `ABSTRACTIONS.md`, `CONTINUITY.md` and `CRITIQUE.md` first.

## Objective

Carry the old bulb across the floor from the foot of the ladder to the disposal crate, arriving
standing within reach of it, still holding the bulb.

## Start state

`START_STATE["S05"]` = S04's success state:

- Robot standing on the floor at the ladder's base, old bulb in hand — root pose from
  `BULB_IN_ROOT_STANDING` (foundation probe 1; the standing cradle, **not** the on-ladder value
  S04 used). Grasping hand pinned by `mdp.hold_grasp_pose` for step 0.
- Ladder standing at the fixture target, empty fixture above.
- Disposal crate at its sampled zone, `DISPOSAL_ZONE_HALF_SIZE = 0.5`.

## Success gate

`arrived_at_disposal` — all of:

| Condition | Value |
|---|---|
| robot root within xy clearance of the crate FOOTPRINT | `DISPOSAL_ARRIVAL_CLEARANCE` = `G1_HORIZONTAL_REACH` |
| facing error to crate bearing | < 0.5 rad |
| root speed | < 1.0 m/s |
| **bulb still held** | hand↔old_bulb contact > 2 N |
| bulb not dropped | `old_bulb_dropped` false |
| robot standing | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` |
| sustained | 0.5 s |

**Settled (#149).** The crate is 0.60 × 0.40 × 0.17 m, so measuring arrival to its ORIGIN scored
the approach side rather than being at it: the same 0.24 m gap from the crate reads 0.54 m off the
short end and 0.44 m off the long face, and three operator takes standing 0.24-0.29 m from the edge
split 1/3 on which face they came from. `base_near` now takes the target's footprint
(`CRATE_FOOTPRINT_HALF_EXTENT`) and measures the clearance to it, with the threshold being
`G1_HORIZONTAL_REACH` — within arm's reach of the crate is exactly when a hand can go over it.
The staging radius keeps its own constant (`DISPOSAL_STANCE_RADIUS`), since where staging PUTS the
robot is a distance from the origin, not a clearance.

Dropping the bulb *en route* must not score, hence the held condition; dropping it *into the
crate* is S06, not S05.

## Rewards

- `approach_progress` = `distance_progress(distance_fn=base_bin_distance)`.
- `arrival_bonus` = `completion_bonus(predicate_fn=arrived_at_disposal)`.
- `contact_penalty` = `hand_contact_force_l2` on the filtered channel.
- `old_bulb_dropped` penalty, `termination_penalty`, base shaping.

`base_bin_distance(env)` as a bare module-level function, per `distance_progress`'s no-closures
requirement.

## Terminations

`time_out`, `success=arrived_at_disposal`, `fell_below`, `fell_over`, `old_bulb_dropped`.
`episode_length_s = 25.0`.

No `ladder_tipped` term: the robot is walking away from the ladder and no longer interacts with
it. State that at the cfg site — its absence is a decision, not an omission.

## Reuse

`old_bulb_disposal_distance` (as the pattern for the base-to-crate wrapper), `_add_parts_bin`,
`BIN_POSITION`, `BIN_BULB_INTERIOR_Z`, `CRATE_MASS_KG = 1.5`, `old_bulb_dropped`,
`OLD_BULB_DROP_HEIGHT = 0.15`, `hand_contact_force_l2`, `distance_progress`, `completion_bonus`.
New: `base_bin_distance`, `arrived_at_disposal`, `CRATE_APPROACH_RADIUS`.

## Visual start-state validation

Frames must show: robot standing on the floor, both feet flat, clear of the ladder's footprint;
the bulb enclosed in the hand; the crate on the floor in its zone, upright, resting on the floor
(not sunk into it — the crate is kinematic and spawned at `BIN_POSITION[2]`, so confirm the
authored origin is its floor-contact plane); a clear walkable path from robot to crate.

Report whether the crate is in the ego camera frustum at t=0 — as a difficulty note, not a defect (the mount is Unitree's spec; see foundation).

## Acceptance

As foundation, plus the **retention gate** over the full 25 s horizon, and: the walkable path is
genuinely clear — the layout's zone sampler guarantees non-overlapping zones with
`ZONE_MARGIN = 0.5`, but not that a straight line between two zones misses the third. If the table
sits between the ladder and the crate, say so; it makes the subtask harder, not broken, but the
plan should record it.

## Blockers

**PR #64** — the still-held condition reads the filtered channel.
**#54** — inherits the provisional start state from S03/S04.
