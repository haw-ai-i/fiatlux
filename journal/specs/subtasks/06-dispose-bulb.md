# S06 — Dispose of the old bulb

`FIATLUX-S06-DisposeBulb-v0` · mode **grasping (release)** · object **old_bulb**
Read `00-foundation.md`, `ABSTRACTIONS.md`, `CONTINUITY.md` and `CRITIQUE.md` first.

## Objective

Put the old bulb into the disposal crate and let go, leaving it at rest inside — unbroken.

## Start state

`START_STATE["S06"]` = S05's success state: robot standing within `CRATE_APPROACH_RADIUS` of the
crate, facing it, old bulb in hand (`BULB_IN_ROOT_STANDING`), crate on the floor in its zone.

## Success gate

`old_bulb_disposed_and_released` — all of:

| Condition | Value | Source |
|---|---|---|
| bulb within the disposal threshold of the crate origin | `DISPOSAL_THRESHOLD = 0.25` m | `old_bulb_disposed` |
| bulb at rest | lin vel < 0.05 m/s, ang vel < 0.10 rad/s | new |
| bulb inside the crate, not on its rim | bulb AABB centre inside the crate's interior AABB | measured at spawn |
| **released** — zero hand↔old_bulb contact | < 1 N | filtered channel |
| bulb not broken | peak contact never exceeded the glass bound | `GLASS_CONTACT_LIMIT_N` |
| robot standing | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` | base |
| sustained | 1.0 s | `sustained` |

The containment condition earns its place: `old_bulb_disposed` is a 25 cm sphere around the crate
*origin*, and the crate is only 0.17 m tall, so a bulb balanced on the rim or resting on the floor
beside it satisfies the distance test.

**It must be containment, not a height equality.** An earlier draft required the root z within 3 cm
of `BIN_BULB_INTERIOR_Z - BULB_STAND_Z_OFFSET` = 0.0187 m, which silently assumes the bulb ends
*standing on its cap*: a bulb lying on its glass reads root z = `0.055 + BULB_LIE_Z_OFFSET` =
**0.0946 m** and fails. `poses.py` argues this bulb self-rights onto its flat cap, so the window was
not necessarily unsatisfiable — but a bulb wedged against a crate wall or still settling would fail a
gate that should only be asking whether it is in the crate. Test the AABBs; it holds in every
orientation.

Note the interaction with `old_bulb_dropped`, which is deliberately *not* a plain height gate for
exactly this reason: legitimately disposing the bulb also ends near the floor, so "dropped"
additionally requires being outside `DISPOSAL_THRESHOLD`. That existing predicate is correct here
as-is; do not re-add a naive height check.

## Rewards

- `disposal_progress` = `distance_progress(distance_fn=old_bulb_disposal_distance)` — the existing
  Replace channel, unchanged.
- `disposal_bonus` = `completion_bonus(predicate_fn=old_bulb_disposed_and_released)`.
- `contact_penalty` = `hand_contact_force_l2` on the filtered channel.
- `old_bulb_dropped` penalty, `termination_penalty`, base shaping.

## Terminations

`time_out`, `success=old_bulb_disposed_and_released`, `fell_below`, `fell_over`,
`old_bulb_dropped`. `episode_length_s = 20.0`.

## Reuse

`old_bulb_disposed`, `old_bulb_disposal_distance`, `old_bulb_dropped`, `DISPOSAL_THRESHOLD`,
`BIN_BULB_INTERIOR_Z`, `BIN_BULB_POSITION`, `BULB_STAND_Z_OFFSET`, `_spawn_open_container`
(the crate's collision geometry, which is what lets a bulb rest *inside* rather than on a lid),
`hand_contact_force_l2`, `distance_progress`, `completion_bonus`, `sustained`.
New: `old_bulb_disposed_and_released`.

## Visual start-state validation

Frames must show: the robot standing at the crate, the bulb enclosed in the hand and held above or
beside the crate opening; the crate upright on the floor with its **interior visible and empty**.
The `_physics` crate variant ships collision geometry and `_spawn_open_container` is what keeps the
opening open — confirm from the frames that the crate is genuinely open, because a closed collider
would make the whole subtask impossible while looking fine in a wireframe.

Then render the *target* state: teleport the bulb to `BIN_BULB_POSITION` relative to the sampled
crate, soak, and confirm it rests inside the crate and stays there. That is the acceptance test for
the success gate's height condition.

## Acceptance

As foundation, plus: the target-state render above settles with drift < 2 mm · a scripted release
above the crate lands the bulb inside it (proves the opening is reachable and the geometry
cooperates) · `old_bulb_disposed_and_released` does **not** fire for a bulb placed on the crate's
rim or on the floor beside it — test both negatives explicitly, since the 25 cm threshold alone
would accept them.

## Blockers

**PR #64** — the released condition and the fragility bound read the filtered channel.
**#54** — inherits the provisional start state from S03.

## Note

This is the chain's natural mid-point checkpoint: after S06 the old bulb is gone and the robot is
free-handed on the floor. S07 restarts a clean navigation leg. If the chain is ever run partially,
S01–S06 and S07–S12 are the two halves to run.
