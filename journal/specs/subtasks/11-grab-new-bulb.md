# S11 — Grab the new bulb

`FIATLUX-S11-GrabNewBulb-v0` · mode **grasping** · object **bulb**
Read `00-foundation.md` and `CONTINUITY.md` first.

## Objective

Pick the fresh bulb up off the bench and hold it securely, without crushing it and without
dropping it.

## Start state

`START_STATE["S11"]` = S10's success state: robot standing beside the bench facing the fresh bulb,
hands free; bulb standing on its cap on the tabletop; old bulb at rest in the crate.

## Success gate

`bulb_grasped` — all of:

| Condition | Value |
|---|---|
| bulb lifted above the tabletop | > 0.05 m above its start z |
| **held** — hand↔bulb contact on ≥ 2 hand bodies | > 2 N each, filtered channel |
| grip within the fragility bounds | `GLASS_CONTACT_LIMIT_N = 50`, `CAP_CONTACT_LIMIT_N = 300` |
| bulb not dropped | `object_dropped`, `FRESH_BULB_DROP_HEIGHT = 0.4` |
| robot standing | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` |
| sustained | 1.0 s |

Lift plus sustained hold, as in S02: contact alone is satisfied by pushing the bulb across the
bench. The 1 s window is the longer one because this is the grasp the rest of the chain depends
on — S12, S13 and S14 all assume a bulb that stays held.

## Grasp geometry — use the measured calibration, do not re-derive it

This is the one grasp on the project that has already been measured, argued about, and fixed. The
numbers live in `scripts/verify_interactions.py` and must be lifted into `fiatlux_task/grasp.py`
(foundation) rather than copied:

- `BULB_CAP_RADIUS_M = 0.021`, `BULB_CAP_CENTRE_M = 0.054` (along local +z from the root)
- `BULB_GLASS_RADIUS_M = 0.040`, `BULB_GLASS_CENTRE_M = 0.131`
- `PALM_GRASP_FORWARD_M = 0.045`, `PALM_LOCAL_AXES` (per-variant palm frame)
- `palm_frame()`, `palm_grasp_pose()`, `place_bulb_under_palm()`

Three facts that cost real time to establish, recorded so they are not rediscovered:

1. **The bulb's root lies outside its geometry** — cap bottom is at +0.036 from the root. Placing
   the root at a seat point leaves the body 13 cm away. Seat a *feature* and back the root out.
2. **The palm body is per-variant.** `G1_PALM_BODY_BY_VARIANT` — Dex3 is `right_hand_palm_link`,
   and the old code silently fell back to `right_wrist_yaw_link`, ~4 cm off, which turned a flat
   press into a 64.8 N edge contact. Resolve it from the map and raise if absent.
3. **The bulb is retained upright on its cap, not laid across the fingers.** Laid across the
   fingers it is pinched against the palm and squirts out — `BULB_UPRIGHT_QUAT` exists for this
   reason and `poses.py` records it. Do not re-attempt the across-the-palm grasp.

Grip: `HAND_CRADLE_DEX3` at `_DEX3_CURL = 1.2`. The curl is bounded on both sides — below ~1.0 the
bulb slips, at ~1.3 the grip exceeds the 50 N glass bound, past ~1.4 the closing fingers eject it.
There is no headroom; if retention fails, the answer is not more curl.

## Rewards

- `reach_progress` = `distance_progress(distance_fn=palm_bulb_distance)`.
- `lift_progress` = `distance_progress(distance_fn=bulb_height_above_table)`.
- `grasp_bonus` = `completion_bonus(predicate_fn=bulb_grasped)`.
- `contact_penalty` = `hand_contact_force_l2` on the filtered channel.
- `bulb_dropped` penalty, `termination_penalty`, base shaping.

## Terminations

`time_out`, `success=bulb_grasped`, `fell_below`, `fell_over`, `bulb_dropped`.
`episode_length_s = 20.0`.

## Reuse

Everything named above, plus `object_contact_forces`, `hand_contact_force_l2`, `ARM_CRADLE`,
`HAND_CRADLE_DEX3`, `BULB_UPRIGHT_QUAT`, `TABLETOP_SURFACE_Z`, `distance_progress`,
`completion_bonus`, `sustained`. New: `palm_bulb_distance`, `bulb_height_above_table`,
`bulb_grasped`.

## Deliverable for the chain

**`BULB_IN_ROOT_STANDING`** (foundation probe 1) — the bulb's root pose in the robot root frame for
the standing cradle grasp. S08 and S12 both start from it. S11 owns measuring and freezing it, from
a scripted grasp, with the probe command recorded at the constant.

## Visual start-state validation

Frames must show: the bulb standing on its cap on the tabletop, upright, not sunk in; the open hand
beside it at bench height, not intersecting the tabletop; the robot clear of the table's collision
footprint.

Then render the *grasped* state and check it hard, at multiple angles: the bulb must be **enclosed
by the digits**, not pinched between a fingertip and the palm, not intersecting the palm mesh, and
not depenetrating outward. A "0.5 cm slip, 150/150 contact frames" result was reported as a
successful grasp twice on this project and was wrong both times — the renders showed the bulb
pinched outside the hand, then squeezing out of the palm. Numbers are not sufficient evidence here.

## Acceptance

As foundation, plus: the grasped-state render inspected and reported at ≥ 3 angles · measured peak
grip force reported and inside the glass bound · `BULB_IN_ROOT_STANDING` frozen · a static hold of
≥ 5 s retains the bulb · `handoff:S11->S12` passes.

## Blockers

**PR #64** — the held condition and both fragility bounds read the filtered channel. This subtask
in particular must not be built before it merges: the unfiltered channel reads the arm resting on
the bench as force on the bulb, which is precisely how the gentle press once scored 64.8 N.
