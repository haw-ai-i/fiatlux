# S10 — Climb up with the bulb in hand

`FIATLUX-S10-ClimbWithBulb-v0` · mode **balance (loaded)** · payload **bulb**
Read `00-foundation.md`, `ABSTRACTIONS.md`, `CONTINUITY.md` and `CRITIQUE.md` first.

## Objective

Climb the ladder to working height while holding the fresh bulb, arriving balanced on the upper
steps with the bulb still in hand and unbroken.

## Start state

`START_STATE["S10"]` = S09's success state:

- Robot standing at the ladder's base, facing the steps, fresh bulb in hand
  (`BULB_IN_ROOT_STANDING`). Grasping hand pinned by `mdp.hold_grasp_pose` for step 0.
- Ladder standing at the fixture target, at rest. Fixture empty above.
- Old bulb at rest in the disposal crate.

## Success gate

`climbed_with_bulb` — S02's `climbed_to_ladder_top` plus the payload conditions:

| Condition | Value |
|---|---|
| `climbed_to_ladder_top` | ladder-derived height + 0.6 m xy radius + 1.5 m/s cap |
| **bulb still held** | hand↔bulb contact > 2 N |
| bulb not dropped | `object_dropped`, `FRESH_BULB_DROP_HEIGHT = 0.4` |
| ladder not tipped | `LADDER_TILT_LIMIT = 0.6` |
| robot not fallen | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` |

## The hard problem: one hand is occupied

S02 climbs with both hands free and its bootstrap reward pays limb-on-ladder contact across feet
*and palms*. Here one hand holds a 35 g bulb it must not crush, so the climb has to be done with
one hand and two feet.

This is the most physically demanding subtask in the chain and the plan should say so up front
rather than discover it. Two consequences for the implementation:

1. **`ladder_contact_fraction` must not penalize the occupied hand** — *once the sensor bug below is
   fixed*. It returns the mean over the sensor's bodies in contact, so with palms present a
   permanently-occupied palm would cap the term at 3/4 and charge the policy for holding the bulb —
   the same class of error as charging it for an arm resting on a bench. Scope the sensor's bodies to
   feet + the free palm, or use a 3-body denominator.

   An earlier draft asserted this as a live problem. It is not, yet: on Dex3 the sensor currently
   resolves **no palms at all** (see below), so the denominator is 2 feet. The concern becomes real
   the moment that is fixed, which is why both belong in the same change.
2. **The grip must survive climbing accelerations**, which is a stronger requirement than S05's or
   S09's steady walk. See Acceptance.

## Rewards

- `climb_progress` = `climb_height_progress` — unchanged.
- `success_bonus` = `completion_bonus(predicate_fn=climbed_with_bulb)`.
- `ladder_contact` = the scoped `ladder_contact_fraction` above, weight 0.25.
- `contact_penalty` = `hand_contact_force_l2` on the filtered channel — the glass bound is the
  live constraint: gripping harder is the obvious way to keep the bulb while climbing, and 50 N is
  where a real bulb's glass gives.
- `bulb_dropped` penalty, `ladder_tipped` penalty, `com_sway`, `termination_penalty`, base
  shaping. No `flat_orientation_l2` (the A-frame lean).

## Terminations

`time_out`, `success=climbed_with_bulb`, `fell_below`, `fell_over`, `ladder_tipped`,
`bulb_dropped`. `episode_length_s = 120.0`.

## Reuse

`climb_height_progress`, `climbed_to_ladder_top` (S02's), `ladder_contact_fraction`,
`add_ladder_contact_sensor`, `_ladder_top_point_w`, `STEP_LADDER_TOP_OFFSET`,
`BULB_IN_ROOT_STANDING`, `object_dropped`, `hand_contact_force_l2`, `GLASS_CONTACT_LIMIT_N`,
`completion_bonus`. New: `climbed_with_bulb`, the scoped contact-fraction variant.

## Deliverable for the chain

**`BULB_IN_ROOT_ON_LADDER`** is consumed here as the *end* state, and S03 owns producing it. If S03
has not run yet, S10 must measure it instead — the two must agree, so whichever lands first freezes
it and the other imports it. Do not let two subtasks each freeze their own copy.

## Visual start-state validation

Frames must show: both feet flat on the floor at the ladder base on the side the steps face; the
bulb enclosed in the hand; the free hand open and available for a rail; no limb intersecting the
ladder. Use `--record_view fixture` too and confirm the (empty) fixture is above the ladder top.

Then render the intended *arrival* state — robot on the upper steps, bulb in hand — and confirm the
bulb clears the ladder's rails and steps in that pose. A bulb held in a fist that has to pass
through a rail on the way up makes the subtask geometrically impossible, and that is worth knowing
before any reward is written.

## Blocking sensor fix — `ladder_contact` sees no hands on Dex3

`add_ladder_contact_sensor` uses `prim_path=".../Robot/.*(ankle_roll|hand_base)_link"`. That matches
Inspire's `*_hand_base_link` palms but **not Dex3's `*_hand_palm_link`** — so on the variant the
benchmark scores, the sensor resolves only the two ankle links and `ladder_contact_fraction` averages
over feet alone. The palms-on-the-ladder half of the bootstrap reward does not exist.

Fix it before scoring anything on that channel: resolve the bodies from `G1_PALM_BODY_BY_VARIANT` and
`G1_FOOT_BODIES` instead of a regex that encodes one variant's naming. This is the twin of the
`palm_body_index` bug PR #64 fixed, one file over, and it is live in Climb and Descend today.
Tracked as **#69**.

## Acceptance

As foundation, plus the **retention gate under motion**, which is the gate that decides whether this
subtask is viable at all: hold the grasp pose and drive the body through a climb's full motion
envelope, then check the bulb is still in the hand. A static hold does not test this. Report the
measurement and the peak grip force.

If the bulb leaves the hand, report it plainly: S10 is unsolvable as specified, and the honest fix
is folding hand retention into #54's mechanic. Increasing `_DEX3_CURL` is not available — 1.2 is
already the calibrated value and 1.3 exceeds the glass bound.

## Blockers

**PR #64** — the still-held condition and the fragility penalty read the filtered channel.
Highest risk of the chain's loaded subtasks. Recommend implementing it **after** S05 and S09,
so their retention measurements are in hand before this one is attempted.

## As built

`subtasks/s10_climb_with_bulb_env_cfg.py`, on `subtask_tiers.balance.ClimbSubtaskCfg`. Constructs.

The scoped contact fraction this plan predicted is implemented as
`balance.LOADED_LADDER_CONTACT_BODIES` — feet plus both variants' *left* palm, passed to
`add_ladder_contact_sensor`, which now takes a body list. Names belonging to the absent hand
variant never resolve, so the live denominator is three.

The gate is `mdp.all_of` over the S02 climb conjunct, `payload_held`, `object_lifted` at
`FRESH_BULB_DROP_HEIGHT`, `robot_standing` and `ladder_near_vertical`.

`BULB_IN_ROOT_ON_LADDER` was **not** measured here — S10 consumes `BULB_IN_ROOT_STANDING` as its
start state and produces the on-ladder pose only as an end state, so the deliverable stays with
S03. Both remain `UNCALIBRATED`.

Not validated. The retention-under-motion gate — the measurement that decides whether this subtask
is viable at all — has not been run.
