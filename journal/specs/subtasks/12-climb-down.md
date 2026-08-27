# S12 — Climb down

`FIATLUX-S12-ClimbDown-v0` · mode **balance** · payload **none**
Read `00-foundation.md`, `ABSTRACTIONS.md`, `CONTINUITY.md` and `CRITIQUE.md` first.

## Objective

With the new bulb installed, come back down the ladder under control and end standing on the
floor. The terminal subtask of the chain.

## Start state

`START_STATE["S12"]` = S11's success state:

- Robot on the ladder's upper steps, balanced, **hands free**.
- Fresh bulb **seated** in the fixture above, retained, at rest.
- Ladder standing at the fixture target, at rest.
- Old bulb at rest in the disposal crate.

## Success gate

`descended_from_ladder` (S04's predicate, with no payload conditions) — all of:

| Condition | Value |
|---|---|
| pelvis below the floor-stance height near the ladder base | ladder-derived (S04) |
| pelvis within xy radius of the ladder base | 0.6 m |
| root speed | < 1.5 m/s |
| ladder not tipped | `LADDER_TILT_LIMIT = 0.6` |
| robot not fallen | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` |
| **fresh bulb still seated** | `bulb_seated(0.015, 0.2)` |

The last condition is what makes S12 more than a descent: dismounting must not knock the freshly
installed bulb out of the fixture. A robot that shakes the ladder hard enough to unseat the bulb on
the way down has undone the task, and the chain's final state should reflect that.

As in S02 and S04, the height bound and xy centre come from the ladder's **live** pose via
`_ladder_top_point_w` / the ladder root, not from `CLIMB_ROBOT_POSITION` — `FIATLUX-Descend-v0`
hardcodes `SUCCESS_XY` from that constant, which describes the default workshop layout.

## Rewards

- `descend_progress` = `descend_height_progress` — unchanged.
- `success_bonus` = `completion_bonus(predicate_fn=descended_and_bulb_intact)`.
- `ladder_contact` = `ladder_contact_fraction`, weight 0.25 — both hands are free here, so the
  unscoped 4-body form is correct (unlike S10).
- `ladder_tipped` penalty, `com_sway`, `ang_vel_xy`, `termination_penalty`, base shaping.
  No `flat_orientation_l2`.
- **`bulb_unseated` penalty** — a new termination-and-penalty pair: the fresh bulb leaving the
  socket during the descent ends the episode. Sized like `ladder_tipped`'s penalty; undoing the
  task's goal should cost about as much as destroying its equipment.

## Terminations

`time_out`, `success=descended_and_bulb_intact`, `fell_below`, `fell_over`, `ladder_tipped`,
`bulb_unseated`. `episode_length_s = 20.0`.

## Reuse

`descend_height_progress`, `descended_from_ladder` (S04's), `ladder_contact_fraction`,
`add_ladder_contact_sensor`, `bulb_seated`, `SEAT_POS_THRESHOLD`, `SEAT_ORI_THRESHOLD`,
`_ladder_top_point_w`, `completion_bonus`. New: `descended_and_bulb_intact`, `bulb_unseated`.

Nearly all of it is S04's, minus the payload. The two descend subtasks share
`descended_from_ladder` and differ only in their gates — the intended shape.

## Visual start-state validation

Frames must show: the robot balanced on the upper steps with both feet on a step and hands free;
the fresh bulb **seated in the fixture** above, cap in the socket, glass hanging down — render this
with `--record_view fixture`, since it is the only view that shows an overhead mount properly; the
ladder at rest with all four feet down; the old bulb at rest inside the disposal crate.

This start state is the chain's terminal scene and doubles as the visual acceptance test for the
whole task. Render it carefully and keep the frames — it is the "what does success look like"
reference for the benchmark.

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

As foundation, plus: the settle soak leaves the seated bulb seated (drift < 2 mm, `bulb_seated`
still true) · `handoff:S11->S12` passes · S12 has no successor, so assert instead that its success
region is the chain's declared terminal state.

## Blockers

**#54** — the start state requires a bulb *retained* in an inverted fixture. Without it, the fresh
bulb falls out of the socket at t=0 and both the start state and the `bulb_seated` success condition
are unreachable. Author it, validate everything else, and mark the seated conditions
**provisional** — the same handling as S03 and S11.

## As built

`subtasks/s12_climb_down_env_cfg.py`, on `subtask_tiers.balance.DescendSubtaskCfg`. Constructs.

Shares `balance_terms.descended_from_ladder` with S04 and differs only in the gate, as intended.
`mdp.bulb_unseated` is the negated seating predicate, added beside `bulb_seated`; it is both the
termination and the penalty, since `all_of` conjoins and cannot negate.

Start state composed from `scene_cfg.seat_bulb_in_fixture` (bulb root pose == socket root pose, no
offset arithmetic at any mount orientation) and `park_old_bulb_in_crate`.

Both hands are free, so the contact bootstrap keeps all four bodies.

Not validated. This start state is the chain's terminal scene and the "what does success look like"
reference for the whole benchmark, and nobody has rendered it. Blocked on #54: without retention
the fresh bulb leaves the socket at t=0 and the gate is unreachable.
