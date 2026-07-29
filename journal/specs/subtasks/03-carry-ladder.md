# S03 — Walk with the ladder to the fixture

`FIATLUX-S03-CarryLadder-v0` · mode **navigation (loaded)** · payload **ladder**
Read `00-foundation.md` and `CONTINUITY.md` first.

## Objective

Holding the ladder, walk it across the room to a spot from which the fixture is workable, and
arrive still holding it and still upright.

## Start state

`START_STATE["S03"]` = S02's success state:

- Robot standing at the ladder's original zone, ladder **held and lifted** — root pose from
  `LADDER_IN_ROOT_CARRIED` (S02's deliverable), composed with the robot root pose.
- Grasping hand pinned to its grasp pose by `mdp.hold_grasp_pose` for step 0, so the ladder is
  not dropped before the policy acts.
- Fixture mounted per the layout draw (ceiling or wall), old bulb seated in it.
- Target is derived from the fixture, not a constant: the wall-mount case needs
  `LADDER_WALL_STANDOFF = 0.4` clearance, the ceiling case is directly beneath.

## Success gate

`ladder_carried_to_fixture` — all of:

| Condition | Value | Source |
|---|---|---|
| ladder top within xy radius of the fixture seat | `LADDER_READY_XY_RADIUS = 0.9` | existing |
| ladder still held | hand↔ladder contact > 5 N | S02's channel |
| ladder not tipped | `LADDER_TILT_LIMIT = 0.6` | existing |
| robot standing | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` | base |
| sustained | 0.5 s | `sustained` |

**Still-held is the load-bearing condition.** Without it, throwing the ladder at the fixture
scores. S03 ends with the ladder in the air near the target; putting it down is S04.

## Rewards

- `carry_progress` = `distance_progress(distance_fn=ladder_fixture_distance)` — the existing
  Replace/Carry channel, unchanged, measuring the ladder top toward the seat point.
- `arrival_bonus` = `completion_bonus(predicate_fn=ladder_carried_to_fixture)`.
- `ladder_dropped` penalty (`object_dropped` on the ladder), `ladder_tipped` penalty,
  `termination_penalty`, base shaping.

## Terminations

`time_out`, `success=ladder_carried_to_fixture`, `fell_below`, `fell_over`, `ladder_tipped`,
`ladder_dropped`. `episode_length_s = 30.0` — the longest traverse of the chain, under load.

## Reuse

`ladder_fixture_distance`, `_ladder_top_point_w`, `_seat_point_w`, `LADDER_READY_XY_RADIUS`,
`LADDER_TILT_LIMIT`, `LADDER_WALL_STANDOFF`, `object_dropped`, `distance_progress`,
`completion_bonus`. New: `ladder_carried_to_fixture`.

## Visual start-state validation

Frames must show: the ladder upright and **clear of the floor**, held at one rail, not
intersecting the robot's legs or torso; the robot's stance able to take the load (feet under the
combined CoM, not splayed); a walkable path to the fixture target. Render from
`--record_view scene`, and for a ceiling mount also `--record_view fixture` to confirm the target
is under the fixture rather than beside it.

## Acceptance

As foundation, plus: **the retention gate** — hold the start pose under zero action for the full
30 s horizon and confirm the ladder is still in the hand. Report the result either way.

## Blockers and risks

- **Ladder mass is settled at `LADDER_MASS_KG = 7.25`** family-wide (foundation). This is the
  subtask it matters most for: Carry's old 3.0 kg existed precisely so that "one arm can move it",
  and that cushion is gone. If a one-armed carry fails at 7.25 kg, the finding is that this subtask
  needs a two-handed grasp — a *start-state* change owned by S02 — not a second mass for one object.
- **PR #64** — the still-held condition reads the filtered contact channel.
- Highest physical risk in the chain: a 7.25 kg dynamic body held at arm's length by a
  position-controlled hand, while walking. If the retention gate fails under zero action, S03 is
  unsolvable as specified and the honest report is that hand retention needs #54's mechanic.
