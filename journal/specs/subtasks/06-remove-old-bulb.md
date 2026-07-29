# S06 — Remove the old bulb while balancing on the ladder

`FIATLUX-S06-RemoveOldBulb-v0` · mode **grasping on balance** · object **old_bulb**
Read `00-foundation.md`, `ABSTRACTIONS.md` and `CONTINUITY.md` first. **Blocked on #54 — read the Blockers section first.**

## Objective

Standing on the upper steps, take hold of the seated old bulb, free it from the inverted fixture,
and end holding it — without falling, without tipping the ladder, and without crushing it.

## Start state

`START_STATE["S06"]` = S05's success state:

- Robot on the ladder's upper steps, pelvis at the top-step height, balanced, hands free.
- Ladder standing at the fixture target, at rest.
- Old bulb **seated** in the fixture: pose == the fixture's own pose, both halves being authored
  assembled at identity (`SOCKET_SEAT_OFFSET == BULB_PLUG_OFFSET == (0, 0, 0.036259)`).
- Fixture **inverted** (`_quat_y_deg(180)` for a ceiling mount), so the bulb hangs downward.

## Success gate

`old_bulb_taken` — all of:

| Condition | Value | Source |
|---|---|---|
| bulb cleared the seat | `REMOVAL_CLEARANCE = 0.10` m | `old_bulb_removed` |
| **bulb held in the hand** | hand↔old_bulb contact > 2 N | filtered channel |
| bulb not dropped | `old_bulb_dropped` false | existing |
| robot not fallen | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` | base |
| ladder not tipped | `LADDER_TILT_LIMIT = 0.6` | existing |
| sustained | 0.5 s | `sustained` |

**The held condition is the entire subtask.** Clearance alone is satisfied by the bulb *falling
out of an inverted socket under gravity* — which, with #54 unimplemented, is exactly what happens
at t=0 with no action at all. A clearance-only gate would score a 100% success rate for a policy
that does nothing. Do not ship the gate without the held condition, and verify by running a
zero-action rollout: it must score 0.

## Fragility

Grip force on the filtered hand↔old_bulb channel, two bounds, no geometric attribution through
the robot body:

- `GLASS_CONTACT_LIMIT_N = 50.0`
- `CAP_CONTACT_LIMIT_N = 300.0`

Real reference: IEC 60968 specifies a 3 N·m cap-to-glass torsion test for E26/E27, which is what
makes the cap tolerant and the glass not. The sensor cannot attribute contact per feature (the
bulb is one rigid body, so PhysX reports per body, not per collider), so the two bounds are
applied as thresholds on the same channel — the decision already taken for the verification
scenarios. Note the limitation at the cfg site.

## Rewards

- `reach_progress` = `distance_progress(distance_fn=palm_old_bulb_distance)`.
- `removal_progress` = `distance_progress(distance_fn=old_bulb_fixture_clearance,
  away_threshold=REMOVAL_CLEARANCE)` — the away-from formulation, because the bulb starts *at* the
  fixture so `d0 ≈ 0` and the normalized form would divide by ~zero. This is why
  `away_threshold` exists.
- `removal_bonus` = `completion_bonus(predicate_fn=old_bulb_taken)`.
- `contact_penalty` = `hand_contact_force_l2` on the filtered channel.
- `old_bulb_dropped` penalty, `ladder_tipped` penalty, `termination_penalty`, base shaping.

## Terminations

`time_out`, `success=old_bulb_taken`, `fell_below`, `fell_over`, `ladder_tipped`,
`old_bulb_dropped`. `episode_length_s = 30.0`.

## Reuse

`old_bulb_removed`, `old_bulb_fixture_clearance`, `_old_bulb_plug_point_w`, `old_bulb_dropped`,
`REMOVAL_CLEARANCE`, `OLD_BULB_DROP_HEIGHT = 0.15`, `object_contact_forces`,
`hand_contact_force_l2`, `HAND_CRADLE_DEX3`, `ARM_CRADLE`, `distance_progress`,
`completion_bonus`. New: `palm_old_bulb_distance`, `old_bulb_taken`.

## Visual start-state validation

Frames must show: the robot balanced on the upper steps, both feet on a step (not one in the air,
not a foot through a step); the old bulb hanging in the inverted fixture, cap up, glass down,
**not intersecting the socket collar**; the bulb within the robot's reach from that stance; and
the bulb visible in the ego camera from the on-ladder pose. Use `--record_view fixture`.

Then the critical one: **run a zero-action rollout and watch what the bulb does.** If it falls out
immediately, report that as the #54 blocker manifesting, and record the fall time.

## Deliverable for the chain

**`BULB_IN_ROOT_ON_LADDER`** (foundation probe 2) — the bulb's root pose in the robot root frame
for the on-ladder cradle grasp. S07 starts from it, so S06 owns measuring and freezing it. The
on-ladder stance carries a torso lean, so this is *not* the standing value from probe 1.

## Blockers

- **#54 (bulb attach/detach)** — the fixture is inverted and nothing retains the bulb. Author the
  subtask, validate the start state, measure the fall, and mark the success gate **provisional**
  until #54 lands. Do not work around it by making the bulb kinematic: a kinematic bulb cannot be
  moved by any action, which is what made `FIATLUX-Remove-v0` unsolvable.
- **`verify_interactions --scenario socket` is 3/6**, all three failures being bulb↔socket
  contact — the same root cause. Do not treat those as S06 regressions.
- **PR #64** — the held condition and the fragility bounds both read the filtered channel.
