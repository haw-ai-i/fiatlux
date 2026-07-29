# S07 — Walk down the ladder with the old bulb in hand

`FIATLUX-S07-DescendWithBulb-v0` · mode **balance (loaded)** · payload **old_bulb**
Read `00-foundation.md` and `CONTINUITY.md` first.

## Objective

Carry the removed old bulb down the ladder to the floor, arriving standing, still holding it,
unbroken.

## Start state

`START_STATE["S07"]` = S06's success state:

- Robot on the upper steps, balanced.
- Old bulb **in hand** — root pose from `BULB_IN_ROOT_ON_LADDER` (S06's deliverable), composed with
  the robot root pose. Grasping hand pinned by `mdp.hold_grasp_pose` for step 0.
- Ladder standing at the fixture target; fixture now empty.

## Success gate

`descended_with_bulb` — all of:

| Condition | Value |
|---|---|
| pelvis below the floor-stance height near the ladder base | ladder-derived, see below |
| pelvis within xy radius of the ladder base | 0.6 m (Descend's tolerance) |
| root speed | < 1.5 m/s |
| **bulb still held** | hand↔old_bulb contact > 2 N |
| bulb not dropped | `old_bulb_dropped` false |
| robot not fallen | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` |
| ladder not tipped | `LADDER_TILT_LIMIT` |

As in S05, the height and xy centre must come from the ladder's **live** pose, not from
`CLIMB_ROBOT_POSITION` — `FIATLUX-Descend-v0` hardcodes `SUCCESS_XY` from that constant and it
describes the default workshop layout, not this chain's placed ladder. New predicate
`descended_from_ladder(payload=...)`, mirroring `climbed_to_ladder_top`.

## Rewards

- `descend_progress` = `descend_height_progress` — unchanged, pays each centimetre lost once.
- `success_bonus` = `completion_bonus(predicate_fn=descended_with_bulb)`.
- `ladder_contact` = `ladder_contact_fraction`, weight 0.25 — the same bootstrap Climb uses;
  a controlled descent keeps limbs on the ladder.
- `contact_penalty` = `hand_contact_force_l2` on the filtered channel (glass bound: a panicked
  grip while balancing is exactly how a real bulb gets crushed).
- `old_bulb_dropped` penalty, `ladder_tipped` penalty, `com_sway`, `termination_penalty`,
  base shaping. No `flat_orientation_l2`.

## Terminations

`time_out`, `success=descended_with_bulb`, `fell_below`, `fell_over`, `ladder_tipped`,
`old_bulb_dropped`. `episode_length_s = 30.0`.

Note `fell_below` and the success height gate both read pelvis height and must not collide:
`FALL_MIN_HEIGHT = 0.35` sits below the floor-stance success height (~0.94 m in Descend's
formulation), so a controlled arrival fires `success` and not `fell_below`. Confirm the ordering
empirically — a hard landing that dips the pelvis under 0.35 m should read as a fall, and that is
correct behaviour, not a bug to tune away.

## Reuse

`descend_height_progress`, `descended_to_target` (as the template for the live-pose variant),
`ladder_contact_fraction`, `old_bulb_dropped`, `hand_contact_force_l2`, `BULB_IN_ROOT_ON_LADDER`,
`GLASS_CONTACT_LIMIT_N`, `CAP_CONTACT_LIMIT_N`, `completion_bonus`.
New: `descended_from_ladder`, `descended_with_bulb`.

## Visual start-state validation

Frames must show: the robot on the upper steps with both feet on a step; the bulb **enclosed in
the hand**, not resting on a fingertip and not intersecting the palm mesh; the fixture above now
empty; the ladder at rest. Render from `--record_view scene` and confirm the descent path down the
steps is unobstructed.

The bulb-in-hand frame is the one to be sceptical about. A bulb pinched against the thumb outside
the hand has passed a 1 cm coded proximity check on this project twice. Look at it.

## Acceptance

As foundation, plus **the retention gate**, which is the real risk here: hold the start pose under
zero action for the full 30 s and confirm the bulb stays in the hand. Then repeat while the body
is in motion — a static hold proves nothing about a descent's accelerations. If the bulb leaves the
hand under either, report it: S07 is then unsolvable as specified, and the honest fix is folding
hand retention into #54's mechanic rather than tightening the finger curl (curl above ~1.3 already
exceeds the 50 N glass bound, and past ~1.4 the closing fingers eject the bulb — the grip has no
headroom left).

## Blockers

**PR #64** — the still-held condition and the fragility penalty read the filtered channel.
**#54** — inherits S06's provisional start state: if the bulb cannot be removed and held, S07's
start state cannot be reached by the chain, though it can still be authored and validated directly.
