# S14 — Screw the bulb in while on the ladder

`FIATLUX-S14-ScrewInBulb-v0` · mode **grasping on balance** · object **bulb** + **socket**
Read `00-foundation.md`, `ABSTRACTIONS.md`, `CONTINUITY.md` and `CRITIQUE.md` first. **Blocked on #54 — read the Blockers section first.**

## Objective

Balanced on the upper steps holding the fresh bulb, seat it in the inverted fixture and leave it
seated — the terminal manipulation of the whole task.

## Start state

`START_STATE["S14"]` = S13's success state:

- Robot on the ladder's upper steps, balanced, fresh bulb in hand (`BULB_IN_ROOT_ON_LADDER`).
- Ladder standing at the fixture target, at rest.
- Fixture **inverted and empty**, its socket mouth facing down.
- Old bulb at rest in the disposal crate.

## Success gate

`bulb_screwed_in` — all of:

| Condition | Value | Source |
|---|---|---|
| `bulb_seated(pos, ori)` | `SEAT_POS_THRESHOLD = 0.015`, `SEAT_ORI_THRESHOLD = 0.2` | existing |
| bulb at rest | lin vel < 0.05 m/s, ang vel < 0.10 rad/s | new |
| **retained after release** | still seated 1.0 s after hand contact drops below 1 N | `sustained` |
| bulb not broken | peak grip never exceeded `GLASS_CONTACT_LIMIT_N = 50` | filtered channel |
| robot not fallen | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` | base |
| ladder not tipped | `LADDER_TILT_LIMIT = 0.6` | existing |

The retained-after-release condition is what makes this "screwed in" rather than "held in the
socket". It is also the condition #54 owns: the fixture is inverted, so a merely-resting bulb falls
straight out the moment the hand lets go.

**Seated is exact here.** Both halves are authored assembled at identity, so
`SOCKET_SEAT_OFFSET == BULB_PLUG_OFFSET == (0, 0, 0.036259)` and *seated ⟺ bulb root pose ==
socket root pose*. There is no offset arithmetic to get wrong at any mount orientation.

**Rotation about the mating axis must be free.** `BULB_PLUG_AXIS == SOCKET_SEAT_AXIS == (0, 0, 1)`,
and seating is scored on alignment *about* that axis — so the screwing motion itself costs nothing.
If the orientation term is built on full-quaternion error instead of axis alignment, the dense
reward fights the screw. `object_socket_orientation_tanh` must be on the axis angle.

## Rewards

- `approach_progress` = `distance_progress(distance_fn=bulb_fixture_distance)` — the existing
  Replace channel: fresh-bulb plug toward the fixture seat.
- `alignment` = `object_socket_orientation_tanh` on the **mating-axis** angle.
- `seating_bonus` = `completion_bonus(predicate_fn=bulb_screwed_in)`.
- `contact_penalty` = `hand_contact_force_l2` on the filtered channel.
- `bulb_dropped` penalty, `ladder_tipped` penalty, `termination_penalty`, base shaping.

Consider exposing the **accumulated signed roll about the mating axis while in contact** as a term
here — it is the natural home for the spec's "accumulated wrist roll" gate, it costs nothing, and
S14 is the only subtask that screws anything.

## Terminations

`time_out`, `success=bulb_screwed_in`, `fell_below`, `fell_over`, `ladder_tipped`, `bulb_dropped`.
`episode_length_s = 40.0` — the longest of the chain; fine insertion under balance.

## Reuse

`bulb_seated`, `bulb_fixture_distance`, `_seat_point_w`, `_plug_point_w`,
`object_socket_distance_tanh`, `object_socket_orientation_tanh`, `SEAT_POS_THRESHOLD`,
`SEAT_ORI_THRESHOLD`, `SOCKET_SEAT_OFFSET`, `BULB_PLUG_OFFSET`, `BULB_PLUG_AXIS`,
`SOCKET_INSERTION_DEPTH = 0.034226`, `hand_contact_force_l2`, `completion_bonus`, `sustained`.
New: `bulb_screwed_in`.

## What must not be touched

The socket's colliders are an **exact triangle mesh** (`physics:approximation = "none"`),
specifically so the screw hole stays open. A convex hull closes it and an exact mesh is illegal on
a dynamic body, so the socket can never be anything but static or kinematic. Do not change the
socket's collision approximation, and do not change the bulb or socket colliders at all — that is
#54's territory (measured there: 0.65 mm radial interference as authored, a seated bulb takes 50 N
axial pull and 0.8 N·m without moving).

Contact offsets are already tuned for insertion (`contact_offset=0.005`, `rest_offset=0.0`,
`torsional_patch_radius=0.005`) against a cap radius of 0.0208 m; the PhysX default of 0.02 m is
half the cap diameter and causes the classic buzzing in the hole. Leave them.

## Visual start-state validation

Frames must show: the robot balanced on the upper steps, both feet on a step; the bulb enclosed in
the hand, cap oriented toward the socket mouth (cap **up**, since the fixture is inverted — a bulb
held cap-down cannot be inserted without a full wrist reversal, and whether the Dex3 wrist has that
range from this stance is worth checking before writing rewards); the socket mouth open and empty,
within reach from the stance. Use `--record_view fixture` — it is the only view that shows an
overhead mount properly.

Report the measured distance from the held bulb's cap to the socket seat point in the start pose.

**This is a blocking probe for the whole ladder half of the chain, not just this subtask.** The margin
is 7.4 cm and measured from the wrong stance:

| Quantity | Value |
|---|---|
| ladder top → ceiling fixture | 1.30 m |
| `G1_OVERHEAD_REACH` | 1.3738 m |
| margin straight up | **0.074 m** |

`G1_OVERHEAD_REACH` was measured standing on a flat surface, arm straight up, every other arm joint
at zero. The robot here is in a balanced crouch on a ladder step, leaning, one hand holding a bulb it
must not crush — each of which costs vertical reach. If the on-ladder reach comes out below 1.30 m,
then climb, remove-the-old-bulb, climb-with-bulb, screw-in and climb-down are **all** unsolvable at
`CEILING_FIXTURE_Z = 3.0` and the constant has to move. Measure it before any of those five is
authored. Tracked as **#69**.

## Acceptance

As foundation, plus: teleport the bulb to the socket's pose, soak 4 s, and confirm `bulb_seated`
stays true — the success predicate's own acceptance test · the retained-after-release condition
tested explicitly with a scripted release · `handoff:S13->S14` and `handoff:S14->S15` pass.

## Blockers

- **#54 (bulb attach/detach)** — the retained-after-release condition cannot pass without it. The
  fixture is inverted; nothing holds the bulb. Author the subtask, validate the start state, test
  every other condition, and mark the gate **provisional**.
- **`verify_interactions --scenario socket` is 3/6**, all three being bulb↔socket contact (137 N,
  never settles). Same root cause. Not an S14 regression.
- **PR #64** — the fragility bound and the release detection read the filtered channel.
