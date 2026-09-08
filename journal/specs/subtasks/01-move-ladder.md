# S01 — Move the ladder to the fixture

`FIATLUX-S01-MoveLadder-v0` · mode **navigation + grasping (unseparated)** · object **ladder**
Read `00-foundation.md`, `ABSTRACTIONS.md`, `CONTINUITY.md` and `CRITIQUE.md` first.

Replaces the retired `S01-ApproachLadder` / `S02-GrabLadder` / `S03-CarryLadder` /
`S04-PlaceLadder`, and is the reason the rest of the chain renumbered from fifteen slots to
twelve. The foundation's rule — every switch between navigation, balance and grasping is a
boundary — is deliberately **not** applied here, on the grounds the foundation itself states:
the rule is a mnemonic for the list, not a generator of it.

## Objective

Get the ladder from wherever the layout drew it to a position where a robot standing on it can
reach the fixture, and leave it standing there.

## Why the four collapsed into one

The four retired gates required, between them: a rail grasped and loaded past ~57 N, the ladder's
feet lifted 2 cm clear of the floor, that grip retained through the whole traverse, the hand
released at the end, and the robot left inside a rectangle on the ladder's step-facing side. Not
one of those is a property of the deliverable. Each is an independent way for an operator who has
already stood the ladder where it belongs to be scored zero, and the last of them —
`robot_at_ladder_base` — constrained a handoff that does not exist at run time, since S02 spawns
its own mounting stance rather than inheriting one.

The cost is real and worth stating: the four boundaries isolated *where* a policy fails, and one
episode cannot. That diagnostic is not lost, it moves — `FIATLUX-Carry-v0` is the coarse-tier task
that already spans approach + grasp + transport + place, and the failure localization the
subtasks were meant to provide for the ladder now has to come from the reward breakdown
(`approach_progress` vs `placement_progress`) rather than from four separate scores.

## Start state

The head of the chain, so the only start state drawn from the randomized layout rather than
composed from a predecessor's success state.

- `apply_replace_preset(scene)` — **uncoupled**. The ladder's floor zone is sampled independently
  of the fixture mount, because closing that gap is the task. Every other subtask draws with
  `couple_ladder_to_fixture=True`; this one must not.
- Robot in its own drawn zone, both hands free, re-aimed at the ladder (`face_robot_at`) rather
  than at the table the preset points it at.
- No payload composition, no `LADDER_IN_ROOT_CARRIED`, no carry arm pose. Those constants are gone
  with the retired legs: nothing spawns the ladder pre-held any more, so the grip-pose gap the
  foundation records as still-open no longer touches this leg at all.

## Success gate

`mdp.sustained(all_of(...), PLACE_SUSTAIN_SECONDS)` — the place tier's 1.0 s window, since the
claim is that the ladder stays standing once whatever was holding it stops.

| Conjunct | Value |
|---|---|
| `ladder_ready(xy_radius=LADDER_READY_XY_RADIUS, tilt_limit=LADDER_TILT_LIMIT)` | ladder top within reach of the seat, ladder upright |
| `ladder_feet_down(tolerance=0.02)` | root on the floor — a ladder *held* in the right place at the right angle is not standing |
| `object_at_rest(ladder, 0.05 m/s, 0.10 rad/s)` | settled, not swinging through |
| `robot_standing(FALL_MIN_HEIGHT, FALL_TILT_LIMIT)` | not scoring on the step the robot collapses |

**Nothing in the gate reads a contact sensor, and the leaf adds none.** Carrying, dragging,
shouldering, pushing along the floor and nudging a foot at a time are indistinguishable to it, by
construction. That is the point of the fold.

The one surviving constraint on *how* the ladder travels is the `ladder_tipped` termination at
`LADDER_TILT_LIMIT` = 0.6 rad, which is what "standing" means and is shared with nine other
leaves, `FIATLUX-Carry-v0` and `FIATLUX-Replace-v0`. A 34° lean while shoving the ladder along is
fine; laying it flat and walking it end over end ends the episode. Relaxing that for this leaf
alone is a one-line change to a leaf-local tilt bound, and would be a `-v1` bump.

## Rewards

Built on `PlaceSubtaskCfg`, so `placement_progress` = `distance_progress(ladder_fixture_distance)`
comes from the tier. The leaf adds two terms:

- `approach_progress` = `distance_progress(base_ladder_distance)` — the navigate tier's canonical
  name and function, declared here because this leaf spans both legs and the breakdown would
  otherwise be silent for the entire walk out to the ladder.
- `ladder_tipped` penalty, matching the termination.

Plus `success_bonus` and the shared shaping tail from `SubtaskRewardsCfg`.

## Terminations

`time_out`, `success`, `fell_below`, `fell_over`, `ladder_tipped`. No `ladder_dropped`: setting
the ladder down is the goal.

`episode_length_s = 120.0`.

## Reuse

`ladder_ready`, `ladder_fixture_distance`, `base_ladder_distance`, `ladder_tipped`,
`place_terms.ladder_feet_down`, `place_terms.object_at_rest`, `place_terms.robot_standing`,
`distance_progress`, `sustained`, `all_of`, `PlaceSubtaskCfg`. Nothing new was written for it.

Retired with the four legs, since nothing else reached them: `mdp.arrived_at_ladder`,
`nav_terms.arrived_carrying_ladder`, `grasp_terms.hand_ladder_distance`,
`grasp_terms.grasp_force_above`, `grasp_terms.ladder_feet_clear`,
`place_terms.robot_at_ladder_base`, and `grasp_poses`' `LADDER_IN_ROOT_CARRIED`,
`LADDER_CARRY_ARM_JOINT_POS`, `LADDER_GRIP_TRIPWIRE_N`. `LADDER_APPROACH_RADIUS` is kept: it is a
scene measurement, not gate machinery, and two live comments cite it to explain their own value.

## Validation — as run

`scripts/verify_scene.py --task FIATLUX-S01-MoveLadder-v0 --num_envs 1 --hold_base --seed 0`:
**43/43 PASS**, init drift 5.4 cm on the robot (the reset jitter) and 0.0 cm on every other
entity. Rendered orbit reviewed frame by frame: robot standing on the floor with both hands free
and empty, facing the ladder; ladder upright on its own feet in its drawn zone, well clear of the
fixture; fresh bulb standing on the table; crate on the floor; old bulb in the wall-mounted
fixture. Nothing floating, nothing interpenetrating.

`scripts/verify_move_ladder.py --headless`: **9/9 PASS**. It reads the gate through the env's own
termination manager, and stages the ladder at the pose the *rest of the chain* spawns
(`couple_ladder_to_fixture=True`, same layout seed) rather than at one it invented:

| Check | Result at seed 0 |
|---|---|
| `start:hands_free` | sensors are `ego_camera`, `hand_contact`, `mid360_lidar` — no grip or release channel exists for a gate to read |
| `start:robot_faces_ladder` | facing error 0.085 rad |
| `start:ladder_on_its_feet` | feet down, at rest |
| `start:ladder_not_yet_ready` | ladder top 6.131 m from the fixture, gate radius 0.670 m |
| `start:gate_open` | the episode does not begin solved |
| `gate:fires_at_chain_placement` | all four conjuncts true, `success` fires at step **50** = the 1.0 s sustain exactly |
| `gate:rejects_held_in_the_air` | same xy, ladder 0.5 m up: `ladder_feet_down` false, no success |
| `gate:rejects_tipped` | 1.0 rad roll: `ladder_tipped` terminates at step 1 |
| `gate:rejects_left_where_drawn` | upright and at rest 6.131 m away: `ladder_ready` false, no success |

The positive control is the load-bearing one: it is the same ladder pose S02–S12 spawn, so a gate
that failed there would mean this subtask and its successors disagree about where the ladder goes.

## Open

- **No rollout has run it.** The horizon, and whether the traverse is achievable at all under the
  ladder's 7.58 kg, are unmeasured. A 0% score here cannot yet be distinguished from an
  unsolvable subtask — the foundation's standing warning, now covering four legs' worth of task
  in one episode.
- **Tier membership thinned.** `GraspSubtaskCfg` is down to one member (S08) and `MateSubtaskCfg`
  to two. Both are kept: the split they encode — terms here, numbers in the leaf — is what keeps
  routine retuning out of shared code, and that does not depend on the member count.
