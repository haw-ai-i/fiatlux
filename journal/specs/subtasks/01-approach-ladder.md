# S01 — Approach the ladder

`FIATLUX-S01-ApproachLadder-v0` · mode **navigation** · payload **none**
Read `00-foundation.md`, `ABSTRACTIONS.md` and `CONTINUITY.md` first.

## Objective

From the Replace layout's robot spawn, walk to the ladder and end standing within grasping
range of a rail, facing it, with the ladder untouched and upright.

## Start state

`START_STATE["S01"]` — the head of the chain, so it is the only start state drawn from the
randomized layout rather than inherited.

- Layout from `apply_replace_preset(scene, rng)` with `set_layout_seed(seed)`, so the
  robot→ladder gap varies between runs from declared entropy. `distance_progress` normalizes by
  the episode's own `d0`, so a close draw cannot outscore a far one.
- Robot: `ROBOT_POSITION[2]` pelvis height at the sampled robot zone centre, yaw facing the
  **ladder** (S01's target), not the table — `apply_replace_preset` currently faces the robot at
  the table because Replace's first target is the fresh bulb. Override the facing here.
- Ladder: dynamic, at its sampled zone, `LADDER_MASS_KG = 7.25` (unified, see foundation).
- Everything else (table, fresh bulb, disposal crate, fixture, old bulb) present as dressing.
- Payload: none. Hands at the standing default.

Measured gap for reference: the WBC walk probe crossed **2.43 m** robot→ladder at seed 0 in
~8 s at ~0.5 m/s, so the horizon below is ~2× the observed traverse.

## Success gate

`arrived_at_ladder` — all of:

| Condition | Value | Source |
|---|---|---|
| robot root within xy radius of ladder root | `LADDER_APPROACH_RADIUS` | **probe (below)** |
| facing error to ladder bearing | < 0.5 rad | new `base_facing_entity` |
| root speed | < 1.0 m/s | the `climbed_to_target` cap convention |
| robot standing | not `fell_below` / `fell_over` | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` |
| ladder upright | not `ladder_tipped` | `LADDER_TILT_LIMIT = 0.6` |

The speed cap and the two negative conditions are what stop the degenerate solution: charging
the ladder, knocking it over, and landing inside the radius must not score.

**Probe (blocking):** `LADDER_APPROACH_RADIUS` does not exist. `G1_OVERHEAD_REACH = 1.3738` is
*vertical*. Measure the G1's horizontal reach to a rail at grasp height by FK, then set the
radius so a robot inside it can touch a rail without stepping. The A-frame footprint is
0.68 × 1.11 m and the ladder root is at its base centre, so expect ~0.7–0.9 m. Freeze it with a
`CALIBRATED` note; S02 consumes the same constant.

## Rewards

- `approach_progress` = `distance_progress(distance_fn=base_ladder_distance)`, weight per
  Replace's ladder channel.
- `arrival_bonus` = `completion_bonus(predicate_fn=arrived_at_ladder)`.
- `ladder_tipped` penalty, `termination_penalty` (`fall_terminated`), plus the base shaping set.

`distance_progress` requires a bare module-level `(env) -> Tensor` for `distance_fn` — no
lambdas, no closures (its own docstring, and why `removal_bulb_disposal_distance` exists as a
wrapper). Add `base_ladder_distance(env)` to `mdp/rewards.py` in that style.

## Terminations

`time_out`, `success=arrived_at_ladder`, `fell_below`, `fell_over`, `ladder_tipped`.
`episode_length_s = 20.0`.

## Reuse

`apply_replace_preset`, `set_layout_seed`, `LADDER_TILT_LIMIT`, `distance_progress`,
`completion_bonus`, `ladder_tipped`, `fall_terminated`, `add_ego_camera`, `add_mid360_lidar`.
New: `base_ladder_distance`, `base_facing_entity`, `arrived_at_ladder`,
`LADDER_APPROACH_RADIUS`.

## Visual start-state validation

Frames must show: robot standing, both feet flat on the floor; the ladder fully deployed,
upright, all four feet on the floor; a clear walkable gap between them with no prop in the path;
and **the ladder inside the ego camera frustum at t=0** — the ego camera is mounted 46° down,
which is the true hardware mount, so confirm rather than assume. Report the measured gap.

If the ladder is not visible at spawn, report it as a start-state finding and stop — do not
adjust the camera. The mount is a benchmark-wide decision, not S01's to make.

## Acceptance

Registered and constructs · `verify_scene.py --task FIATLUX-S01-ApproachLadder-v0` all checks
pass · settle drift < 2 mm · visual checklist reported frame by frame · `handoff:S01->S02`
passes · zero-action and random-action rollouts complete without a solver explosion.

## Notes

Nothing here is blocked. S01 is the cheapest of the 15 and the natural first implementation —
it exercises the shared base, the layout seeding, and the visual gate end to end before any
grasp or balance machinery is involved.
