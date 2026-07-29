# S02 — Grab the ladder

`FIATLUX-S02-GrabLadder-v0` · mode **grasping** · object **ladder**
Read `00-foundation.md`, `ABSTRACTIONS.md` and `CONTINUITY.md` first.

## Objective

Standing at a rail, close a hand on the ladder and take its weight — proven by lifting it clear
of the floor without tipping it.

## Start state

`START_STATE["S02"]` = S01's success state, frozen:

- Robot at `LADDER_APPROACH_RADIUS` from the ladder root, facing it, standing, hands at the
  default open pose.
- Ladder dynamic, upright, feet on the floor, at the layout's sampled ladder zone.
- Payload: none yet.

## Success gate

`ladder_grasped` — all of:

| Condition | Value |
|---|---|
| hand↔ladder contact on ≥ 2 hand bodies | > 5 N each, filtered channel |
| ladder root lifted above its start z | > 0.03 m |
| ladder not tipped | `LADDER_TILT_LIMIT = 0.6` |
| robot standing | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` |
| sustained | 0.5 s (`sustained` helper, foundation) |

**Lift is the gate, not contact.** Contact alone is satisfied by leaning on the ladder; taking
its weight is what a grasp is. The 3 cm threshold is above the ladder's resting contact jitter
and well below a carry height.

## Sensing

The `hand_contact` sensor is filtered to the *manipulated objects*, and in this subtask the
ladder is one. Extend the filter in the preset:

```python
scene.hand_contact.filter_prim_paths_expr = [
    *scene.hand_contact.filter_prim_paths_expr, "{ENV_REGEX_NS}/Ladder",
]
```

The filter target **must be a rigid body** — this preset's ladder is dynamic
(`_collision_rigid`), so it qualifies. A static or `AssetBaseCfg` ladder can never appear in
`force_matrix_w`; PhysX returns zero silently, and reading that silence as "no contact" has
already cost this project two misdiagnoses. Assert non-zero contact in the probe before
trusting the channel.

Contact bound: the ladder is not fragile, so the bulb's cap/glass limits do not apply. Bound
grip force at **500 N** purely as a runaway-solver tripwire, and say so at the constant.

## Rewards

- `grasp_approach` = `distance_progress(distance_fn=palm_ladder_distance)` — palm to nearest
  rail, dense shaping toward contact.
- `contact_bootstrap` = `ladder_contact_fraction`-style term on the hand sensor, weight small
  enough to be unfarmable against the completion bonus (Climb's 0.25 vs 500 is the precedent).
- `grasp_bonus` = `completion_bonus(predicate_fn=ladder_grasped)`.
- `ladder_tipped` penalty, `termination_penalty`, base shaping.

## Terminations

`time_out`, `success=ladder_grasped`, `fell_below`, `fell_over`, `ladder_tipped`.
`episode_length_s = 15.0`.

## Reuse

`_spawn_usd_as_rigid_body_frictional` (the ladder's high-friction grip material — the affordance
this subtask depends on), `STEP_LADDER_RIGID_USD`, `HAND_CRADLE_DEX3` / `HAND_FLAT_DEX3` as pose
references, `object_contact_forces`, `distance_progress`, `completion_bonus`, `ladder_tipped`.
New: `palm_ladder_distance`, `ladder_grasped`, `sustained`.

## Deliverable for the chain

**`LADDER_IN_ROOT_CARRIED`** (foundation probe 3) — the ladder's root pose in the robot root
frame at the moment `ladder_grasped` fires. S03 starts from it, so S02 owns measuring and
freezing it. Take it from a scripted grasp, not from a policy rollout, and record the probe
command in the constant's comment.

## Visual start-state validation

Frames must show: the robot's hand at rail height beside the ladder, not inside it; the ladder
upright with all feet on the floor; no interpenetration between the hand and the rail at t=0.
Then, separately, render the *grasped* state used to derive `LADDER_IN_ROOT_CARRIED` and confirm
the rail is enclosed by the digits — the Dex3 thumb tip reaches 6.5 cm off the palm plane and
the fingertips 12.4 cm out, so a rail can sit *against* the fingers while looking held. This is
the exact failure that was reported as a successful grasp twice before renders settled it.

## Acceptance

As foundation, plus: the filtered hand↔ladder channel reads non-zero under a scripted grasp
(prove the filter works before scoring on it) · `LADDER_IN_ROOT_CARRIED` measured, frozen, and
visually confirmed · `handoff:S02->S03` passes.

## Blockers

**PR #64** must merge first — the whole success gate reads the filtered contact channel.
