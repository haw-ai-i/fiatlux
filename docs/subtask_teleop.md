# Subtask teleoperation

Every benchmark subtask `FIATLUX-SNN-<Name>-v0` has a teleop twin
`FIATLUX-SNN-<Name>-Teleop-v0`. The twin subclasses the RL cfg and swaps only the action
interface, so changes to a subtask flow through to its twin automatically.

- Recipe: `source/fiatlux_teleop/fiatlux_teleop/subtask_teleop.py`
- Per-task cfgs: `source/fiatlux_teleop/fiatlux_teleop/subtasks/`
- Driver: `scripts/teleop/sonic_teleop.py`
- VR launcher: `scripts/teleop/restart_sonic_teleop.sh`
- Tests: `source/fiatlux_teleop/tests/test_subtask_teleop_table.py` (pure AST, no Isaac)

## What the twin changes

| | RL env | Teleop twin |
|---|---|---|
| Actions | `joint_pos`, all joints | bimanual IK (`arm_action`, `left_arm_action`) + binary grips |
| Terminations | task's own set | all cleared — the session is operator-paced |
| Legs | policy | SONIC walk/balance ONNX, driven outside the action manager |
| Camera | task viewer | pelvis-anchored `XrCfg` follow camera |

Scene, assets, events, `sim.dt` and decimation are untouched.

## Running

```bash
# keyboard, no headset
python scripts/teleop/sonic_teleop.py --task FIATLUX-S06-RemoveOldBulb-Teleop-v0 --input keyboard

# VR over CloudXR
NV_CXR_ENDPOINT_IP=<ip> FIATLUX_TASK=FIATLUX-S06-RemoveOldBulb-Teleop-v0 \
  bash scripts/teleop/restart_sonic_teleop.sh
```

Always use `restart_sonic_teleop.sh` for VR. It restarts the CloudXR runtime and clears its
state files. Reusing a runtime across Isaac restarts leaves signalling working while media
negotiation fails — the client connects, looks healthy, and drops after about 30 seconds.

Keyboard: arrows walk, `SPACE` stops, `TAB` switches arm, `W/S A/D Q/E` move the end effector,
`U/O I/K J/L` rotate the wrist, `G` grips, `C` toggles recording, `R` resets.

## The 15 subtasks

Heights are the pelvis after the driver's settle. On-ladder tasks are staged at the tread
(1.18 m) plus `TOP_STANCE_PELVIS_OFFSET` (0.787 m) = **1.97 m** nominal.

| | Task | Start | Holding at spawn | On reset | Ladder zone |
|---|---|---|---|---|---|
| S01 | ApproachLadder | floor, 0.74 | — | robot ±5 cm | own draw |
| S02 | GrabLadder | floor, 0.74, in grasp range of a rail | — | robot ±5 cm | own draw |
| S03 | CarryLadder | floor, 0.70 | ladder, composed into grip | robot ±5 cm | own draw |
| S04 | PlaceLadder | floor, 0.70, at the fixture | ladder, composed into grip | robot ±5 cm | own draw |
| S05 | ClimbLadder | floor, 0.74, at the steps | — | robot ±5 cm | fixture anchor |
| S06 | RemoveOldBulb | ladder, 1.97 | — | robot ±5 cm | fixture anchor |
| S07 | DescendWithBulb | ladder, 1.97 | old bulb, on open palm | **pinned** | fixture anchor |
| S08 | CarryBulbToDisposal | floor, 0.74 | old bulb, on open palm | **pinned** | fixture anchor |
| S09 | DisposeBulb | floor, 0.74, at the crate | old bulb, on open palm | robot ±5 cm | fixture anchor |
| S10 | ApproachNewBulb | floor, 0.74 | — | robot ±5 cm | fixture anchor |
| S11 | GrabNewBulb | floor, 0.67, at the bench | — | robot ±5 cm | fixture anchor |
| S12 | CarryBulbToLadder | floor, 0.74 | fresh bulb, on open palm | **pinned** | fixture anchor |
| S13 | ClimbWithBulb | floor, 0.74, at the steps | fresh bulb, on open palm | **pinned** | fixture anchor |
| S14 | ScrewInBulb | ladder, 1.97 | fresh bulb, on open palm | **pinned** | fixture anchor |
| S15 | ClimbDown | ladder, 1.97 nominal | — | robot ±5 cm | fixture anchor |

**Own draw** means the ladder takes its own zone from the layout sampler. **Fixture anchor**
means `couple_ladder_to_fixture=True` — the ladder is placed at the fixture's reserved anchor
so the socket is always reachable. Whether that anchor sits directly under the fixture or
offset from it depends on the draw: a ceiling mount puts the ladder underneath, a wall mount
puts it out in front.

**Pinned** tasks set both `position_range` and `pose_range` to `(0.0, 0.0)`, so reset returns
the robot to exactly its staged pose. The comment in `s13_climb_with_bulb_env_cfg.py` gives the
reason: the open-palm payload staging is too fragile to survive the noise.

## Randomization: two levels

Layout and reset randomization are separate, and confusing them wastes time.

### Per process launch — the room

`apply_replace_preset` runs inside the cfg's `__post_init__`, so this is drawn **once per
scene build**:

- fixture mount, ceiling or wall, sampled first so its ladder anchor can be reserved
- ladder yaw, uniform 0–360°
- four non-overlapping zone centres — robot, table, ladder, disposal — inside
  `ROOM_FLOOR_MIN (-4.0, -3.0)` to `ROOM_FLOOR_MAX (4.0, 4.2)`, retried up to 64 times

The function's own docstring states the consequence: *"one layout serves every env and every
episode of a run, and varying it is a between-runs affair."* This is intended. It also means
every env in a parallel run shares one room — verified with `num_envs=3`, where all three had
an identical ladder position and differed only by the robot's reset jitter.

### Per reset — the robot and the lights

- robot root: x, y ±5 cm, yaw ±0.1 rad (about ±6°) — zero on the five pinned tasks
- robot joints: ±0.05 rad — zero on the same five
- dome light 600–1400 and key light 800–2200, both re-aimed; room tint; hand grip material

**`R` does not give you a new room.** Verified by resetting four times and reading back every
asset position: identical to three decimals. To change the layout, relaunch.

## Layout seed

The room is drawn from OS entropy unless seeded, which makes a bad draw impossible to hand to
anyone else. The driver therefore always seeds, and always prints what it used:

```bash
--layout_seed 42       # that exact room, every time
--layout_seed random   # draw one (default); the drawn seed is printed
--layout_seed none     # unseeded, as before
```

`FIATLUX_LAYOUT_SEED` passes the same value through `restart_sonic_teleop.sh`.

The seed is stored in each demo bag's `meta.json`, so a recorded demo carries the room it was
collected in. It was previously hardcoded to `0`.

Seeding must happen before `parse_env_cfg` — the layout is sampled during `__post_init__`, so
seeding afterwards silently does nothing. A test guards the ordering.

## Hand variants

Both G1 hands work on all 15 tasks: `--hand inspire` or `--hand dex3`. The swap happens inside
the recipe before the action terms are built, so grips are authored against the target hand.

Two benchmark-side fixes were needed, both because `swap_robot_variant` rewrites **joint**
names only:

- `mdp/grasp_terms.py` hardcoded the Inspire palm body. It now resolves the palm from the
  mounted articulation, matching what `nav_terms.py` already did.
- `nav_terms.settle_carried_payload_live` takes a `hand_variant` parameter defaulting to
  `"inspire"`, and no task passes it. `swap_robot_variant` now inspects each term's signature
  and sets that parameter.

Without those, six subtasks failed at env creation under Dex3 with
`Not all regular expressions are matched: right_hand_base_link`.

## Known issues

**Payloads are not attached.** The eight carrying tasks stage their payload unsecured — the
bulb rests on an open palm (`settle_carried_payload_live`, friction only), and the ladder is a
free rigid body positioned in the grip. `s03_carry_ladder_env_cfg.py` carries an explicit TODO
saying nothing holds it. Close the grip immediately on spawn or the payload drops.

**A fall becomes a launch.** SONIC has no fall recovery. Once the pelvis is down the policy is
out of distribution and its output is still applied as joint position targets, so the robot is
driven across the room at 4–8 m/s. The same tasks in the RL env peak at 0.5–2 m/s. Freezing the
legs at the default stance below 0.45 m cuts peak speed by about 55% in testing; not applied,
because the threshold would also fire during a legitimate deep crouch.

**Bad draws happen.** Some layouts collapse before the operator has control — the robot is
placed 2 m up on a free-standing ladder, and if that perch does not settle, both go down. The
benchmark models this: `ladder_tipped` is both a −200 reward and a termination. Relaunching
redraws; resetting does not. With the seed printed, a bad draw is now reproducible.

**S15 slips during settle.** It is staged at the same 1.97 m tread height as the other
on-ladder tasks but has been observed settling to 1.44–1.58 m. Cause not yet established.

**Hands-off sweeps mislead on carrying tasks.** With no operator, grips never close, so the
payload drops and the robot trips on it. A "fell" verdict there measures the dropped payload,
not the task.
