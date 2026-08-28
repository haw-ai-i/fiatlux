# Subtask teleoperation

Every benchmark subtask `FIATLUX-SNN-<Name>-v0` has a teleop twin
`FIATLUX-SNN-<Name>-Teleop-v0`. The twin subclasses the RL cfg and swaps only the action
interface, so changes to a subtask flow through to its twin automatically.

- Recipe: `source/fiatlux_teleop/fiatlux_teleop/subtask_teleop.py`
- Per-task cfgs: `source/fiatlux_teleop/fiatlux_teleop/subtasks/`
- Driver: `scripts/teleop/sonic_teleop.py`
- VR launcher: `scripts/teleop/restart_sonic_teleop.sh`

## What the twin changes

| | RL env | Teleop twin |
|---|---|---|
| Actions | `joint_pos`, all joints | bimanual IK (`arm_action`, `left_arm_action`) + binary grips |
| Terminations | task's own set | failures and timeout cleared; `success` kept |
| Legs | policy | SONIC walk/balance ONNX, driven outside the action manager |
| Camera | task viewer | pelvis-anchored `XrCfg` follow camera |

Scene, assets, events, `sim.dt` and decimation are untouched.

`success` is kept deliberately. It is the only place a subtask's success predicate is
evaluated, and `recording.term_flag` records it per step as `success_term` — which is what
`scripts/score.py` reads. Clearing it does not disable scoring; `term_flag` falls back to an
all-False vector, so every recorded demo silently scores `success_rate 0.0`.

## Running

```bash
# keyboard, no headset
python scripts/teleop/sonic_teleop.py --task FIATLUX-S03-RemoveOldBulb-Teleop-v0 --input keyboard

# VR over CloudXR
NV_CXR_ENDPOINT_IP=<ip> FIATLUX_TASK=FIATLUX-S03-RemoveOldBulb-Teleop-v0 \
  bash scripts/teleop/restart_sonic_teleop.sh
```

Always use `restart_sonic_teleop.sh` for VR. It restarts the CloudXR runtime and clears its
state files. Reusing a runtime across Isaac restarts leaves signalling working while media
negotiation fails — the client connects, looks healthy, and drops after about 30 seconds.

Keyboard: arrows walk, `SPACE` stops, `TAB` switches arm, `W/S A/D Q/E` move the end effector,
`U/O I/K J/L` rotate the wrist, `G` grips, `C` toggles recording, `R` resets. Forward reach
saturates around 0.35 m from the pelvis — past that the arm is at its kinematic limit.

## The 12 subtasks

The four ladder legs were folded into `S01-MoveLadder`; everything after it shifted down by
three. On-ladder tasks are staged at the tread (1.18 m) plus `TOP_STANCE_PELVIS_OFFSET`
(0.787 m) = **1.97 m** nominal.

| | Task | Start | Holding at spawn | On reset |
|---|---|---|---|---|
| S01 | MoveLadder | floor, own zone | — | robot ±5 cm |
| S02 | ClimbLadder | floor, at the steps | — | robot ±5 cm |
| S03 | RemoveOldBulb | ladder, 1.97 | — | robot ±5 cm |
| S04 | DescendWithBulb | ladder, 1.97 | old bulb, on open palm | **pinned** |
| S05 | CarryBulbToDisposal | floor | old bulb, on open palm | **pinned** |
| S06 | DisposeBulb | floor, at the crate | old bulb, on open palm | robot ±5 cm |
| S07 | ApproachNewBulb | floor | — | robot ±5 cm |
| S08 | GrabNewBulb | floor, at the bench | — | robot ±5 cm |
| S09 | CarryBulbToLadder | floor | fresh bulb, on open palm | **pinned** |
| S10 | ClimbWithBulb | floor, at the steps | fresh bulb, on open palm | **pinned** |
| S11 | ScrewInBulb | ladder, 1.97 | fresh bulb, on open palm | **pinned** |
| S12 | ClimbDown | ladder, 1.97 | — | robot ±5 cm |

**S01-MoveLadder** is the longest to teleoperate: one episode covers walking to the ladder,
taking it, moving it, and standing it under the fixture. It is also the only leaf that leaves
`couple_ladder_to_fixture` off — positioning the ladder is the task, so the ladder gets its own
independently-sampled zone rather than the fixture's anchor.

**Pinned** tasks set both `position_range` and `pose_range` to `(0.0, 0.0)`, so reset returns
the robot to exactly its staged pose. The open-palm payload staging is too fragile to survive
the noise.

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

- robot root: x, y ±5 cm, yaw ±0.1 rad (about ±6°) — zero on the pinned tasks
- robot joints: ±0.05 rad — zero on the same tasks
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
seeding afterwards silently does nothing.

## Hand variants

Both G1 hands work on every subtask: `--hand inspire` or `--hand dex3`. The swap happens inside
the recipe before the action terms are built, so grips are authored against the target hand.

Two benchmark-side fixes were needed, both because `swap_robot_variant` rewrites **joint**
names only — never body names, never variant-valued parameters:

- `mdp/grasp_terms.py` held the right palm in a module constant. It now resolves the palm from
  the mounted articulation, mirroring the `G1_PALM_BODY_BY_VARIANT` pattern used elsewhere.
- `swap_robot_variant` now retargets terms that select bodies through a `hand_variant`
  parameter, by inspecting each term's signature.

Without those, the grasp-tier subtasks failed at env creation under Dex3 with
`Not all regular expressions are matched: right_hand_base_link`.

## Demo scoring

A take is bounded by the record toggle: press to start, press again to stop. On stop the
episode is closed, the bag is flushed, and the benchmark's own scorer runs over it — the score
is both printed and merged into `meta.json`.

```
[sonic] RECORDING OFF -- bag updated: 1 episode(s) -> .../run.h5
[sonic]   score: success 1/1 (100%)  mean_score=1.00  clean=100%  broken=0%  dropped=0%
```

`score.py` reads the **last step** of each episode, which is why `success` must terminate: it
puts the success flag exactly where the scorer looks. The per-episode score is binary (1.0 or
0.0, minus penalties for a crushed or dropped payload); continuous signal — `reward`,
`pos_error`, `contact_force` — is recorded per step in the bag alongside it.

## Known issues

**Payloads are not attached.** The carrying tasks stage their payload unsecured — the bulb
rests on an open palm (`settle_carried_payload_live`, friction only), and the ladder is a free
rigid body positioned in the grip. Close the grip immediately on spawn or the payload drops.

**A fall becomes a launch.** SONIC has no fall recovery. Once the pelvis is down the policy is
out of distribution and its output is still applied as joint position targets, so the robot is
driven across the room at 4–8 m/s. The same tasks in the RL env peak at 0.5–2 m/s. Freezing the
legs at the default stance below 0.45 m cuts peak speed by about 55% in testing; not applied,
because the threshold would also fire during a legitimate deep crouch.

**Bad draws happen.** Some layouts collapse before the operator has control — the robot is
placed 2 m up on a free-standing ladder, and if that perch does not settle, both go down. The
benchmark models this: `ladder_tipped` is both a −200 reward and a termination. Relaunching
redraws; resetting does not. With the seed printed, a bad draw is reproducible.

**Hands-off sweeps mislead on carrying tasks.** With no operator, grips never close, so the
payload drops and the robot trips on it. A "fell" verdict there measures the dropped payload,
not the task.
