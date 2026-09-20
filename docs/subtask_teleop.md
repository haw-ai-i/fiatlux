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
| Legs | policy | GEAR-SONIC walk/balance ONNX, driven outside the action manager |
| Camera | task viewer | pelvis-anchored `XrCfg` follow camera |

Scene, assets, events, `sim.dt` and decimation are untouched.

`success` is kept deliberately. It is the only place a subtask's success predicate is
evaluated, and `recording.term_flag` records it per step as `success_term` — which is what
`scripts/score.py` reads. Clearing it does not disable scoring; `term_flag` falls back to an
all-False vector, so every recorded demo silently scores `success_rate 0.0`.

## Running

Run these **from the repo root** — the driver needs `PYTHONPATH` pointing at this checkout's
`source/`, which is what the setup script's printed command sets for you. The VR launcher is the
exception: it finds its own repo and `cd`s there, so it works from anywhere, and the copy you
invoke decides which branch runs (each worktree has its own).

```bash
# keyboard, no headset
python scripts/teleop/sonic_teleop.py --task FIATLUX-S07-ApproachNewBulb-Teleop-v0 --input keyboard

# VR over CloudXR -- one command, and the usual one. Two things to set: the task, and the
# address the headset dials. Recording is armed, so a bag and a video are written from the
# moment you press the right face button.
NV_CXR_ENDPOINT_IP=<ip> FIATLUX_TASK=FIATLUX-S07-ApproachNewBulb-Teleop-v0 \
  bash scripts/teleop/restart_sonic_teleop.sh

# ...and to collect into a dataset folder of your own (keeps the tree below):
NV_CXR_ENDPOINT_IP=<ip> FIATLUX_TASK=FIATLUX-S07-ApproachNewBulb-Teleop-v0 \
  FIATLUX_CAPTURES_DIR=~/my-dataset bash scripts/teleop/restart_sonic_teleop.sh
```

**Where takes land.** Unset, the root is `../teleop-captures` **beside** the repo — outside the
git tree, so sessions never pollute it. `FIATLUX_CAPTURES_DIR` moves that root, keeping the tree.
Under the root, one dimension per level:

```
<captures>/<task>/<hand>/<kind>/<input>/<YYYY-MM-DD>/<HHMMSS>/epNN_score<X.XX>/
teleop-captures/FIATLUX-S07-ApproachNewBulb-Teleop-v0/dex3/hdf5/vr/2026-09-17/143052/ep00_score1.00/
```

One `epNN_score<X.XX>/` folder per record-on..off take, each holding `run.h5`, `meta.json`,
`score_report.txt` and the videos — so a listing reads as per-demo results, and
`<task>/<hand>/<kind>/` is always a schema-homogeneous training set (dex3 bags have 43 joint
columns, inspire 53; never mixable).

What the launcher takes. These are environment variables, not flags -- the full flag reference is
[Driver flags](#driver-flags) below, and anything not listed here keeps the driver's own default.

| variable | default | |
|---|---|---|
| `FIATLUX_TASK` | **required** | which subtask; unset prints all twelve |
| `NV_CXR_ENDPOINT_IP` | **required** | address the headset dials; unset prints this box's |
| `FIATLUX_HAND` | `dex3` | or `inspire` |
| `FIATLUX_LAYOUT_SEED` | random | an int reproduces that exact room |
| `FIATLUX_RECORD` | `bag` | `none` to drive without writing anything |
| `FIATLUX_RECORD_START` | `toggle` | `auto` records from launch instead of on the button |
| `FIATLUX_RECORD_SETTLE` | off | include the ~90 spawn-settle steps, for spawn-time bugs |
| `FIATLUX_CAPTURES_DIR` | `../teleop-captures` | dataset root; keeps the `<task>/<hand>/…` tree under it |

`FIATLUX_TASK` is any subtask id with `-Teleop-v0` on the end -- the launcher lists all twelve if
you leave it unset:

| | | | |
|---|---|---|---|
| `FIATLUX-S01-MoveLadder-Teleop-v0` | `FIATLUX-S04-DescendWithBulb-Teleop-v0` | `FIATLUX-S07-ApproachNewBulb-Teleop-v0` | `FIATLUX-S10-ClimbWithBulb-Teleop-v0` |
| `FIATLUX-S02-ClimbLadder-Teleop-v0` | `FIATLUX-S05-CarryBulbToDisposal-Teleop-v0` | `FIATLUX-S08-GrabNewBulb-Teleop-v0` | `FIATLUX-S11-ScrewInBulb-Teleop-v0` |
| `FIATLUX-S03-RemoveOldBulb-Teleop-v0` | `FIATLUX-S06-DisposeBulb-Teleop-v0` | `FIATLUX-S09-CarryBulbToLadder-Teleop-v0` | `FIATLUX-S12-ClimbDown-Teleop-v0` |

Reproduce a room -- the seed is printed at launch and stored in every bag's `meta.json`, so this
is how a take, a bug report or an A/B comparison gets repeated on the exact same layout:

```bash
FIATLUX_LAYOUT_SEED=561366545 NV_CXR_ENDPOINT_IP=<ip> \
  FIATLUX_TASK=FIATLUX-S07-ApproachNewBulb-Teleop-v0 bash scripts/teleop/restart_sonic_teleop.sh
```

`NV_CXR_ENDPOINT_IP` is the address the **headset** dials, so it depends on where the headset is,
not on this box. Same WiFi: use this box's LAN address -- a direct hop, nothing to install on the
headset. Any other network: use its tailnet address, with Tailscale running on the headset too.
Leave it unset and the launcher lists what this box has, labelled:

```
$ FIATLUX_TASK=FIATLUX-S07-ApproachNewBulb-Teleop-v0 bash scripts/teleop/restart_sonic_teleop.sh
set NV_CXR_ENDPOINT_IP to the address the headset can reach:
    100.x.y.z    tailnet -- headset anywhere, needs Tailscale on it too
    192.168.x.y  LAN -- headset on this WiFi, nothing to install
```

`<ip>` above is whichever of those two the headset can reach.

Getting it wrong is the failure where the client page loads fine and CONNECT then hangs: the page
arrived over a route the media stream cannot use.

Always use `restart_sonic_teleop.sh` for VR. It restarts the CloudXR runtime and clears its
state files. Reusing a runtime across Isaac restarts leaves signalling working while media
negotiation fails — the client connects, looks healthy, and drops after about 30 seconds.

### Headsets

No headset software is installed: the client is the web page CloudXR serves at
`https://<ip>:48322/client/`, so any headset whose browser does WebXR can drive the sim. The
driver reads Isaac Lab's abstract controller row (thumbstick, trigger, squeeze, two face
buttons), not a vendor SDK, so the controls below are the same on every headset — only the
button *names* differ.

| headset | Device Profile | also switch on | face buttons |
|---|---|---|---|
| Quest 3S / Quest 3 | `Quest 3S` / `Quest 3` | Quest Texture Optimization, Quest Color Workaround | left `X`/`Y`, right `A`/`B` |
| Quest 2 | `Quest 2` | Quest Texture Optimization | left `X`/`Y`, right `A`/`B` |
| Pico 4 Ultra | `Pico 4 Ultra` | — | left/right face buttons |

Every index the driver reads lands in the same place on both, measured on a Quest 3S over
486 controller rows (`FIATLUX_XR_DEBUG=1`): sticks 0/1, trigger 2, grip squeeze 3, and the two
face buttons 4/5 — left X/Y lean, right A stops the walk, right B toggles a take. No code
change is needed for a Quest.

Quest Color Workaround is the Display P3 fix; without it a Quest 3/3S stream looks washed
out. Sideloading Tailscale (only needed when the headset is on a different network) differs:
the PICO Browser can install an APK directly, the Quest Browser cannot — use `adb install`
with Developer Mode on.

Keyboard — click the Isaac Sim viewport to focus it first. `TAB` switches the active arm (R ↔ L):

| | |
|---|---|
| `W/S` `A/D` `Q/E` | move the active arm in X / Y / Z |
| `U/O` `I/K` `J/L` | roll / pitch / yaw the wrist |
| `G` | toggle grip |
| arrows | walk (↑↓ forward/back, ←→ turn) |
| `,` `.` | strafe left / right |
| `T` / `Y` | lean forward / back |
| `SPACE` | stop walking |
| `H` | toggle the rail hand (only on tasks that brace one) |
| `C` | start / stop recording |
| `R` reset · `ESCAPE` quit | |

Forward reach saturates around 0.35 m from the pelvis — past that the arm is at its kinematic
limit. On the six in-hand legs the right grip **starts closed** on the seated bulb, so the first
`G` releases it.

### Driver flags

`sonic_teleop.py --help` is the reference; this is the map. Defaults in bold.

| flag | values | what it does |
|---|---|---|
| `--task` | env id | which twin to drive. **Required** — there is no default, since the old one was a legacy scene that is not one of the 12 subtasks |
| `--input` | **`vr`** / `keyboard` | headset over CloudXR, or the desktop keys above |
| `--hand` | **`dex3`** / `inspire` | which G1 hand the env is built with; the driver prints the one it actually got |
| `--layout_seed` | int / **`random`** | the room layout; the seed in use is printed and stored in `meta.json` |
| `--walk_scale` | m/s, **1.0** | walking speed at full stick |
| `--record` | **`none`** / `bag` | write a demo bag per take |
| `--record-video` | | also write `video.mp4` (third-person) and `ego.mp4` (head camera) |
| `--record-start` | **`auto`** / `toggle` | record from launch, or start off until B / `C` |
| `--record-settle` | | include the ~90-step startup settle in the take, so spawn-time failures are in the footage |
| `--record-format` | **`hdf5`** / `npz` | bag format |
| `--record-images`, `--images-stride` | , **5** | the env's own cameras as JPEGs, every Nth step |
| `--out` | path | dump takes flat into ONE folder (`<dir>/epNN_score<X.XX>/`), no tree — for a one-off, not for collecting a dataset. Unset uses the tree below |
| `--camera` | **`auto`** / `follow` / `static` / `fixture` / `bench` / `crate` | the third-person shot. `auto` picks per task: socket side view on S03/S11, bench side view on S07/S08, crate side view on S06, chase cam framing the task's objects elsewhere |
| `--stop-on-success` / `--no-stop-on-success` | **on** | close the take the moment the success gate latches |
| `--max_steps` | int, **0** = run until quit | stop after N loop steps, writing the bag cleanly. Killing the process instead skips the final write |
| `--keys` | `NAME@SECONDS,...` | scripted key presses for hands-off runs, e.g. `"G@4,R@8,ESCAPE@15"`; keyboard input only. Pair with `--max_steps` |
| `--teleop_device` | **`controller_rel`** | which XR device config to drive the arms with |
| `--walk_onnx`, `--balance_onnx` | paths | the SONIC policies (`$SONIC_POLICY_DIR`) |
| `--num_envs` | **1** | |
| `FIATLUX_XR_DEBUG=1` | env var | print the raw controller row (`stickX stickY trigger squeeze btn0 btn1 pad`) for both hands whenever it changes — how you map a new headset's buttons |

`restart_sonic_teleop.sh` is the VR entry point and reaches these through `FIATLUX_*` environment
variables rather than flags: `FIATLUX_HAND`, `FIATLUX_LAYOUT_SEED`, `FIATLUX_WALK_SCALE`,
`FIATLUX_CAMERA`, `FIATLUX_RECORD`, `FIATLUX_RECORD_FORMAT`, `FIATLUX_RECORD_START`,
`FIATLUX_RECORD_SETTLE`, `FIATLUX_RECORD_VIDEO` and `FIATLUX_CAPTURES_DIR` -- those fall back to the driver
defaults above when unset. Two do NOT have a default and the launcher stops if either is missing:
`FIATLUX_TASK` and `NV_CXR_ENDPOINT_IP`; see [Running](#running).

## The 12 subtasks

All twelve subtasks have a registered teleop adapter under `subtasks/`. That is not the same as
reaching the gate: a teleoperated take clears the success gate on eight of them (S01, S03, S05,
S06, S07, S08, S09, S11). The four climbing legs -- S02, S04, S10, S12 -- have no take that
satisfies their gate yet.

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

## The objects

Five things exist in every subtask scene. The names in brackets are the scene asset names, which
are what appear in a bag's columns and in error messages.

| What you see | Asset | Notes |
|---|---|---|
| Step ladder | `ladder` | A **free rigid body** — nothing bolts it down. It can tip, and `ladder_tipped` is a real termination. |
| Ceiling/wall fixture | `socket` | Where bulbs go. Ceiling- or wall-mounted per draw, and the two sit at different heights: **2.37 m** (`CEILING_FIXTURE_Z`) and **2.2 m** (`WALL_MOUNT_Z`). The ceiling one was raised by #147 so its bulb is within reach from the ladder. |
| Old bulb | `old_bulb` | Starts **seated in the fixture** (axial-detent retention, issue #167). The one you remove and throw away. |
| Fresh bulb | `bulb` | Starts **on the bench**. The one you install. |
| Disposal crate | `bin` | The **only** container in the scene. The old bulb goes in here. Not the bench. |
| Bench | `table` | Holds the fresh bulb. Not a target for anything. |

There is one crate and one bench, so "the crate" is never ambiguous — but note the two bulbs are
distinct assets with opposite jobs: `old_bulb` comes **out** of the fixture and goes **into** the
crate; `bulb` comes **off** the bench and goes **into** the fixture.

## What counts as complete

Each subtask's `success` termination is a conjunction — **every** row must hold at the same
instant. Tasks marked *sustained* additionally require the whole conjunction to hold
continuously for `GRASP_SUSTAIN_SECONDS` (0.5 s), so a momentary brush does not score.

Shared thresholds: **robot standing** = pelvis above 0.35 m and tilt under 1.0 rad;
**ladder near-vertical** = tilt under 0.6 rad; **at rest** = under 0.05 m/s and 0.10 rad/s;
**held** = grip contact force over 1.0 N; **released** = under 1.0 N.

| | Task | Gate | Complete when |
|---|---|---|---|
| S01 | MoveLadder | sustained | the stance that ladder pose would produce could grasp the bulb: its shoulder within **0.419 m** of the bulb's body centre in **3-D**, facing it within **45°**, and standing off far enough that the fixture is not inside its torso · ladder upright · feet down within **2 cm** of the floor · ladder at rest · robot standing |
| S02 | ClimbLadder | all_of | pelvis within **0.15 m** (`LADDER_TOP_STANCE_TOLERANCE`) of the top stance height · within **0.6 m** of the ladder in xy · moving under **1.5 m/s** · standing · ladder vertical |
| S03 | RemoveOldBulb | sustained | old bulb **0.10 m** clear of the fixture after release · held (>1 N) · lifted above **0.15 m** · standing · ladder vertical |
| S04 | DescendWithBulb | all_of | pelvis below the floor-stance height, within **0.6 m** of the ladder, under **1.5 m/s** · bulb held · lifted · standing · ladder vertical |
| S05 | CarryBulbToDisposal | all_of | within **0.5 m** of the disposal crate (`bin`) · facing it within **0.5 rad** (`ARRIVAL_FACING_TOLERANCE`) · moving under **1.0 m/s** (`ARRIVAL_MAX_SPEED`) · bulb still held |
| S06 | DisposeBulb | sustained | old bulb inside the disposal crate (`bin`) · at rest · **released** (<1 N) · standing |
| S07 | ApproachNewBulb | all_of | within reach of the fresh bulb (`bulb`, on the bench) · facing it within **0.5 rad** (`ARRIVAL_FACING_TOLERANCE`) · under **1.0 m/s** (`ARRIVAL_MAX_SPEED`) |
| S08 | GrabNewBulb | sustained | fresh bulb lifted **3 cm** off the bench · **≥2 hand bodies** in contact (>1 N each) · total grip force under **50 N** · standing |
| S09 | CarryBulbToLadder | all_of | within mounting range of the ladder · facing it within **0.5 rad** · under **1.0 m/s** · bulb held · ladder upright |
| S10 | ClimbWithBulb | all_of | at top stance (as S02) · bulb held · lifted · standing · ladder vertical |
| S11 | ScrewInBulb | sustained | fresh bulb **attached** (seated, detent holding it) · at rest · **released** (grip <1 N) · standing · ladder vertical |
| S12 | ClimbDown | all_of | descended to floor stance (as S04) · **fresh bulb still seated** in the fixture · standing · ladder vertical |

Three patterns worth internalising before operating:

**You have to let go.** S06, S11 and S08's release-adjacent checks require grip force *below*
1 N. Holding the bulb in the bin is not disposal; holding it in the socket is not seating.

**Speed gates exist.** Arrival tasks reject a score while the robot is still moving faster than
1.0 m/s (1.5 m/s on the ladder), so charging at the target and stopping short of settled will
not fire.

**S08 is the only task with an upper force bound.** Over 50 N total contact and the glass is
counted as crushed — that is also the `broken` penalty in the score, so a take can complete the
motion and still score 0.00.

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

## Recording and scoring

A take is bounded by the record toggle: press to start, press again to stop. Each take gets its
own folder, sealed with its own score once the bag and video are closed:

```
teleop-captures/<task>/<hand>/hdf5/vr/2026-08-27/143052/
  ep00_score1.00/    run.h5   meta.json   score_report.txt   video.mp4   ego.mp4   (+ poster PNGs)
  ep01_score0.00/    ...
  ep02_score1.00/    ...
```

The session folder keeps its plain timestamp — the score belongs on the take, and a session
average is dragged down by a single mis-press. Each `meta.json` carries that take's own score
(`episodes: 1`), the layout seed, joint order and action terms, so a demo can be replayed into
the room it came from.

The score is also printed the moment you stop, so a take's result is known without opening
anything:

```
[sonic] RECORDING OFF -- take saved: .../ep01_score0.00
[sonic]   score: success 0/1 (0%)  mean_score=0.00  clean=0%  broken=0%  dropped=0%
```

Takes are flushed at record-off, at `R`, and at exit — never held until the end, since Kit's
SIGINT handler exits past `finally`/`atexit`.

`score.py` reads the **last step** of each episode, which is why `success` must remain a
termination: it puts the success flag exactly where the scorer looks. The per-episode score is
binary (1.0 or 0.0, minus penalties for a crushed or dropped payload); continuous signal —
`reward`, `pos_error`, `contact_force` — is recorded per step in the bag alongside it.

Every take is kept and scored, including accidental ones. One step is 20 ms at the 50 Hz
control rate, so a double-press writes a real folder with a one-frame bag — visible by its
`_score0.00` suffix and by `episode_lengths` in its meta.
