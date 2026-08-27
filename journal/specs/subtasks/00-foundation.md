# Subtask re-discretization — foundation (issue #66)

Read this before any of the subtask plans. It owns everything they share: the primitive
reduction, the shared env base, the start-state mechanism, the handoff contract, and the
visual validation gate every subtask agent must pass.

> **There are twelve subtasks, not fifteen.** Everything below is written against the original
> fifteen, and its numbering is the original numbering. The four ladder legs were later folded
> into one and the chain renumbered; see **"As built — the four ladder legs are one subtask"**
> at the end of this file for the map, and `01-move-ladder.md` for the leg itself. Nothing else
> in this document changed.

## The rule

**Every switch between navigation, balance, and grasping is a subtask boundary.** A subtask is one
mode; the moment the robot must change mode, the episode is over.

Read that as a **mnemonic for the list, not a rule that generates it.** The modes are not mutually
exclusive and this document's own headers admit it — carrying the ladder is "navigation (loaded)",
i.e. navigating *while* grasping, and removing the old bulb is "grasping on balance". If a subtask can
be two modes at once then "a switch between modes" does not by itself define a boundary. **The
fifteen-item list is the specification** (reviewed and endorsed as such); the rule describes its
shape. Do not "correct" the list by applying the rule literally.

This is a property of the environment. No plan may reference a WBC, a VLA, an ONNX locomotion
pair, or policy stitching — those are solution structure. Each subtask is a standalone
`ManagerBasedRLEnv` with its own start state, reward channels, and success gate, and must be
solvable by any policy that can act on its observation group.

## Structure: 15 first-class tasks, one shared base

**Each subtask is its own task, its own module, its own independently versioned gym id** — the
same convention as the existing family (`climb_env_cfg.py` → `FIATLUX-Climb-v0`). One subtask's
contract can then change and bump to `-v1` without touching the other fourteen, which is the
whole point of versioning them separately: retuning S03's success radius must not silently
invalidate scores for S08.

```
tasks/manager_based/fiatlux_task/subtasks/
    s01_approach_ladder_env_cfg.py   ->  class S01ApproachLadderEnvCfg
    s02_grab_ladder_env_cfg.py       ->  class S02GrabLadderEnvCfg
    ...                                  (15 modules, 15 classes, 15 ids)
```

What is **not** duplicated 15 times is the boilerplate the six existing RL cfgs already
re-declare independently — ~200 of each file's ~300 lines: two observation groups, the event set,
the smoothness/discipline reward tail, the solver block, the `fell_below`/`fell_over` pair. They are
*not* verbatim copies; they have already drifted, with one concept carrying up to five different
term names across the family (`ABSTRACTIONS.md` measures it). Climb and Descend half-acknowledge the
problem by importing `FALL_MIN_HEIGHT`/`FALL_TILT_LIMIT` across module boundaries. That shared
material moves into one base class, which also fixes the **canonical name** for each concept:

```
SubtaskEnvCfg(ManagerBasedRLEnvCfg)      # obs, events, shaping, solver, fall gates
  └── S01ApproachLadderEnvCfg, S02GrabLadderEnvCfg, ... (one subclass per subtask)
```

Each of the 15 subclasses states, in its own file, only what makes it that subtask: its start
state, its success gate, its task reward channels, its extra terminations, its
`episode_length_s`, and its orbit framing. Everything a reader needs to understand *that* task is
in *that* file; nothing that is identical across all fifteen is written fifteen times.

Where two subtasks genuinely share task logic — S05/S13 both climb, S07/S15 both descend, S01/S03/
S08/S10/S12 all navigate to something — they share **mdp reward functions**, not cfg classes.
`mdp/rewards.py` is already the place this project puts shared task logic
(`climb_height_progress`, `distance_progress`, `completion_bonus` are each used by several
tasks). A parameterized cfg class would couple the five navigation subtasks' versions together;
a shared reward function does not.

The full factoring — what the base owns, the six mode intermediates, the leaf contract, and what
deliberately stays un-abstracted — is in **`ABSTRACTIONS.md`**, together with the measured drift
across the six existing RL cfgs that motivates it. Summary: `SubtaskEnvCfg` holds:

- **`ObservationsCfg`** — `policy` (base_ang_vel, projected_gravity, base_lin_vel, base_height,
  joint_pos_rel, joint_vel_rel, hand_contact, ego_rgb, lidar_ranges, last_action) and
  `privileged` (root pose, root lin vel, plus per-subtask entity poses). Keep the group names
  `policy` / `privileged` — rsl_rl's `obs_groups` routing is keyed to them.
- **`EventCfg`** — `reset_all`, `reset_robot_joints`, `reset_robot_root`,
  `randomize_sky_intensity`, `randomize_key_light`, `randomize_material_tint`,
  `randomize_hand_material`, plus the three new terms below.
- **Shaping rewards** — `com_sway`, `ang_vel_xy`, `action_rate`, `joint_acc`,
  `ankle_pos_limits`, `joint_deviation_waist`, `joint_deviation_fingers`, `robot_fall`. The last is
  the canonical name for the fall penalty that the family currently spells two ways
  (`termination_penalty` in 2/6, `robot_fall` in 4/6).
- **Solver / rate block** — `decimation=4`, `sim.dt=1/200`, `solver_type=1`,
  `min_position_iteration_count=8`, `min_velocity_iteration_count=1`,
  `bounce_threshold_velocity=0.2`, `enable_stabilization=True`.
- **Fall gates** — one pair of `FALL_MIN_HEIGHT` / `FALL_TILT_LIMIT` constants. Today Climb
  says 0.35 m and Carry says 0.40 m for the same robot; the base picks **0.35** (Climb's, the
  one probed against a deep mounting crouch) and the discrepancy dies with the fork.
- **`disable_randomization()`** — the `--no_randomize` contract, one implementation.

**Enforcement**, so the fork cannot silently grow back: a non-GPU test
(`tests/test_subtask_contract.py`, no `isaacsim_ci` marker) walks every registered subtask cfg
class and asserts each one's observation-group term names, event term names, and shaping reward
term names are exactly the base's — any subclass that redefines a shared term fails the test.

## Start states — randomize what is fat, calibrate what is thin

A start state is **randomized per component, not per subtask.** The criterion is whether that
component's feasible set is fat enough to sample from:

| Component | Treatment |
|---|---|
| robot root xy / yaw, **standing on the floor** | **randomized** — ±5 cm, ±0.1 rad, the existing `reset_robot_root` range |
| joint jitter for a floor stance | **randomized** — ±0.05 rad, the existing `reset_robot_joints` range |
| entity layout (ladder zone, table, crate, fixture mount) | **randomized** — the existing `apply_replace_preset` draw |
| lights, material tint | **randomized** — the existing three DR axes |
| robot pose **while balanced on the ladder** | **deterministic**, one manually calibrated pose |
| payload pose **in the hand** | **deterministic**, one manually calibrated offset |

The reason for the split is that a balanced-on-the-ladder-holding-a-bulb state sits on a thin,
coupled manifold: both feet on a *specific* step, CoM over the support polygon, the bulb inside a
roughly centimetre-scale window given Dex3's geometry (thumb tip 6.5 cm off the palm plane,
fingertips 12.4 cm out), no limb through a rail, all at rest at once. Sampling axis-aligned ranges
independently would draw mostly from the infeasible interior of the bounding box, injecting
penetration and imbalance — and the resulting distribution would be neither known nor reproducible.

So those components get **one deterministic, hand-calibrated pose** — the convention the repo
already uses for `LADDER_STANCE_ROOT_POS` / `LADDER_STANCE_ROOT_ROT` / `LADDER_STANCE_JOINTS` /
`TOP_ROBOT_POSITION`, calibrated exactly the way `poses.py` documents: run the probe, iterate the
constants until the contacts register, freeze with a `CALIBRATED <date>` note.

### Two calibrated artefacts, not six

The deterministic poses must be **shared, not authored per subtask.** Four subtasks start with the
robot balanced on the upper steps (remove-the-old-bulb, descend-with-bulb, screw-in, climb-down),
differing only in whether a payload is in the hand; three start standing on the floor with a payload
(carry-the-ladder, carry-bulb-to-disposal, carry-bulb-to-ladder).

So there are exactly two calibrated artefacts, each with an optional payload offset:

| Artefact | Used by |
|---|---|
| **one on-ladder stance** (root pose + joints) | remove-the-old-bulb, descend-with-bulb, screw-in, climb-down |
| **one standing cradle** (payload offset in the robot root frame) | carry-the-ladder, carry-bulb-to-disposal, carry-bulb-to-ladder, climb-with-bulb |

This is the cheapest risk reduction available in the whole plan. Every hand-calibrated state is a
chance to freeze something physically wrong, and this project's recent history is a catalogue of
exactly that — a bulb 13 cm from the palm, a palm body 4 cm off, a bulb pinched outside the hand and
reported as held, twice. Six chances become two. Whichever subtask lands first measures and freezes
them; the rest import.

**This is explicitly a for-now decision.** Widening those start states later means generating a pool
of physics-validated states offline (or harvesting them from a controller that solves the
predecessor) and drawing an index at reset. Nothing here forecloses that: the mechanism below reads
a range of zero as a point, so a pool is a later substitution behind the same interface, not a
rewrite.

Do **not** compute a payload pose from live forward kinematics inside a reset event: at reset the
joint state has been written but physics has not stepped, so `robot.data.body_pos_w` still holds
the previous episode's poses and the payload lands in the wrong place. This is the exact class of
bug that put the bulb 13 cm from the palm before (root outside geometry) and 4 cm off (wrong palm
body).

Mechanism, in `fiatlux_task/subtask_states.py`:

```python
@dataclass(frozen=True)
class SubtaskStartCfg:
    robot_root: Vec3;  robot_root_range: Vec3 = (0.0, 0.0, 0.0)   # zero range == deterministic
    robot_rot: Quat;   robot_yaw_range: float = 0.0
    robot_joints: dict[str, float]                    # pattern -> radians, over the standing default
    joint_range: float = 0.0
    payload: str | None = None                        # "bulb" | "old_bulb" | "ladder" | None
    payload_in_root: tuple[Vec3, Quat] | None = None  # payload root pose IN THE ROBOT ROOT FRAME
    entities: dict[str, tuple[Vec3, Quat]] = field(default_factory=dict)
    entity_ranges: dict[str, tuple[Vec3, float]] = field(default_factory=dict)
```

Ranges **default to zero**, so a component is deterministic unless a subtask opts it into
randomization. That way the thin cases are the cheap default and widening is always a deliberate,
reviewable act — the opposite of the current risk, where an inherited range silently covers
infeasible states. `payload_in_root` is a rigid offset in the robot's root frame, so the reset event
only composes `robot_root_pose ⊗ payload_in_root` — known quantities, valid before the first
physics step, and correct under root jitter because the same transform carries the payload with the
robot.

**Any range that is non-zero needs a validated boundary.** DR covers variation inside the feasible
set and cannot manufacture feasibility: the A-frame's steps face one way, so S05's floor-stance
range may cover the step-facing arc and must not cover the back. Sample the extremes, render them,
confirm the subtask is still doable — part of the visual gate below, not an afterthought.

Three new event terms:

- `mdp.randomize_start_state` (`mode="reset"`) — samples the non-zero ranges. Rides the seeded torch
  default generator like every other DR term, with a **constant per-reset draw count** regardless of
  which ranges are zero (the rule `randomize_material_tint` already follows), so same-seed runs
  reproduce and adding a range later does not shift the stream for the others.
- `mdp.place_payload_in_hand` (`mode="reset"`, ordered **after** `reset_robot_root`) — writes the
  payload's root state from the composed pose, zero velocity.
- `mdp.hold_grasp_pose` — pins the grasping hand's joint targets to the start pose's grasp so the
  payload is not dropped on step 0 before the policy has produced an action.

### Phase 0 probes (blocking, do these first)

Three numbers do not exist yet. Measure them once, freeze them with a `CALIBRATED <date>` note,
and record the probe command in the constant's comment.

**Probe 0 is no longer needed — the reach gap was settled from existing constants.** The fixture
was at 3.00 m while `MAX_REACHABLE_MOUNT_Z` was computed from the robot's *feet on the ladder's
1.70 m top platform*, a stance no task requires. Climb's own success gate accepts a pelvis at
1.70 m, i.e. feet at 0.91 m on a middle step, from which the fingertips reach 2.284 m — leaving the
fixture **0.72 m out of reach** while the import-time guard passed.

Now derived from the stance the tasks actually work from:

| Constant | Value |
|---|---|
| `LADDER_WORK_PELVIS_Z` (= Climb's `SUCCESS_HEIGHT`) | 1.700 m |
| `LADDER_WORK_FOOT_Z` | 0.910 m |
| `MAX_REACHABLE_MOUNT_Z` | 2.284 m |
| `CEILING_FIXTURE_Z` = `WALL_MOUNT_Z` | **2.200 m** (8.4 cm margin) |
| `LADDER_READY_XY_RADIUS` | 0.422 m |

Two import-time guards now bracket the fixture height from both sides: it must be **reachable** from
the working stance, and it must remain **above standing floor reach** (1.374 m) so the ladder is
still required. Clearing the climb gate and being able to work the fixture are now the same
condition, which is what the earlier split of those two numbers failed to guarantee.

1. **`BULB_IN_ROOT_STANDING`** — bulb root pose in the robot root frame for `ARM_CRADLE` +
   `HAND_CRADLE_DEX3`, robot standing. Derive it from the already-validated palm-frame
   arithmetic in `scripts/verify_interactions.py` (`palm_frame`, `palm_grasp_pose`,
   `BULB_CAP_RADIUS_M = 0.021`, `BULB_CAP_CENTRE_M = 0.054`, `PALM_GRASP_FORWARD_M = 0.045`,
   `PALM_LOCAL_AXES`), then express the result in the root frame and freeze it.
2. **`BULB_IN_ROOT_ON_LADDER`** — the same, with the ladder stance's torso lean applied (the
   arm cradle and the lean compose; do not assume 1 is reusable).
3. **`LADDER_IN_ROOT_CARRIED`** — ladder root pose in the robot root frame for a one-rail
   grasp, upright, clear of the legs. No existing calibration covers this; subtask 2 owns it
   and subtask 3 consumes it.

**Move the calibration code, don't copy it.** `palm_frame`, `palm_grasp_pose`,
`quat_from_matrix`, and the measured bulb/palm geometry constants currently live in
`scripts/verify_interactions.py`, where no env cfg can reach them. Lift them into
`fiatlux_task/grasp.py` and have the script import from there. A calibration that only a script
can use is why these numbers were re-derived wrong twice.

### Payload retention is an open physical risk

Subtasks 3, 7, 8, 12, 13 all move the whole body while holding something. The grip is a
position-control artifact: `HAND_CRADLE_DEX3` closes the Dex3 to curl 1.2, and retention was
measured only *statically* — the calibration note records that below ~1.0 the bulb slips and
past ~1.4 the closing fingers eject it. Nobody has measured whether it survives a climb's body
accelerations.

**Gate every loaded subtask on this**, before writing any reward: hold the start pose under
zero action through the subtask's full motion envelope and check the payload is still in the
hand. If it is not, the subtask is unsolvable by construction and the honest options are
(a) score the drop as the failure mode it is and accept a low ceiling, or (b) fold hand
retention into the #54 attach mechanic. Report the measurement; do not silently pick one.

## The handoff contract — coverage, not equality

The naive contract is "subtask *N*'s success state **equals** subtask *N+1*'s start state". That is
the wrong contract. Demanding equality within a tolerance makes the chain brittle, forces an
argument about per-entity tolerances, and produces failures that are settling noise rather than
defects — the ladder alone gets climbed four times.

The right contract is **coverage**:

> `SUCCESS_REGION[N]` ⊆ `support(START_DIST[N+1])`

Anywhere subtask *N* can legitimately finish must be somewhere subtask *N+1* can legitimately
start. Randomization is how the successor's support is widened to achieve that, so exact continuity
stops mattering: pose mismatches between a predecessor's outcome and a successor's nominal start are
absorbed by the successor's own DR range, which it wants for robustness regardless.

In `subtask_states.py`:

```python
SUBTASK_CHAIN: list[str]                        # the 15 ids, in order
START_DIST: dict[str, SubtaskStartCfg]          # centre + ranges
SUCCESS_REGION: dict[str, SuccessRegion]        # the success gate as a region, not a bool
```

`SuccessRegion` holds the same thresholds the success termination uses (xy centre + radius, height
bound, tilt limit, payload-held flag). New check **`handoff:<N>-><N+1>`**: sample K states from
`SUCCESS_REGION[N]`, including its corners, and assert each lies inside `START_DIST[N+1]`'s support.
Pure geometry over frozen constants, so it runs without a GPU — put it in
`tests/test_subtask_contract.py`. It catches the failure this re-discretization is most exposed to:
retuning subtask 5's success region until it pokes outside subtask 6's start range.

### What coverage does *not* fix

Two residues, and they are the ones that matter:

1. **Feasibility.** DR covers variation *inside* the feasible set; it cannot manufacture
   feasibility. Widening S05's spawn range to include "behind the A-frame" does not make the back of
   a step ladder climbable — it just adds unsolvable episodes. So each non-zero range needs a
   validated boundary, and a predecessor whose success region extends outside it has a real gap
   (`CONTINUITY.md` C2).
2. **Presence and topology.** No pose range covers "the object is not there". If subtask *N* ends
   with the old bulb on the floor and subtask *N+1* is *remove the old bulb from the fixture*, that
   is not a distribution mismatch, it is a different world (`CONTINUITY.md` C3 — the #54 case).

### How much the check is actually worth — stated plainly

Two decisions weaken it, and the earlier draft of this file oversold it as "the check that catches
the failure mode this re-discretization is most exposed to". It is not.

- The balance start states are **deterministic point masses**, so for the S05→S06, S06→S07,
  S13→S14 and S14→S15 boundaries coverage cannot hold in any meaningful sense — a point is inside a
  region or it is not, and there is no distribution to cover anything.
- The 15 run as **independent envs** with no chained rollout, so no runtime behaviour depends on the
  handoff at all. Nothing breaks at run time if it fails.

What is left is a **static consistency check over frozen constants**: it fires when someone edits
S05's success height until S06's start pose falls outside it. That is worth keeping — it is free and
it catches a silent orphaning — but it is a lint, not a regression guard, and it cannot fail
spontaneously. The plans should not lean on it as evidence that the chain composes.

### Seeds

With coverage semantics, subtasks do **not** need a shared layout seed to train: each randomizes
its own layout, and the chain is a claim about distributions. A shared `set_layout_seed(seed)`
matters only when rolling the 15 out as one continuous demo or evaluation. Scores still may not be
compared across seeds, because the geometry differs.

## Naming and registration

```
FIATLUX-S01-ApproachLadder-v0     FIATLUX-S09-DisposeBulb-v0
FIATLUX-S02-GrabLadder-v0         FIATLUX-S10-ApproachNewBulb-v0
FIATLUX-S03-CarryLadder-v0        FIATLUX-S11-GrabNewBulb-v0
FIATLUX-S04-PlaceLadder-v0        FIATLUX-S12-CarryBulbToLadder-v0
FIATLUX-S05-ClimbLadder-v0        FIATLUX-S13-ClimbWithBulb-v0
FIATLUX-S06-RemoveOldBulb-v0      FIATLUX-S14-ScrewInBulb-v0
FIATLUX-S07-DescendWithBulb-v0    FIATLUX-S15-ClimbDown-v0
FIATLUX-S08-CarryBulbToDisposal-v0
```

Numbered because the chain order is load-bearing — the handoff check, the docs, and
`SUBTASK_IDS` all iterate it. Registration stays lazy (string entry points only):
`fiatlux_task.tasks` swallows import errors during its auto-import walk, so one eagerly
imported cfg that fails would silently drop every registration in the module.

**Versioning policy.** The `-v0` suffix is a per-task benchmark contract, exactly as for the
existing family. A subtask bumps to `-v1` — new id, old id kept registered — whenever a change
makes previously recorded scores incomparable: its success thresholds, its start state, its
observation-group layout, or its reward channel set. Changes that cannot move a score (docstrings,
refactors, a new privileged term) do not bump. Because the fifteen are separate classes in
separate modules, a bump is local to one file and one id; the handoff check
(`handoff:<N>-><N+1>`) is what catches a bump that orphans a neighbour's start state.

One consequence to accept deliberately: a change to `SubtaskEnvCfg` itself — the shared
observation group, say — is contract-breaking for **all fifteen** at once and bumps all fifteen.
That is the honest accounting (the observation really did change for every task), and it is a
reason to keep only genuinely invariant material in the base.

## Nothing is retired — the two tiers measure different things

**Every existing task stays registered.** An earlier draft proposed retiring Carry, Climb and
Descend as "superseded"; that was wrong, and the reason is the point of having both tiers.

The existing tasks are **coarser on purpose**. Carry alone spans approach, grasp, transport under
load, and place — it requires the policy to *sequence four mode switches by itself*, with no episode
boundary handing it the next start state. That is a strictly harder problem than any of the four
subtasks it decomposes into, and it is a different capability: **coordination across mode
transitions**, not competence within one mode.

So the family becomes a **difficulty ladder**, and the two tiers answer different questions:

| Tier | Question it answers |
|---|---|
| the 15 subtasks | Can the policy do each mode at all? Isolates *where* it fails. |
| Carry / Climb / Descend | Can it chain several modes without an episode boundary to help? |
| `FIATLUX-Replace-v0` | Can it do the whole job end to end, unaided? |

That progression is worth more than any single tier. A policy scoring well on the subtasks and zero
on Carry localizes the deficit precisely to transition handling — a diagnosis neither tier gives
alone, and exactly the kind of finding the GR00T baseline could not produce (0% everywhere tells you
nothing about which capability is missing).

| Existing | Fate |
|---|---|
| `FIATLUX-Carry-v0` | **keep.** Approach + grasp + carry + place with no boundaries between them — the multi-transition test for what S01–S04 isolate. |
| `FIATLUX-Climb-v0` | **keep.** Fixed-layout climb; S05's is ladder-pose-relative and follows a placement, so they are not the same task. |
| `FIATLUX-Descend-v0` | **keep.** Same relationship to S15. |
| `FIATLUX-Insert-v0`, `-Install-v0`, `-Remove-v0` | **keep.** Tabletop manipulation drills at bench height; their calibrated bench geometry (`TABLETOP_SURFACE_Z = 0.9941`) is the reference S11/S14 borrow from. |
| `FIATLUX-Replace-v0` | **keep, unchanged.** The flat full task is the benchmark; the 15 are its decomposition, not its replacement. |
| `FIATLUX-Base-v0` | **keep** (scene-only scaffold). |

Consequences to accept deliberately:

- **Shared constants change both tiers at once.** Tightening `LADDER_READY_XY_RADIUS` retunes Carry
  and Replace as well as S04, because they import it. That is correct — it was a geometric error in
  all three — but it means a fix in the subtask work is a contract change for the coarse tier too,
  and must be reported as one.
- **The `SubtaskEnvCfg` base is not retrofitted onto the old tasks in this work.** They keep their
  own duplicated cfg blocks for now. Migrating them is a separate, later change, and doing it
  would bump their versions.
- **Scoring must not mix tiers.** A cross-tier average is meaningless: the subtasks and Carry are
  not independent samples of the same difficulty. Report per tier.

## Visual start-state validation — mandatory, per subtask

Every subtask agent must **look at rendered frames of its own start state** before writing a
single reward term, and must report what it saw. A coded assertion that the bulb's root is
within 1 cm of the palm has passed while the bulb was visibly pinched outside the hand; that
happened twice on this project. Frames are the gate.

```bash
cd /home/esports/fiatlux
PYTHONPATH= OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python3 -u scripts/verify_scene.py \
  --headless --enable_cameras --task <SUBTASK_ID> --num_envs 1 --hold_base \
  --record --record_view scene --record_steps 240
# overhead-mount subtasks (S05, S06, S13, S14) additionally:
#   --record_view fixture
ffmpeg -i logs/<mp4> -vf "select='not(mod(n,30))'" -vsync 0 /tmp/.../frame_%02d.jpg
```

**Renders catch what nobody thought to assert; they cannot adjudicate millimetres.** At 1280×720 on an
orbit camera, nobody can distinguish 3 mm of interpenetration from contact, and "enclosed by the
digits" is a judgement call. So every visual item below is paired with the numeric check that actually
decides it, and **both must be reported**:

| Visual claim | The check that decides it |
|---|---|
| "nothing intersects" | `omni.physx` `overlap_mesh` shape-level overlap, reporting offending pairs |
| "resting, not floating" | settle soak: < 2 mm drift, < 1 cm/s, over 150 steps |
| "the payload is held" | filtered hand↔object contact force, non-zero on ≥ 2 digit bodies |
| "touching the floor / a step" | contact force on the relevant body, non-zero |
| "in the ego frustum" | project the target's world position through the camera intrinsics |

The earlier draft made frames *the* gate because coded checks had produced false confidence twice. That
was an overcorrection: the lesson was to use both, since renders and assertions fail in opposite
directions.

Then read the frames and confirm, in writing:

1. Every entity touches what it is supposed to touch — feet on floor or on a specific step,
   ladder feet on floor, crate on floor. Nothing floats, nothing intersects.
2. A held payload is **inside** the hand — enclosed by the digits, not pinched against a
   fingertip, not intersecting the palm mesh.
3. The robot pose is the intended one and is physically plausible (no hyperextension, no limb
   inside the ladder or the bench).
4. The ego camera view (`--enable_cameras`) contains the subtask's target. If it does not,
   **say so and stop** — that is a start-state defect, not a policy problem.
5. The success target is where the plan says it is.

Plus two numeric gates, both already-established patterns:

- **settle** — soak 150 steps under zero action with `--hold_base`: every tracked entity moves
  < 2 mm and ends below 1 cm/s. A start state that needs settling gets fixed at the constant.
- **retention** (loaded subtasks only) — the payload is still in the hand after the soak.

## What these envs are for

**Policy evaluation — they are the benchmark.** Not an RL training suite and not a home for scripted
controllers. That decides where the effort goes:

- **Start-state fidelity and success-gate correctness are the deliverable.** A gate that fires on the
  wrong thing, or a start state that is physically impossible, corrupts every number the benchmark
  produces. This is where the visual and numeric validation below earns its cost.
- **Reward channels exist for the score breakdown, not for training.** `replace_env_cfg` documents
  the per-term episode sums in `extras['log']` as the breakdown, which is why the canonical names
  matter. Weight balancing for trainability is not in scope.
- **No `PPORunnerCfg` per subtask.** Nothing here needs `scripts/rsl_rl/{train,play}.py` to reach it.
- **No scripted baseline per subtask.**

One consequence to hold onto, since dropping the scripted baseline removes the guard against it: a
subtask nobody can solve reports 0%, and **0% cannot be distinguished from a broken subtask by the
score alone.** The validation gates are therefore the only thing standing between a real result and
a meaningless one — which is an argument for taking them seriously, not for adding baselines back.

## Scoring is not yet defined — open

Fifteen success gates do not make a benchmark score. `scripts/score.py` currently scores one task
(seated / broken / dropped). Nothing in these plans says how 15 subtask results aggregate, and the
naive answer is wrong: the 15 are **not equal difficulty and must not be averaged**. Five of them
(S01, S03, S08, S10, S12) are "walk to X", so a policy that can only walk would score 5/15 = 33%
while the hardest single subtask (S14, fine insertion under balance) is worth 1/15. Partial credit is
also undefined — the reward channels carry graded progress but the score has no place to put it.

This needs a decision before any number is quoted. Options, not resolved here: per-subtask
independent reporting with no aggregate; difficulty-weighted aggregate; or longest-prefix (how far
along the chain a policy gets), which matches how the task actually reads.

## Sequencing: one vertical slice before fifteen files

Take **approach-the-ladder end to end** — env, start state, success gate, visual *and* numeric
validation, and an entry in whatever scoring scheme gets decided — and only then start the second
subtask. Otherwise the cfg-hierarchy risk and the undefined scoring get discovered fifteen times
instead of once.

**One decision must be made before that slice starts, not during it:**

- **Spike the cfg hierarchy** — base + one intermediate + one leaf, confirm all five managers
  construct and that term ordering matches field declaration order. Cheap now, expensive after
  fifteen files exist.

### Settled: the ego camera stays on Unitree's mount

The camera is already attached to the G1 USD's own `d435_link` with zero positional offset and
identity rotation — Unitree's authored RealSense D435 housing, fixed to the torso (the G1 has no neck
joint). Its downward pitch is therefore **the hardware spec, not a tuning choice**, and the FOV is
already matched to the real D435's published 69.4° horizontal.

So no plan may adjust it, and no plan may stop on it. If a subtask's target is outside the frustum at
t=0, that is a true fact about what the real robot can see, not a setup defect: record it as task
difficulty and continue. This closes the open question the earlier drafts deferred fifteen times.

## What counts as done

## As built — what the fifteen actually are, and what the plan got wrong

All fifteen exist, are registered, and construct (15/15). What follows is only where the code
diverges from these plans, or where writing it turned up something the plans did not know.

**The completion bonus was missing from four shipped subtasks.** `success_bonus` was declared on
the navigate tier alone, so S02, S04, S09 and S11 — grasp and place leaves, all written and
verified before this was noticed — had a sparse success gate and no sparse reward for reaching it.
Nothing failed loudly: the `success` termination still fired, episodes still ended, the log still
carried a `success` row. Only the reward was absent. It now lives on `SubtaskRewardsCfg`, which is
where the ABSTRACTIONS "one canonical name per channel" rule always implied it belonged; declaring
it one layer down defeated the rule while appearing to follow it.

Worth generalizing: a channel promised by a base class and declared in a sibling is invisible to
every check that reads a leaf. The construction harness now prints each cfg's full reward and
termination name sets, which is what surfaced it.

**`ladder_contact_fraction` charges a loaded policy for holding its payload.** S13's plan raised
this and predicted it would become live once the Dex3 palm-naming fix landed. It did. The term
means over the sensor's bodies, so a hand occupied for the whole episode caps it below 1.0 with no
action available to recover the difference. `add_ladder_contact_sensor` now takes a body list;
S07 and S13 pass feet plus the free palm. S05 and S15 keep all four.

The wider lesson the plans should have stated once instead of four times: any *fraction over
bodies* term encodes an assumption about which limbs are free, and that assumption changes per
subtask. The denominator is part of the task definition, not a sensor detail.

**The fragility bound cannot be a success conjunct.** Both S06 and S14 specified "peak grip never
exceeded `GLASS_CONTACT_LIMIT_N`" as a gate condition. A conjunct is read at the scoring step, so a
bulb crushed at second three and seated at second twenty still scores — the condition as specified
tests the grip at the wrong moment, and tracking a peak would need state inside a predicate that
`all_of` instantiates as a plain function. It is a **termination** instead: crossing the bound ends
the episode and puts the gate out of reach for the rest of it. Same shape as `ladder_tipped`, and
for the same reason.

**Two sustain windows in the mate mode, not one.** Taking hold (S06) only has to prove the grasp is
not a glancing contact — the grasp tier's 0.5 s. Leaving something installed (S14) has to prove it
stays there once the hand is off — the place tier's 1.0 s. These are different claims about
physics and the plans were right to give them different numbers; the tier names them
`MATE_GRASP_SUSTAIN_SECONDS` and `MATE_RELEASE_SUSTAIN_SECONDS` rather than picking one.

**The held-force threshold was specified as 2 N and shipped as 1 N.** S06, S07 and S13 each wrote
"hand↔bulb contact > 2 N"; `nav_terms.GRIP_FORCE_THRESHOLD_N` is 1.0, chosen as the mirror of the
place tier's release cutoff. Both numbers are provisional and neither is derived from anything, so
the family uses one — otherwise "still held" and "let go" would describe an overlapping band of
forces in which a payload is both.

**S14's alignment is a new term, not a rewrite of the old one.** The plan asked for
`object_socket_orientation_tanh` to be rebuilt on the mating axis. That function is live in Insert,
Install and Replace, and rebuilding it would silently move their reward surfaces.
`mate_terms.bulb_axis_alignment_tanh` is the axis-angle form and only S14 uses it. Rebuilding the
shared one belongs to the scene-physics ticket that owns C.2, not here.

**The ladder subtasks draw their layout with `couple_ladder_to_fixture=True`.** The preset's own
docstring calls coupling a debug/curriculum aid, which is true for the full task — positioning the
ladder is part of it. For S05–S07 and S13–S15 the ladder has *already been positioned*, so the
coupled draw is the chain's state and the uncoupled one would be wrong. The flag's docstring
describes the caller it was written for, not the flag.

**Still open, unchanged by this work:**

- **The grip-pose gap.** No subtask pins the grasping hand. Every loaded start state places its
  payload beside an open hand, so it falls at reset and the still-held conjuncts are false from
  step one — S03, S07, S08, S12, S13, S14. The envs are correct; the start states are not
  reachable without hand retention. Explicitly deferred.
- **Every payload pose is `UNCALIBRATED`** (`grasp_poses.py`), including the two the balance and
  mate subtasks consume.
- **Nothing has been rendered.** Thirteen of fifteen start states have never been looked at, and
  the one bug the numeric checks missed on S01 — the robot facing 3.05 rad from its target, with
  seven checks passing — was caught only by a rendered ego frame.
- **Scoring is still undefined.** See the section above; five of fifteen are "walk to X", so an
  unweighted mean hands a walk-only policy a third of the benchmark.

## As built — the four ladder legs are one subtask

Later than everything above, and it supersedes the fifteen-item list. `S01-ApproachLadder`,
`S02-GrabLadder`, `S03-CarryLadder` and `S04-PlaceLadder` are now one
`FIATLUX-S01-MoveLadder-v0`: **the ladder ends up standing close enough to the fixture**, and
nothing about how it got there. The plan for it is `01-move-ladder.md`.

The reason is one this document already half-states in its own "What these envs are for" section.
These are evaluation envs, so a gate that fires on the wrong thing corrupts every number. Between
them the four gates required a rail grasped and loaded, the feet lifted clear, the grip retained,
the hand released, and the robot left on a particular side — five conditions on *technique*, none
of them a property of the deliverable, each an independent way to score zero after already
standing the ladder where it belongs. For an operator driving the benchmark by hand that is four
separate cliffs in front of one job. The gate now reads the ladder's pose and nothing else: no
contact sensor is wired into the leaf at all, so carrying, dragging, shouldering and pushing are
indistinguishable to it by construction. `ladder_tipped` (0.6 rad) survives, because it is what
"standing" means.

Stated plainly, since this document argued the other way: **the four boundaries isolated where a
policy fails on the ladder, and one episode cannot.** That diagnostic moves rather than
disappearing — `FIATLUX-Carry-v0` remains the coarse-tier task that spans the same four modes, and
the leg's own breakdown (`approach_progress` vs `placement_progress`) is what now separates
"never reached the ladder" from "reached it and could not move it".

The chain renumbered to stay contiguous:

| Was | Now | | Was | Now |
|---|---|---|---|---|
| S01 ApproachLadder | **S01 MoveLadder** | | S09 DisposeBulb | S06 DisposeBulb |
| S02 GrabLadder | ″ | | S10 ApproachNewBulb | S07 ApproachNewBulb |
| S03 CarryLadder | ″ | | S11 GrabNewBulb | S08 GrabNewBulb |
| S04 PlaceLadder | ″ | | S12 CarryBulbToLadder | S09 CarryBulbToLadder |
| S05 ClimbLadder | S02 ClimbLadder | | S13 ClimbWithBulb | S10 ClimbWithBulb |
| S06 RemoveOldBulb | S03 RemoveOldBulb | | S14 ScrewInBulb | S11 ScrewInBulb |
| S07 DescendWithBulb | S04 DescendWithBulb | | S15 ClimbDown | S12 ClimbDown |
| S08 CarryBulbToDisposal | S05 CarryBulbToDisposal | | | |

Consequences to accept deliberately:

- **Every id changed, so every recorded subtask score is orphaned.** Under the versioning policy
  above this is not a `-v1` bump: the old numbering names a different chain, so the old ids are
  gone rather than kept registered beside new ones. Twelve `-v0` contracts, none comparable with
  anything recorded before this change.
- **Two tiers thinned.** `GraspSubtaskCfg` is down to one member (S08) and `PlaceSubtaskCfg`
  regained S01 under its new name. Both stay: the split a tier encodes — which terms exist here,
  which numbers live in the leaf — is what keeps routine retuning out of shared code, and that
  argument never depended on having two members.
- **Dead gate machinery was removed, not left behind.** `mdp.arrived_at_ladder`,
  `nav_terms.arrived_carrying_ladder`, `grasp_terms.hand_ladder_distance`,
  `grasp_terms.grasp_force_above`, `grasp_terms.ladder_feet_clear`,
  `place_terms.robot_at_ladder_base`, and `grasp_poses`' `LADDER_IN_ROOT_CARRIED` /
  `LADDER_CARRY_ARM_JOINT_POS` / `LADDER_GRIP_TRIPWIRE_N` had no remaining caller.
- **One item on the still-open list is closed by this, not fixed.** The grip-pose gap listed above
  named S03 first: its start state spawned a ladder beside an open hand, so the still-held
  conjunct was false from step one. That start state no longer exists — S01 picks the ladder up
  itself. The gap remains real for the bulb-carrying legs.
- **`CRITIQUE.md`, `CONTINUITY.md` and `ISSUES_2026-08-22.md` keep the old numbering.** They are
  dated review records; renumbering them would falsify what was reviewed.

## As built — the on-ladder stance used to spawn inside the fixture

Separate from the fold, and it hit every leaf that stands on the tread (S02, S03, S04, S10, S11,
S12). The working stance puts the pelvis at 1.967 m; the fixture hangs at 2.200 m. That is 0.233 m
— the fixture is at the robot's **chest**, not overhead, and the torso reaches 0.152 m out from the
pelvis axis at that height. A **ceiling** mount anchored its ladder zone at the point directly
beneath the fixture, and the stance stands over the ladder's root, so the socket spawned inside the
torso (`imu_in_torso` 0.039 m from the fixture axis, measured). Being a kinematic rigid body with
an exact triangle-mesh collider, it held the robot up: under zero action the pelvis moved
1.967 → 1.950 m over 3 s where a wall draw's identical stance collapses to 1.194 m. 13 of 24
sampled seeds, i.e. every ceiling draw.

The plans never caught it because each of the two numbers is defensible alone. `CEILING_FIXTURE_Z`
= 2.2 m is derived from `MAX_REACHABLE_MOUNT_Z`, which assumes feet at `LADDER_WORK_FOOT_Z` =
1.03 m — the gate-tolerant height, deliberately *not* the tread. `stand_robot_on_ladder_top` puts
the feet on the tread at 1.18 m, because that is where the platform collider is. Nothing compared
the two, and the reach guards check height only ("nothing here bounds the HORIZONTAL distance", as
`MAX_REACHABLE_MOUNT_Z`'s own comment says).

Fixed in the sampler: both mount kinds now stand the ladder `LADDER_FIXTURE_STANDOFF` off the
fixture, and `scene_cfg` gained the missing guard — a horizontal one, tying the standoff to the
measured torso and fixture extents plus the reset jitter. Checked by
`scripts/verify_ladder_stance.py`, which sweeps layout seeds geometrically and then instantiates
one draw of each mount kind to confirm the robot falls under zero action rather than hanging.

The general lesson, since this is the second time it has bitten: **a start state validated one
axis at a time is not validated.** The vertical reach guard passed, the layout feasibility check
passed, and the subtask constructed and rendered — the robot was simply inside the fixture. Read them against the
  map above.

## Not in scope: "move the ladder back" (S16)

Raised in review: the chain could end by returning the ladder to storage. Skipped, because it is
mechanically S03 with a different target and adds no new mode boundary.

One consequence to record rather than hide: the chain is therefore **not idempotent**. Its terminal
state leaves the ladder under the fixture, not where it started, so the world cannot be replayed
without a reset. That is fine for 15 independent envs and would matter only if the chain were ever
run twice back-to-back.

## Episode lengths are provisional

Every horizon in the fifteen plans except the navigation ones is a guess. The only measured datum is
the walk probe — 2.43 m in ~8 s at ~0.5 m/s — which justifies the navigation horizons and nothing
else. The manipulation-under-balance numbers (screw-in at 40 s, remove-the-old-bulb at 30 s) have no
evidence behind them and must be labelled provisional until a real rollout sets them.

## Blockers

- **#54** — S06 and S14 act on a bulb in an inverted fixture. `mdp/attach.py` does not exist,
  and `verify_interactions --scenario socket` is 3/6 on exactly this. Author both, validate
  their start states, mark their success gates provisional.
- **PR #64** — the filtered hand-contact channel. Every grasp subtask scores grip force on
  `force_matrix_w` summed over the manipulated objects. Start implementation after it merges.
- ~~Ladder mass~~ **settled** — the family had two masses for one object: `LADDER_MASS = 3.0` kg
  (Carry, "modest, so one arm can move it") and `LADDER_MASS_KG = 12.0` kg (Replace). Unified on
  **`LADDER_MASS_KG = 7.25`**, which is what `STEP_LADDER_RIGID_USD` itself carries
  (`omniverse_ladder_rigid.py`'s `2 + 3*h` heuristic at the asset's 1.75 m) and what a real
  1.75 m fiberglass A-frame weighs. `LADDER_MASS` is gone.

  Consequence to expect: Carry's 3 kg existed to keep a one-armed lift easy, so at 7.25 kg S02's
  grasp and S03's carry may need two hands. That is a start-state finding for S02 to report, not a
  reason to reintroduce a second mass.
