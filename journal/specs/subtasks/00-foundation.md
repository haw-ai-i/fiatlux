# Subtask re-discretization — foundation (issue #66)

Read this before any of the 15 subtask plans. It owns everything they share: the primitive
reduction, the shared env base, the start-state mechanism, the handoff contract, and the
visual validation gate every subtask agent must pass.

## The rule

**Every switch between navigation, balance, and grasping is a subtask boundary.** Nothing else
is. A subtask is one mode; the moment the robot must change mode, the episode is over.

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

## Start states — randomized ranges around a frozen centre

A subtask's start state is a **distribution**, not a pose. Several of the centres are states no
static layout constant can express today (*standing at the ladder base holding a bulb*, *on the
upper steps holding a bulb*), so each is a frozen measured centre — the convention the repo
already uses for `LADDER_STANCE_ROOT_POS` / `TOP_ROBOT_POSITION` / `TABLETOP_SURFACE_Z` — plus a
randomization range around it.

The range is the load-bearing part. A policy that only works from one exact pose is worthless, and
**a start distribution wide enough to contain its predecessor's actual outcomes removes the need
for the handoff to be exact.** That is what makes the chain robust to the previous subtask
finishing sloppily, and it is why the contract below is stated as coverage rather than equality.

Do **not** compute a payload pose from live forward kinematics inside a reset event: at reset the
joint state has been written but physics has not stepped, so `robot.data.body_pos_w` still holds
the previous episode's poses and the payload lands in the wrong place. This is the exact class of
bug that put the bulb 13 cm from the palm before (root outside geometry) and 4 cm off (wrong palm
body).

Mechanism, in `fiatlux_task/subtask_states.py`:

```python
@dataclass(frozen=True)
class SubtaskStartCfg:
    robot_root: Vec3;  robot_root_range: Vec3          # centre + half-extent
    robot_rot: Quat;   robot_yaw_range: float
    robot_joints: dict[str, float]                     # pattern -> radians, over the standing default
    joint_range: float                                 # +/- rad on every actuated joint
    payload: str | None                                # "bulb" | "old_bulb" | "ladder" | None
    payload_in_root: tuple[Vec3, Quat] | None          # payload root pose IN THE ROBOT ROOT FRAME
    payload_pos_range: Vec3 | None                     # + rotation range, about the same frame
    payload_rot_range: float | None
    entities: dict[str, tuple[Vec3, Quat]]             # world poses for everything else
    entity_ranges: dict[str, tuple[Vec3, float]]
```

`payload_in_root` is a rigid offset in the robot's root frame, so the reset event only composes
`robot_root_pose ⊗ sample(payload_in_root)` — known quantities, valid before the first physics
step, and correct under root jitter because the same transform carries the payload along with the
robot.

**Every sampled state must be feasible, which is the one thing DR cannot manufacture.** Widening a
range past the point where the subtask is possible does not buy robustness, it injects unsolvable
episodes. The A-frame ladder is the concrete case: its steps face one way, so S05's robot-spawn
range may cover the step-facing arc and must not cover the back. Each range therefore needs a
*validated* boundary — sample the extremes, render them, confirm the subtask is still doable — and
that check is part of the visual gate below, not an afterthought.

Two new event terms:

- `mdp.randomize_start_state` (`mode="reset"`) — samples the ranges above. Rides the seeded torch
  default generator like every other DR term, with a constant per-reset draw count regardless of
  which optional entities a preset keeps (the rule `randomize_material_tint` already follows), so
  same-seed runs reproduce.
- `mdp.place_payload_in_hand` (`mode="reset"`, ordered **after** `reset_robot_root`) — writes the
  payload's root state from the composed pose, zero velocity.
- `mdp.hold_grasp_pose` — pins the grasping hand's joint targets to the start pose's grasp so the
  payload is not dropped on step 0 before the policy has produced an action.

Two new event terms:

- `mdp.place_payload_in_hand` (`mode="reset"`, ordered **after** `reset_robot_root`) — writes
  the payload's root state from the composed pose, zero velocity.
- `mdp.hold_grasp_pose` — pins the grasping hand's joint targets to the start pose's grasp so
  the payload is not dropped on step 0 before the policy has produced an action.

### Phase 0 probes (blocking, do these first)

Three numbers do not exist yet. Measure them once, freeze them with a `CALIBRATED <date>` note,
and record the probe command in the constant's comment:

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
   a step ladder climbable — it just adds unsolvable episodes. So each range needs a validated
   feasibility boundary, and a predecessor whose success region extends outside it has a real gap
   (`CONTINUITY.md` C2).
2. **Presence and topology.** No pose range covers "the object is not there". If subtask *N* ends
   with the old bulb on the floor and subtask *N+1* is *remove the old bulb from the fixture*, that
   is not a distribution mismatch, it is a different world (`CONTINUITY.md` C3 — the #54 case).

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

## Fate of the existing tasks

The new chain supersedes three current tasks and leaves the rest alone:

| Existing | Fate |
|---|---|
| `FIATLUX-Carry-v0` | superseded by S01–S04. Keep registered until all four pass, then retire. |
| `FIATLUX-Climb-v0` | superseded by S05. Same. |
| `FIATLUX-Descend-v0` | superseded by S15. Same. |
| `FIATLUX-Insert-v0`, `-Install-v0`, `-Remove-v0` | **keep.** Tabletop manipulation drills at bench height; they are not steps of the ladder chain and their calibrated bench geometry (`TABLETOP_SURFACE_Z = 0.9941`) is the reference S11/S14 borrow from. |
| `FIATLUX-Replace-v0` | **keep, unchanged.** The flat full task is the benchmark; the 15 subtasks are the decomposition, not a replacement. |
| `FIATLUX-Base-v0` | keep (scene-only scaffold). |

Retirement is a separate commit from the additions, so a bisect can tell "new subtask broke"
from "removing the old task broke something that depended on it".

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

## What counts as done

A subtask is complete when: the env is registered and constructs; `verify_scene.py --task <id>`
passes all checks; the visual checklist above is reported frame by frame; `handoff:` passes
against both neighbours; and a zero-action and a random-action rollout both run the full
episode without a solver explosion. **A converged policy is not required** — and per the
benchmark's own bar, a policy failing to solve a subtask is not a defect. Broken physics, a
broken start state, a broken feed, or a success gate that fires on the wrong thing are.

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
