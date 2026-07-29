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

What is **not** duplicated 15 times is the boilerplate that Climb/Descend/Carry/Replace already
agree on verbatim today: the same two observation groups, the same five event terms, the same six
smoothness/discipline reward terms, the same solver block, the same `fell_below`/`fell_over`
pair — ~200 of each file's ~300 lines. Climb and Descend already half-acknowledge this by
importing `FALL_MIN_HEIGHT`/`FALL_TILT_LIMIT` across module boundaries. That shared material
moves into one base class:

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

`SubtaskEnvCfg` holds:

- **`ObservationsCfg`** — `policy` (base_ang_vel, projected_gravity, base_lin_vel, base_height,
  joint_pos_rel, joint_vel_rel, hand_contact, ego_rgb, lidar_ranges, last_action) and
  `privileged` (root pose, root lin vel, plus per-subtask entity poses). Keep the group names
  `policy` / `privileged` — rsl_rl's `obs_groups` routing is keyed to them.
- **`EventCfg`** — `reset_all`, `reset_robot_joints`, `reset_robot_root`,
  `randomize_sky_intensity`, `randomize_key_light`, `randomize_material_tint`,
  `randomize_hand_material`. Plus the two new terms below.
- **Shaping rewards** — `com_sway`, `ang_vel_xy`, `action_rate`, `joint_acc`,
  `ankle_pos_limits`, `joint_deviation_waist`, `joint_deviation_fingers`, `termination_penalty`.
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

## Start states — frozen, cfg-time, no live FK

A subtask's start state is the previous subtask's end state, and several of them are states no
static layout constant can express today: *standing at the ladder base holding a bulb*,
*on the upper steps holding a bulb*.

Author them as **frozen measured constants**, the convention the repo already uses for
`LADDER_STANCE_ROOT_POS` / `TOP_ROBOT_POSITION` / `TABLETOP_SURFACE_Z`. Do **not** compute a
payload pose from live forward kinematics inside a reset event: at reset the joint state has
been written but physics has not stepped, so `robot.data.body_pos_w` still holds the previous
episode's poses and the payload lands in the wrong place. This is the exact class of bug that
put the bulb 13 cm from the palm before (root outside geometry) and 4 cm off (wrong palm body).

Mechanism, in `fiatlux_task/subtask_states.py`:

```python
@dataclass(frozen=True)
class SubtaskState:
    robot_root: tuple[float, float, float]
    robot_rot: tuple[float, float, float, float]
    robot_joints: dict[str, float]          # pattern -> radians, over the standing default
    payload: str | None                     # "bulb" | "old_bulb" | "ladder" | None
    payload_in_root: tuple[Vec3, Quat] | None   # payload root pose IN THE ROBOT ROOT FRAME
    entities: dict[str, tuple[Vec3, Quat]]      # world poses for everything else
```

`payload_in_root` is a rigid offset in the robot's root frame, so the reset event only has to
compose `robot_root_pose ⊗ payload_in_root` — a transform of known quantities, valid before the
first physics step, and correct under the ±5 cm / ±0.1 rad root jitter because the same
transform carries the payload along with the robot.

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

## The handoff contract

The point of the split is that subtask *N*'s success state is subtask *N+1*'s start state. Make
it checkable, or this is 15 disconnected envs sharing a room.

In `subtask_states.py`:

```python
SUBTASK_CHAIN: list[str]                       # the 15 ids, in order
START_STATE: dict[str, SubtaskState]
SUCCESS_REGION: dict[str, SuccessRegion]       # the success gate as a region, not a bool
```

`SuccessRegion` holds the same thresholds the success termination uses (xy centre + radius,
height bound, tilt limit, payload-held flag). New `verify_scene.py` check
**`handoff:<N>-><N+1>`**: for each adjacent pair, assert `START_STATE[N+1]` lies inside
`SUCCESS_REGION[N]` — robot root within the xy radius and height bound, payload state equal,
and every shared entity pose equal to within 1 cm. Pure geometry on frozen constants, so it
costs nothing and runs without a GPU; put it in `tests/test_subtask_contract.py` too.

This is the check that catches the failure mode this re-discretization is most exposed to:
retuning subtask 5's success height and silently orphaning subtask 6's start state.

Three rules the continuity audit (`CONTINUITY.md`) derived, which the check depends on:

1. **A subtask that ends holding something must pin the payload to its successor's start pose** —
   within 0.03 m and 0.2 rad of the frozen `payload_in_root` constant — as a condition of its own
   success gate. Ending on a bare threshold ("lifted > 3 cm") while the successor starts from a
   frozen pose leaves a gap the chain jumps silently. See `CONTINUITY.md` C1 for the per-subtask
   table.
2. **A chain is identified by `(seed, subtask ids)`.** The Replace layout is drawn once per cfg
   build, and each subtask builds its own cfg in its own process, so two subtasks at different
   seeds are two different rooms. Run a chain at one `set_layout_seed(seed)` throughout, evaluate
   `handoff:` per seed, and never compare subtask scores across seeds.
3. **Handoff tolerances are per entity, not global.** 1 cm for static and kinematic entities; the
   ladder gets **5 cm / 0.1 rad** because it is dynamic and gets climbed four times in the chain,
   plus a separate `ladder_tipped` check for the thing that actually matters. Set the ladder number
   from S05's measured post-climb drift rather than guessing.

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
