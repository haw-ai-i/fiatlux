# Task Specifications

# `FIATLUX-Replace-v0` — the benchmark task

Defined in
`source/fiatlux_task/fiatlux_task/tasks/manager_based/fiatlux_task/replace_env_cfg.py`.
The full light-bulb replacement, scored as **one flat RL episode** (no policy stitching
or stage chaining — that is solution structure, not benchmark structure). Design record:
`journal/specs/full-task-benchmark-plan.md`.

## Scene (randomized per build)

The **replace preset** of the shared family scene (`scene_cfg.apply_replace_preset`):
robot, ladder, table (with the **fresh bulb** on it), and the **disposal crate** are each
randomized into their own non-overlapping floor "safe zone"; the **fixture** (the same
Omniverse socket the Insert task uses, `SOCKET_USD`) mounts randomly on the ceiling or a wall with the
**old bulb** seated in it. The layout is sampled once per scene build; per-episode resets
return to it (plus the reset jitter below).

- The **ladder is dynamic** (**10.9 kg**, `LADDER_MASS_KG` -- stamped in code, not read from the asset) — only in this preset — so knocking it over is a real,
  penalized, episode-ending event. Ladder placement is *independent* of the fixture by
  default; `ReplaceEnvCfg.couple_ladder_to_fixture = True` is an explicit debug/curriculum
  opt-in that spawns it reachably near the fixture.
- **Both bulbs are dynamic**, governed by the `mdp.bulb_attachment` state machine
  (unification spec Phase 4; issue #167 superseded the original bayonet design, issue #54).
  Two states per bulb: `FREE` (unconstrained) and `SEATED` (a continuous wrench holds it at
  the seat: a magnet-shaped axial term, strongest at the seat and decaying with distance
  -- issue #171, see below -- plus a much gentler lateral + tilt centering term). `FREE ->
  SEATED` fires on reaching the seat aligned (position + tilt tolerance) with the socket
  empty; real bulb-socket collision -- filtered out under the old bayonet, now enabled
  everywhere -- constrains lateral position and orientation as its primary mechanism, the
  wrench's lateral/tilt term only assisting. `SEATED -> FREE` (release) fires on a real,
  physics-driven axial pull past `release_threshold`. No twist/lock/rotation state: this
  asset has no physical lug or groove, so the old bayonet's clock-angle semantics were never
  modeling a real feature. `fresh_bulb_attached` and `success` read the attachment state, so
  every score channel is achievable. Remove/Install do not yet use this mechanic: their bulbs
  simply lift out of / drop into the socket.
  - **Lateral + tilt centering (issue #171)**: the original design left lateral position and
    orientation entirely to real contact. Teleop evidence found a seated bulb visibly
    tilts/swings -- the bore's 2.69mm radial clearance is real, necessary slop (tightening
    it even to 1.86mm breaks force-driven insertion outright, confirmed with
    `scripts/diagnose_contact_axial.py`), so the fix is a much gentler additional
    spring-damper on lateral position and tilt while seated, not a tighter bore. Gated on
    the same seated condition as the axial term, so it cannot affect insertion.
  - **Axial retention is a magnet, not a spring (issue #171)**: a linear spring is weakest
    exactly at the seat and grows with distance -- backwards from a magnetic/detent catch,
    which is strongest at contact and falls off with distance. The axial term's shape now
    matches the latter: peak force (`hold_force`, 1.5 N) right at the seat, falling off linearly
    to zero at `bore_depth` (0.025 m -- measured, the withdrawal at which the plug clears the
    socket throat), so there is no bore left for the magnet to act across. This also fixed a real ceiling-mount failure along the way: a ceiling
    fixture is inverted (seat axis points down), so gravity pulls a seated bulb OUTWARD
    along it, and the original linear spring's steady-state sag under that load left almost
    no margin before `release_threshold` -- a ceiling-seated bulb fell out unassisted within
    under a second. `hold_force`/`bore_depth` are sized (not a gravity feedforward -- a
    passive mechanism doesn't cancel gravity outright, and a ceiling-hung bulb should sag
    more than a resting one, same as any real spring/friction/magnet) so worst-case sag sits
    comfortably clear of `release_threshold`. Table/wall mounts sag less than ceiling ones
    under their own weight, correctly.
  - **No twist term -- one was tried twice and removed (issue #171)**: a ceiling-seated bulb
    was found spinning about the seat axis at 1-19 rad/s for ~2.9 s, no operator or contact,
    before ejecting. Two fixes were tried on the theory that twist was under-damped: a viscous
    term, then a Coulomb-like constant-magnitude friction (`twist_friction = 0.5 N*m`). **Both
    were wrong, and the second one WAS the bug.** A term-by-term ablation
    (`scripts/ablate_attach_forces.py`) found that zeroing `twist_friction` -- and no other term
    -- removes the spin: 95-97 rad/s with it, 0.7-3.0 rad/s without, holding 4/5 repeats instead
    of 0/5; with nothing applied at all the bulb shows only ~1.2 rad/s. The plug's moment of
    inertia about the seat axis is 2.9e-05 kg*m^2, so one step of 0.5 N*m changes the twist rate
    by 344 rad/s while the law reverses sign at a 0.1 rad/s deadband -- it overshoots zero by
    ~3400x every step and re-accelerates the other way. Coulomb chatter, not friction; the
    original teleop bag shows twist flipping sign on 138 of 140 consecutive control steps at
    +/-16 rad/s, a clean alternation at exactly the 50 Hz control rate. **There is no twist term
    now, in any form.** Rotation about the seat axis is left to the socket's real contact
    friction, which the asset supplies (static 1.2 / dynamic 1.0, resolving onto all 2 bulb and
    8 socket colliders) -- and issue #90's "rotation is free" stands. The one sound part of the
    first attempt was kept: angular velocity is still split into twist and tilt components so
    tilt damping uses only the latter. Any future twist term must bound its impulse by
    `I*|omega|/step_dt`.

## Goal

Insert the fresh bulb into the fixture, remove the old bulb from the fixture, and place
the old bulb in the disposal crate. Full success = fresh bulb **attached** (seated and held
by the axial detent, per `mdp.bulb_attachment` -- a magnet, not a spring; see below) **and** old bulb in the crate. Seating
alone no longer scores.

## Actions

Whole-body joint-position targets (all DoF incl. fingers, like Climb): the task spans
locomotion, ladder work, and manipulation.

## Observations — `standard` vs `privileged` modes

Two groups (named `policy`/`privileged` for rsl_rl's routing; the paper calls the modes
**standard** and **privileged** — "cheatcode" is retired naming that survives only in the
`basic_cheatcode` baseline's identifier below):

- **`policy` = standard mode** (sensor-realizable only): IMU (base angular velocity,
  projected gravity), estimated base height + linear velocity (the documented
  estimator-realizable exception, as in Climb), joint pos/vel, hand contact forces,
  **head-mounted (`d435_link`) RGB camera features** and **head-mounted (`mid360_link`)
  lidar ranges** (camera needs `--enable_cameras`), last action. Corruption enabled.
- **`privileged` mode** (exact simulator state): world poses of the robot,
  ladder, fixture, fresh bulb, old bulb, and disposal crate, plus the four score-relevant
  distances (`replace_score_distances`). Critic-only during RL
  (`ReplacePPORunnerCfg.obs_groups`); a privileged-mode policy may consume it directly.

Smoke-test policies (`fiatlux_task/policy.py`): `basic_standard` consumes only the
standard group and holds posture; `basic_cheatcode` additionally asserts and reads the
privileged group. Both prove the episode/scoring loop end-to-end; neither solves the task.

Standard mode admits *raw* sensor access too: a policy may read the `ego_camera` frames
and proprioception directly from the scene (rather than the flattened, corrupted,
feature-extracted `policy` group) as long as it touches nothing privileged — that is how
the `groot` VLA baseline consumes the same sensors (`fiatlux_task/groot.py`).

## Language instruction

VLA-style policies receive the task as a natural-language instruction
(`--instruction` on `eval.py` / `record_run.py`). The canonical sentence
(`fiatlux_task.groot.DEFAULT_INSTRUCTION`):

> Replace the light bulb: take the fresh bulb from the table, insert it into the light
> fixture, then put the old bulb in the yellow crate.

## Rewards (the score breakdown)

Every channel is its own named term, so `Episode_Reward/<term>` sums **are** the score
breakdown. Dense terms pay *increments of the episode's best normalized progress* —
`(d0 − d) / d0` clamped to [0, 1] with `d0` captured at reset — so randomized spawn
distances cannot dominate the score (a lucky close spawn and an unlucky far one both cap
at 1.0). Completion bonuses pay once per episode.

| Term | Kind | Purpose |
| --- | --- | --- |
| `ladder_progress` (+) | dense | ladder top → fixture, normalized progress |
| `fresh_bulb_progress` (+) | dense | fresh-bulb plug → fixture seat, normalized progress |
| `old_bulb_removal` (+) | dense | old-bulb clearance from the seat vs an absolute 0.10 m threshold (its d0 ≈ 0, so toward-style normalization can't apply) |
| `old_bulb_disposal_progress` (+) | dense | old bulb → disposal crate, normalized progress |
| `ladder_ready` (+) | completion | ladder upright and placed so the top-tread stance's shoulder is within 0.419 m of the seated bulb and facing it (`LADDER_READY_REACH`) |
| `fresh_bulb_inserted` (+) | completion | fresh bulb seated (the attach gate; Insert's 1.5 cm / 0.2 rad seating tolerances are enforced at attach time) |
| `old_bulb_removed` (+) | completion | old bulb released and cleared the seat by 0.10 m (a held bulb reads as seated) |
| `old_bulb_disposed` (+) | completion | old bulb inside the disposal crate (containment, any orientation) |
| `success_bonus` (+) | sparse | full replacement (fires on the terminating step) |
| `robot_fall`, `ladder_tipped`, `fresh_bulb_dropped`, `old_bulb_dropped` (−) | penalty | each fires once — the same predicate also terminates |
| `contact_penalty` (−) | penalty | hand contact force (fragile-handling proxy) |
| `com_sway`, `ang_vel_xy`, `action_rate`, `joint_acc`, ankle limits, waist/finger deviation (−) | shaping | stability / smoothness |

## Success & termination

- **Success** (`attached_replacement_success`): fresh bulb seated (attachment state,
  not raw seating geometry) **and** old bulb inside the disposal crate (containment, any orientation).
- **Robot fall**: root below 0.35 m or tilt beyond 1.0 rad (family thresholds).
- **Ladder tipped**: ladder up-axis beyond 0.6 rad from vertical.
- **Fresh bulb dropped**: below 0.4 m. **Old bulb dropped**: below 0.15 m *and* away
  from the crate (a disposed bulb legitimately rests near the floor inside it).
- **Timeout**: `episode_length_s = 1440 s` — the twelve subtask budgets (120 s each) summed.
  Part of the evaluation protocol (`docs/scoring.md`); a submission may not change it.

## Randomization

Scene layout (zones, fixture mount, ladder yaw, robot yaw) per scene build; robot root
xy (±5 cm) / yaw (±0.1 rad), joints (±0.05 rad), light intensities, key-light direction
(pitch ±15° / yaw ±30° about its authored 40° tilt), HDRI sky azimuth (0–360°), and a
global room albedo tint (HSV multiplier on the bound materials' diffuse inputs) per
reset. Prop-scale randomization is a scaffold-env default and an RL opt-in (prestartup
USD writes require `replicate_physics=False`); see `base_env_cfg.py`'s EventCfg.

# `FIATLUX-Insert-v0`

A development aid that predates the twelve-subtask decomposition, not part of what the
paper reports on. Defined in
`source/fiatlux_task/fiatlux_task/tasks/manager_based/fiatlux_task/g1_bulb_env_cfg.py`.

## Scene

The **tabletop preset** of the shared family scene (`scene_cfg.py: G1ReplaceSceneCfg`):

- **Robot:** Unitree G1 (`assets/unitree_g1/wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd`),
  legged/free base, right arm + Inspire hand actuated.
- **Bulb:** graspable dynamic rigid body, the Omniverse A19 bulb
  (`assets/omniverse_bulb/LightBulb_bulb_z_rigid.usda`, 0.035 kg), standing on its screw
  cap at hand height on the table.
- **Socket:** kinematic fixture on the table, the matching Omniverse socket with a guide
  sleeve inside its bore (`assets/omniverse_bulb/LightBulb_socket_z_static_sleeve.usda`,
  authored over the stock socket by `scripts/omniverse/omniverse_socket_guide_sleeve.py`;
  the stock mouth ring alone lets a seated bulb lean 22 deg and jam). Its screw hole is an exact
  triangle-mesh collider, so a bulb genuinely enters and rests in it -- which also makes
  the socket permanently ineligible to be dynamic (a PhysX rule). Both halves are authored
  assembled at identity, so *seated* is exactly *bulb pose == socket pose*.
- Packing table; ground plane; per-env Simple Room with real wall/ceiling colliders + HDRI
  sky dome (randomized intensity); no ladder (that's the workshop preset's business).

## Actions

Joint-position targets on the G1 right arm (`G1_ARM_JOINTS`), scaled around the
default pose. Hardware-realizable for sim-to-real.

## Observations

Two groups:

- **`policy`** (sensor-realizable, the real-robot interface): arm joint pos/vel,
  end-effector pose, wrist-camera RGB features, hand contact forces, last action.
  Corruption (noise) enabled.
- **`privileged`** (ground-truth, for the critic / scripted baselines only): bulb
  world pose, socket world pose.

## Rewards

| Term | Purpose |
| --- | --- |
| `object_socket_distance` (−) | coarse L2 bulb→socket distance |
| `object_socket_distance_tanh` (+) | dense reaching |
| `object_socket_distance_exp` (+) | sharp seating reward close-in |
| `object_socket_orientation_tanh` (+) | axis alignment |
| `bulb_seated` (+) | sparse success bonus |
| `hand_contact_force_l2` (−) | compliant insertion |
| `action_rate`, `joint_vel`, `joint_acc`, `joint_pos_limits` (−) | smoothness / safety |

## Success & termination

- **Success** (`bulb_seated`): bulb within `pos_threshold` (1.5 cm) **and**
  `ori_threshold` (0.2 rad) of the socket.
- **Fall** (`fell_below` / `fell_over`): root below **0.35 m** (standing pelvis is
  0.79 m; a collapsed robot reads < 0.30 m) or tilt beyond **1.0 rad**. This is the
  family's fall-detection RL gate: solver-kick episodes against the kinematic table
  end immediately (thresholds shared from `climb_env_cfg.py`).
- **Timeout**: `episode_length_s = 15 s`.
- **Bulb dropped**: bulb falls below `min_height`.

## Randomization (on reset)

Socket pose (±3–5 cm), bulb start pose (±2 cm), arm joints (±0.05 rad), dome-light
intensity and sky azimuth (0–360°), key-light direction (pitch ±15° / yaw ±30°), and a
global room albedo tint. Prop-scale randomization is an RL opt-in (see
`base_env_cfg.py`'s EventCfg; requires `replicate_physics=False`).

# `FIATLUX-Climb-v0`

A development aid that predates the twelve-subtask decomposition, not part of what the
paper reports on (the paper's own climbing evaluation runs the `S02`/`S04`/`S10`/`S12`
subtasks — see `docs/scoring.md`). Defined in
`source/fiatlux_task/fiatlux_task/tasks/manager_based/fiatlux_task/climb_env_cfg.py`.

## Scene

The **at-height preset** of the shared family scene: the G1 spawns at the base of the
kinematic work-site step ladder (`assets.py: STEP_LADDER_USD`, 0.608 × 0.979 × 1.861 m at
(1.6, 0, 0), steps facing the robot), an elevated fixture stands in at (1.9, 0, 2.80)
(`ELEVATED_SOCKET_USD`, now an alias of `SOCKET_USD` -- the decorative chandelier it once named
is gone; success here is geometric either way), and the bulb is parked on the floor. A dedicated contact sensor (`ladder_contact`) filters the feet +
palms (`G1_FOOT_BODIES` + `G1_PALM_BODIES`) against the ladder body. Runs at the family
control rate (50 Hz), `episode_length_s = SUBTASK_EPISODE_LENGTH_S` (120 s).

## Actions

Whole-body joint-position targets (all 53 DoF incl. fingers), scaled (0.5) around the
default standing pose. The per-phase action-space split (whole-body here vs arm+hand in
Insert) is a family design decision (unification spec).

## Observations

Two groups:

- **`policy`**: IMU terms (base angular velocity, projected gravity), **estimated base
  height and linear velocity** — a documented *estimator-realizable exception* to the
  sensor-only contract: the real G1 publishes both from its kinematic-inertial state
  estimator (the same argument Isaac Lab's velocity tasks make) — joint pos/vel, per-limb
  ladder contact forces (feet + palms), a head-mounted (`d435_link`) RGB camera, a
  head-mounted (`mid360_link`) lidar (ground + ladder ranges), last action. Corruption
  enabled; camera needs `--enable_cameras`.
- **`privileged`** (critic / scripted baselines only): robot root pose + linear velocity,
  ladder pose. Routed to the critic via `ClimbPPORunnerCfg.obs_groups` (rsl_rl does not
  auto-route a group named `privileged`).

## Rewards

| Term | Purpose |
| --- | --- |
| `climb_height_progress` (+) | progressive ascent: each centimetre of *new* best root height paid once |
| `climbed_to_target` (+) | one-time success bonus |
| `ladder_contact_fraction` (+) | small bootstrap for limb-on-ladder contact (filtered sensor) |
| `com_sway_l2` (−) | whole-body CoM horizontal-velocity (sway) penalty |
| `ang_vel_xy_l2` (−) | roll/pitch rate (wobble, not the static climbing lean) |
| `fall_terminated` (−) | one-time fall penalty, fires exactly on the `fell_below` / `fell_over` step |
| `action_rate`, `joint_acc`, ankle `joint_pos_limits`, waist/finger `joint_deviation_l1` (−) | smoothness / joint discipline |

`flat_orientation_l2` is deliberately absent: climbing an A-frame requires a sustained
forward lean.

## Success & termination

- **Success** (`climbed_to_target`): root above **1.82 m** (`LADDER_WORK_PELVIS_Z`, the
  top stance minus `LADDER_TOP_STANCE_TOLERANCE`), horizontally within **0.6 m** of the
  upper steps (1.6, 0), at
  root speed < **1.5 m/s** (rejects ballistic fly-throughs).
- **Fall** (`fell_below` / `fell_over`): root below **0.35 m** (standing pelvis is
  0.79 m; a collapsed robot reads < 0.30 m) or tilt beyond **1.0 rad**. This is the
  family's fall-detection RL gate: solver-kick episodes end immediately.
- **Timeout**: `episode_length_s = SUBTASK_EPISODE_LENGTH_S` (120 s).

## Randomization (on reset)

Robot root xy (±5 cm) and yaw (±0.1 rad), joints (±0.05 rad), dome/key-light intensity,
key-light direction (pitch ±15° / yaw ±30°), sky azimuth (0–360°), and a global room
albedo tint. All within `verify_scene.py`'s 0.10 m init-drift tolerance.
