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
randomized into their own non-overlapping floor "safe zone"; the **fixture** (the
validated BEHAVIOR-1K socket-lamp) mounts randomly on the ceiling or a wall with the
**old bulb** seated in it. The layout is sampled once per scene build; per-episode resets
return to it (plus the reset jitter below).

- The **ladder is dynamic** (12 kg) — only in this preset — so knocking it over is a real,
  penalized, episode-ending event. Ladder placement is *independent* of the fixture by
  default; `ReplaceEnvCfg.couple_ladder_to_fixture = True` is an explicit debug/curriculum
  opt-in that spawns it reachably near the fixture.
- The **old bulb is kinematic**: the "screwed in" stand-in until the attach/detach
  mechanic exists (unification spec Phase 4). Its removal/disposal channels are scored
  and reported but not yet achievable by any policy — the same gap Remove/Install carry.

## Goal

Insert the fresh bulb into the fixture, remove the old bulb from the fixture, and place
the old bulb in the disposal crate. Full success = fresh bulb seated **and** old bulb in
the crate.

## Actions

Whole-body joint-position targets (all DoF incl. fingers, like Climb): the task spans
locomotion, ladder work, and manipulation.

## Observations — `standard` vs `cheatcode` modes

Two groups (named `policy`/`privileged` for rsl_rl's routing; the benchmark calls the
modes **standard** and **cheatcode**):

- **`policy` = standard mode** (sensor-realizable only): IMU (base angular velocity,
  projected gravity), estimated base height + linear velocity (the documented
  estimator-realizable exception, as in Climb), joint pos/vel, hand contact forces,
  **torso-mounted RGB camera features** (needs `--enable_cameras`), last action.
  Corruption enabled.
- **`privileged` = cheatcode mode** (exact simulator state): world poses of the robot,
  ladder, fixture, fresh bulb, old bulb, and disposal crate, plus the four score-relevant
  distances (`replace_score_distances`). Critic-only during RL
  (`ReplacePPORunnerCfg.obs_groups`); a cheatcode policy may consume it directly.

Smoke-test policies (`fiatlux_task/policy.py`): `basic_standard` consumes only the
standard group and holds posture; `basic_cheatcode` additionally asserts and reads the
privileged group. Both prove the episode/scoring loop end-to-end; neither solves the task.

Standard mode admits *raw* sensor access too: a policy may read the torso camera frames
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
| `ladder_ready` (+) | completion | upright ladder top horizontally within 0.9 m of the fixture |
| `fresh_bulb_inserted` (+) | completion | fresh bulb seated (Insert's 1.5 cm / 0.2 rad tolerances) |
| `old_bulb_removed` (+) | completion | old bulb cleared the seat by 0.10 m |
| `old_bulb_disposed` (+) | completion | old bulb within 0.25 m of the crate |
| `success_bonus` (+) | sparse | full replacement (fires on the terminating step) |
| `robot_fall`, `ladder_tipped`, `fresh_bulb_dropped`, `old_bulb_dropped` (−) | penalty | each fires once — the same predicate also terminates |
| `contact_penalty` (−) | penalty | hand contact force (fragile-handling proxy) |
| `com_sway`, `ang_vel_xy`, `action_rate`, `joint_acc`, ankle limits, waist/finger deviation (−) | shaping | stability / smoothness |

## Success & termination

- **Success** (`full_replacement_success`): fresh bulb seated **and** old bulb within
  0.25 m of the disposal crate.
- **Robot fall**: root below 0.35 m or tilt beyond 1.0 rad (family thresholds).
- **Ladder tipped**: ladder up-axis beyond 0.6 rad from vertical.
- **Fresh bulb dropped**: below 0.4 m. **Old bulb dropped**: below 0.15 m *and* away
  from the crate (a disposed bulb legitimately rests near the floor inside it).
- **Timeout**: `episode_length_s = 40 s` (the full approach → ladder → insert → dispose
  horizon).

## Randomization

Scene layout (zones, fixture mount, ladder yaw, robot yaw) per scene build; robot root
xy (±5 cm) / yaw (±0.1 rad), joints (±0.05 rad), light intensities, key-light direction
(pitch ±15° / yaw ±30° about its authored 40° tilt), HDRI sky azimuth (0–360°), and a
global room albedo tint (HSV multiplier on the bound materials' diffuse inputs) per
reset. Prop-scale randomization is a scaffold-env default and an RL opt-in (prestartup
USD writes require `replicate_physics=False`); see `base_env_cfg.py`'s EventCfg.

# `FIATLUX-Insert-v0`

Defined in
`source/fiatlux_task/fiatlux_task/tasks/manager_based/fiatlux_task/g1_bulb_env_cfg.py`.

## Scene

The **tabletop preset** of the shared family scene (`scene_cfg.py: G1ReplaceSceneCfg`):

- **Robot:** Unitree G1 (`assets/unitree_g1/wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd`),
  legged/free base, right arm + Inspire hand actuated.
- **Bulb:** graspable rigid body, a BEHAVIOR-1K light bulb
  (`assets/behavior1k_bulb/ymomhw/usd/ymomhw.usd`), at hand height on the table.
- **Socket:** kinematic lamp/socket fixture on the table, a BEHAVIOR-1K table lamp
  stripped to a single rigid body (`assets/behavior1k_lamp/bbentu/usd/bbentu.usd`; see
  `fiatlux_task/scenes.py: spawn_b1k_single_body` / issue #14).
- Packing table; ground plane; Simple Room backdrop + HDRI sky dome (randomized
  intensity); no ladder (that's the workshop preset's business).

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
  0.75 m; a collapsed robot reads < 0.30 m) or tilt beyond **1.0 rad**. This is the
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

Defined in
`source/fiatlux_task/fiatlux_task/tasks/manager_based/fiatlux_task/climb_env_cfg.py`.

## Scene

The **at-height preset** of the shared family scene: the G1 spawns at the base of the
kinematic work-site step ladder (`assets.py: STEP_LADDER_USD`, 0.68 × 1.11 × 1.75 m at
(1.6, 0, 0), steps facing the robot), an elevated BEHAVIOR-1K chandelier stands in for
the fixture at (1.9, 0, 2.80) (visual dressing — success is geometric), and the bulb is
parked on the floor. A dedicated contact sensor (`ladder_contact`) filters the feet +
palms (`G1_FOOT_BODIES` + `G1_PALM_BODIES`) against the ladder body. Runs at the family
control rate (50 Hz), `episode_length_s = 20`.

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
  ladder contact forces (feet + palms), last action. Corruption enabled.
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

- **Success** (`climbed_to_target`): root above **1.70 m** (the descend task's start
  height minus 0.15 m), horizontally within **0.6 m** of the upper steps (1.35, 0), at
  root speed < **1.5 m/s** (rejects ballistic fly-throughs).
- **Fall** (`fell_below` / `fell_over`): root below **0.35 m** (standing pelvis is
  0.75 m; a collapsed robot reads < 0.30 m) or tilt beyond **1.0 rad**. This is the
  family's fall-detection RL gate: solver-kick episodes end immediately.
- **Timeout**: `episode_length_s = 20 s`.

## Randomization (on reset)

Robot root xy (±5 cm) and yaw (±0.1 rad), joints (±0.05 rad), dome/key-light intensity,
key-light direction (pitch ±15° / yaw ±30°), sky azimuth (0–360°), and a global room
albedo tint. All within `verify_scene.py`'s 0.10 m init-drift tolerance.
