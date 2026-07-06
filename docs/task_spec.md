# Task Specifications

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
- **Timeout**: `episode_length_s = 15 s`.
- **Bulb dropped**: bulb falls below `min_height`.

## Randomization (on reset)

Socket pose (±3–5 cm), bulb start pose (±2 cm), arm joints (±0.05 rad), dome-light
intensity and color.

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

Robot root xy (±5 cm) and yaw (±0.1 rad), joints (±0.05 rad), dome/key-light intensity.
All within `verify_scene.py`'s 0.10 m init-drift tolerance.
