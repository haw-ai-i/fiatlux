# Task Specification — `FIATLUX-Insert-v0`

Defined in
`source/fiatlux_task/fiatlux_task/tasks/manager_based/fiatlux_task/g1_bulb_env_cfg.py`.

## Scene

- **Robot:** Unitree G1 (`assets/unitree_g1/wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd`),
  legged/free base, right arm + Inspire hand actuated.
- **Bulb:** graspable rigid body, a BEHAVIOR-1K light bulb
  (`assets/behavior1k_bulb/ymomhw/usd/ymomhw.usd`).
- **Socket:** kinematic lamp/socket fixture, a BEHAVIOR-1K table lamp stripped to a
  single rigid body (`assets/behavior1k_lamp/bbentu/usd/bbentu.usd`; see
  `g1_bulb_env_cfg.py` / issue #14).
- Ground plane + randomized dome light.

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
