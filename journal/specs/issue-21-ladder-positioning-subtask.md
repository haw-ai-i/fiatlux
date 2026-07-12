# Issue 21: Ladder handling / positioning subtask — `FIATLUX-Carry-v0`

## Goal

Promote the `FIATLUX-Carry-v0` scaffold into a full RL subtask: the G1 **grasps a ladder and
positions it upright at a target near the light-bulb fixture** (the pre-climb step). Two
deliverables:

1. **Ladder grasping** — grasp affordance + interaction physics for carrying/positioning.
2. **Target placement verification** — reward terms + success criteria for placing the ladder
   within stable, correct alignment near the fixture.

## Scope (locked decisions)

- **Approach + grasp + position (whole-body)** — the ladder starts ~2 m out in front; the
  robot **walks to it**, grasps a rail, and carries it upright to the target under the light.
  Uses a **whole-body** action space (`joint_names=[".*"]` — legs + torso + arms + Inspire
  hands, exactly like the full Replace task), so locomotion and manipulation are both
  available. **Note:** whole-body locomotion + manipulation from scratch is hard to train (the
  same difficulty class as the flat Replace task); this issue delivers the *environment*
  (assets, layout, grasp affordance, whole-body obs/rewards, success criteria), not a converged
  policy.
- **Physics friction grasp** via the Inspire right hand (no attach/weld) — the env provides
  the affordance (a high-friction material on a dynamic ladder body); the policy learns to grip.

## Assets used

Referenced by constant in `source/fiatlux_task/fiatlux_task/assets.py`. Binary USDs live on GCS
(`gs://fiatlux/assets/…`) and sync locally via `assets/download_assets.sh`.

| Role | Constant (`fiatlux_task.assets`) | Asset (USD) | Notes |
|---|---|---|---|
| **Ladder** (graspable manipuland) | `STEP_LADDER_RIGID_USD` | `omniverse_ladder/HeavyDutyFRPStep_A/HeavyDutyFiberglassStepLadder_A01_PR_NVD_01_collision_rigid.usd` | The orange fiberglass **A-frame step ladder** — the **preconfigured `_collision_rigid` overlay** (sublayers the `_collision` variant's meshes + colliders and pre-applies a single **dynamic `RigidBodyAPI` + `MassAPI`**). cm-authored → spawn scale `0.01`, base at z=0. The Carry spawner tunes its props (solver/sleep; mass → `LADDER_MASS` = 3 kg) and binds a high-friction grip material. (Base/Climb use the plain `STEP_LADDER_USD` `_collision` variant, made kinematic.) |
| **Fixture — socket** | `SOCKET_USD` | `behavior1k_lamp/ehjsdz/ehjsdz.usd` | The **same BEHAVIOR-1K lamp** the Insert/Replace tasks use (the validated `bulblampF` socket). Spawned via `spawn_b1k_single_body`, kinematic. |
| **Fixture — bulb** | `BULB_USD` | `behavior1k_bulb/kfmkwd/kfmkwd.usd` | The **same BEHAVIOR-1K bulb** the Insert/Replace tasks use (the validated `bulblampM` plug). Spawned via `spawn_b1k_single_body`; made kinematic here (overhead context, not the manipuland). |
| **Robot** | `G1_INSPIRE_CFG` (`fiatlux_task.robots.g1`) | `unitree_g1/wholebody_inspire/...` | **Whole-body** control — all joints actioned (`joint_names=[".*"]`: legs + torso + arms + Inspire hands), like the full Replace task, so the robot can walk to the ladder and carry it. |

The light fixture the positioned ladder leads to is the **same `SOCKET_USD` + `BULB_USD` lamp
pair the Insert and Replace tasks use** (not a Carry-specific asset) — **mounted on the ceiling
directly above the target and flipped bulb-down, exactly how `apply_replace_preset` mounts its
ceiling fixture** (`(x, y, ROOM_CEILING_Z)` + `_quat_y_deg(180)`, with the bulb seated in the
flipped socket via the shared `SOCKET_SEAT_OFFSET` / `BULB_PLUG_OFFSET` math). Both kinematic —
visual context for "position the ladder under the fixture". **Layout:** the robot stands back
near the origin; the ladder **starts ~2 m out in front, a long distance from the robot and far
from the light** (`POSITION_LADDER_START_POS = (1.50, 0.85, 0)`); the fixed **target is directly
beneath the fixture** (`TARGET_LADDER_POSITION = (0.55, -0.30, 0)`, the fixture's x,y at floor
z), so the robot approaches the ladder and carries it to under the light (see the
approach/locomotion caveat in **Scope**).

## Implementation

- **Scene / preset** — `scene_cfg.py`:
  - `apply_position_preset(scene)`: spawns the `ladder` from the **preconfigured**
    `STEP_LADDER_RIGID_USD` (already a dynamic rigid body + mass) via
    `_spawn_usd_as_rigid_body_frictional`, which **modifies** the existing body (solver/sleep
    props, mass → `LADDER_MASS`) **and** binds a high-friction `RigidBodyMaterialCfg` —
    `UsdFileCfg` has no `physics_material` field, so it is bound inside the spawner. It no longer
    stamps `RigidBodyAPI`/`MassAPI` (the `_collision_rigid` asset ships them). Ladder starts
    upright ~2 m out; mounts the
    **default `socket`/`bulb` (B1K lamp pair, `spawn_b1k_single_body`) on the ceiling above the
    target** (same as `apply_replace_preset`). The `hand_contact` sensor stays for the net-force
    obs + compliance penalty (no ladder force-matrix filter).
  - Constants: `POSITION_ROBOT_POSITION`, `POSITION_LADDER_START_POS/_YAW`,
    `TARGET_LADDER_POSITION`, `LADDER_MASS`.
- **Env** — `carry_env_cfg.py` → `CarryEnvCfg(ManagerBasedRLEnvCfg)`, mirroring the full
  Replace task's whole-body recipe: whole-body actions (`joint_names=[".*"]`); policy obs
  (IMU + estimator + full proprio + hand contact) + privileged obs (robot / ladder / fixture
  poses); reset events; rewards + terminations (below).
- **MDP terms** — Carry **reuses the full Replace task's ladder scoring** (no Carry-specific
  reward code; the functions ship in `mdp/rewards.py` from issue #20):
  - `distance_progress(ladder_fixture_distance)` — dense, randomization-fair progress of the
    ladder top toward the fixture socket seat,
  - `completion_bonus(ladder_ready)` — sparse bonus when the ladder is upright and its top is
    horizontally within `LADDER_READY_XY_RADIUS` of the fixture,
  - `ladder_tipped` — penalty (and termination) past `LADDER_TILT_LIMIT`,
  - plus `fall_terminated` and the standard compliance / smoothness shaping.
- **Terminations**: `time_out`, `success=ladder_ready`, `ladder_tipped`, `ladder_dropped`
  (`object_dropped`), `fell_below`/`fell_over` (built-in bool fall terms).
- **Registration + agent** — `__init__.py` points `FIATLUX-Carry-v0` at the RL cfg +
  `agents/rsl_rl_ppo_cfg.py::CarryPPORunnerCfg` (explicit `obs_groups` so the privileged
  ladder-pose group reaches the critic).
- **Verifier** — `scripts/verify_scene.py`: the `init:<name>_at_cfg_pose` drift check is now
  range-aware (`_reset_pose_slack`) — it widens the tolerance by an entity's reset `pose_range`
  so the ladder's wide start-pose DR is not mis-flagged as cfg-vs-stage drift.

## Verification

```bash
conda activate env_isaaclab && export PYTHONPATH=$PWD/source/fiatlux_task
python scripts/verify_scene.py --headless --task FIATLUX-Carry-v0 --num_envs 1   # 30/30 PASS
python scripts/verify_scene.py --record --headless --task FIATLUX-Carry-v0  # eyeball poster/orbit
python scripts/record_run.py --task FIATLUX-Carry-v0 --policy random --episodes 1 --record bag --out logs/runs/carry
python scripts/rsl_rl/train.py --task FIATLUX-Carry-v0 --num_envs 32 --max_iterations 3 --headless  # PPO smoke
```

Verified: **30/30** verify checks, ruff-clean, **no regression** (Climb 30/30, Replace 42/42,
Insert unchanged), and the PPO smoke iterates with the **whole-body action space (53 DoF)**,
the reused ladder scoring + Climb/Replace stability shaping, and the critic receiving the
privileged robot/ladder/fixture poses.
