# S05 — Climb the ladder

`FIATLUX-S05-ClimbLadder-v0` · mode **balance** · payload **none**
Read `00-foundation.md`, `ABSTRACTIONS.md`, `CONTINUITY.md` and `CRITIQUE.md` first.

## Objective

From the foot of the placed ladder, climb until standing on the upper steps at working height for
the fixture, under control.

## Start state

`START_STATE["S05"]` = S04's success state:

- Ladder standing, at rest, at the fixture-derived target — **not** `LADDER_POSITION`.
- Robot standing at the ladder's base, facing the steps, hands free.
- Old bulb seated in the inverted fixture above.

## Success gate

`climbed_to_ladder_top` — a **new** predicate. `climbed_to_target` takes a hardcoded `xy_center`
from `TOP_ROBOT_POSITION`, which is the default workshop layout's ladder. Here the ladder is
wherever the chain put it, so the gate must read the ladder's live pose:

| Condition | Value |
|---|---|
| pelvis height above the ladder's top-step height | `_ladder_top_point_w()[2] - 0.15` |
| pelvis within xy radius of the ladder top | 0.6 m (Climb's tolerance) |
| root speed | < 1.5 m/s (rejects a solver-kicked fly-through) |
| ladder not tipped | `LADDER_TILT_LIMIT = 0.6` |
| robot not fallen | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` |

Derive the height bound from `STEP_LADDER_TOP_OFFSET = (0, 0, 1.70)` applied to the ladder's live
root pose, exactly as `_ladder_top_point_w` already does. The `-0.15` slack is Climb's, so a solid
stance on the top steps scores rather than requiring the pelvis exactly at rail height.

## Rewards

- `climb_progress` = `climb_height_progress` — unchanged, pays each centimetre of new height once.
- `success_bonus` = `completion_bonus(predicate_fn=climbed_to_ladder_top)`.
- `ladder_contact` = `ladder_contact_fraction(sensor_cfg="ladder_contact", threshold=1.0)`,
  weight 0.25 — the touch-the-ladder bootstrap, unfarmable against the task total.
- `com_sway`, `ang_vel_xy`, `termination_penalty`, `ladder_tipped` penalty, base shaping.
- **No `flat_orientation_l2`** — climbing an A-frame requires a sustained forward lean, and
  Climb's cfg documents this. Do not add it.

## Terminations

`time_out`, `success=climbed_to_ladder_top`, `fell_below`, `fell_over`, `ladder_tipped`.
`episode_length_s = 20.0`.

## Sensing

`add_ladder_contact_sensor(scene)` — feet + palms filtered against `{ENV_REGEX_NS}/Ladder`.

Note the ladder is **dynamic** in this chain (S02–S04 required it), whereas Climb's at-height
preset uses a *kinematic* ladder. That is a real physics difference, not a detail: a dynamic
ladder can be climbed off-centre and tip, which is why `ladder_tipped` is a termination here and
is absent from `FIATLUX-Climb-v0`. Keep it dynamic — a kinematic ladder would make the placement
subtasks upstream meaningless.

## Reuse

`climb_height_progress`, `ladder_contact_fraction`, `_ladder_top_point_w`,
`STEP_LADDER_TOP_OFFSET`, `add_ladder_contact_sensor`, `LADDER_STANCE_*` (as the start-pose
reference for the mounting stance), `fall_terminated`, `completion_bonus`.
New: `climbed_to_ladder_top`.

## Visual start-state validation

Frames must show: both feet flat on the floor at the ladder's base, on the side the steps face
(the A-frame's steps face one way — a robot spawned at the back rail cannot climb); no limb
intersecting a rail or step; the ladder at rest with all four feet down. Use `--record_view
fixture` as well and confirm the fixture is above the ladder, since the mount is sampled.

Report which side of the ladder the robot is on and how it was determined — the ladder's yaw comes
from the layout draw (`ladder_yaw`), so "facing the steps" is a function of the sample, not a
constant.

## Blocking sensor fix — `ladder_contact` sees no hands on Dex3

`add_ladder_contact_sensor` uses `prim_path=".../Robot/.*(ankle_roll|hand_base)_link"`. That matches
Inspire's `*_hand_base_link` palms but **not Dex3's `*_hand_palm_link`** — so on the variant the
benchmark scores, the sensor resolves only the two ankle links and `ladder_contact_fraction` averages
over feet alone. The palms-on-the-ladder half of the bootstrap reward does not exist.

Fix it before scoring anything on that channel: resolve the bodies from `G1_PALM_BODY_BY_VARIANT` and
`G1_FOOT_BODIES` instead of a regex that encodes one variant's naming. This is the twin of the
`palm_body_index` bug PR #64 fixed, one file over, and it is live in Climb and Descend today.
Tracked as **#69**.

## Acceptance

As foundation, plus: `handoff:S04->S05` and `handoff:S05->S06` pass · a scripted mounting stance
(`LADDER_STANCE_JOINTS`) registers non-zero `ladder_contact` on the feet, proving the sensor and
the ladder's collider actually meet · the settle soak does not tip the ladder.

## Relationship to `FIATLUX-Climb-v0` — both stay

Not a replacement. `FIATLUX-Climb-v0` climbs a *kinematic* ladder at a fixed layout position with a
hardcoded success centre; this subtask climbs a *dynamic* ladder wherever a placement left it, with
the gate derived from the ladder's live pose and a tipping termination. Different physics, different
difficulty, both kept — see the foundation's difficulty-ladder note.

## As built

`subtasks/s05_climb_ladder_env_cfg.py`, on `subtask_tiers.balance.ClimbSubtaskCfg`. Constructs.

The gate is `mdp.all_of` over three conjuncts — `balance_terms.climbed_to_ladder_top` (height slack
0.15 m below the live top point, 0.6 m xy, 1.5 m/s), `place_terms.robot_standing`,
`grasp_terms.ladder_near_vertical` — with no `sustained` wrapper; the speed cap already excludes a
fly-through and the fall gates are separate terminations.

The stance is placed by rotating the workshop preset's own validated `CLIMB_ROBOT_POSITION` offset
onto the sampled ladder yaw, not by copying its world coordinates. The blocking Dex3 palm-naming
fix landed earlier in this branch, so the contact bootstrap keeps its full four-body denominator
here — both hands are free.

Not validated: no render, no rollout. Nobody has confirmed which side of the ladder the robot
lands on for a given draw.
