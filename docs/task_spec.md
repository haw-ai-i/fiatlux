# Task Specifications

# `FIATLUX-Replace-v0` — the benchmark task

Defined in
`source/fiatlux_task/fiatlux_task/tasks/manager_based/fiatlux_task/replace_env_cfg.py`.
The full light-bulb replacement, scored as **one flat RL episode** (no policy stitching
or stage chaining — that is solution structure, not benchmark structure). Design notes are
in that file's module docstring.

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
  every score channel is achievable. No physics-in-the-loop regression test currently covers
  this mechanic (issue #237).
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
    matches the latter: peak force (`hold_force`) right at the seat, decaying past
    `hold_range`. This also fixed a real ceiling-mount failure along the way: a ceiling
    fixture is inverted (seat axis points down), so gravity pulls a seated bulb OUTWARD
    along it, and the original linear spring's steady-state sag under that load left almost
    no margin before `release_threshold` -- a ceiling-seated bulb fell out unassisted within
    under a second. `hold_force`/`hold_range` are sized (not a gravity feedforward -- a
    passive mechanism doesn't cancel gravity outright, and a ceiling-hung bulb should sag
    more than a resting one, same as any real spring/friction/magnet) so worst-case sag sits
    comfortably clear of `release_threshold`. Table/wall mounts sag less than ceiling ones
    under their own weight, correctly.
  - **Twist friction (issue #171)**: a ceiling-seated bulb was found spinning about the seat
    axis at 1-19 rad/s for ~2.9s, no operator or contact, before ejecting -- the tilt torque's
    damping shared tilt's tiny torque budget, which couldn't arrest a real spin. Twist
    (rotation about the seat axis, no target angle) now gets its own, Coulomb-like friction
    term (`twist_friction`, roughly constant magnitude, not velocity-proportional -- a
    viscous version tried first settled into a stable but nonzero spin under real contact and
    got worse, not better, as its gain was raised). Reliably stops the self-ejection across
    the full reported range, but does not reliably drive the residual spin itself to zero --
    that looks like a real 3D contact effect, open follow-up.

## Goal

Insert the fresh bulb into the fixture, remove the old bulb from the fixture, and place
the old bulb in the disposal crate. Full success = fresh bulb **attached** (seated and held
by the retention spring, per `mdp.bulb_attachment`) **and** old bulb in the crate. Seating
alone no longer scores.

## Actions

Whole-body joint-position targets (all DoF incl. fingers, like Climb): the task spans
locomotion, ladder work, and manipulation.

## Observations — `standard` vs `cheatcode` modes

Two groups (named `policy`/`privileged` for rsl_rl's routing; the benchmark calls the
modes **standard** and **cheatcode**):

- **`policy` = standard mode** (sensor-realizable only): IMU (base angular velocity,
  projected gravity), estimated base height + linear velocity (the documented
  estimator-realizable exception, as in Climb), joint pos/vel, hand contact forces,
  **head-mounted (`d435_link`) RGB camera features** and **head-mounted (`mid360_link`)
  lidar ranges** (camera needs `--enable_cameras`), last action. Corruption enabled.
- **`privileged` = cheatcode mode** (exact simulator state): world poses of the robot,
  ladder, fixture, fresh bulb, old bulb, and disposal crate, plus the four score-relevant
  distances (`replace_score_distances`). Critic-only during RL
  (`ReplacePPORunnerCfg.obs_groups`); a cheatcode policy may consume it directly.

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
| `ladder_ready` (+) | completion | upright ladder top horizontally within 0.9 m of the fixture |
| `fresh_bulb_inserted` (+) | completion | fresh bulb screwed in (the attach gate; `mdp.bulb_attachment`'s 1.5 cm / 0.2 rad seating tolerances are enforced at attach time) |
| `old_bulb_removed` (+) | completion | old bulb unscrewed and cleared the seat by 0.10 m (a held bulb reads as seated) |
| `old_bulb_disposed` (+) | completion | old bulb inside the disposal crate (containment, any orientation) |
| `success_bonus` (+) | sparse | full replacement (fires on the terminating step) |
| `robot_fall`, `ladder_tipped`, `fresh_bulb_dropped`, `old_bulb_dropped` (−) | penalty | each fires once — the same predicate also terminates |
| `contact_penalty` (−) | penalty | hand contact force (fragile-handling proxy) |
| `com_sway`, `ang_vel_xy`, `action_rate`, `joint_acc`, ankle limits, waist/finger deviation (−) | shaping | stability / smoothness |

## Success & termination

- **Success** (`attached_replacement_success`): fresh bulb screwed in (attachment state,
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
reset. Prop-scale randomization is an RL opt-in (prestartup USD writes require
`replicate_physics=False`); see `replace_env_cfg.py`'s `EventCfg`.

