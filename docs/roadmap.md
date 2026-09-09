# Roadmap

The scored benchmark is the **full replacement task** (`FIATLUX-Replace-v0`); the
**insertion** (`FIATLUX-Insert-v0`), **climbing** (`FIATLUX-Climb-v0`), and
**ladder-positioning** (`FIATLUX-Carry-v0`: walk to the ladder, grasp it, and carry it
upright to a target — whole-body RL, `carry_env_cfg.py`) subtasks remain as functional
development environments, not benchmark targets. Unfinished items below are listed so the
extension seams are intentional.

## 1. Climbing subtask — `FIATLUX-Climb-v0` — ✅ DONE (2026-07-06)

G1 climbs the work-site step ladder (`fiatlux_task.assets.STEP_LADDER_USD`, the
at-height preset) to the fixture height. All three deliverables landed in
`climb_env_cfg.py`:

- ~~Upgrade the scaffold to `ManagerBasedRLEnvCfg`; add a whole-body / locomotion
  action space (the G1 base is already free).~~ Whole-body joint-position targets
  (all 53 DoF).
- ~~Add a fall-detection termination (base height / orientation thresholds).~~
  `fell_below` (root z < 0.35 m) + `fell_over` (tilt > 1.0 rad).
- ~~Reward: progressive height + hand/foot–rung contact + CoM-sway penalty.~~
  `mdp.climb_height_progress` / `mdp.ladder_contact_fraction` (filtered contact
  sensor on feet+palms) / `mdp.com_sway_l2`, see `docs/task_spec.md`.

Remaining polish for later: a start-state curriculum (mounted poses from
`fiatlux_task/poses.py`), and the phase-handoff state bank (unification spec
Phase 5).

## 2. Full task — `FIATLUX-Replace-v0` — ✅ PRIMARY BENCHMARK (2026-07-09)

Promoted from scene-only scaffold to the scored full-task RL environment per
`journal/specs/full-task-benchmark-plan.md` (this deliberately reverses the earlier
"stays off the roadmap" call for the *task itself*; the policy-stitching part of that
descoping still stands — it is one flat RL episode, chaining is solution structure).

- ~~Build and verify the full combined-family scene: robot, ladder, table+bulb, and the
  elevated fixture together, each randomized into its own non-overlapping floor "safe
  zone" per scene build, fixture randomly ceiling- or wall-mounted.~~ DONE (2026-07-07):
  `scene_cfg.apply_replace_preset`.
- ~~Reward/termination logic: normalized-progress scoring (spawn-distance fair), sparse
  completions, fall/tip/drop penalties, full-success termination; `standard` (sensor) vs
  `cheatcode` (privileged) observation modes; `basic_standard`/`basic_cheatcode`
  smoke-test policies.~~ DONE (2026-07-09): `replace_env_cfg.py`, see
  `docs/task_spec.md` / `docs/scoring.md`.
- Policy stitching / staged-curriculum chaining: not planned (solution structure).
- Old-bulb attach/detach mechanic (unification spec Phase 4): the original bayonet state
  machine (issue #54) was superseded (issue #167, 2026-09-08) by a simpler two-state
  `mdp.bulb_attachment` FREE/SEATED axial detent -- real bulb-socket collision (re-enabled
  globally; the old collision filter is gone) now constrains lateral position and
  orientation on its own, so the only thing left to script is retention: a continuous
  spring-damper WRENCH while seated, release on a real physics-driven axial pull past
  `release_threshold`. No twist/lock semantics (this asset has no physical lug/groove; the
  bayonet never modeled a real feature). Wired for Replace, the S01/S03/S11 subtask-teleop
  tasks, and `FIATLUX-Insert-v0` (RL). **Not yet wired for `FIATLUX-Insert-Teleop-v0`**: that
  task swaps in a differently-scaled OMNI socket/bulb asset whose seat/plug geometry hasn't
  been measured against the family asset's calibrated offsets, so retention there needs its
  own calibration pass first (see the TODO in `insert_teleop_env_cfg.py`).
  - **Lateral + tilt centering (issue #171, 2026-09-08)**: real teleop evidence found a
    seated bulb visibly tilts/swings -- the axial-only design left lateral position and
    orientation entirely to real contact, and the bore's necessary radial clearance (2.69mm)
    is real slop, not a defect (`scripts/diagnose_contact_axial.py` confirmed tightening it
    even to 1.86mm breaks force-driven insertion outright, since the bore has no lead-in
    chamfer). Fixed in software instead: a SEATED bulb now also gets a much gentler lateral
    spring-damper and a tilt spring-damper torque, both far weaker than the axial term so
    they damp wobble without fighting real contact or affecting insertion (gated on
    `seated_now` exactly like the axial term). Gains are rough starting points, same as the
    original axial ones -- needs real-teleop retuning before trusting the numbers.
  - **Gravity feedforward (issue #171, second finding, 2026-09-08)**: re-teleop after the
    centering fix found wall mounts hold cleanly, but a ceiling-mounted bulb falls out
    unassisted within under a second, no operator or contact. Cause: a ceiling fixture is
    inverted, so the seat axis points down and gravity acts entirely along it, in the
    OUTWARD/release direction -- the axial spring's steady-state hold distance under that
    load (~1.7cm at the shipped gain) left almost no margin before `release_threshold` (2cm)
    on its own, before any transient. Fixed by adding a gravity feedforward to the axial term
    that cancels gravity's own component along the seat axis every step, so the spring only
    ever corrects deviations rather than also holding static weight -- steady-state sag is
    now ~0 at any mount orientation, not a margin tuned around whichever one was tested.
    Wall mounts are unaffected (gravity is ~perpendicular to a horizontal seat axis there, so
    the feedforward is ~0). Uses a hardcoded bulb mass and gravity vector (module has no
    direct read of the sim's configured gravity); needs a regression pass on S11
    (screw-in, ceiling) to confirm the insert-then-hold failure reported alongside this is
    the same root cause.
  Follow-up: put Remove/Install on the same mechanic; their
  bulbs are already dynamic but currently lift straight out of / drop straight into the
  socket (issue #76 Step 2).

## 3. Learned-policy support

- Imitation pre-training (ACT / Diffusion) from teleop or scripted "cheat-code"
  demos, using the `privileged` observation group.
- The current PPO config (`agents/rsl_rl_ppo_cfg.py`) covers RL fine-tuning.
- If demo recording is re-added, keep its dataset tooling (LeRobot/HDF5) optional
  and out of the core install.

## 4. Sim-to-real (physical G1)

Added as a **separate optional deployment adapter**, never the old ROS/Zenoh
harness:

- **Action bridge:** policy joint-position targets → Unitree SDK joint commands.
- **Observation bridge:** real proprioception + wrist camera + F/T → the `policy`
  observation vector (the `privileged` group is sim-only).
- **Perception:** estimate socket/bulb pose (ArUco or segmentation) to replace the
  ground-truth poses the scripted baseline uses.
- **Safety:** E-stop + contact-force limits (the env already penalizes contact
  force, so a hardware threshold maps cleanly).

The two design constraints that keep this cheap — hardware-realizable actions and a
sensor-realizable default observation group — are already baked into
`g1_bulb_env_cfg.py`.
