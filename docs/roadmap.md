# Roadmap

The scored benchmark has two framings of the same job: **`FIATLUX-Replace-v0`**, the full
replacement as one flat episode, and the **twelve subtasks** (`FIATLUX-S01-MoveLadder-v0` …
`FIATLUX-S12-ClimbDown-v0`), the same chain cut into legs with their own success gates and
their own score model (`docs/scoring.md`). Each subtask also has a `-Training-v0` tier and a
`-Teleop-v0` twin (`docs/subtask_teleop.md`).

The older coarse envs — **insertion** (`FIATLUX-Insert-v0`), **climbing**
(`FIATLUX-Climb-v0`), and **ladder-positioning** (`FIATLUX-Carry-v0`: walk to the ladder,
grasp it, and carry it upright to a target — whole-body RL, `carry_env_cfg.py`) — remain as
functional development environments, not benchmark targets. Unfinished items below are listed
so the extension seams are intentional.

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
  - **Axial retention against gravity (issue #171, second finding, 2026-09-08)**: re-teleop
    after the centering fix found wall mounts hold cleanly, but a ceiling-mounted bulb falls
    out unassisted within under a second, no operator or contact. Cause: a ceiling fixture is
    inverted, so the seat axis points down and the bulb's own weight acts entirely along it,
    in the OUTWARD/release direction -- the axial spring's steady-state hold distance under
    that load (~1.7cm at the original 20 N/m gain) left almost no margin before
    `release_threshold` (2cm) on its own, before any transient. First attempt added a gravity
    feedforward that cancelled it outright (steady-state sag ~0 everywhere) -- rejected on
    review: a passive retention mechanism (spring, friction, magnet) doesn't know its own
    orientation and null out whatever load that implies, so a bulb hanging against gravity
    SHOULD sag more than one resting with it. Second attempt just raised the linear spring's
    stiffness -- also reconsidered: a spring is weakest exactly at the seat and grows with
    distance, backwards from what this is meant to model (a magnetic/detent catch, strongest
    at contact, falling off with distance). Landed on a magnet-shaped axial law instead:
    magnitude peaks at `hold_force` right at the seat and falls off with withdrawal. That shape
    has since been revised once more -- the law now falls off **linearly to zero at
    `bore_depth`** (0.025 m, MEASURED by `scripts/measure_bore_geometry.py`: the withdrawal at
    which the plug clears the socket throat, past which there is no bore to be inside of),
    rather than decaying asymptotically past a `hold_range`, which left the ceiling case an
    unstable gravity balance point. Current sizing, against the bulb's 0.035 kg / 0.343 N:
    `hold_force` 1.5 N is 4.4x its weight at the bore bottom and still 1.75x at
    `release_threshold`, putting the gravity crossing at 19.3 mm -- outside the threshold, so on
    an inverted ceiling mount gravity alone cannot walk the bulb out. The steepest slope the law
    reaches is `hold_force/bore_depth` = 60 N/m, under the semi-implicit stability bound
    (mass/step_dt^2, ~88 N/m at 50 Hz) with 31% to spare. `release_threshold` has moved twice
    since (20 -> 8 -> 15 mm); `mdp/attach.py`'s module docstring carries the reasoning for each.
    Table/wall mounts still sag less than ceiling ones under their
    own weight, correctly. Needs a regression pass on S11 (screw-in, ceiling) to confirm the
    insert-then-hold failure reported alongside this is the same root cause, and real-teleop
    validation that the new force-vs-distance shape (firm at contact, easier once separated)
    actually feels different from the spring it replaced.
  - **Twist: no term at all, after two were tried and removed (issue #171, third finding)**: a
    ceiling-seated bulb was found spinning about the seat axis at 1-19 rad/s for ~2.9s, no
    operator or contact, before abruptly ejecting. A viscous term was tried first and settled
    into a nonzero equilibrium spin that got WORSE as its gain rose; it was replaced with a
    Coulomb-like constant-magnitude friction (`twist_friction = 0.5 N*m`). **Both were wrong,
    and the Coulomb term WAS the bug.** A term-by-term ablation
    (`scripts/ablate_attach_forces.py`, seed 3, forced ceiling mount, zero action, no injected
    spin) found that zeroing `twist_friction` -- and no other term -- removes the spin: 95-97
    rad/s with it, 0.7-3.0 rad/s without, holding 4/5 repeats instead of 0/5, while every other
    single-term ablation still spun at ~95 rad/s. With nothing applied at all the bulb shows
    ~1.2 rad/s, so the term was generating the spin, not failing to suppress it. The mechanism
    is a unit-scale error: the plug's moment of inertia about the seat axis is 2.9e-05 kg*m^2,
    so one control step of 0.5 N*m changes the twist rate by 344 rad/s, while the law reverses
    sign whenever `|twist_rate|` crosses the 0.1 rad/s deadband -- overshooting zero by ~3400x
    every step and re-accelerating the other way. A sign-flipping friction law is dissipative
    only if its impulse cannot exceed the momentum it opposes (`tau <= I*|omega|/dt`); this one
    exceeded it by ~350x. Coulomb chatter, not friction -- and the original teleop bag confirms
    it directly: twist changes sign on 138 of 140 consecutive control steps at +/-16 rad/s, a
    clean alternation at exactly the 50 Hz control rate. What the operator saw as a spin was
    that alternation. **There is no twist term now, in any form**: rotation about the seat axis
    is left to the socket's real contact friction, which the asset already supplies (verified,
    not assumed -- static 1.2 / dynamic 1.0, resolving at runtime onto all 2 bulb and all 8
    socket colliders), and issue #90's "rotation is free" stands as written. The angular-velocity
    SPLIT into twist and tilt components was the one sound part of the first attempt and is
    kept, so tilt damping spends its tiny budget only on the component it has a target for.
    Should a twist term ever be wanted again it must bound its impulse by `I*|omega|/step_dt`.
    (`scripts/verify_twist_damping.py` predates the ablation and tests the removed term.)
  Follow-up: put Remove/Install on the same mechanic; their
  bulbs are already dynamic but currently lift straight out of / drop straight into the
  socket (issue #76 Step 2).

## 3. Learned-policy support

- Imitation pre-training (ACT / Diffusion) from teleop or scripted "cheat-code"
  demos, using the `privileged` observation group. **The demos exist now**: the subtask
  teleop twins record scored HDF5 bags per take (`docs/subtask_teleop.md`), so this is a
  consumer-side gap, not a collection one.
- The current PPO config (`agents/rsl_rl_ppo_cfg.py`) covers RL fine-tuning.
- Demo recording lives in `source/fiatlux_teleop/` behind the `teleop` extra, keeping its
  dataset tooling out of the core install.

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
