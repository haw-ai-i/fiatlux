# Roadmap

The scored benchmark is the **full replacement task** (`FIATLUX-Replace-v0`), decomposed into
twelve subtasks (`FIATLUX-S01-MoveLadder-v0` .. `FIATLUX-S12-ClimbDown-v0`). Unfinished items
below are listed so the extension seams are intentional.

## 1. Climbing subtask — ✅ DONE (2026-07-06), now `FIATLUX-S02-ClimbLadder-v0`

G1 climbs the work-site step ladder (`fiatlux_task.assets.STEP_LADDER_USD`, the
at-height preset) to the fixture height. All three deliverables shipped as part of the
climb subtask:

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
  bayonet never modeled a real feature). Wired for Replace and the S01/S03/S11 subtask-teleop
  tasks.
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
    `F(axial) = -tanh(axial/deadband) * hold_force/(1 + |axial|/hold_range) - spring_d *
    axial_rate` -- magnitude peaks at `hold_force` right at the seat, decays past
    `hold_range`, with a small `tanh` deadband replacing a literal `sign()` to avoid a
    direction-flip chatter risk exactly at rest. `hold_force`/`hold_range` are sized so the
    local stiffness at the seat (their ratio) stays under the same semi-implicit stability
    ceiling (~87 N/m) that bounded the spring, and so worst-case ceiling sag (~4.6mm) sits
    well clear of `release_threshold` -- `hold_force` ends up modest in absolute terms (~1.5x
    the bulb's weight) as a direct consequence: a magnet-shaped peak occurs exactly where its
    stability-relevant stiffness is evaluated, unlike a spring's cap sitting far out along an
    otherwise-gentle curve, so there's no way to get a strong peak, fast falloff, and the same
    stability margin at once. Table/wall mounts still sag less than ceiling ones under their
    own weight, correctly. Needs a regression pass on S11 (screw-in, ceiling) to confirm the
    insert-then-hold failure reported alongside this is the same root cause, and real-teleop
    validation that the new force-vs-distance shape (firm at contact, easier once separated)
    actually feels different from the spring it replaced.
  - **Twist friction (issue #171, third finding, 2026-09-08)**: real teleop found a
    ceiling-seated bulb spinning about the seat axis at 1-19 rad/s for a sustained ~2.9s, no
    operator or contact, before abruptly ejecting. Cause: the tilt torque's damping used the
    FULL angular velocity but shared tilt's tiny `max_torque` (0.05 N*m) budget -- arresting
    even 10 rad/s needed several times that, so it saturated uselessly every step. Split
    twist (rotation about the seat axis, no target angle -- issue #90) from tilt
    (misalignment, which does have a target) and gave twist its own budget. A first, viscous
    version of that (`-twist_d * twist_rate`) settled into a stable but NONZERO equilibrium
    spin under real contact, and raising its gain made the equilibrium worse at some tested
    magnitudes -- evidence of the wrong force law, not just an under-sized one. Replaced with
    Coulomb-like FRICTION (`twist_friction`, roughly constant magnitude, not
    velocity-proportional) instead, matching how real contact friction actually behaves.
    Verified (`scripts/verify_twist_damping.py`): both the viscous and friction versions
    reliably stop the actual reported failure (self-ejection) across the full 1-19 rad/s
    range, holding 5+ simulated seconds -- but NEITHER reliably drives the residual spin
    itself to zero; it persists at some nonzero, sometimes noisy rate. That residual looks
    like a real 3D contact effect (a loosely-toleranced plug precessing/rattling in the bore)
    rather than something a single-axis torque law can fully resolve -- open follow-up, not
    treated as solved, though the critical failure (detachment) is fixed.
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
`subtask_env_cfg.py` (and `replace_env_cfg.py`).
