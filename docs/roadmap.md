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
- Old-bulb attach/detach mechanic (unification spec Phase 4): LANDED for Replace
  (issue #54, revised 2026-07-31) as the `mdp.bulb_attachment` bayonet state machine.
  It constrains each bulb to axial-only insertion/ejection or rotation-only locking,
  switching only when the bulb itself moves at the fully inserted junction. Insertion
  depth and lock angle are per-env scalar-or-range parameters for future domain
  randomization. All Replace score channels are achievable. Follow-up: put Remove/Install
  on the same mechanic; their bulbs are already dynamic but currently lift straight out
  of / drop straight into the socket.

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
