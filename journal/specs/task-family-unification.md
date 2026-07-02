# Task-Family Unification Plan — one world, many starting points

Status: Phases 0–3 LANDED (2026-07-02); Phase 3.5 (per-task start layouts, no RL) LANDED
(2026-07-02); Phases 4–5 DEFERRED. Owner: fiatlux maintainers.

Phase 3.5 — per-task start layouts: each scaffold starts in its phase of the Replace story
(carry: ladder stored at the wall; climb/descend: elevated chandelier fixture
(`assets.ELEVATED_SOCKET_USD`) with robot at base/top; remove: old bulb seated kinematic in
the fixture; install: empty fixture + fresh bulb in a parts crate (`assets.CRATE_USD`, new
optional `bin` entity)). verify_scene replaced the hardcoded preset pose asserts with
presence-per-preset (`scene_preset` attr) + a generic cfg-vs-stage init-state drift check,
and records each task with cfg-declared `orbit_*` framing.

Phase 3.5 regrouping (user feedback): Remove + Install moved onto Insert's bench world --
the manipulation trio shares the tabletop (Remove: bulb seated in the table lamp + empty
floor crate; Install: empty socket + bulb in the crate); the at-height chandelier world is
climb/descend-only. When Phase 4's screw mechanic lands, it anchors at the table lamp's
seat pose for all three bench tasks.

Phase 3.5 ladder revision (user feedback): all three B1K ladders are authored
lying/leaning (probe: shfvtl 2.41 long x 1.67 high, pivot rests at z=0.47 -- it had been
floating at z=0.85); the work-site ladder is now a deployed free-standing Omniverse
SimReady A-frame (`assets.STEP_LADDER_USD`, HeavyDutyFRPStep 0.68x1.11x1.75 m, cm-authored
-> spawn scale 0.01, `_collision.usd` carries visuals AND colliders, RigidBodyAPI applied
at spawn via `_spawn_usd_as_rigid_body` -- SimReady packs author colliders only). shfvtl
stays as Carry's stored cargo in its natural lying pose. Remove now shares Install's exact
work site (crate included, empty). 27 more SimReady ladder designs are local for later
variety/DR via MultiUsdFileCfg. Its fiberglass base MDL resolves via a missing relative
pack path (renders in fallback red; decals/textures bundled) -- cosmetic, noted.

Implementation notes from landing (decisions log additions):

- Manager blocks (Actions/Obs/Rewards/Terminations) stayed in ``g1_bulb_env_cfg.py``
  rather than moving to shared modules -- factoring happens when Install becomes their
  second consumer (avoid single-consumer abstractions).
- ``clone_in_fabric`` must stay False family-wide: the ``hand_contact`` sensor's PhysX
  contact-reporter cannot attach to fabric-cloned env prims.
- The Phase 0 Insert verification surfaced a real, PRE-EXISTING physics defect: an
  uncontrolled free-base G1 collapsing onto the kinematic table picks up violent solver
  kicks (observed up to ~1200 m root height / 470 m/s; matches the ~385 kN contact
  spikes in earlier score output). Two genuine mitigations landed -- family PhysX solver
  floors + ``enable_stabilization`` (~100x better worst case) and a hardware-realistic
  Inspire finger torque cap (100 -> 2 N.m in ``robots/g1.py``) -- but the instability is
  stochastic and persists (legs' 300 N.m PD vs the table edge). KNOWN ISSUE, deferred to
  the RL gate: the proper fix is task design (a robot-fell termination ends such episodes
  immediately) plus possibly compliant gains. Until then the acceptance battery verifies
  the Insert *scene* with ``--hold_base`` (the tool's documented mode for policy-less
  humanoids); the free-base run reproduces the defect on demand.
- verify_scene checks are RL-aware: within-episode kinematic drift (auto-reset events
  legitimately re-pose the socket) and reset-randomization-tolerant default-pose check.

## Goal

Merge `FIATLUX-Insert-v0` and the ladder family into **one task family** backed by one
parametrized scene, one shared manager vocabulary, and per-subtask cfgs that override only
what defines their phase — while keeping the advantages of both sides:

| Keep from Insert (RL side)                     | Keep from the ladder family (scaffold side) |
| ---------------------------------------------- | -------------------------------------------- |
| Trainability (rewards/terminations, PPO cfg)   | One scene source of truth                    |
| Replicated physics at training scale           | Modular subtask slots (subclass cfgs)        |
| Manipulation sensors (wrist cam, hand contact) | `verify_scene.py` coverage of every member   |
| Sensor-realizable vs `privileged` obs contract | Dressing + lighting/fixture randomization    |
| eval / record / score toolchain                | Cheap non-RL loading                         |

Design principle: **a subtask = the shared world + a phase initial state + a phase
reward/termination set + an action subset.** Subtasks are windows into the one long-horizon
Replace episode: Carry → Climb → Remove → Install → Descend; tabletop Insert is the
manipulation phase practiced on a bench.

"One family" is at the *class/preset* level, NOT the scene-instance level: training envs do
not spawn furniture their task never touches.

## Current state (anchors)

- Leaf module `tasks/manager_based/fiatlux_task/` (spec-conformant layout — do not move).
- `ladder_scene_cfg.py: G1LadderSceneCfg(DressedSceneCfg)` — ground(µ=1.0), key_light,
  robot(0,0,0.75), ladder(1.5,0,0.85) kinematic, `lamp`(-0.8,0,0.20) kinematic, bulb on
  floor, per-env random `fixture` (needs `replicate_physics=False`).
- `g1_ladder_env_cfg.py: G1LadderEnvCfg(ManagerBasedEnvCfg)` — proprio obs, whole-body
  53-joint actions, reset + enabled light-intensity randomization, dt=1/200 dec=4 (50 Hz).
- `g1_bulb_env_cfg.py: G1BulbInsertEnvCfg(ManagerBasedRLEnvCfg)` — own `G1BulbSceneCfg`
  (table, `socket` on table @1.20 m, bulb @1.05 m, hand_contact, wrist cam), arm+hand
  19-joint actions, rewards/terminations (`success`/`bulb_dropped`/`time_out` — names read
  by `recording.py`/`score.py`), dt=1/120 dec=4 (30 Hz), 15 s episodes, ground µ=default.
- Shared already: `scenes.py` (DressedSceneCfg, spawn_b1k_single_body), `robots/g1.py`,
  `assets.py`, `mdp/` (incl. `randomize_light_properties`), `viz.py`, `policy.py`,
  `recording.py` (expects entities `robot`/`bulb`/`socket`, sensor `hand_contact`).
- `verify_scene.py` — drives non-RL `(obs, extras)` step; hardcodes
  `TRACKED = [robot, ladder, lamp, bulb]`; 24 checks, real exit codes.

## Phases

### Phase 0 — verify_scene becomes family-wide (prep, ~1 h)

`scripts/verify_scene.py`:
- Tolerate RL envs: `res = env.step(actions)` → `obs = res[0]` (5-tuple vs 2-tuple).
- Build the tracked-entity list dynamically from the scene cfg instead of the hardcoded
  list: check each of `robot / ladder / socket / bulb / table` only when the cfg attribute
  exists and is not None. (Also covers presets that drop entities.)
- Add per-preset **initial-state checks** (the merge's acceptance signal): tabletop →
  table present, socket root z ≈ 1.20 ± 0.1 m, bulb ≈ 1.05 ± 0.1 m, no ladder; workshop →
  ladder present, socket on the floor (z < 0.5 m), no table. Derived from the scene cfg's
  `init_state` vs the spawned prims, so drift between cfg and stage is caught.

Verify: all 6 current ids still PASS, exit 0.

### Phase 1 — one parametrized scene (~half day)

1. `git mv ladder_scene_cfg.py scene_cfg.py`; class → `G1ReplaceSceneCfg` (the Replace
   world; keep `G1LadderSceneCfg = G1ReplaceSceneCfg` alias one release).
   `git mv g1_ladder_env_cfg.py base_env_cfg.py` (class `G1LadderEnvCfg` →
   `FamilyBaseEnvCfg`, alias kept). Update the leaf `__init__.py` entry-point strings.
   (In-leaf renames only — the spec's target layout is untouched.)
2. Rename scene entity **`lamp` → `socket`** (matches Insert's mdp/rewards/recording
   vocabulary; verify_scene tracked list is dynamic after Phase 0).
3. Add optional entities to the one scene: `table` (from `g1_bulb_env_cfg.py`, kinematic)
   — spawned only by the tabletop preset; `ladder` — dropped by the tabletop preset.
4. Preset helpers in `scene_cfg.py` (applied from env cfg `__post_init__`, since
   `InteractiveSceneCfg` treats every field as an entity — same pattern as the existing
   fixture guard):
   - `apply_workshop_preset(scene)` — today's ladder layout: ladder + socket-lamp on the
     floor, no table. (The at-height "elevated" mount arrives in Phase 4.)
   - `apply_tabletop_preset(scene)` — Insert layout: table @(0.40,-0.10,0), socket on
     table @(0.45,0,1.20), bulb @(0.35,-0.20,1.05), robot at the table, `ladder = None`.
5. Dressing-randomization flag on the base env cfg:
   `enable_dressing_randomization: bool = True`; when False → `scene.fixture = None`,
   `replicate_physics = True`, `clone_in_fabric = True`. Scaffolds keep True; RL training
   cfgs set False. (Resolves the fixture-vs-replication tension with no compromise.)
6. `hand_contact` ContactSensor moves into the shared scene (recording expects it;
   climbing will want contact sensing anyway). Wrist camera stays task-cfg-added (render
   cost only where needed).

Verify: verify_scene all 6 workshop ids 24/24; one `--record` orbit — visually identical
to the current Base video (modulo sampled fixture/lighting).

### Phase 2 — Insert joins the family (~half day)

`g1_bulb_env_cfg.py` keeps its Rewards/Terminations/EventCfg and `G1BulbInsertEnvCfg`, but:
- drops its private `G1BulbSceneCfg`; scene = `G1ReplaceSceneCfg` + tabletop preset,
  `enable_dressing_randomization = False` (replicated physics for training).
- Actions/Obs come from shared blocks factored into `base_env_cfg.py`:
  `WholeBodyActionsCfg` (family default) and `ArmHandActionsCfg` (manipulation phases);
  `ProprioObsCfg` and `ManipulationObsCfg` (proprio + wrist cam + contact + `privileged`).
- Keep entity names (`socket`, `bulb`) and termination names (`success`, `bulb_dropped`,
  `time_out`) — `recording.py`, `score.py`, `eval.py` stay untouched.
- **Control-rate caveat**: keep Insert at dt=1/120, dec=4, 15 s (baseline parity) via its
  `__post_init__` override; reconciling to the family's 50 Hz is a separate, deliberate
  decision (changes trained-policy dynamics and recorded metrics).
- Ground friction unifies to µ=1.0 (family ground): re-baseline Insert metrics (below).

Verify (no-RL scope — initial states + recordings only): `list_envs` 7 ids;
`verify_scene --task FIATLUX-Insert-v0` PASSES incl. the tabletop initial-state checks
(new capability); `verify_scene --record` orbit MPKs/posters for one workshop id and for
Insert — tabletop looks like today's Insert scene, workshop like today's Base scene.
Insert's rewards/terminations ride along *unchanged but unexercised*; the RL regression
gate (`eval --policy zero` before/after metric parity, `record_run`+`score`, train smoke)
is DEFERRED to whenever RL work resumes, and is required before trusting any training
run on the merged scene.

### Phase 3 — docs + registration polish (~1 h)

- `FIATLUX-Insert-v0` id is preserved (it *is* the tabletop preset of the manipulation
  phase). When Phase 4 lands, `FIATLUX-Install-v0` = same manipulation blocks on the
  elevated preset.
- Update README (task table: one family), CLAUDE.md (two-env-kinds section → one family,
  preset story, dressing flag), `docs/task_spec.md`, this spec → promote to `docs/` when
  stable.

### Phase 4 — screw mechanic + elevated preset (DEFERRED — first RL construction)

1. Shared mdp util (new `mdp/attach.py`): make/break `FixedJoint` between bulb and socket
   seat pose, gated by alignment + accumulated wrist roll ("screw"), per the TODOs already
   in `remove_env_cfg.py` / `install_env_cfg.py`. NOT threaded geometry.
2. `apply_elevated_preset(scene)`: socket mounted at ~2.2 m (wall/ceiling bracket prim),
   ladder placed beneath, no table.
3. `install_env_cfg.py` → `ManagerBasedRLEnvCfg`, reusing Insert's reward/termination
   blocks (factor them from `g1_bulb_env_cfg.py` into `mdp/` or a shared `task_blocks`
   module first) + the attach util + elevated preset + whole-body actions.
   `remove_env_cfg.py` mirrors it (break instead of make; bulb starts seated).
4. Climb/Descend/Carry gain their rewards/terminations independently (fall termination,
   height progress, rung contact — per roadmap); Carry flips the ladder to dynamic.

Verify: verify_scene per task (add per-preset checks: socket height, ladder presence);
eval smoke per new RL task; recorded orbit per preset.

### Phase 5 — phase handoff + Replace-v0 (DEFERRED)

1. State-bank util in the package: capture `(robot joint/root state, ladder/bulb/socket
   root states)` at each task's `success` into an npz bank; an `EventTerm(mode="reset")`
   samples a bank as the next phase's start distribution (Carry success states seed Climb,
   etc.). Falls back to hand-authored start ranges when a bank is absent.
2. `FIATLUX-Replace-v0`: elevated preset, whole-body actions, staged reward =
   phase-gated sum of the subtask terms, terminations from the union; register in the leaf
   `__init__.py` (slot already reserved).

## Risks / decisions log

- **Entity rename `lamp`→`socket`** touches only the scene + dynamic verify list; mdp and
  toolchain already say `socket`.
- **Old Insert checkpoints break** (obs/scene change). None are load-bearing yet; note in
  the changelog when Phase 2 lands.
- **Control-rate split** (30 Hz Insert vs 50 Hz family) is kept initially, on purpose.
- **Action-space split** (19 vs 53 joints) is per-phase by design; cross-phase policy
  transfer would need padding/masking — out of scope.
- **`replicate_physics`** is now a per-cfg decision via the dressing flag; document that
  scaled training must keep the flag False.
- Spec layout (`tasks/manager_based/fiatlux_task/`) is preserved throughout; only in-leaf
  file renames.

## Acceptance — current scope (Phases 0–3, no RL construction or testing)

- One scene class + two presets (tabletop / workshop), one shared block set; elevated
  preset deferred with Phase 4.
- 7 ids, all members of one family; `verify_scene.py --task <any id>` passes, including
  the per-preset initial-state checks; orbit recordings exist for both presets.
- `uvx pre-commit run --all-files` clean; every phase lands as its own verified commit.

## Acceptance — deferred (Phases 4–5, when RL work resumes)

- RL regression gate re-run for Insert on the merged scene (eval parity, record+score,
  train smoke) BEFORE any training is trusted.
- Elevated preset + screw mechanic; Remove/Install RL; Replace-v0; 8 ids total.
