# Fiatlux Repository Structure Spec

## Purpose

Define the target repository structure for Fiatlux as a small, reproducible Isaac Lab benchmark for humanoid light-bulb replacement.

The reference layout:
NVIDIA's [IsaacLabEvalTasks](https://github.com/isaac-sim/IsaacLabEvalTasks)

The near-term deliverable is an Isaac Lab scene that initializes the robot and task assets, then grows into a benchmark with standard evaluation scripts.

**Publication target:** the benchmark itself is the publishable result (Jun 22 sync; M2SV cited as a model). The deadline is **August 15, 2026**. Everything that does not directly support a runnable, evaluable scene by that date is roadmap.

## Design Goals

1. Keep the benchmark surface small.
   Fiatlux should be understandable as one Isaac Lab extension plus thin helper
   scripts. A new contributor should not need to understand the old AIC stack.

2. Make Isaac Lab the core runtime.
   The benchmark may later have sim-to-real adapters, but the repo's main path
   should be a pure Python / Isaac Lab extension with Gym-registered env ids.

3. Separate benchmark code, assets, docs, and project memory.
   Runtime code belongs under `source/`. Binary assets are pulled by scripts.
   User-facing docs belong under `docs/`. Internal meeting notes, scratch specs, and
   decision records belong under `journal/`.

4. Support embodiment flexibility without blocking the first physical setup.
   Unitree G1 with the available hand assets can be the first supported
   embodiment, but task definitions should avoid hard-coding assumptions that
   prevent adding other embodiments.

## Reference Repo Lessons

`IsaacLabEvalTasks` is a useful model as it keeps a clear split between:

- `source/<extension>/`: the Isaac Lab extension package.
- `source/<extension>/<package>/tasks/`: Gym task registration and environment
  config modules.
- `source/<extension>/<package>/tests/`: environment smoke tests.
- `scripts/`: runnable evaluation, policy, conversion, and helper entrypoints.
- `scripts/config/`: policy/model-specific runtime config.
- `doc/`: installation, datasets, checkpoints, evaluation, and troubleshooting.
- External assets/checkpoints/datasets that are referenced by docs and scripts,
  not committed directly.

Fiatlux should follow the same shape, but stay narrower. We do not need a
general industrial-task package or GR00T-specific layout until those features
actually exist.

## Target Layout

```text
fiatlux/
  README.md
  pyproject.toml
  LICENSE
  LICENSE.isaaclab

  assets/
    .gitignore
    README.md
    download_assets.sh
    behavior1k_uploaded_manifest.csv
    # Binary USD, mesh, texture, and key material are never committed.

  docs/
    overview.md
    getting_started.md
    task_spec.md
    scoring.md
    roadmap.md
    troubleshooting.md              # add once setup failures recur

  journal/
    specs/
      *.md                          # design decisions not yet promoted to docs/
    syncs/
      *.transcript.md
      *.md

  scripts/
    list_envs.py
    zero_agent.py
    random_agent.py
    teleop.py
    eval.py
    behavior1k_asset_intake.py
    rsl_rl/
      train.py
      play.py
      cli_args.py
    policies/                       # add when nontrivial policy adapters exist
      __init__.py
      policy_base.py
    evaluators/                     # add when eval.py outgrows one file
      __init__.py
      evaluator_base.py
    config/                         # add for policy-specific YAML/JSON configs

  source/
    fiatlux_task/
      config/
        extension.toml
      pyproject.toml
      setup.py
      fiatlux_task/
        __init__.py
        tasks/
          __init__.py               # imports/registers Fiatlux envs
          manager_based/
            fiatlux_task/
              __init__.py           # gym.register(...)
              g1_bulb_env_cfg.py
              mdp/
                __init__.py
                observations.py
                rewards.py
                events.py
                terminations.py     # add if done logic outgrows env cfg
              agents/
                __init__.py
                rsl_rl_ppo_cfg.py
        assets/                     # only lightweight Python asset constants
          __init__.py
        tests/
          test_env_registration.py
          test_random_rollout.py
```

## Directory Contracts

### `source/fiatlux_task/`

It should contain the Isaac Lab extension and all code
needed to register, instantiate, and step Fiatlux environments.

Rules:

- All Gym env ids are registered through `fiatlux_task.tasks`.
- Environment configs live under `fiatlux_task/tasks/...`.
- MDP terms should be grouped by purpose: observations, rewards, events, and
  terminations.
- Code in `source/` may reference assets by path, but must not download,
  decrypt, upload, or generate heavy assets as an import side effect.
- Optional sim-to-real adapters do not belong in the core task package until
  there is a tested reason to put them there.

Near-term tasks:

- `FIATLUX-Insert-v0`: first runnable manipulation benchmark.
- `FIATLUX-Climb-v0`: roadmap until ladder locomotion exists.
- `FIATLUX-Replace-v0`: roadmap until the combined episode exists.

### `scripts/`

Scripts are command-line entrypoints for humans and CI. They should be thin.
Task logic should move into `source/fiatlux_task` once it is reusable.

Keep now:

- `list_envs.py`: registration sanity check.
- `zero_agent.py` and `random_agent.py`: baseline floors.
- `teleop.py`: manual interaction if it still runs.
- `eval.py`: standardized metrics.
- `rsl_rl/`: training and play wrappers.
- `behavior1k_asset_intake.py`: one-time BEHAVIOR-1K data intake tool; not a
  benchmark entrypoint. Remove or move to `tools/` once intake is complete.

Add later only when needed:

- `scripts/policies/` for model wrappers such as GR00T, ACT, diffusion policy,
  or custom checkpoint inference.
- `scripts/evaluators/` if `eval.py` needs multiple evaluator classes.
- `scripts/config/` for policy/model-specific action, observation, or modality
  configs.

### `assets/`

`assets/` is a local cache boundary, not a source-code package.

Rules:

- Commit scripts, READMEs, and manifests.
- Do not commit decrypted USDs, meshes, textures, checkpoints, keys, or large
  dataset payloads.
- Pull approved assets from GCS with `assets/download_assets.sh`.
- Keep the manifest specific enough that a contributor can tell which asset
  groups are required for each benchmark phase.
- If third-party license terms require click-through or local decryption, the
  repo should document the flow and keep restricted material outside git.

Recommended sublayout after download:

```text
assets/
  behavior1k_bulb/
  behavior1k_bulb_broken/
  behavior1k_lamp/
  behavior1k_ladder/
  behavior1k_<scene_dressing>/
  unitree_g1/                       # recovered from prior Isaac Lab G1 run, not from GCS
```

### `docs/`

`docs/` is the contributor-facing manual.

Rules:

- `README.md` should be short and current.
- `docs/getting_started.md` should be the canonical setup path.
- `docs/task_spec.md` should describe implemented task behavior, or clearly
  mark sections as planned.
- `docs/scoring.md` should define the stable benchmark protocol.
- `docs/roadmap.md` should contain future tasks and optional sim-to-real work.
- Avoid copied AIC phase documents unless they are rewritten for Fiatlux.
- Internal tracking and asset collection notes belong in `journal/specs/`, not here.

### `journal/`

`journal/` is project memory, not runtime documentation.

Rules:

- Keep sync transcripts, summaries, exploratory notes, and decision specs here.
- Use `journal/specs/` for design decisions that have not yet been promoted into
  user-facing docs.
- Once a decision becomes stable user documentation, summarize it in `docs/` and
  leave the full rationale in `journal/`.

## Embodiment Policy

The first supported robot can be Unitree G1 with the hand configuration available
to the team. However, the benchmark should keep these seams explicit:

- Robot asset and joint/body names live in a small set of constants or config
  modules, not scattered across rewards and scripts.
- Action space definitions should be documented in terms of expected command
  semantics, for example joint-position targets, not only one hardware SDK.
- Observation groups should distinguish sensor-realizable policy observations
  from privileged simulator state.
- Success metrics should be task-level, for example bulb seated in socket, so a
  future embodiment can be compared without changing the headline metric.

This keeps the current physical setup supported while reducing the risk that the
benchmark is only usable by labs with exactly the same hand asset.

## Testing Expectations

Add tests under `source/fiatlux_task/fiatlux_task/tests/` following the
`IsaacLabEvalTasks` pattern.

Minimum checks:

- Environment registration imports without errors.
- `FIATLUX-Insert-v0` can instantiate in headless Isaac Lab.
- A random-action rollout for a small number of steps returns finite
  observations, rewards, dones, and infos.
- Evaluation metrics are JSON-serializable and stable for a fixed seed.

Tests that require Isaac Sim/GPU can be marked separately so lightweight CI can
still run linting and pure-Python checks.

## What Was Cut from AIC

The Jun 24 sync decided to nuke the AIC-derived structure rather than migrate it. The following are explicitly removed and should not be reintroduced without a concrete reason:

- **Gazebo / bring-up package** — not needed; Isaac Lab is the only runtime.
- **ROS** — deferred until sim-to-real transfer requires it.
- **Multi-simulation setup** — not needed for a single-env benchmark.
- **Phase 1 / Phase 2 / Phase 3 doc files** — AIC leftover; replaced by `docs/` and `journal/`.
- **`fiatlux_utils/isaaclab/` adapter layer** — wrong structure; Isaac Lab extension pattern used instead.

The baseline is the prior Isaac Lab G1 run, not the AIC codebase.

## Migration Plan

1. Delete AIC remnants.
   Remove any file that is a renamed/lightly edited copy from AIC with no Fiatlux-specific content.
   (Jun 24 sync: "just nuke everything that doesn't look like it belongs.")

2. Make docs honest.
   Ensure `README.md` and `docs/*.md` distinguish implemented behavior from
   roadmap behavior.

3. Stabilize the first runnable scene.
   The immediate milestone is a scene that initializes Unitree G1 and the
   required bulb/lamp/ladder assets in Isaac Lab by **August 15, 2026**.

4. Add registration and rollout tests.
   Mirror the reference repo's environment smoke-test style, scaled down to the
   available Fiatlux task ids.

5. Grow optional policy and sim-to-real folders only after they have real code.
   Do not pre-create large empty structures unless a near-term PR will use them.

## Open Questions

- Should `FIATLUX-Insert-v0` stay under
  `tasks/manager_based/fiatlux_task/`, or should task folders move toward a
  semantic path such as `tasks/manipulation/lightbulb/`?
- Where will the canonical Unitree G1 asset come from, and should it be synced
  through the same GCS asset contract as BEHAVIOR-1K assets?
- Which hand configuration is the first supported embodiment, and what is the
  smallest config interface needed to swap it later?
- Should `journal/` eventually be ingested by an agent memory system? **Decided (Jun 24):** MD files in the repo are sufficient for now. No agent stack yet, but the structure is intentionally compatible with future ingestion.
