# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fiatlux is an Isaac Lab benchmark for humanoid light-bulb replacement: a Unitree G1
inserts a bulb into a socket. Pure Python / Isaac Lab extension — no ROS. One functional
RL task (`FIATLUX-Insert-v0`) plus a family of non-RL ladder-scene scaffolds
(`FIATLUX-{Base,Carry,Climb,Descend,Remove,Install}-v0`). `FIATLUX-Replace-v0` is roadmap.

## Commands

### On this machine (NVIDIA Brev Isaac Sim launchable) — read first

This box runs the pre-installed Isaac Lab at `/workspace/isaaclab`, **not** the uv
environment. Do not use `uv run python`; run every script with the bundled launcher:

```bash
source /isaac-sim/.local/nvidia-gl-580/env.sh    # Vulkan/GL fix — required for any rendering, cameras, --record
/workspace/isaaclab/isaaclab.sh -p scripts/<script>.py ...
```

- `fiatlux_task` is pip-installed editable into the kit python
  (`/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/fiatlux_task`).
- The container ships only the NVIDIA *compute* userspace libs. The graphics libs
  (matching host driver 580.159.03) were extracted from `libnvidia-gl-580-server` into
  `/isaac-sim/.local/nvidia-gl-580/`; `env.sh` points `VK_DRIVER_FILES`/`LD_LIBRARY_PATH`
  at them. Without it, Kit logs `VkResult: ERROR_INCOMPATIBLE_DRIVER` and RTX rendering
  is unavailable (headless physics still works).
- `gsutil`/`gcloud` live at `/isaac-sim/google-cloud-sdk/bin/` (not on PATH); needed by
  `assets/download_assets.sh`.
- Kit hard-exits on app close, so block-buffered stdout is lost when redirecting output
  to a file — keep `PYTHONUNBUFFERED=1` (env.sh sets it) or tables like `list_envs.py`'s
  silently vanish.

### Standard (uv) workflow

Everything runs through `uv` from the repo root (Python 3.11 pinned; Isaac Sim 5.1 +
Isaac Lab 2.3.2 + torch cu128 are all pinned in `uv.lock` — never install Isaac Lab manually):

```bash
uv sync                        # full environment (~10 GB first run; UV_HTTP_TIMEOUT=1200 if a wheel times out)
./assets/download_assets.sh    # USDs from GCS (needs gsutil); FIATLUX_ASSET_BUCKET to override bucket
```

Assets are git-ignored; point the env at an existing asset dir with `FIATLUX_ASSETS_DIR`.
First non-interactive Isaac Sim launch hangs on the EULA prompt — set `OMNI_KIT_ACCEPT_EULA=YES`.

```bash
uv run python scripts/list_envs.py                                  # sanity-check registrations
uv run python scripts/zero_agent.py --task FIATLUX-Insert-v0        # launch scene (also random_agent.py, teleop.py)

# Ladder-family scaffolds are verified with verify_scene.py ONLY (they are non-RL envs;
# train/eval/teleop assume RL envs). Prints PASS/FAIL table, exits non-zero on failure.
uv run python scripts/verify_scene.py --headless                    # defaults to FIATLUX-Base-v0
uv run python scripts/verify_scene.py --headless --task FIATLUX-Climb-v0
uv run python scripts/verify_scene.py --record --hold_base --headless --num_envs 1  # orbiting MP4 -> logs/verify/

# Evaluate / record / score (Insert task). --enable_cameras is required when recording:
# the env carries a wrist-camera sensor. score.py replays a recorded bag, no simulator.
uv run python scripts/eval.py --task FIATLUX-Insert-v0 --policy random --episodes 20 --seed 0
uv run python scripts/record_run.py --task FIATLUX-Insert-v0 --policy random \
    --episodes 2 --record bag --headless --enable_cameras --out logs/runs/random0
uv run python scripts/score.py logs/runs/random0

# Train / roll out PPO (rsl_rl; play.py also exports a TorchScript exported/policy.pt)
uv run python scripts/rsl_rl/train.py --task FIATLUX-Insert-v0
uv run python scripts/rsl_rl/play.py  --task FIATLUX-Insert-v0
```

Lint (CI runs `ruff check source scripts`; pre-commit also runs ruff-format, codespell,
and auto-inserts the BSD-3 license header from `.github/LICENSE_HEADER.txt` into py/yaml files):

```bash
uvx pre-commit run --all-files
```

Ruff: line length 120, custom isort sections (omniverse `omni`/`pxr`/`isaacsim` imports and
each `isaaclab*` extension are separate groups — match existing import blocks).

There are no tests yet; the planned location is `source/fiatlux_task/fiatlux_task/tests/`
(see `journal/specs/repository-structure.md`), with GPU-requiring tests marked `isaacsim_ci`.

## Architecture

Two-layer layout: the root `pyproject.toml` is a uv-managed *environment* (`package = false`)
that pins the stack; `source/fiatlux_task/` is the actual Isaac Lab extension, installed
editable. Scripts in `scripts/` are thin CLI entrypoints; reusable logic belongs in the package.

**Task registration** — `fiatlux_task/__init__.py` imports `fiatlux_task.tasks`, whose
`__init__.py` auto-walks subpackages via `isaaclab_tasks.utils.import_packages`; each leaf
`tasks/manager_based/fiatlux_task/__init__.py` calls `gym.register(...)`. Two rules follow:

- Registrations use **string entry points only** (`"module:ClassName"`), never eager cfg
  imports — the auto-import walk swallows import errors, so an eagerly imported cfg that
  fails (e.g. missing USD) silently drops every registration in that module.
- After touching registration or cfg imports, run `scripts/list_envs.py` and confirm all
  FIATLUX ids are present.

**Two env kinds, two toolchains** — `FIATLUX-Insert-v0` is a full `ManagerBasedRLEnv`
(`g1_bulb_env_cfg.py`: scene + rewards/terminations in `mdp/`, PPO cfg in `agents/`). The
ladder family shares one **non-RL** `ManagerBasedEnv` base (`g1_ladder_env_cfg.py` on top of
`ladder_scene_cfg.py`) with observation/action/event managers only — no rewards, no training;
the per-task cfgs (`carry_env_cfg.py`, etc.) are subclasses holding TODO scaffolds. Task
work later upgrades a scaffold to `ManagerBasedRLEnvCfg` (see `docs/roadmap.md`).

**Shared building blocks** (import from these; don't re-derive):

- `fiatlux_task/assets.py` — single source of truth for USD paths (G1, BEHAVIOR-1K bulb
  `ymomhw` / lamp-as-socket `bbentu` / ladder `shfvtl`, room dressing). Swap asset variants
  here only.
- `fiatlux_task/robots/g1.py` — task-agnostic G1 `ArticulationCfg` + joint/body name
  constants (`G1_ARM_JOINTS`, `G1_EE_BODY`, …). `prim_path` is deliberately `MISSING`;
  scenes supply it via `G1_INSPIRE_CFG.replace(prim_path=...)`. Keep joint names here,
  not scattered in rewards/scripts.
- `fiatlux_task/policy.py` — `make_policy(spec, env)` returns `policy(obs) -> actions`;
  specs: `"zero"`, `"random"`, a TorchScript `"<path>.pt"`, or `"rsl_rl[:<ckpt>]"`. All
  eval/record scripts consume policies through this, so new baselines plug in here.
- `fiatlux_task/recording.py` — the trajectory "bag" (HDF5/npz + `meta.json`) that
  `record_run.py` writes and `score.py` reads. Caveat encoded there: Isaac Lab auto-resets
  finished episodes inside `step`, so success/drop are captured from termination-manager
  flags, not from post-step poses.
- `fiatlux_task/scenes.py` — scene vocabulary shared by the Insert and ladder scenes:
  `DressedSceneCfg` (HDRI `dome_light` + Simple Room backdrop) and `spawn_b1k_single_body`.
  Task scenes subclass the base and add only their furniture; per-scene divergence
  (ground friction, key light, sensors) is intentional and stays in the task cfgs.
- `fiatlux_task/viz.py` — the one video-capture implementation (RTX sensor camera +
  imageio MP4 + poster PNG). `record_orbit` = turntable scene inspection
  (`verify_scene.py --record`); `VideoRecorder` = frames captured inside a rollout loop so
  video and bag describe the same run (`record_run.py`, incl. `--cam orbit`).

**Sim-to-real contracts** (preserve when adding obs/actions): the default `policy`
observation group stays sensor-realizable (proprioception, wrist camera, contact forces);
ground-truth poses go in the separate `privileged` group (critic/scripted baselines only).
Actions are joint-position targets that map onto the Unitree SDK.

**BEHAVIOR-1K socket workaround** — lamps/ladders are multi-body objects; scenes spawn
them as single rigid bodies by deactivating their `meta__*` helper links via
`fiatlux_task/scenes.py: spawn_b1k_single_body` (issue #14).

**docs/ vs journal/** — `docs/` is the contributor-facing manual (task spec, scoring
protocol, roadmap); `journal/` is internal project memory (design specs not yet promoted
to docs, meeting sync notes). Decisions graduate from `journal/specs/` into `docs/`.

## License

Apache-2.0 project; files derived from Isaac Lab are BSD-3 (`LICENSE.isaaclab`). Pre-commit
inserts the header automatically — don't hand-edit copyright years.
