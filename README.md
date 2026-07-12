# Fiatlux Benchmark

A minimal Isaac Lab benchmark for **humanoid light-bulb replacement**: a Unitree
G1 robot inserts a light bulb into a socket. It is a plain Python / Isaac Lab
extension — no ROS, no distributed harness — so it plugs into the standard
`train` / `play` / `teleop` / `eval` scripts.

> **Status (honest):** the **full replacement** (`FIATLUX-Replace-v0`) is the scored
> benchmark task — randomized room layout, normalized-progress scoring,
> standard/cheatcode observation modes. The **insertion** (`FIATLUX-Insert-v0`) and
> **climbing** (`FIATLUX-Climb-v0`) subtasks are functional RL tasks kept as
> development aids; the rest of the family exists as loadable non-RL scene
> **scaffolds**. The old bulb's unscrew mechanic is still a kinematic stand-in, so
> Replace's removal/disposal score channels are wired but not yet achievable. See
> [docs/roadmap.md](docs/roadmap.md).

## Task hierarchy

| Env id | Description | Status |
| --- | --- | --- |
| `FIATLUX-Replace-v0` | **the benchmark**: full replacement — insert fresh bulb, remove old bulb, dispose of it (randomized room) | ✅ functional |
| `FIATLUX-Insert-v0` | G1 seats a bulb into a socket (tabletop manipulation) | ✅ functional |
| `FIATLUX-Climb-v0` | G1 climbs the step ladder to the fixture height (whole-body RL) | ✅ functional |
| `FIATLUX-Base-v0` | shared G1 + ladder + lamp + bulb scene, no task logic | 🧱 scaffold (non-RL) |
| `FIATLUX-Carry-v0` | G1 walks to a ladder, grasps it, and carries it upright to a target (whole-body RL) | ✅ functional |
| `FIATLUX-Descend-v0` | bipedal ladder descent | 🧱 scaffold (non-RL) |
| `FIATLUX-Remove-v0` | unscrew / remove the seated bulb | 🧱 scaffold (non-RL) |
| `FIATLUX-Install-v0` | seat a new bulb at the fixture (the at-fixture counterpart of `Insert`) | 🧱 scaffold (non-RL) |

All seven ids are members of **one task family** backed by **one scene** with preset
layouts; the scaffolds share a non-RL base env (observation/action/event managers only).
The train / eval / record scripts apply to the RL members; `scripts/verify_scene.py`
covers every member.

## Repository layout

```
fiatlux/
├── source/fiatlux_task/      # the Isaac Lab extension package (the benchmark)
│   ├── .../fiatlux_task/scenes.py   # shared scene vocabulary (room dressing + B1K spawner)
│   ├── .../fiatlux_task/viz.py      # shared video capture (orbit / rollout MP4s + posters)
│   └── .../manager_based/fiatlux_task/
│       ├── scene_cfg.py         # THE family scene + tabletop/workshop presets
│       ├── base_env_cfg.py      # shared non-RL base env (managers only)
│       ├── g1_bulb_env_cfg.py   # Insert task MDP (RL, tabletop preset)
│       ├── climb_env_cfg.py     # Climb task MDP (RL, at-height preset)
│       ├── carry_env_cfg.py     # Carry task MDP (RL, ladder-positioning preset)
│       ├── *_env_cfg.py         # descend / remove / install scaffolds
│       ├── mdp/                 # rewards, events, observations
│       ├── agents/              # rsl_rl PPO config
│       └── __init__.py          # gym.register(...) x7
├── scripts/                  # zero / random / teleop / list_envs / rsl_rl / eval / verify_scene
├── assets/                   # download_assets.sh (pulls USDs from GCS; git-ignored)
└── docs/                     # overview, getting_started, task_spec, scoring, roadmap
```

## Quick start

See [docs/getting_started.md](docs/getting_started.md) for the full setup.

Requires [uv](https://docs.astral.sh/uv/), an NVIDIA GPU with a CUDA 12.8-capable
driver, and `gsutil` (Google Cloud SDK) for the assets.

```bash
# 1. Build the full environment (Isaac Sim 5.1 + Isaac Lab 2.3.2 + this package).
#    Everything is pinned in uv.lock -- no manual Isaac Lab install needed.
#    First run pulls ~10 GB; if a big CUDA wheel stalls: UV_HTTP_TIMEOUT=1200 uv sync
uv sync

# 2. Pull the USD assets (G1, bulb/socket, ladder) from the bucket:
./assets/download_assets.sh

# 3. Sanity-check registration and launch a baseline:
uv run python scripts/list_envs.py                                        # all 7 FIATLUX ids
uv run python scripts/verify_scene.py --headless                          # FIATLUX-Base-v0 checks
uv run python scripts/verify_scene.py --headless --task FIATLUX-Climb-v0  # any family member

# 4. Evaluate (standardized, reproducible):
uv run python scripts/eval.py --task FIATLUX-Insert-v0 --policy random --episodes 20 --seed 0

# 5. Record a run, then score it offline (no simulator needed for scoring).
#    --enable_cameras is required: the env carries a wrist-camera sensor.
uv run python scripts/record_run.py --task FIATLUX-Insert-v0 --policy random \
    --episodes 2 --record bag --headless --enable_cameras --out logs/runs/random0
uv run python scripts/score.py logs/runs/random0

# 6. Train a policy:
uv run python scripts/rsl_rl/train.py --task FIATLUX-Insert-v0
uv run python scripts/rsl_rl/train.py --task FIATLUX-Climb-v0
```

## Sim-to-real

The benchmark stays pure-Python; a physical-G1 deployment adapter is intentionally
kept separate (future work). To keep that path cheap, the env uses a
**hardware-realizable action space** (joint-position targets) and a default
**sensor-realizable observation group**, with ground-truth ("cheat") observations
isolated in a separate `privileged` group. See [docs/roadmap.md](docs/roadmap.md).

## License

Apache-2.0 (see `LICENSE`). Files derived from Isaac Lab are BSD-3 (`LICENSE.isaaclab`).
