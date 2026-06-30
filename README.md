# Fiatlux Benchmark

A minimal Isaac Lab benchmark for **humanoid light-bulb replacement**: a Unitree
G1 robot inserts a light bulb into a socket. It is a plain Python / Isaac Lab
extension — no ROS, no distributed harness — so it plugs into the standard
`train` / `play` / `teleop` / `eval` scripts.

> **Status (honest):** the **insertion** subtask (`FIATLUX-Insert-v0`) is the
> functional target. Ladder **climbing** and the combined **replace** task are
> defined in the roadmap but not yet implemented. See [docs/roadmap.md](docs/roadmap.md).

## Task hierarchy

| Env id | Description | Status |
| --- | --- | --- |
| `FIATLUX-Insert-v0` | G1 seats a bulb into a socket (manipulation only) | ✅ functional |
| `FIATLUX-Climb-v0` | G1 climbs a ladder to the fixture | 🚧 roadmap |
| `FIATLUX-Replace-v0` | end-to-end climb + insert | 🚧 roadmap |

## Repository layout

```
fiatlux/
├── source/fiatlux_task/      # the Isaac Lab extension package (the benchmark)
│   └── .../manager_based/fiatlux_task/
│       ├── g1_bulb_env_cfg.py   # G1 + bulb + socket scene & MDP
│       ├── mdp/                 # rewards, events, observations
│       ├── agents/              # rsl_rl PPO config
│       └── __init__.py          # gym.register(...)
├── scripts/                  # zero / random / teleop / list_envs / rsl_rl / eval
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
uv run python scripts/list_envs.py
uv run python scripts/zero_agent.py --task FIATLUX-Insert-v0

# 4. Evaluate (standardized, reproducible):
uv run python scripts/eval.py --task FIATLUX-Insert-v0 --policy random --episodes 20 --seed 0

# 5. Record a run, then score it offline (no simulator needed for scoring).
#    --enable_cameras is required: the env carries a wrist-camera sensor.
uv run python scripts/record_run.py --task FIATLUX-Insert-v0 --policy random \
    --episodes 2 --record bag --headless --enable_cameras --out logs/runs/random0
uv run python scripts/score.py logs/runs/random0

# 6. Train a policy:
uv run python scripts/rsl_rl/train.py --task FIATLUX-Insert-v0
```

## Sim-to-real

The benchmark stays pure-Python; a physical-G1 deployment adapter is intentionally
kept separate (future work). To keep that path cheap, the env uses a
**hardware-realizable action space** (joint-position targets) and a default
**sensor-realizable observation group**, with ground-truth ("cheat") observations
isolated in a separate `privileged` group. See [docs/roadmap.md](docs/roadmap.md).

## License

Apache-2.0 (see `LICENSE`). Files derived from Isaac Lab are BSD-3 (`LICENSE.isaaclab`).
