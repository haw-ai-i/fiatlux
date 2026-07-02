# Fiatlux Benchmark

A minimal Isaac Lab benchmark for **humanoid light-bulb replacement**: a Unitree
G1 robot inserts a light bulb into a socket. It is a plain Python / Isaac Lab
extension — no ROS, no distributed harness — so it plugs into the standard
`train` / `play` / `teleop` / `eval` scripts.

> **Status (honest):** the **insertion** subtask (`FIATLUX-Insert-v0`) is the
> functional target. The **ladder task family** (base scene + carry / climb /
> descend / remove / install) exists as loadable non-RL scene **scaffolds** — no
> rewards or training yet. The combined **replace** task is still roadmap.
> See [docs/roadmap.md](docs/roadmap.md).

## Task hierarchy

| Env id | Description | Status |
| --- | --- | --- |
| `FIATLUX-Insert-v0` | G1 seats a bulb into a socket (tabletop manipulation) | ✅ functional |
| `FIATLUX-Base-v0` | shared G1 + ladder + lamp + bulb scene, no task logic | 🧱 scaffold (non-RL) |
| `FIATLUX-Carry-v0` | grab and position the ladder | 🧱 scaffold (non-RL) |
| `FIATLUX-Climb-v0` | G1 climbs the ladder to the fixture | 🧱 scaffold (non-RL) |
| `FIATLUX-Descend-v0` | bipedal ladder descent | 🧱 scaffold (non-RL) |
| `FIATLUX-Remove-v0` | unscrew / remove the seated bulb | 🧱 scaffold (non-RL) |
| `FIATLUX-Install-v0` | seat a new bulb at the fixture (the at-fixture counterpart of `Insert`) | 🧱 scaffold (non-RL) |
| `FIATLUX-Replace-v0` | end-to-end climb + insert | 🚧 roadmap |

Scaffolds share one **non-RL** base env (observation/action/event managers only) and
are exercised with `scripts/verify_scene.py` — **not** the train / eval / teleop
scripts, which assume RL envs.

## Ladder task family (new)

The six scaffold envs live in `source/.../tasks/manager_based/fiatlux_task/` (alongside
the Insert task) and share one scene built entirely from the bucket assets: the same
Inspire-hand G1 and BEHAVIOR-1K bulb/lamp the insertion task uses, plus the primary
BEHAVIOR-1K climb ladder (`shfvtl`, wired as `LADDER_USD` in `fiatlux_task.assets`).
The scene carries the same Simple Room + HDRI-sky dressing as the insertion task, plus
a per-env random BEHAVIOR-1K ceiling fixture and per-reset lighting randomization (the
fixtures are the opt-in `./assets/download_assets.sh --scene-dressing` asset group and
are skipped gracefully when absent). To test them:

```bash
./assets/download_assets.sh    # unchanged -- it already syncs the ladder assets
uv run python scripts/list_envs.py                                        # all 7 FIATLUX ids
uv run python scripts/verify_scene.py --headless                          # FIATLUX-Base-v0 checks
uv run python scripts/verify_scene.py --headless --task FIATLUX-Climb-v0  # any family member
# actually SEE the scene on a headless box: orbiting MP4 -> logs/verify/
uv run python scripts/verify_scene.py --record --hold_base --headless --num_envs 1
```

`verify_scene.py` loads the env, steps it under a zero/default-hold policy, and prints
a PASS/FAIL table (assets present, robot sanity, gravity/settling, collision coverage,
contact/penetration), exiting non-zero on failure.

## Repository layout

```
fiatlux/
├── source/fiatlux_task/      # the Isaac Lab extension package (the benchmark)
│   ├── .../fiatlux_task/scenes.py   # shared scene vocabulary (room dressing + B1K spawner)
│   ├── .../fiatlux_task/viz.py      # shared video capture (orbit / rollout MP4s + posters)
│   └── .../manager_based/fiatlux_task/
│       ├── g1_bulb_env_cfg.py   # G1 + bulb + socket scene & MDP (Insert)
│       ├── ladder_scene_cfg.py  # G1 + ladder + lamp + bulb scene (bucket USDs)
│       ├── g1_ladder_env_cfg.py # shared non-RL ladder base env (managers only)
│       ├── *_env_cfg.py         # carry / climb / descend / remove / install scaffolds
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
uv run python scripts/list_envs.py
uv run python scripts/zero_agent.py --task FIATLUX-Insert-v0
uv run python scripts/verify_scene.py --headless   # ladder-family scene checks

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
