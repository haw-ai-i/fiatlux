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

All seven ids are members of **one task family** backed by **one scene** with preset
layouts; the scaffolds share a non-RL base env (observation/action/event managers only).
The train / eval / record scripts apply to the RL members; `scripts/verify_scene.py`
covers every member.

## One task family: one scene, preset layouts

The family scene (`tasks/.../scene_cfg.py: G1ReplaceSceneCfg`) is built entirely from
the bucket assets: the Inspire-hand G1, the BEHAVIOR-1K bulb and socket-lamp, plus the
primary BEHAVIOR-1K climb ladder (`shfvtl`, wired as `LADDER_USD` in
`fiatlux_task.assets`). A *preset* picks the layout for the task's phase of the
replacement story:

- **tabletop** — `FIATLUX-Insert-v0` (RL): packing table, socket on top, bulb at hand
  height, no ladder; the manipulation bench.
- **workshop and its per-task variants** — the non-RL scaffolds (on
  `base_env_cfg.py: FamilyBaseEnvCfg`) each start in their phase of the replacement story:
  - `Base` — the reference floor layout: ladder at the work spot, socket-lamp + bulb on the floor.
  - `Carry` — ladder *stored* by the room wall, robot beside it; work area across the room.
  - `Climb` — chandelier fixture overhead, robot at the ladder's base.
  - `Descend` — same elevated fixture, robot already at the upper steps.
  - `Remove` — old bulb seated in the fixture (kinematic stand-in for "screwed in"), robot at height.
  - `Install` — empty fixture, fresh bulb in a parts crate at the ladder base, robot at height.

The scene carries the Simple Room + HDRI-sky dressing everywhere, plus (scaffolds only)
a per-env random BEHAVIOR-1K ceiling fixture and per-reset lighting randomization (the
fixtures are the opt-in `./assets/download_assets.sh --scene-dressing` asset group, are
skipped gracefully when absent, and are disabled in training cfgs to keep replicated
physics). To test any member:

```bash
./assets/download_assets.sh    # unchanged -- it already syncs the ladder assets
uv run python scripts/list_envs.py                                        # all 7 FIATLUX ids
uv run python scripts/verify_scene.py --headless                          # FIATLUX-Base-v0 checks
uv run python scripts/verify_scene.py --headless --task FIATLUX-Climb-v0  # any family member
# Insert needs cameras (wrist sensor) + a held base (a policy-less free-base humanoid
# collapsing onto the table is a known-unstable regime; see the unification spec)
uv run python scripts/verify_scene.py --headless --enable_cameras --hold_base --task FIATLUX-Insert-v0
# actually SEE the scene on a headless box: orbiting MP4 -> logs/verify/
uv run python scripts/verify_scene.py --record --hold_base --headless --num_envs 1
```

`verify_scene.py` loads the env, steps it under a zero/default-hold policy, and prints
a PASS/FAIL table (assets present, preset initial state, robot sanity, gravity/settling,
collision coverage, contact/penetration), exiting non-zero on failure.

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
