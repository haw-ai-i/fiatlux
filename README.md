# Fiatlux Benchmark

A minimal Isaac Lab benchmark for **humanoid light-bulb replacement**: a Unitree
G1 robot inserts a light bulb into a socket. It is a plain Python / Isaac Lab
extension — no ROS, no distributed harness — so it plugs into the standard
`train` / `play` / `teleop` / `eval` scripts.

> **Status (honest):** the **full replacement** (`FIATLUX-Replace-v0`) is the scored
> benchmark task — randomized room layout, normalized-progress scoring,
> standard/cheatcode observation modes. The benchmark is the full-length
> `FIATLUX-Replace-v0` plus the twelve subtasks it decomposes into. The bulb/socket retention mechanic is implemented (`mdp.bulb_attachment`),
> so Replace's removal and disposal score channels are now achievable. See
> [docs/roadmap.md](docs/roadmap.md).

## Task hierarchy

| Env id | Description | Status |
| --- | --- | --- |
| `FIATLUX-Replace-v0` | **the benchmark**: full replacement — fetch the ladder, swap the bulb, dispose of the old one (randomized room) | ✅ functional |
| `FIATLUX-S01-MoveLadder-v0` | carry the ladder to the fixture and stand it up | ✅ functional |
| `FIATLUX-S02-ClimbLadder-v0` | climb to working height, hands free | ✅ functional |
| `FIATLUX-S03-RemoveOldBulb-v0` | free the old bulb from the fixture | ✅ functional |
| `FIATLUX-S04-DescendWithBulb-v0` | carry the old bulb down the ladder | ✅ functional |
| `FIATLUX-S05-CarryBulbToDisposal-v0` | carry it to the disposal crate | ✅ functional |
| `FIATLUX-S06-DisposeBulb-v0` | put it in the crate and let go | ✅ functional |
| `FIATLUX-S07-ApproachNewBulb-v0` | walk to the fresh bulb on the bench | ✅ functional |
| `FIATLUX-S08-GrabNewBulb-v0` | pick it up without crushing it | ✅ functional |
| `FIATLUX-S09-CarryBulbToLadder-v0` | carry it back to the ladder | ✅ functional |
| `FIATLUX-S10-ClimbWithBulb-v0` | climb one-handed holding the bulb | ✅ functional |
| `FIATLUX-S11-ScrewInBulb-v0` | seat the fresh bulb in the fixture | ✅ functional |
| `FIATLUX-S12-ClimbDown-v0` | come back down, bulb still seated | ✅ functional |

The benchmark is **one full-length task and the twelve subtasks it decomposes into**, backed
by one scene with preset layouts. Each subtask starts from its predecessor's end state, so a
policy can be trained and scored on any leg independently; `FIATLUX-Replace-v0` runs the whole
chain. Each subtask also has a `-Training-v0` variant (replicated physics, shaping rewards) and
a `-Teleop-v0` variant for operator recording.

The train / eval / record scripts apply to every id; `scripts/verify_scene.py` covers them all.

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
├── assets/                   # download_assets.sh (pulls USDs from the HF dataset; git-ignored)
└── docs/                     # overview, getting_started, task_spec, scoring, roadmap
```

## Quick start

See [docs/getting_started.md](docs/getting_started.md) for the full setup.

Requires [uv](https://docs.astral.sh/uv/) (also used to fetch the assets via `uvx`, from the
haw-ai-i/fiatlux-assets HF dataset) and an NVIDIA GPU with a CUDA 12.8-capable driver.

```bash
# 1. Build the full environment (Isaac Sim 5.1 + Isaac Lab 2.3.2 + this package).
#    Everything is pinned in uv.lock -- no manual Isaac Lab install needed.
#    First run pulls ~10 GB; if a big CUDA wheel stalls: UV_HTTP_TIMEOUT=1200 uv sync
uv sync

# 2. Pull the USD assets (G1, bulb/socket, ladder) from the HF dataset:
./assets/download_assets.sh

# 3. Sanity-check registration and launch a baseline:
uv run python scripts/list_envs.py                                        # all 7 FIATLUX ids
uv run python scripts/verify_scene.py --headless --task FIATLUX-Replace-v0   # scene checks
uv run python scripts/verify_scene.py --headless --task FIATLUX-S02-ClimbLadder-v0  # any member

# 4. Evaluate (standardized, reproducible):
uv run python scripts/eval.py --task FIATLUX-S08-GrabNewBulb-v0 --policy random --episodes 20 --seed 0

# 5. Record a run, then score it offline (no simulator needed for scoring).
#    --enable_cameras is required: the env carries a wrist-camera sensor.
uv run python scripts/record_run.py --task FIATLUX-S08-GrabNewBulb-v0 --policy random \
    --episodes 2 --record bag --headless --enable_cameras --out logs/runs/random0
uv run python scripts/score.py logs/runs/random0

# 6. Train a policy:
uv run python scripts/rsl_rl/train.py --task FIATLUX-S08-GrabNewBulb-Training-v0
uv run python scripts/rsl_rl/train.py --task FIATLUX-S02-ClimbLadder-Training-v0

# 7. Run the GR00T N1.7 model baseline on the benchmark (needs the external
#    PolicyServer -- setup in journal/specs/groot-sonic-baseline.md):
uv sync --extra groot
scripts/groot/serve.sh &   # terminal 1: the VLA server (own venv, HF token required)
uv run python scripts/eval.py --task FIATLUX-Replace-v0 --policy groot \
    --episodes 20 --seed 0 --enable_cameras

# 8. Teleoperate the tasks (whole-body: SONIC walking + bimanual arms; keyboard or
#    Pico VR) and record scored demo sessions -- guide: source/fiatlux_teleop/README.md
./scripts/teleop/setup_sim_teleop.sh            # one-command setup (keyboard tier; `vr` adds CloudXR)
PYTHONPATH=source/fiatlux_task:source/fiatlux_teleop \
uv run --extra teleop python scripts/teleop/sonic_teleop.py \
    --task FIATLUX-Carry-Teleop-v0 --input keyboard
```

## Sim-to-real

The benchmark stays pure-Python; a physical-G1 deployment adapter is intentionally
kept separate (future work). To keep that path cheap, the env uses a
**hardware-realizable action space** (joint-position targets) and a default
**sensor-realizable observation group**, with ground-truth ("cheat") observations
isolated in a separate `privileged` group. See [docs/roadmap.md](docs/roadmap.md).

## License

Apache-2.0 (see `LICENSE`). Files derived from Isaac Lab are BSD-3 (`LICENSE.isaaclab`).
