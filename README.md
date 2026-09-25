# Fiatlux Benchmark

A minimal Isaac Lab benchmark for **humanoid ladder climbing and light-bulb replacement**: a
Unitree G1 robot positions a step ladder under a ceiling or wall fixture, climbs it, exchanges
the spent bulb for a fresh one, and drops the spent one in a disposal crate. It is a plain
Python / Isaac Lab extension — no ROS, no distributed harness — so it plugs into the standard
`train` / `play` / `teleop` / `eval` scripts.

> **Status (honest):** the **full replacement** (`FIATLUX-Replace-v0`) is the scored
> benchmark task — randomized room layout, normalized-progress scoring,
> standard/cheatcode observation modes. The benchmark is the full-length
> `FIATLUX-Replace-v0` plus the twelve subtasks it decomposes into. The bulb/socket retention mechanic is implemented (`mdp.bulb_attachment`),
> so Replace's removal and disposal score channels are now achievable. See
> [docs/roadmap.md](docs/roadmap.md).

## Task hierarchy

**The twelve subtasks** — the chain, in order. Each id below exists three times: `-v0` (RL),
`-Training-v0` (the training tier), and `-Teleop-v0` (the human-demo twin).

| | Task | | | Task |
| --- | --- | --- | --- | --- |
| S01 | `MoveLadder` — carry the ladder under the fixture | | S07 | `ApproachNewBulb` — walk to the bench |
| S02 | `ClimbLadder` — ascend to the work stance | | S08 | `GrabNewBulb` — pick the fresh bulb up |
| S03 | `RemoveOldBulb` — unseat it at height | | S09 | `CarryBulbToLadder` — walk it back |
| S04 | `DescendWithBulb` — climb down holding it | | S10 | `ClimbWithBulb` — ascend holding it |
| S05 | `CarryBulbToDisposal` — walk it to the crate | | S11 | `ScrewInBulb` — seat it and let go |
| S06 | `DisposeBulb` — drop it in and release | | S12 | `ClimbDown` — descend, bulb left seated |

**The flat task and the coarse envs.** All but `Base-v0` are functional RL envs with their own
rewards and terminations; `Base-v0` registers the non-RL `ManagerBasedEnv` and exists to load the
scene.

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

The eval / record scripts and `scripts/verify_scene.py` apply to every id; `scripts/rsl_rl/`
train / play need a PPO config, which only `FIATLUX-Replace-v0` has so far.

## Repository layout

```
fiatlux/
├── source/fiatlux_task/      # the Isaac Lab extension package (the benchmark)
│   ├── .../fiatlux_task/assets.py   # every USD path the env loads
│   ├── .../fiatlux_task/sensors.py  # wrist / ego cameras + mid360 lidar cfgs
│   ├── .../fiatlux_task/scenes.py   # shared scene vocabulary (the room backdrop + sky dome)
│   ├── .../fiatlux_task/viz.py      # shared video capture (orbit / rollout MP4s + posters)
│   ├── .../fiatlux_task/subtask_score.py  # the subtask headline score
│   └── .../manager_based/fiatlux_task/
│       ├── scene_cfg.py         # THE family scene + tabletop/position/replace presets
│       ├── replace_env_cfg.py   # FIATLUX-Replace-v0 MDP (the benchmark task)
│       ├── subtask_env_cfg.py   # shared recipe for the twelve subtasks
│       ├── subtasks/            # one thin cfg file per subtask
│       ├── mdp/                 # rewards, events, observations
│       ├── agents/              # rsl_rl PPO config
│       └── __init__.py          # gym.register() for Replace-v0 + the twelve subtasks
├── source/fiatlux_teleop/    # teleop twins (gym.register(...) x16) + recording, behind the
│                             #   `teleop` extra -- not a core benchmark dependency
├── scripts/                  # zero / random / eval / record_run / score / score_subtasks /
│                             #   list_envs / rsl_rl / verify_* / teleop / omniverse
├── assets/                   # download_assets.sh (pulls USDs from the HF dataset; git-ignored)
└── docs/                     # overview, getting_started, task_spec, subtask_teleop, scoring,
                              #   roadmap, asset + collision provenance
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
uv run python scripts/list_envs.py                                        # every registered FIATLUX id
uv run python scripts/verify_scene.py --headless --enable_cameras --task FIATLUX-Replace-v0   # scene checks
uv run python scripts/verify_scene.py --headless --enable_cameras --task FIATLUX-S02-ClimbLadder-v0  # any member

# 4. Evaluate (standardized, reproducible):
uv run python scripts/eval.py --task FIATLUX-S08-GrabNewBulb-v0 --policy random --episodes 20 --seed 0

# 5. Record a run, then score it offline (no simulator needed for scoring).
#    --enable_cameras is required: the env carries an ego-camera sensor.
uv run python scripts/record_run.py --task FIATLUX-S08-GrabNewBulb-v0 --policy random \
    --episodes 2 --record bag --headless --enable_cameras --out logs/runs/random0
uv run python scripts/score.py logs/runs/random0

# 6. Train a policy (only FIATLUX-Replace-v0 ships an rsl_rl PPO config so far):
uv run python scripts/rsl_rl/train.py --task FIATLUX-Replace-v0

# 7. Run the GR00T N1.7 model baseline on the benchmark (needs the external
#    PolicyServer -- setup and required env vars are in scripts/groot/serve.sh's header):
uv sync --extra groot
scripts/groot/serve.sh &   # terminal 1: the VLA server (own venv, HF token required)
uv run python scripts/eval.py --task FIATLUX-Replace-v0 --policy groot \
    --episodes 20 --seed 0 --enable_cameras

# 8. Teleoperate the tasks (whole-body: SONIC walking + bimanual arms; keyboard or
#    Pico VR) and record scored demo sessions -- guide: source/fiatlux_teleop/README.md
./scripts/teleop/setup_sim_teleop.sh            # one-command setup (keyboard tier; `vr` adds CloudXR)
PYTHONPATH=source/fiatlux_task:source/fiatlux_teleop \
uv run --extra teleop python scripts/teleop/sonic_teleop.py \
    --task FIATLUX-S07-ApproachNewBulb-Teleop-v0 --input keyboard
```

## Sim-to-real

The benchmark stays pure-Python; a deployment adapter that bridges its joint-position
targets to Unitree SDK commands on a physical G1 is future work. To keep that path cheap,
the env uses a **hardware-realizable action space** (joint-position targets) and a default
**sensor-realizable observation group**, with ground-truth observations isolated in a
separate `privileged` group. See [docs/roadmap.md](docs/roadmap.md).

## Paper

Fiatlux is described in *Fiatlux: A Long-Horizon Benchmark for Humanoid Ladder Climbing and
Light-Bulb Replacement* (Pavel Bushuyeu, Yujin Chen, Anton Nikolaev, Brian Shu, Igor Molybog;
the first four authors contributed equally, order alphabetical by surname). The code and the
teleoperated recordings used to specify and check the success gates are at
[fiatlux.github.io](https://fiatlux.github.io).

```bibtex
@misc{fiatlux,
  title  = {Fiatlux: A Long-Horizon Benchmark for Humanoid Ladder Climbing and Light-Bulb Replacement},
  author = {Bushuyeu, Pavel and Chen, Yujin and Nikolaev, Anton and Shu, Brian and Molybog, Igor},
  year   = {2026},
  note   = {The first four authors contributed equally; author order is alphabetical by surname.},
  url    = {https://fiatlux.github.io},
}
```

## License

Apache-2.0 (see `LICENSE`). Files derived from Isaac Lab are BSD-3 (`LICENSE.isaaclab`).
