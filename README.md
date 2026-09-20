# Fiatlux Benchmark

A minimal Isaac Lab benchmark for **humanoid ladder climbing and light-bulb replacement**: a
Unitree G1 robot positions a step ladder under a ceiling or wall fixture, climbs it, exchanges
the spent bulb for a fresh one, and drops the spent one in a disposal crate. It is a plain
Python / Isaac Lab extension — no ROS, no distributed harness — so it plugs into the standard
`train` / `play` / `teleop` / `eval` scripts.

> **Status (honest):** the benchmark has two framings of the same job. **`FIATLUX-Replace-v0`**
> is the full replacement as one flat scored episode; the **twelve subtasks**
> (`FIATLUX-S01-MoveLadder-v0` … `FIATLUX-S12-ClimbDown-v0`) are the same chain cut into legs,
> each with its own success gate, its own score (`docs/scoring.md`), a `-Training-v0` RL tier
> and a `-Teleop-v0` twin for human demonstration. The older coarse envs (`Insert`, `Climb`,
> `Carry`) are functional development aids, not benchmark targets, as are `Descend`, `Remove`
> and `Install` — each now covered by a subtask. `Base` is the non-RL scene-only env. The
> bulb/socket retention mechanic is implemented (`mdp.bulb_attachment`), so Replace's removal
> and disposal score channels are achievable. See [docs/roadmap.md](docs/roadmap.md).

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
| `FIATLUX-Replace-v0` | **the flat benchmark**: the whole chain in one episode (randomized room) | ✅ RL |
| `FIATLUX-Insert-v0` | G1 seats a bulb into a socket (tabletop manipulation) | ✅ RL · has a teleop twin |
| `FIATLUX-Climb-v0` | G1 climbs the step ladder to the fixture height (whole-body RL) | ✅ RL |
| `FIATLUX-Carry-v0` | G1 walks to a ladder, grasps it, and carries it upright to a target | ✅ RL · has a teleop twin |
| `FIATLUX-Descend-v0` | bipedal ladder descent — Climb's reward scheme mirrored | ✅ RL · superseded by S04/S12 |
| `FIATLUX-Remove-v0` | remove the seated bulb from the bench-lamp socket (`Insert`'s bench, not the elevated fixture) | ✅ RL · superseded by S03 |
| `FIATLUX-Install-v0` | seat a new bulb into that same empty bench-lamp socket | ✅ RL · superseded by S11 |
| `FIATLUX-Base-v0` | shared scene, no task logic — `verify_scene.py`'s default target | 🧱 non-RL |

`Replace-v0` and `Climb-v0` are also **load-bearing for the subtasks**, which import their
thresholds (`FRESH_BULB_DROP_HEIGHT`, `REMOVAL_CLEARANCE`, `SEAT_*_THRESHOLD`,
`bulb_attachment_event`; `FALL_MIN_HEIGHT` / `FALL_TILT_LIMIT`) rather than redeclaring them.
`Descend`, `Remove` and `Install` have no such dependants and no teleop twin — a subtask covers
each of their jobs now, so they are candidates for retirement.

Every id is a member of **one task family** backed by **one scene** with preset layouts; the
all share one non-RL base env (observation/action/event managers only). `scripts/list_envs.py`
prints the live list — 32 ids from `fiatlux_task` plus 16 teleop ids from `fiatlux_teleop`. The
train / eval / record scripts apply to the RL members; `scripts/verify_scene.py` covers every
member.

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
│       ├── scene_cfg.py         # THE family scene + tabletop/workshop/replace presets
│       ├── base_env_cfg.py      # shared non-RL base env (managers only)
│       ├── subtask_env_cfg.py   # shared subtask MDP (obs/actions/events/gates)
│       ├── subtask_tiers/       # behaviour shared by a group of legs (balance, carry, grasp…)
│       ├── subtasks/            # s01_…_env_cfg.py … s12_…_env_cfg.py + the training tier
│       ├── replace_env_cfg.py   # the flat benchmark task
│       ├── g1_bulb_env_cfg.py   # Insert task MDP (RL, tabletop preset)
│       ├── climb_env_cfg.py / carry_env_cfg.py   # Climb / Carry MDPs (RL)
│       ├── mdp/                 # rewards, events, observations, the attach state machine
│       ├── agents/              # rsl_rl PPO config
│       └── __init__.py          # gym.register(...) x32
├── source/fiatlux_teleop/    # teleop twins (gym.register(...) x16) + recording, behind the
│                             #   `teleop` extra -- not a core benchmark dependency
├── scripts/                  # zero / random / eval / record_run / score / score_subtasks /
│                             #   list_envs / rsl_rl / verify_* / teleop / omniverse
├── assets/                   # download_assets.sh (pulls USDs from GCS; git-ignored)
├── journal/                  # design specs and references (not user docs)
└── docs/                     # overview, getting_started, task_spec, subtask_teleop, scoring,
                              #   roadmap, asset + collision provenance
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

# 3. Sanity-check registration and launch a baseline. Every task carries a camera sensor,
#    so --enable_cameras is required even here (headless verification, no video):
uv run python scripts/list_envs.py                                                        # every registered FIATLUX id
uv run python scripts/verify_scene.py --headless --enable_cameras                         # FIATLUX-Base-v0 checks
uv run python scripts/verify_scene.py --headless --enable_cameras --task FIATLUX-Climb-v0  # any family member

# 4. Evaluate (standardized, reproducible):
uv run python scripts/eval.py --task FIATLUX-Insert-v0 --policy random --episodes 20 --seed 0 \
    --enable_cameras

# 5. Record a run, then score it offline (no simulator needed for scoring).
#    --enable_cameras is required: the env carries a wrist-camera sensor.
uv run python scripts/record_run.py --task FIATLUX-Insert-v0 --policy random \
    --episodes 2 --record bag --headless --enable_cameras --out logs/runs/random0
uv run python scripts/score.py logs/runs/random0

# 6. Train a policy:
uv run python scripts/rsl_rl/train.py --task FIATLUX-Insert-v0 --enable_cameras
uv run python scripts/rsl_rl/train.py --task FIATLUX-Climb-v0 --enable_cameras

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
    --task FIATLUX-Carry-Teleop-v0 --input keyboard
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
