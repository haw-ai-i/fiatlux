# Getting Started

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (the only thing you install by hand).
- An NVIDIA GPU with a driver new enough for CUDA 12.8 (the pinned torch build).
- `gsutil` (Google Cloud SDK), authenticated, for the assets in step 2.

## 1. Build the environment

The repo is a self-contained `uv` project: the entire stack — Isaac Sim 5.1, Isaac
Lab 2.3.2, PyTorch (cu128), and the `fiatlux_task` package — is pinned in `uv.lock`.
There is **no manual Isaac Lab install**; from the repo root just run:

```bash
uv sync
```

This downloads Isaac Sim and the Isaac Lab wheels (~10 GB on first run) from the
NVIDIA package index and installs `source/fiatlux_task` in editable mode. Prefix the
run commands below with `uv run` to use this environment.

> Some of the CUDA wheels are multiple GB; if `uv sync` aborts with a network
> timeout, raise uv's per-download limit: `UV_HTTP_TIMEOUT=1200 uv sync` (it resumes
> from whatever already downloaded).

## 2. Download the assets

The G1, bulb/socket, and ladder USDs are pulled from a GCS bucket (they are not
checked into git). This also pulls the table/warehouse/HDRI sky that dress up
the scene by default (see `assets/README.md`):

```bash
./assets/download_assets.sh
# override the bucket if needed:
FIATLUX_ASSET_BUCKET=gs://my-bucket/assets ./assets/download_assets.sh
```

This produces:

```
assets/unitree_g1/wholebody_inspire/...usd      # the robot (G1_USD); _dex3 variant alongside
assets/omniverse_bulb/LightBulb_bulb_z_rigid.usda           # the graspable bulb
assets/omniverse_bulb/LightBulb_socket_z_static_sleeve.usda # the socket it seats into
assets/omniverse_ladder/AlumStep_D/...collision.usd         # the ladder the robot climbs
assets/isaac_packing_table/, assets/isaac_room/, assets/isaac_skies/
```

(See `source/fiatlux_task/fiatlux_task/assets.py` for the exact paths the env
loads -- swap the G1 hand variant there if needed.)

If your assets live elsewhere, point the env at them with
`export FIATLUX_ASSETS_DIR=/path/to/assets`.

## 3. Run

Every FIATLUX task carries a camera sensor (wrist and/or head-mounted), so `--enable_cameras`
is required to build any of them, even without `--headless` or video recording:

```bash
uv run python scripts/list_envs.py                                                # list registered tasks
uv run python scripts/zero_agent.py --task FIATLUX-Insert-v0 --enable_cameras    # launch the scene
uv run python scripts/eval.py --task FIATLUX-Insert-v0 --policy random --episodes 20 --enable_cameras
uv run python scripts/rsl_rl/train.py --task FIATLUX-Insert-v0 --enable_cameras  # train PPO
uv run python scripts/rsl_rl/play.py  --task FIATLUX-Insert-v0 --enable_cameras  # roll out a checkpoint
```

## 4. Record and score a run

A single rollout can be recorded to a trajectory "bag" and scored offline, without
re-running the simulator:

`--enable_cameras` is required because the env carries a wrist-camera sensor;
scoring reads the recorded bag and needs no simulator.

```bash
uv run python scripts/record_run.py --task FIATLUX-Insert-v0 --policy random \
    --episodes 2 --record bag --headless --enable_cameras --out logs/runs/random0
uv run python scripts/score.py logs/runs/random0
```

Pass `--record video` (or `--record both` to get the bag too) to also render an MP4
of the rollout to `<out>/video/`:

```bash
uv run python scripts/record_run.py --task FIATLUX-Insert-v0 --policy random \
    --episodes 2 --record both --headless --enable_cameras --out logs/runs/random0
```

This writes `logs/runs/random0/video/run.mp4` plus a `run_poster.png` still (PNGs
preview inline in most editors). `--cam` picks the camera: the fixed `third_person`
(default) / `closeup` / `hand` viewpoints, `orbit` for a 360° turntable of the scene — the
same scene-inspection view `verify_scene.py --record` produces for the ladder
family — `fixture`, a low orbit that looks **up** at the mounted socket, or `ego`
for the robot's own head camera. Use `fixture` (also `verify_scene.py --record
--record_view fixture`) whenever an overhead mount is in question: every other
camera here points at the floor and the bench, and none of them can see a fixture
at 2.37 m (ceiling, `CEILING_FIXTURE_Z`) or 2.2 m (wall, `WALL_MOUNT_Z`) at all. On a headless/remote box there's no display to preview the MP4 on, so pull
the file to your machine (e.g. `scp` or a remote-file-browsing editor) and play it
locally.

## 5. Verify the physics

Two verifiers guard the benchmark's physical modeling. Run them after asset or env
changes and as a pre-flight before scoring runs — they are deliberately **not** wired
into per-commit CI (a full Isaac Sim launch per check is too costly there):

```bash
# the scene is solid: assets present, colliders exist, nothing explodes or sinks
uv run python scripts/verify_scene.py --headless --enable_cameras --task FIATLUX-Base-v0

# the graded interactions are modeled correctly: seated bulb rests stably, the
# success pose is attainable, a hand press stays gentle and doesn't launch the
# bulb, break/drop detection fires exactly when it should, the robot can lean
# on the ladder -- all policy-free, driven by calibrated poses (fiatlux_task.poses)
uv run python scripts/verify_interactions.py --headless --scenario all
```

Both exit non-zero on FAIL. `verify_interactions.py --scenario <name> --probe`
prints poses, contact forces and asset bboxes for recalibrating
`fiatlux_task/poses.py`; `--video out.mp4` renders the scenario for visual
inspection; `--record-bag <dir>` writes the fragility episodes as standard
trajectory bags (scoreable with `scripts/score.py`).

## 6. Run the unit tests

The offline tests need no simulator and no GPU — they run in about two seconds:

```bash
uv run pytest                          # 82 pass, 2 skipped (those two need Kit)
uv run pytest -m isaacsim_ci           # just the two; run this one under Isaac Sim
```

A bare `pytest` is green anywhere: the two tests that need a bootstrapped Kit runtime skip
themselves rather than fail, and say so.

`pytest` collects only `source/fiatlux_task/fiatlux_task/tests/`. That is deliberate: some env
cfgs are named after their task rather than their role (`test_lightbulb_mechanism_env_cfg.py` is
the cfg for `FIATLUX-TestLightbulbMechanism-Teleop-v0`), and importing one during collection
aborts the whole run.

## Troubleshooting

- **Env not found / empty list** — `uv sync` did not complete, or
  `import fiatlux_task.tasks` failed.
- **USD not found** — run `./assets/download_assets.sh` or set `FIATLUX_ASSETS_DIR`.
- **Joint/body name errors** — the constants in `fiatlux_task/robots/g1.py`
  (`G1_ARM_JOINTS`, `G1_EE_BODY`, the palm/foot body lists) must match the joints/links in
  your G1 USD.
- **Hangs at `Do you accept the EULA? (Yes/No):`** — first launch of Isaac Sim's Kit
  runtime prompts interactively and there's no stdin in a non-interactive/background
  shell. Set `OMNI_KIT_ACCEPT_EULA=YES` in the environment (accepts once, writes an
  `EULA_ACCEPTED` marker, and is silent on subsequent runs).
