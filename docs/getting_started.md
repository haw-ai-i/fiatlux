# Getting Started

## 1. Install Isaac Lab

Follow the official guide: <https://isaac-sim.github.io/IsaacLab>. Verify it works
with one of the shipped examples before continuing. This benchmark targets Isaac
Sim 4.5/5.x and the `isaaclab` / `isaaclab_rl` / `isaaclab_tasks` packages.

## 2. Install the Fiatlux task package

From the repo root, into the Isaac Lab Python environment:

```bash
pip install -e source/fiatlux_task
```

## 3. Download the assets

The G1, bulb/socket, and ladder USDs are pulled from a GCS bucket (they are not
checked into git):

```bash
./assets/download_assets.sh
# override the bucket if needed:
FIATLUX_ASSET_BUCKET=gs://my-bucket/assets ./assets/download_assets.sh
```

This produces:

```
assets/unitree_g1/g1.usd
assets/bulb_socket/{bulb.usd,socket.usd}
assets/ladder/ladder.usd        # for the climbing subtask (roadmap)
```

If your assets live elsewhere, point the env at them with
`export FIATLUX_ASSETS_DIR=/path/to/assets`.

## 4. Run

```bash
python scripts/list_envs.py                                  # list registered tasks
python scripts/zero_agent.py --task FIATLUX-Insert-v0        # launch the scene
python scripts/eval.py --task FIATLUX-Insert-v0 --policy random --episodes 20
python scripts/rsl_rl/train.py --task FIATLUX-Insert-v0      # train PPO
python scripts/rsl_rl/play.py  --task FIATLUX-Insert-v0      # roll out a checkpoint
```

## Troubleshooting

- **Env not found / empty list** — `pip install -e source/fiatlux_task` did not run
  in the Isaac Lab env, or `import fiatlux_task.tasks` failed.
- **USD not found** — run `./assets/download_assets.sh` or set `FIATLUX_ASSETS_DIR`.
- **Joint/body name errors** — the constants in `g1_bulb_env_cfg.py`
  (`G1_ARM_JOINTS`, `G1_EE_BODY`) must match the joints/links in your G1 USD.
