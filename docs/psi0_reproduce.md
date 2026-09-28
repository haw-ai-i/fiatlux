# Reproducing the Psi-0 experiments

Every step behind `docs/psi0_baseline.md` (zero-shot Psi-0 on all twelve subtasks, against a
same-robot `zero` floor) and `docs/psi0_finetune.md` (the S06 fine-tune), in order, with the command
that runs it, what it writes, and roughly how long it takes on one RTX 3090. The *why* of each
choice is in those two decision logs; this page is only the *how*.

All commands run from the fiatlux repo root unless they say otherwise. Paths default to the
layout used for the reported runs (`~/tools/...` for third-party code and models, `~/psi0_ft/...`
for the fine-tune workspace); every script reads them from environment variables of the same name
if you want them elsewhere.

## 0. Requirements

- Linux, one NVIDIA GPU with **24 GB** (RTX 3090 used). Isaac Sim (~7 GB) and one Psi-0 server
  (~7-9 GB) run together; fine-tuning (~21 GB) needs the GPU to itself.
- ~60 GB disk: Isaac Sim + Isaac Lab venv (~10 GB), Psi0 venv (~8 GB), Psi-0 checkpoint (11 GB),
  fine-tune checkpoints (12 GB each), bags/videos.
- `git`, [`uv`](https://docs.astral.sh/uv/), the Hugging Face CLI (`hf`), `curl`, `ffmpeg`
  (optional, for looking at videos).
- `HF_TOKEN` with read access to the **private** `haw-ai-i/fiatlux-teleoperation` dataset (fine-tune
  only). Everything else downloaded here is public.

## 1. fiatlux itself

As in `docs/getting_started.md`, plus the optional `groot` extra (onnxruntime for the in-process
SONIC decoder/encoder):

```bash
uv sync --extra groot
./assets/download_assets.sh
git checkout feat/psi0-zero-shot          # this branch (PR #261)
```

`.venv/bin/python` below is this venv.

## 2. Psi0, its checkpoint, SONIC, and the data: `scripts/psi0/setup.sh`

One idempotent script, six steps (`STEPS="1 2 3" scripts/psi0/setup.sh` to run a subset):

| step | what | where |
| --- | --- | --- |
| 1 | clone Psi0 at `4f3720d`, delete its unparseable `uv.lock`, build `.venv-psi` (py3.11, torch 2.7); write the minimal `.env` its trainer asserts | `~/tools/Psi0` |
| 2 | the released Psi-0 SONIC checkpoint `multi-task.psi-dream.2609092156` (run dir, 11 GB); the HF config/processor files of its post-trained VLM (for the fine-tune warm start); pre-cache `Qwen/Qwen3-VL-2B-Instruct` (config/processor) and `openai/clip-vit-large-patch14` | `~/tools/psi0_checkpoints`, `~/psi0_ft/hf_from` |
| 3 | GEAR-SONIC `release` decoder + encoder ONNX | `~/tools/sonic_models/policy/release` |
| 4 | Psi-0's public UnifoLM SONIC val pack, converted to per-episode `.npz` (`scripts/psi0/unifolm_to_npz.py`) | `~/psi0_ft/unifolm_npz` |
| 6 | the S06 dispose-bulb VR teleop takes (private; needs `HF_TOKEN`) | `~/psi0_ft/raw/{success,fail}` |

```bash
HF_TOKEN=... scripts/psi0/setup.sh        # ~15-30 min, mostly the uv sync and the 11 GB download
```

## 3. Unit tests (no GPU)

```bash
.venv/bin/python -m pytest source/fiatlux_task/fiatlux_task/tests/test_psi0_adapter.py \
    source/fiatlux_task/fiatlux_task/tests/test_groot_adapter.py
```

## 4. Serve Psi-0: `scripts/psi0/serve.sh`

```bash
scripts/psi0/serve.sh &                   # :8014, ~70 s to load, ~7-9 GB VRAM, ~460 ms per /act
curl -s localhost:8014/info               # action_dim 80, action_exec_horizon 15, rtc test_time
```

`serve.sh` runs `scripts/psi0/serve_http.py` (upstream's HTTP server + the CLIP pooled-instruction
input these checkpoints need) in the Psi0 venv. Knobs: `PSI0_RUN_DIR`, `PSI0_CKPT_STEP`, `PSI0_PORT`,
`PSI0_EXEC_HORIZON` (default 15), `PSI0_RTC` (default 1). The first request for an unseen
instruction takes ~20 s (CLIP load).

## 5. Check the token path below the VLA: `scripts/psi0/verify_sonic_tracking.py`

Replays Psi-0's own recorded tokens through fiatlux's `SonicDecoder` and scores joint tracking
against the real robot's recorded joints (docs/psi0_baseline.md #17). No server needed.

```bash
.venv/bin/python scripts/psi0/verify_sonic_tracking.py --headless \
    --npz ~/psi0_ft/unifolm_npz/episode_00000{8,6,0}.npz --out ~/psi0_ft/replay_unifolm.json
```

Expected: moving arm joints correlate 0.8-0.99 with the recording; arm RMSE 0.13-0.16 rad vs
0.31-0.49 rad for a frozen robot. ~3 min.

## 6. One rollout by hand

```bash
.venv/bin/python scripts/record_run.py --task FIATLUX-S06-DisposeBulb-v0 --policy psi0 --robot dex3 \
    --episodes 1 --seed 0 --record both --cam ego --video_length 6000 \
    --enable_cameras --headless --out logs/runs/psi0_try
.venv/bin/python scripts/score.py logs/runs/psi0_try
```

`--policy psi0` (or `psi0:<host:port>`) needs `--robot dex3`. `record_run.py` applies
`policy.prepare_env_cfg`, which renders the ego camera at 480x270 for this policy, and the
instruction defaults to the task's sentence in `psi0.TASK_INSTRUCTIONS` (`--instruction` overrides).
A full 6000-step episode takes 12-20 min: the simulator, not the model, is the bottleneck.

## 7. The zero-shot protocol: `scripts/psi0/interleave.sh`

Psi-0 and the same-robot `zero` floor, alternated one seed at a time over all twelve subtasks
(seeds 0-3, one episode each, ego video for seed 0), each run scored by `scripts/score.py`.
Resumable: a run with a `score.json` is skipped, so re-invoking after an interruption continues.

```bash
scripts/psi0/serve.sh &                   # :8014 must be up
scripts/psi0/interleave.sh                # ~8-9 h: ~1.5 h per Psi-0 seed, ~0.5 h per zero seed
```

Writes `logs/runs/psi0_zeroshot/<task>/seed<N>/` and `logs/runs/zero_dex3/<task>/seed<N>/` (bag
`run.h5`, `meta.json`, `score.json`, `record.log`, and `video/run.mp4` for seed 0). The building
block is `scripts/psi0/sweep.sh` (`PSI0_SPEC`, `PSI0_OUT`, `SEEDS`, `VIDEO_SEEDS`, optional subtask
prefixes as arguments, e.g. `scripts/psi0/sweep.sh S06 S08`).

## 8. Results tables

```bash
.venv/bin/python scripts/psi0/summarize.py logs/runs/psi0_zeroshot     # per subtask + per-seed weighted
.venv/bin/python scripts/psi0/summarize.py logs/runs/zero_dex3
.venv/bin/python scripts/psi0/compare_baselines.py zero=logs/runs/zero_dex3 \
    psi0=logs/runs/psi0_zeroshot psi0_ft=logs/runs/psi0_ft_s06
.venv/bin/python scripts/psi0/explain_gate.py logs/runs/psi0_zeroshot/FIATLUX-S06-DisposeBulb-v0/seed0/run.h5
```

Reported: Psi-0 zero-shot 0.090 +- 0.039, `zero` on Dex3 0.084 +- 0.029 (weighted, 4 seeds each;
0/48 successes each). `explain_gate.py` shows which success conjuncts an episode held and when,
which is how an S06 release into the crate that scored like a miss was diagnosed.

## 9. The S06 fine-tune

Stop the :8014 server first; training needs ~21 GB.

| step | command | venv | time |
| --- | --- | --- | --- |
| a. teleop bags -> SONIC tokens, state, finger targets (50 Hz) | `.venv/bin/python scripts/psi0/encode_sonic_tokens.py --takes ~/psi0_ft/raw/success --out ~/psi0_ft/encoded/success` | fiatlux | 1 min |
| a'. check the encoder layout against Psi-0's own tokens (expect ~66 % dims equal vs ~28 % control) | `.venv/bin/python scripts/psi0/unifolm_reencode.py ~/psi0_ft/unifolm_npz` | fiatlux | 1 min |
| b. -> Psi0 LeRobot pack (30 Hz, 480x270 head video, checkpoint-anchored stats) | `~/tools/Psi0/.venv-psi/bin/python scripts/psi0/build_psi0_pack.py --encoded ~/psi0_ft/encoded/success --takes ~/psi0_ft/raw/success --out ~/psi0_ft/packs/s06_dispose_teleop --ckpt-run-dir ~/tools/psi0_checkpoints/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156 --task "put the light bulb into the yellow crate and let go of it"` | Psi0 | 2 min |
| c. closed-loop replay of the encoded tokens in sim (GPU) | `.venv/bin/python scripts/psi0/replay_tokens.py --npz ~/psi0_ft/encoded/success/<take>.npz --meta ~/psi0_ft/raw/success/<take>/meta.json --out ~/psi0_ft/replay/<take> --headless --enable_cameras` | fiatlux | 3 min/take |
| d. warm start: split the released checkpoint | `scripts/psi0/export_warmstart.sh` (-> `~/psi0_ft/init/psi_dream_40k`) | Psi0 | 3 min |
| e. CPU checks: data path, warm-start keys | `DRY_RUN=1 scripts/psi0/finetune_s06.sh` then `CHECK_INIT=1 scripts/psi0/finetune_s06.sh` | Psi0 | 2 min |
| f. train (action expert only, VLM frozen, batch 8 x 2, 2000 steps) | `TS=$(date +%y%m%d%H%M) scripts/psi0/finetune_s06.sh` (-> `~/psi0_ft/runs/finetune/s06ft...<TS>/`) | Psi0 | 23 min |
| g. evaluate on S06, seeds 0-3 (serves :8015 itself) | `RUN_DIR=~/psi0_ft/runs/finetune/<run> STEP=2000 scripts/psi0/eval_finetuned.sh` (-> `logs/runs/psi0_ft_s06`) | both | 10-60 min |
| g'. same, with video for every seed | `VIDEO_SEEDS="0 1 2 3" PSI0_OUT=logs/runs/psi0_ft_s06_video RUN_DIR=... scripts/psi0/eval_finetuned.sh` | both | same |

Reported: 0.250 mean `subtask_score` over 4 seeds (zero-shot on the same seeds 0.271, `zero` 0.125,
GR00T-on-Dex3 0.333), 0/4 successes. Sampling is stochastic, so a re-run reproduces the behaviour
and the score distribution, not identical frames: the video re-run of seed 2 again released into
the crate.

## 10. Using the released fine-tune weights instead of training

Only the action expert changed, so the upload is the action head plus the run-dir files:
`gs://fiatlux/tmp/psi0_s06_ft_2026-09-27/` (`action_header_ckpt2000.safetensors`, 2.7 GB;
`argv.txt`, `run_config.json`, `clip_pooled_cache.pt`, `dataset_statistics.json`, `README.md`).
`scripts/psi0/split_ft_ckpt.py` rebuilds the full `model.safetensors` from it and the released base
(verified tensor-for-tensor against the original):

```bash
gsutil -m cp -r gs://fiatlux/tmp/psi0_s06_ft_2026-09-27 ~/psi0_ft/
RUN=~/psi0_ft/runs/finetune/s06ft_ckpt2000 && mkdir -p $RUN/checkpoints/ckpt_2000
cp ~/psi0_ft/psi0_s06_ft_2026-09-27/{argv.txt,run_config.json,clip_pooled_cache.pt,dataset_statistics.json} $RUN/
~/tools/Psi0/.venv-psi/bin/python scripts/psi0/split_ft_ckpt.py assemble \
    --head ~/psi0_ft/psi0_s06_ft_2026-09-27/action_header_ckpt2000.safetensors \
    --base ~/tools/psi0_checkpoints/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156/checkpoints/ckpt_40000/model.safetensors \
    --out $RUN/checkpoints/ckpt_2000/model.safetensors
RUN_DIR=$RUN STEP=2000 scripts/psi0/eval_finetuned.sh
```

`split_ft_ckpt.py extract` is the reverse (and checks that no VLM tensor changed).

## 11. Where the reported artifacts live

- Code, docs: branch `feat/psi0-zero-shot`, PR haw-ai-i/fiatlux#261.
- Videos (ego camera, seed 0 of every sweep + the fine-tuned re-runs):
  `gs://fiatlux/tmp/fiatlux_psi0_videos_2026-09-27/`.
- Fine-tune weights: `gs://fiatlux/tmp/psi0_s06_ft_2026-09-27/` (section 10).
- Bags and logs of every run: iolani-3, `~/fiatlux-worktrees/psi0/logs/runs/{psi0_zeroshot,zero_dex3,psi0_ft_s06_video}`
  and `~/fiatlux-worktrees/psi0-ft/logs/runs/psi0_ft_s06`; the full training run dir (with optimizer
  state, `ckpt_1000` and `ckpt_2000`) in `~/psi0_ft/runs/finetune/`.

## Entry points

| script | runs in | purpose |
| --- | --- | --- |
| `scripts/psi0/setup.sh` | shell | Psi0 repo + venv, checkpoint, SONIC ONNX, UnifoLM pack, teleop takes |
| `scripts/psi0/serve.sh` -> `serve_http.py` | Psi0 | the Psi-0 HTTP policy server |
| `scripts/record_run.py --policy psi0` | fiatlux | one recorded rollout (the adapter: `fiatlux_task/psi0.py`) |
| `scripts/psi0/sweep.sh` | fiatlux | subtasks x seeds, recorded and scored, resumable |
| `scripts/psi0/interleave.sh` | fiatlux | the reported zero-shot protocol (Psi-0 and `zero`, seed by seed) |
| `scripts/psi0/summarize.py` | fiatlux | per-subtask and per-seed weighted table for one sweep |
| `scripts/psi0/compare_baselines.py` | fiatlux | the same, several sweeps side by side |
| `scripts/psi0/explain_gate.py` | fiatlux | which success conjuncts an episode held, and when |
| `scripts/psi0/verify_sonic_tracking.py` | fiatlux | SonicDecoder vs Psi-0's recorded tokens and joints |
| `scripts/psi0/unifolm_to_npz.py` | Psi0 | UnifoLM pack -> `.npz` |
| `scripts/psi0/encode_sonic_tokens.py` | fiatlux | teleop bags -> SONIC tokens (fine-tune stage A) |
| `scripts/psi0/unifolm_reencode.py` | fiatlux | encoder-layout check against Psi-0's own tokens |
| `scripts/psi0/build_psi0_pack.py` | Psi0 | encoded takes -> LeRobot pack (stage B) |
| `scripts/psi0/replay_tokens.py` | fiatlux | closed-loop replay of encoded takes in S06 |
| `scripts/psi0/export_warmstart.sh` | Psi0 | released checkpoint -> warm-start dir |
| `scripts/psi0/finetune_s06.sh` (+ `train_sdpa.py`, `dryrun_pack.py`, `check_init.py`) | Psi0 | train; `DRY_RUN=1` / `CHECK_INIT=1` CPU checks |
| `scripts/psi0/eval_finetuned.sh` | both | serve a fine-tuned run dir and sweep S06 |
| `scripts/psi0/split_ft_ckpt.py` | Psi0 | action head <-> full checkpoint |
