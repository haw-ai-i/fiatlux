# Psi-0 baseline (`--policy psi0`)

Zero-shot evaluation of USC PSI Lab's **Psi-0** VLA ([paper](https://arxiv.org/abs/2603.12263),
[code](https://github.com/physical-superintelligence-lab/Psi0) at `4f3720d`,
[weights](https://huggingface.co/USC-PSI-Lab/psi-model)) on the FIATLUX subtasks. It is the
second open-weight model baseline after `groot`, chosen because Psi-0 is the only released VLA whose
training data includes the **Unitree G1 with Dex3 hands**, which is the robot `--robot dex3` spawns.

- Adapter: `source/fiatlux_task/fiatlux_task/psi0.py` (spec `psi0[:<host:port>]`, default
  `localhost:8014`).
- Server: `scripts/psi0/serve.sh`, which runs `scripts/psi0/serve_http.py` in the Psi0 repo's own
  venv.
- Sweep: `scripts/psi0/sweep.sh` (12 subtasks x seeds 0-3, one episode each, bag + `score.py`).

```bash
# once: Psi0 repo + venv + checkpoint (see serve.sh's header), then
PSI0_PORT=8014 scripts/psi0/serve.sh &                 # ~7 GB VRAM, ~70 s to load
scripts/psi0/sweep.sh                                  # or: record_run.py --policy psi0 --robot dex3 ...
```

## What runs

| Piece | Choice | Source |
| --- | --- | --- |
| Checkpoint | `psi0/sonic-checkpoints/multi-task.psi-dream.2609092156`, step 40000 | README "Checkpoints" table |
| Model | Qwen3-VL-2B + 12-block flow action expert, `combined_temb` (CLIP-L pooled instruction) | `run_config.json` |
| Image | one head-camera frame, key `observation.images.head`, resized to 270x480 | `run_config.json` repack/model |
| State | 45-D: 29 body joints (leg, waist, arm), 14 Dex3 joints, 2 neck | see below |
| Action | (30 rows at 30 Hz) x 80: SONIC token 64, Dex3 14, neck 2 | `argv.txt` `action-keys` |
| Body control | fiatlux's existing `SonicDecoder` (GEAR-SONIC `release` decoder, 50 Hz) | `groot.py` |
| Serving | HTTP `POST /act`, test-time RTC, 15 rows returned per query, 10 flow steps | `serve.sh` |

## Decision log

Everything below was a judgment call. Where the answer came from documentation or the checkpoint's
own files, the source is named; nothing here was reverse-engineered from weights or behaviour.

1. **Which checkpoint.** The README lists three SONIC-era entries. `postpre.sonic1.0.unifolm.*` is
   an intermediate post-training stage (VLM and action expert as separate files, no run directory);
   `sonic-checkpoints/multi-task.psi-dream.2609092156` is the fine-tuned, deployable model and ships
   the complete run directory the servers need (`argv.txt`, `run_config.json`,
   `checkpoints/ckpt_40000/model.safetensors`, `clip_pooled_cache.pt`, `dataset_statistics.json`).
   Picked the latter. Its fine-tune data is 5 h of Psi-Dream G1+Dex3 teleop across 8 household task
   families (pick-and-place, trash throwing, drawers, laundry, pillows, sweeping, shoes, chairs),
   on top of 50 h of UnifoLM G1 data retargeted to SONIC tokens. None of it is a light bulb, a
   ladder, or a ceiling fixture, so this is a genuine zero-shot test.

2. **Upstream `uv.lock` does not parse** at `4f3720d` (`duplicate key` at line 2100: a duplicated
   `[package.optional-dependencies]` table, a merge artefact; the lock also mixes a Tsinghua mirror
   with PyPI). Deleted it and let `uv sync` re-resolve from `pyproject.toml`, which pins everything
   that matters for inference (`torch==2.7.0`, `torchvision==0.22.0`, `transformers==4.57.1`,
   `triton==3.3.0`, the `lerobot` git commit). The corrupt lock is kept next to the logs for
   reference.

3. **No flash-attn.** Not installed (it needs a CUDA toolkit build or a matching wheel, and the
   README treats it as a separate step). `Psi0Model.from_pretrained` falls back to PyTorch SDPA
   when flash-attn is absent (`src/psi/models/psi0.py:1745`); numerically this is the same
   attention, only slower. Measured: ~460 ms per `/act` on the RTX 3090.

4. **Upstream's HTTP server cannot serve this checkpoint as shipped.** The checkpoint is trained
   with `--model.combined-temb`, so every forward pass needs `pooled_projections` (a frozen CLIP-L
   embedding of the instruction). Upstream's WebSocket server (`serve_psi0_sonic.py`) computes it
   with `PooledTextEncoderCache`; upstream's HTTP server (`serve_psi0_sonic_http.py` ->
   `serve_psi0_simple.Server`) never passes it, and the first request fails inside the action
   head (`linear(): argument 'input' must be Tensor, not NoneType`). The WebSocket server is not an
   option here: it runs its own wall-clock 30 Hz control loop, while the benchmark pauses the
   simulator during every request. `scripts/psi0/serve_http.py` is upstream's HTTP server plus
   upstream's own `PooledTextEncoderCache`, handed to the `pooled_projections` argument all three
   predict methods already take. An instruction outside the training cache is embedded on the fly
   with the frozen `openai/clip-vit-large-patch14`, exactly as the WebSocket server does.

5. **Chunking / RTC.** The checkpoint predicts 30 rows (1 s at 30 Hz) and was trained without
   RTC (`--model.no-rtc`); Psi0's release note says it is meant to be deployed with test-time RTC,
   and its deployment server replans after at least 15 executed rows (`MIN_EXEC_HORIZON = 15`,
   guidance alpha 0.9). The HTTP server's test-time RTC with `--action-exec-horizon 15` is the
   synchronous equivalent: 15 rows executed, then the next chunk is guided toward the unexecuted
   half of the previous one. The inference delay is exactly 0 because the sim clock is paused
   during the request, so no latency is modelled, the same as for `groot`. The first query of each
   episode sends `history["reset"]` so an episode never splices onto the previous one.

6. **Flow steps.** 10, the `ServerConfig` default ("what the released psi0 checkpoints were
   validated with"). The WebSocket deploy server uses 8 for latency; latency does not matter here.

7. **45-D state layout.** For this checkpoint the state is the "legacy `g1_sonic_lerobot_0810`
   order": `leg waist arm hand neck` (Psi0 `scripts/data/backfill_joint_names.py`), each group in
   URDF order, both hands thumb -> middle -> index (`scripts/data/merge_posttrain_sonic.py`: "Both
   hands are thumb -> MIDDLE -> INDEX"). This was cross-checked against the checkpoint's own
   `state_min`/`state_max` rather than taken on trust, because `merge_sonic_v1.py`'s docstring
   describes a *different* order (`hands arms legs waist neck`) for the newer `psix_sonic_v1`
   packs. The stats settle it: slots 3 and 9 are the only strictly positive leg slots (the knees);
   slots 32-35 are all <= 0 and 39-42 all >= 0 (the mirrored curl directions of the left and right
   index/middle fingers); 29-31 and 36-38 have the thumb ranges. A third, independent source
   agrees: the public UnifoLM SONIC pack Psi-0 was post-trained on
   (`USC-PSI-Lab/psi-data:sonic/unifolm_sonic_lerobot_val.zip`) *names* its 43 state columns in
   `meta/info.json`, legs, waist, arms, then both hands thumb/middle/index, and this checkpoint
   appends the 2 neck slots. The 80-D action's hand block is the same 14 joints in the same order.

8. **No neck.** This G1 has no neck joints and its head camera is fixed. The two neck state slots
   are sent as 0 (inside the training range), and the two neck action channels are dropped.

9. **Camera geometry.** The benchmark's ego camera is 256x256 at the D435's 69.4 deg horizontal
   FOV, sized for GR00T. Psi-0 resizes every frame to 270x480 without preserving aspect, so a square
   frame would reach it stretched 1.78x. `prepare_env_cfg` (called by `record_run.py` before the
   env is built) renders the ego camera at 480x270 with the same horizontal aperture, the D435's
   own 16:9 colour mode (69.4 x 42.5 deg). It changes the sensor's resolution only, never the task.
   What cannot be matched: Psi-Dream's camera sits on an actuated neck the policy aims itself;
   this robot's D435 is fixed to the torso and pitched steeply down. In S08 the fresh bulb on the
   tabletop is out of frame at spawn, and in S01 the view while standing is floor only: at the
   G1's ~48 deg mount pitch the top edge of a 42.5 deg-tall frame is still ~26 deg below the
   horizon (the square 69.4 deg frame's top edge is ~13 deg below it, which is also no horizon).
   For reference, the UnifoLM post-train pack is 640x480 (4:3), which training stretches 1.33x
   to 480x270; the Psi-Dream fine-tune pack is not public. A native 16:9 frame is the smallest
   distortion on offer. The alternative, sending the square frame and letting the server stretch
   it 1.78x, would keep 27 deg more vertical view and was not chosen.

10. **Token quantization and timing.** The flow head outputs a continuous token; Psi0's own robot
    client snaps it onto SONIC's FSQ grid (range +-0.625, step 1/16) before publishing it
    (`real/SONIC/run_psi0_rtc_sonic_dex1.py` -> `mock_psi0_client_rtc.fsq_quantize`). The adapter
    does the same. Each 30 Hz row is held over the 50 Hz ticks it covers (row `floor(n * 30 / 50)`
    at tick `n`), which is what SONIC's ZMQ token stream does: it stores the latest token and the
    50 Hz loop reads it.

11. **Which SONIC decoder.** The `release` (v1.0) decoder `nvidia/GEAR-SONIC:model_decoder.onnx`,
    the one `SonicDecoder` already targets. The checkpoint is a `sonic1.0` post-train on the
    1/16-grid v1 token; `merge_sonic_v1.py` says outright that the continuous `sonic_v1_1` token
    is a different space.

12. **Startup.** No standing-latent settle before the first query. The real-robot procedure
    engages SONIC's standing reference first, and the adapter first did the same for 1 s. That was
    wrong for this benchmark: half the subtasks start with the bulb already in hand, and 1 s of
    the standing latent swung the arms to SONIC's rest pose and opened the fingers, dropping the
    bulb before Psi-0 acted. S06 ended at step ~48, the same as the `zero` policy. The first query
    now happens on the tick after the one-tick startup hold, with the fingers holding their spawn
    pose until the first chunk lands.

13. **Instructions.** Psi-0 was trained on one short imperative sentence per task (its pooled
    cache holds ~100 of them, e.g. "pick up the paper ball and turn left and throw it into the trash
    can"). The benchmark had no per-subtask sentences, only `groot`'s whole-task paragraph, so
    `psi0.TASK_INSTRUCTIONS` adds one plain sentence per subtask, written from each subtask's own
    docstring and not tuned against results. `--instruction` overrides it.

14. **`dataset_name`.** Sent as `g1sonic0810`, the checkpoint's `repack.dataset_name`. The HTTP
    server does not use it (a single set of normalization stats), so it is sent for fidelity only.

15. **Protocol.** Same as the `zero`/`random`/`groot` runs it is compared against: `--robot
    dex3`, one episode per (subtask, seed), seeds 0-3, full 120 s horizon, `score.py` per bag, one
    `score_subtasks.py` roll-up per seed. Seed 0 additionally records the ego-camera video.

16. **Run order and the GPU budget.** iolani-3 has one 24 GB RTX 3090: Isaac Sim (~8 GB) +
    this server (~7 GB) + two stale processes (~3 GB) leave no room for a fine-tune. The simulator,
    not the model, sets the pace: a full 6000-step episode takes 12-20 min for *any* policy on this
    box (the `zero` sweep's S02 runs took 12-15 min each). So the sweep runs seed-major (all twelve
    subtasks for seed 0, then seed 1, ...) so that any cut-off still leaves complete per-seed
    roll-ups, and it is paused to hand the GPU to the S06 fine-tune (`docs/psi0_finetune.md`),
    whose CPU-only data preparation starts in parallel with the sweep rather than after it.

## Results

_Filled in by the sweep; see below._
