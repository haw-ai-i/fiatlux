# Psi-0 fine-tune on the S06 dispose-bulb teleop takes

Follow-up to the zero-shot `psi0` baseline (`docs/psi0_baseline.md` on `feat/psi0-zero-shot`):
fine-tune the same released checkpoint on the benchmark's own S06 (`FIATLUX-S06-DisposeBulb-v0`)
VR teleop demonstrations, then evaluate it with the same adapter, server, and sweep.

| Stage | Script | Runs in |
| --- | --- | --- |
| A. teleop bag -> SONIC tokens + state + finger targets (50 Hz) | `scripts/psi0/encode_sonic_tokens.py` | fiatlux venv (CPU) |
| B. -> Psi0 LeRobot v2.1 pack (30 Hz) | `scripts/psi0/build_psi0_pack.py` | Psi0 venv (CPU) |
| token check in sim | `scripts/psi0/replay_tokens.py` | fiatlux venv (GPU) |
| data-path dry run | `DRY_RUN=1 scripts/psi0/finetune_s06.sh` | Psi0 venv (CPU) |
| train | `scripts/psi0/finetune_s06.sh` (+ `train_sdpa.py`) | Psi0 venv (GPU) |
| evaluate | `PSI0_RUN_DIR=... PSI0_PORT=8015 scripts/psi0/serve.sh` + `scripts/psi0/sweep.sh S06` | both |

## Data

`haw-ai-i/fiatlux-teleoperation`, `2026-09-13-dex3-teleop-takes/FIATLUX-S06-DisposeBulb-Teleop-v0/`
(paginated listing read to the end): **12 takes, 10 `success/`, 2 `fail/`**. Each take has
`run.h5` (one demo, 50 Hz), `meta.json`, `ego.mp4` (512x512, 50 fps, one frame per recorded step,
verified: frame count == step count for every take), `video.mp4`, `score_report.txt`. The takes are
short: 172-499 steps (3.4-10 s), 3,022 steps in total -> **1,693 frames at 30 Hz**, about 56 s of
demonstration. Recorded at fiatlux commit `02cf17e` (pre-history-flatten), task
`FIATLUX-S06-DisposeBulb-Teleop-v0`, Dex3 robot.

## Decision log

1. **Success takes only.** The 10 `success/` takes are used; the 2 `fail/` takes are encoded (kept
   under `~/psi0_ft/encoded/fail` on iolani-3) but left out of the pack. Behaviour cloning on a
   failed attempt teaches the failure; with 10 good demos there is no case for it.

2. **Actions had to be produced; there is no public retargeting script, so the documented encoder
   interface was implemented.** The teleop twins drive the arms by IK and the legs by the decoupled
   walk/balance WBC (`docs/subtask_teleop.md`), so no GEAR-SONIC token was ever recorded, and
   Psi-0 SONIC's action is `[token 64 | Dex3 14 | neck 2]`. Searched first for the documented way:
   Psi0's `scripts/data/simple_to_sonic_lerobot.py` only remaps packs that already carry tokens;
   Psi0's GR00T-WholeBodyControl fork records tokens live from the C++ controller
   (`gear_sonic/scripts/run_data_exporter.py`, `action.motion_token` <- `proprio["token_state"]`);
   the UnifoLM retargeting behind `postpre.sonic1.0.*` is not in either repo. What *is* documented
   is how GEAR-SONIC's deploy stack builds the encoder's input from a joint-space reference clip,
   so `encode_sonic_tokens.py` implements exactly that and nothing more:
   - `nvidia/GEAR-SONIC:model_encoder.onnx` (the `release`/v1.0 encoder paired with the decoder
     `SonicDecoder` uses), input 1762-D = the encoder observation list of
     `observation_config.yaml` in order; `g1` mode (id 0) fills `encoder_mode_4`,
     `motion_joint_positions_10frame_step5`, `motion_joint_velocities_10frame_step5`,
     `motion_anchor_orientation_10frame_step5`, and every other block stays zero
     (`GatherEncoderObservations`: "Not required for this mode - leave as zero").
   - Gatherer semantics from `gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp`
     (`GatherMotionJointPositionsMultiFrame`, `...VelocitiesMultiFrame`,
     `GatherMotionAnchorOrientationMutiFrame`, `GatherEncoderMode`): frames `t, t+5, ..., t+45`
     at 50 Hz clamped to the last frame; 29 joints in IsaacLab order, absolute (gear_sonic
     `token_losses.py`: "`[dof_pos(29), dof_vel(29)]` in IsaacLab order"); anchor =
     `conj(q_base) * q_ref_root`, first two rotation-matrix columns flattened row-major.
   - The reference is the take's own *executed* whole-body trajectory (`joint_pos`,
     `robot_root_quat`), not the IK targets: it is the motion that actually disposed of the bulb
     and it is physically consistent across legs, waist, and arms. Robot base and reference root
     are the same recorded pelvis, so the deploy stack's `apply_delta_heading` is the identity.
     Velocities are central differences of the recorded positions.
   - The encoder output is already on the FSQ grid (verified: 0 off-grid error on every take).

3. **Encoder-layout validation (CPU, before any training).**
   - *Against Psi0's own retargeted tokens.* Re-encoding the joint trajectories of Psi0's public
     `USC-PSI-Lab/psi-data:sonic/unifolm_sonic_lerobot_val` pack (9 episodes, which carry the
     `action.body_token` Psi0 produced) with this layout reproduces **66-67 % of token dims
     exactly, vs 28 % for a shuffled-frame control**; mean |diff| 0.023, about a third of a grid
     step. So joint order, block offsets, and the mode are right, and the residual is ±1-step
     disagreement. The likely cause is the anchor orientation: that pack stores no root
     orientation, so the check used identity. The S06 takes do record the pelvis orientation, so
     they do not have that gap. Velocity handling barely matters (gradient vs zero: 66.2 vs 66.4 %;
     encoding at 30 vs 50 Hz: 67.4 vs 66.2 %).
   - *Against the decoder.* For each S06 take, the released SONIC decoder fed token_t and the
     demo's own proprio history puts its joint targets closer to the demo's pose 5 steps later
     than it does with SONIC's standing latent (0.08-0.12 vs 0.16-0.23 rad, all 10 takes) or with a
     token from the other half of the take (0.08-0.19 rad).
   - *Standing pose.* Encoding a motionless clip in SONIC's default pose does **not** reproduce
     `STAND_TOKEN` (19/64 dims equal). This is not evidence against the layout: gear_sonic
     documents that token as a checkpoint-specific "known safe standing pose" for VLA start-up
     (`gear_sonic/utils/inference/initial_poses.py`), not as the encoding of the default pose.
   - *In sim:* see "Token replay" below.

4. **Pack schema mirrors Psi0's released SONIC packs** (`unifolm_sonic_lerobot_*`):
   `observation.state` 43-D (legs, waist, arms, Dex3 thumb->middle->index, the psi0.py order),
   `action` 36-D (Dex3 targets 14 | arm pose 14 | 8 unused torso/base slots), `action.body_token`
   64-D, `tasks.jsonl` / `episodes.jsonl` / `episodes_stats.jsonl` / `stats.json` /
   `stats_psi0.json` / `modality.json`. The repack reads `action.body_token ++ action[:14]` and
   pads to 80 with Psi0's own `pad_to_len`, whose mask removes the neck from the loss
   (`finetune.py`: `loss_action = (loss_action * mask).sum(1)`) instead of training the neck
   toward 0. One deliberate difference: the video key is `observation.images.head`, the key this
   checkpoint and `fiatlux_task.psi0` use, not `...egocentric`. The instruction comes from
   `tasks.jsonl` through LeRobot's built-in `task` field (the SONIC repack's default
   `instruction_key`) rather than the checkpoint's `annotation.task` column.

5. **Finger targets are the commanded ones** (`joint_pos_target` of the Dex3 joints, i.e. the
   binary grip presets), matching how Psi0 builds its SONIC action (`raw_sonic_to_psi_lerobot.py`
   takes `teleop.*_hand_joints` targets when present). The state is the executed `joint_pos`.

6. **Images.** The teleop twin renders the same D435 sensor (`focal_length 15.13`,
   `horizontal_aperture 20.955`, 69.4 deg HFOV) at 512x512 (`subtask_teleop.py`: "512x512 rather
   than the sensor module's 256"). Its centre 512x288 crop is the D435's 16:9 geometry
   (69.4 x 42.5 deg), exactly what the zero-shot eval renders natively at 480x270, so the crop is
   scaled to 480x270 and stored at the model's input size. The model's resize is then the identity,
   and eval frames have the same geometry as training frames. Spot-checked visually.

7. **50 Hz -> 30 Hz.** Frame `j` takes image, state, token, and finger targets from recorded step
   `round(j * 50 / 30)`, so each row stays internally consistent. FSQ codes cannot be interpolated.

8. **Normalization bounds: the checkpoint's own, widened only where the demos leave them.**
   A 10-take pack's own min/max would re-map every normalized dimension the action head was
   trained on. So `stats_psi0.json` carries the fine-tuned checkpoint's `run_config.json` bounds,
   unioned with the demo range:
   - token: 1 of 64 dims widened; 0.008 % of values were outside, so the demo tokens sit inside
     Psi-Dream's token distribution;
   - Dex3 targets: 4 of 14 widened, 16 % of values outside, from the teleop grasp preset closing
     the right index/middle/thumb deeper (to 1.5-1.7 rad) than Psi-Dream's operators did;
   - state: 13 of 43 widened, 12 % outside. The largest factors are the left fingers resting at
     exactly 0 (Psi-Dream's never quite reach 0), the right grasp, right wrist pitch to -0.95
     (Psi-Dream: -0.52), and waist pitch to 0.23 (0.18). No sign flips, which would have indicated a
     layout error.
   The state entry is 45-D: the 43 stored dims plus the checkpoint's own neck bounds, so the
   zero-padded neck normalizes to exactly what the checkpoint saw (0.48 / 0.75 in the dry run). A
   43-D entry would be padded with 0/0 bounds and read as a constant.

9. **Training setup: one RTX 3090, frozen VLM.** Base recipe: Psi0's
   `finetune-sonic-psi-dream-baseline.sh`, which produced the warm-start checkpoint. It tunes the
   VLM with per-component learning rates on 8 GPUs; tuning the 2.13 B-parameter VLM needs fp32
   master weights, grads, and Adam (~34 GB), which do not fit in 24 GB. With `tune_vlm` off, the
   trainer loads the VLM frozen in bf16 (`FinetuneTrainer.vlm_param_dtype`), and only the
   674 M-parameter action expert trains (fp32 + Adam ~11 GB). Everything else is the recipe's:
   flow matching, 12 blocks cross-attending VLM layers 3...28, `combined_temb` CLIP pooled
   instruction, state as action token with a learned null token (drop 0.1), state jitter ±10
   frames p=0.5, state noise 0.05, image + view augmentation, lr 1e-4 cosine, grad clip 1.0,
   bf16 autocast. Warm start: `scripts/export_psi0_ckpt.py` on `ckpt_40000` (VLM exported to
   bf16), with the HF config/processor files of the released post-trained VLM
   (`postpre.sonic1.0.unifolm.2609092156.40k`); its `vocab_size` matches the checkpoint's
   `embed_tokens` (153792).

10. **SDPA instead of flash-attn for training.** The trainers hard-code
    `attn_implementation="flash_attention_2"`; flash-attn is not installed (the zero-shot inference
    already runs SDPA, `docs/psi0_baseline.md` #3). `train_sdpa.py` swaps the argument to `sdpa`
    before running Psi0's own `scripts/train.py`. The model config's `data_flatten`/`data_packing`,
    the only flash-attn-specific-looking options, are read nowhere. Installing flash-attn instead
    would have changed the zero-shot server's attention kernel on its next restart.

11. **`.env`.** Psi0's `scripts/train.py` asserts a `.env` exists. A minimal one was added to the
    Psi0 checkout on iolani-3 (`PSI_HOME`, tokenizer/AV/protobuf quieting flags). It sets nothing
    the server reads (no `HF_HOME`, no offline switch); the offline switch is set per run by
    `finetune_s06.sh`.

12. **Validation split = the training pack.** With 10 takes, holding one out costs 10 % of the
    data for a loss number that says little about closed-loop success. Closed-loop evaluation is
    the metric; the val loss is only a sanity signal.

13. **Warm start verified before any GPU time.** The trainer loads the action header with
    `load_state_dict(..., strict=False)`, which would silently leave a mismatched block randomly
    initialized. Built the CLI-configured model on CPU (meta-device VLM) and compared: 336/336
    header tensors present in the export, 0 missing, 0 unexpected, 0 shape mismatches.

14. **Batch 8 x grad-accumulation 2** (effective 16, the recipe's per-GPU batch). Batch 16 ran out
    of memory on the first step: 20.4 GB for the trainer, with 2.3 GB held by three processes that
    already occupied the GPU. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as the OOM message
    suggests.

15. **DeepSpeed told to stay out.** The first real launch died at step 0 in `trainer.evaluate()`:
    accelerate's `unwrap_model` imports `deepspeed` whenever the package is installed, and its
    import-time op check raises `MissingCUDAException: CUDA_HOME does not exist` (no CUDA toolkit on
    iolani-3). The run is plain single-GPU DDP, so `train_sdpa.py` reports DeepSpeed as unavailable
    to accelerate.

16. **2,000 optimizer steps.** At 1.5 it/s this takes ~22 min, well under the 90-min budget. The
    budget would have allowed ~8,000 steps, but that is ~75 epochs of a 56 s pack. 2,000 steps is
    ~19 epochs, the same order as Psi0's own fine-tunes (psi-dream: 40k steps x 128 over ~540k
    frames, ~9.5 epochs). Checkpoints at 1,000 and 2,000.

## Token replay

`replay_tokens.py` drives the Dex3 G1 in `FIATLUX-S06-DisposeBulb-v0` (the take's own layout
seed, the RL env rather than the teleop twin) with exactly the encoded tokens (50 Hz, one per
recorded step, through `SonicDecoder`) and the commanded finger targets:

| take | demo steps | replay ended | by | mean abs joint err (arm) | demo arm motion |
| --- | --- | --- | --- | --- | --- |
| `2026-09-13_234341_ep00` | 276 | step 265 | **`success`**: bulb disposed in the crate | 0.080 (0.107) rad | 0.158 rad |
| `2026-09-14_184617_ep01` | 267 | step 214 | `old_bulb_struck` | 0.060 (0.080) rad | 0.055 rad |

The first replay reproduces the whole task from tokens alone. The second tracks the demo equally
closely but ends in a failure termination. The start state is the RL env's reset, not the take's
exact first frame (start-pose error 0.13 rad for both), so an open-loop replay is not expected to
succeed every time. This is the closed-loop evidence that the tokens encode the demonstrated
motion.

## Training

Run dir (iolani-3): `~/psi0_ft/runs/finetune/s06ft.g1soni.flow1000.cosine.lr1.0e-04.b16.gpus1.2609262230`
(`argv.txt`, `run_config.json`, `clip_pooled_cache.pt`, `dataset_statistics.json`,
`checkpoints/ckpt_{1000,2000}`, 12 GB each with optimizer state).

- 2,000 optimizer steps, batch 8 x accumulation 2, lr 1e-4 cosine (warmup 100) to 0, bf16 autocast,
  frozen bf16 VLM, 673.8 M trainable parameters, RTX 3090. **23 min wall time** (1.44-1.85 it/s),
  ~21 GB peak.
- Training loss (flow-matching, per-step values from the progress bar, 200-step windows):

  | steps | 1-200 | 201-400 | 401-600 | 601-800 | 801-1000 | 1001-1200 | 1201-1400 | 1401-1600 | 1601-1800 | 1801-2000 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | mean | 2.225 | 1.095 | 1.148 | 0.939 | 0.844 | 0.817 | 0.689 | 0.723 | 0.605 | 0.600 |
  | median | 1.250 | 0.759 | 0.794 | 0.651 | 0.526 | 0.487 | 0.427 | 0.424 | 0.378 | 0.373 |

  Validation passes ran at steps 0/1000/1999 on the training pack (decision 12), but with no
  tracker configured (`--log.report_to` unset, no wandb on iolani-3) their values were not
  printed.

## Evaluation

Same adapter (`--policy psi0:localhost:8015 --robot dex3`), same server settings (test-time RTC,
15 of 30 rows per query, 10 flow steps), same instruction ("put the light bulb into the yellow crate
and let go of it", which is also the pack's task string), same protocol (`sweep.sh S06`, seeds 0-3,
one episode each, `score.py`). Checkpoint `ckpt_2000`. Bags:
`~/fiatlux-worktrees/psi0-ft/logs/runs/psi0_ft_s06/` on iolani-3.

| S06 seed | fine-tuned subtask_score | success | gate_progress | steps | what happened |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.333 | 0 | 0.667 | 129 | released at step 53 from 0.77 m beside the crate; episode ended mid-fall |
| 1 | 0.000 | 0 | 0.000 | 6000 | held the bulb the whole 120 s, never released (timeout) |
| 2 | 0.333 | 0 | 0.667 | 94 | released at step 53 from 0.78 m; `old_bulb_in_bin` fired at step 92, episode ended at 94 |
| 3 | 0.333 | 0 | 0.667 | 101 | released at step 73 from 0.75 m; episode ended mid-fall |
| **mean** | **0.250** | **0/4** | **0.500** | | |

Comparison on S06 (same protocol):

| policy | robot | subtask_score (per seed) | success | gate_progress | source |
| --- | --- | --- | --- | --- | --- |
| Psi-0 fine-tuned (this) | Dex3 | **0.250** (0.333, 0, 0.333, 0.333) | 0/4 | 0.500 | above |
| Psi-0 zero-shot | Dex3 | 0.271 (0.167, 0.250, 0.333, 0.333) | 0/4 | 0.542 | `logs/runs/psi0_zeroshot`, seeds 0-3 |
| `zero` | Dex3 | 0.125 (0.167, 0.167, 0, 0.167) | 0/4 | 0.250 | `logs/runs/zero_dex3`, seeds 0-3, current code |
| GR00T N1.7 zero-shot | Dex3 | 0.333 (every seed) | 0/4 | 0.667 | `rerun_2026-09-26/groot_dex3_S06_*` |
| `zero` | **Inspire** | 0.208 (0.333, 0.167, 0.167, 0.167) | 0/4 | 0.417 | `rerun_2026-09-26/zero_S06_*` (default robot) |

The 2026-09-26 `zero` rerun used the default Inspire hand. The same-robot, current-code Dex3
`zero` rows came later, from the interleaved sweep described in `docs/psi0_baseline.md` #18.

**Reading it.** No success in any seed, and **the fine-tune did not raise the S06 score**: 0.250 over 4
seeds against zero-shot Psi-0's 0.271 over the same 4 seeds (both above the Dex3 `zero` floor's
0.125, both below GR00T's 0.333). The spread between all of these is one or two gate conjuncts per seed. What changed
is the behaviour. Zero-shot seed 0 held the bulb aloft for the full 120 s; in zero-shot seeds 1 and
2 the bulb fell (peak 4.0 m/s) and the episode ended at steps 142 and 216, never over the crate. In
3 of 4 seeds the fine-tuned policy moves the bulb and opens the hand within 1.0-1.5 s. In seed 2
the bulb reached the crate footprint (`old_bulb_in_bin` at step 92), the only S06 episode of any
Psi-0 variant to do so. What ends those three episodes is the release itself. The bulb is
dropped from 0.75-0.78 m and is still falling at 3.7-3.9 m/s when a termination that is neither
success, drop, nor timeout fires. The bag does not record its name; this is consistent with the
glass-impact term `old_bulb_struck`, the one the second token replay ended on.

**That release is what the demos teach.** All 10 successful takes drop the bulb from 0.63-0.90 m
(peak 3.4-4.0 m/s) into the crate. They could still score success because the teleop twins clear
every failure termination but `success` (`docs/subtask_teleop.md`), so an impact check never ran
during teleop. The RL env that `sweep.sh` evaluates keeps it, and the first token replay shows a
0.74 m drop *can* pass in the RL env when the bulb lands cleanly. This is a benchmark/data finding
worth raising: S06 demos recorded under the teleop twin can contain releases the benchmark's own env
terminates on.

**Caveats.** Four episodes; ~56 s of demonstration; frozen VLM (only the action expert adapted);
one checkpoint evaluated (`ckpt_1000` exists and was not evaluated, to hand the GPU back for the
zero-shot seeds 1-3).
