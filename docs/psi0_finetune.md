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

_Sections below are filled in as the GPU work runs._

## Token replay

## Training

## Evaluation
