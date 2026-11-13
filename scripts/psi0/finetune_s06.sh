#!/usr/bin/env bash
# Fine-tune the Psi-0 SONIC checkpoint on the S06 dispose-bulb teleop pack, on ONE 24 GB GPU.
#
# Base recipe: Psi0's scripts/train/psi0/finetune-sonic-psi-dream-baseline.sh, the recipe that
# produced the checkpoint this warm-starts from (multi-task.psi-dream.2609092156). Differences,
# each logged in docs/psi0_finetune.md:
#   - VLM frozen (no --model.tune-vlm): tuning the 2.1B VLM needs fp32 master weights + Adam
#     (~34 GB); frozen, the trainer loads it in bf16 and only the 674M action expert trains.
#   - one GPU, batch BATCH x ACCUM instead of 16 x 8 GPUs; STEPS/WARMUP scaled to a 1.7k-frame pack.
#   - SDPA attention (train_sdpa.py) instead of flash-attn.
#   - pack columns follow Psi0's unifolm_sonic_lerobot schema: action.body_token ++ action[:14],
#     neck padded to 80 (and masked out of the loss) by the repack; instruction from tasks.jsonl.
#   - no task-sample weights (one task), no wandb.
# Everything else -- augmentation, state drop/null token, architecture flags -- is the recipe's.
#
#   PACK_ROOT=~/psi0_ft/packs INIT_DIR=~/psi0_ft/init/psi_dream_40k scripts/psi0/finetune_s06.sh
set -euo pipefail

PSI0_REPO="${PSI0_REPO:-$HOME/tools/Psi0}"
PACK_ROOT="${PACK_ROOT:-$HOME/psi0_ft/packs}"
PACK_ID="${PACK_ID:-s06_dispose_teleop}"
INIT_DIR="${INIT_DIR:-$HOME/psi0_ft/init/psi_dream_40k}"
OUT_DIR="${OUT_DIR:-$HOME/psi0_ft/runs}"
EXP="${EXP:-s06ft}"
TS="${TS:-$(date +%y%m%d%H%M)}"
BATCH="${BATCH:-8}"   # x ACCUM 2 = the recipe's 16; batch 16 OOMs on a 24 GB card (docs #14)
ACCUM="${ACCUM:-2}"
STEPS="${STEPS:-2000}"
WARMUP="${WARMUP:-100}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
LR="${LR:-1e-4}"
DRY_RUN="${DRY_RUN:-}"   # DRY_RUN=1: CPU data-path check; CHECK_INIT=1: CPU warm-start key check

export PSI0_REPO HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" CUDA_LAUNCH_BLOCKING=0 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
STATS="$PACK_ROOT/$PACK_ID/meta/stats_psi0.json"
[ -s "$STATS" ] || { echo "FATAL: missing $STATS" >&2; exit 1; }
for f in config.json model.safetensors action_header.safetensors; do
    [ -s "$INIT_DIR/$f" ] || { echo "FATAL: missing $INIT_DIR/$f (scripts/export_psi0_ckpt.py)" >&2; exit 1; }
done

args=(
    finetune_sonic_psi0_config
    --seed=292285
    --exp="$EXP"
    --timestamp="$TS"
    --train.resume_from_checkpoint=latest
    --train.name=finetune
    --train.output_dir="$OUT_DIR"
    --train.data_parallel=ddp
    --train.mixed_precision=bf16
    --train.train_batch_size="$BATCH"
    --train.max_checkpoints_to_keep=2
    --train.gradient_accumulation_steps="$ACCUM"
    --train.learning_rate="$LR"
    --train.max_training_steps="$STEPS"
    --train.warmup_ratio=None
    --train.warmup_steps="$WARMUP"
    --train.checkpointing_steps="$CKPT_EVERY"
    --train.validation_steps="$CKPT_EVERY"
    --train.val_num_batches=5
    --train.max_grad_norm=1.0
    --train.lr_scheduler_type=cosine
    --train.lr_scheduler_kwargs.weight_decay=1e-6
    --train.lr_scheduler_kwargs.betas 0.95 0.999
    --data.root_dir="$PACK_ROOT"
    --data.train_repo_ids="$PACK_ID"
    --data.val_repo_ids="$PACK_ID"
    --data.transform.repack.image-keys observation.images.head
    --data.transform.repack.action-keys action.body_token "action[:14]"
    --data.transform.repack.dataset-name=g1sonic0810
    --data.transform.repack.pad-action-dim=80
    --data.transform.repack.pad-state-dim=45
    --data.transform.repack.instruction-key=task
    --data.transform.repack.state-temporal-jitter=10
    --data.transform.repack.state-temporal-jitter-prob=0.5
    --data.transform.field.stat-path="$STATS"
    --data.transform.field.state-noise-std=0.05
    --data.transform.field.stat-action-keys action.body_token "action[:14]"
    --data.transform.field.action_norm_type=bounds
    --data.transform.field.normalize-state
    --data.transform.field.pad-action-dim=80
    --data.transform.field.pad-state-dim=45
    --data.transform.model.img-aug
    --data.transform.model.view-aug
    --data.transform.model.view-aug-min-scale=0.85
    --data.transform.model.view-aug-prob=1.0
    --data.transform.model.resize.size 270 480
    --data.transform.model.center_crop.size 270 480
    --model.model_name_or_path="$INIT_DIR"
    --model.pretrained-action-header-path="$INIT_DIR"
    --model.noise-scheduler=flow
    --model.train-diffusion-steps=1000
    --model.n_conditions=0
    --model.action-chunk-size=30
    --model.action-dim=80
    --model.action-exec-horizon=30
    --model.observation-horizon=1
    --model.odim=45
    --model.dropout=0.0
    --model.state-feature-dropout=0.0
    --model.view_feature_dim=2048
    --model.gradient-checkpointing
    --model.no-use_film
    --model.qk-norm=rms_norm
    --model.combined-temb
    --model.num-blocks=12
    --model.vlm-layer-indices 3 5 8 10 12 14 17 19 21 23 26 28
    --model.state-drop-prob=0.1
    --model.state-as-action-token
    --model.state-null-token
    --model.pooled-text-encoder=clip
    --model.pooled-text-encoder-path=openai/clip-vit-large-patch14
    --model.pooled-projection-dim=768
    --model.pooled-cache-path=clip_pooled_cache.pt
    --model.no-rtc
    --model.max-delay=8
)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PSI0_REPO"
if [ -n "$DRY_RUN" ]; then
    exec "$PSI0_REPO/.venv-psi/bin/python" "$HERE/dryrun_pack.py" "${args[@]}"
fi
if [ -n "${CHECK_INIT:-}" ]; then
    exec "$PSI0_REPO/.venv-psi/bin/python" "$HERE/check_init.py" "${args[@]}"
fi
exec "$PSI0_REPO/.venv-psi/bin/torchrun" --nnodes=1 --nproc_per_node=1 --master_port="${MASTER_PORT:-29611}" \
    "$HERE/train_sdpa.py" "${args[@]}"
