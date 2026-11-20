#!/bin/bash
# BASELINE: LadderMan exactly as published. Their reference clip, their G1 with mitt hands, their
# default ladder (65.5 deg, 0.25 m rungs), their 7,500 iterations -- no task or algo overrides.
#
# Runs from LadderMan_stock/, a `git archive` of upstream commit 9a8d5ed with none of our files in
# it. The conda env is shared with our working copy, whose editable install points at THAT copy,
# so PYTHONPATH puts the clean tree first and the job refuses to run if the import resolves
# anywhere else. The only arguments are where to write and "no wandb".
#
# A requeue or a resubmission resumes from this run family's newest checkpoint.
#SBATCH -J lm-stock
#SBATCH -p kill-shared
#SBATCH --requeue
#SBATCH --gres=gpu:1   # pinned to nv-rtx-a4000 after submission with scontrol; the typed request is refused at submit
#SBATCH -t 3-00:00:00
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -o lm-stock.%j.out
set -eo pipefail
ROOT=/mnt/lustre/koa/scratch/$USER/ladderman
STOCK=$ROOT/LadderMan_stock
export PIP_CACHE_DIR=$ROOT/pipcache XDG_CACHE_HOME=$ROOT/cache XDG_DATA_HOME=$ROOT/share
export TMPDIR=$ROOT/tmp OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
source "$ROOT/LadderMan/thirdparty/miniconda3/etc/profile.d/conda.sh"
conda activate ladderman
cd "$STOCK"
export PYTHONPATH="$STOCK${PYTHONPATH:+:$PYTHONPATH}"

MINE=$(squeue -h -u "$USER" -t RUNNING -o '%i %j' | grep -E 'lm-stock' | grep -v "^$SLURM_JOB_ID " || true)
if [ -n "$MINE" ]; then echo "[lm] another lm-stock is RUNNING -- standing down: $MINE"; exit 0; fi

WHERE=$(python -c 'import active_adaptation, os; print(os.path.dirname(os.path.dirname(active_adaptation.__file__)))')
echo "[lm] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) $(date -u)"
echo "[lm] active_adaptation imported from $WHERE"
[ "$WHERE" = "$STOCK" ] || { echo "[lm] NOT the clean tree -- refusing to run"; exit 1; }
[ ! -e "$STOCK/active_adaptation/assets/g1_fiatlux.py" ] && [ ! -e "$STOCK/active_adaptation/envs/ladder_fiatlux.py" ] \
  || { echo "[lm] our files are in the clean tree -- refusing to run"; exit 1; }

OUT="$ROOT/teacher/stock-$SLURM_JOB_ID"
RESUME=()
CKPT=$(ls -t "$ROOT"/teacher/stock-*/checkpoints/checkpoint_[0-9]*.pt 2>/dev/null | head -1 || true)
if [ -n "$CKPT" ]; then
  # train.py counts max_iters from the restored iteration, so a resumed run is told only what is left of 7,500
  DONE=$(basename "$CKPT" .pt | sed 's/checkpoint_//'); LEFT=$((7500 - DONE))
  echo "[lm] resuming from $CKPT: $DONE done, $LEFT left"
  [ "$LEFT" -gt 0 ] || { echo "[lm] nothing left to train"; exit 0; }
  RESUME=(checkpoint_path="$CKPT" algo.max_iters="$LEFT")
else
  echo "[lm] no checkpoint found, starting fresh (their default: 7,500 iterations)"
fi

exec python scripts/train.py algo=teacher task=teacher "${RESUME[@]}" \
    hydra.run.dir="$OUT" \
    wandb.mode=disabled
