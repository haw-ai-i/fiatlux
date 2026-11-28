#!/bin/bash
# Stage 2: the fine-tune. The recipe is tuned for a 24 GB card -- VLM frozen, batch 8 x 2, and
# its own header notes batch 16 OOMs there -- so an A30, which is also the only card on the
# non-preemptible `gpu` partition that fits. The H100/H200s live on `kill-shared`, where a job
# can be preempted mid-run.
#SBATCH -J psi0-s06ft
#SBATCH -p gpu
#SBATCH --gres=gpu:NV-A30:1
#SBATCH -t 12:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o psi0-koa.%x.%j.out
set -eo pipefail
: "${PSI0_KOA_REPO:?set PSI0_KOA_REPO to the fiatlux checkout before sbatch}"
echo "[ft] node=$(hostname) started=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec bash "$PSI0_KOA_REPO/scripts/psi0/koa/finetune_koa.sh"
